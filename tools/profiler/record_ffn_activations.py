# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Record FFN activation tensors during vLLM inference.

For each transformer layer this script captures:

  * ``gate_up_proj`` **input**  – hidden states entering the MLP,
    shape ``[num_tokens, hidden_size]``

  * ``gate_up_proj`` **output (gate half)** – the raw gate logits *before*
    the SiLU activation, shape ``[num_tokens, intermediate_size]``.
    Stored as ``gate_raw``.  SiLU(x) ≈ 0 for x ≲ −4, so this is the
    correct signal for identifying genuinely-inactive neurons.

  * Per-neuron **fraction of tokens where gate_raw > 0** (scalar per
    intermediate neuron, averaged across all tokens) for quick analysis of
    neuron utilisation.  Stored as ``neuron_activity``.

Tensors are streamed to temporary files on disk during inference so that
peak RAM is bounded to roughly one batch's activations regardless of how
many prompts are processed.  The model is deleted before the final
concatenation step to reclaim its memory.  Results are written to a
NumPy ``.npz`` archive with keys of the form::

    layer<N>/<quantity>

where ``<quantity>`` is one of ``gate_up_input``, ``gate_raw``,
``neuron_activity``.

Usage::

    python tools/profiler/record_ffn_activations.py \\
        --model ibm-granite/granite-4.2-3b \\
        --prompts "The capital of France is" "Once upon a time" \\
        --output ffn_activations.npz

    # Record only layers 0 and 15 to keep the file small:
    python tools/profiler/record_ffn_activations.py \\
        --model ibm-granite/granite-4.2-3b \\
        --prompts "Hello world" \\
        --layers 0 15 \\
        --output ffn_activations.npz

    # Use a calibration set file, sample 128 random chunks:
    python tools/profiler/record_ffn_activations.py \\
        --model ibm-granite/granite-4.2-3b \\
        --calibration-set bartowski-imatrix-v5-semantic.txt \\
        --num-chunks 128 \\
        --output ffn_activations.npz

    # Load and inspect results:
    import numpy as np
    data = np.load("ffn_activations.npz")
    print(data["layer0/neuron_activity"].shape)  # (intermediate_size,)
"""

import argparse
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Hook state
# ---------------------------------------------------------------------------


class _FFNRecorder:
    """Streams FFN activation tensors to temp files during inference.

    Each tensor chunk (one per hook call) is immediately written to a
    per-layer temporary file via ``np.save``, so peak RAM is bounded to
    roughly one batch's activations.  Call ``to_numpy()`` after inference
    (and after deleting the model) to concatenate the temp files into
    the final arrays.

    Args:
        save_tensors: When True (default), ``gate_up_input`` and
            ``down_input`` are streamed to disk and concatenated at the end.
            When False, only the running ``neuron_activity`` mean is kept.
    """

    def __init__(self, save_tensors: bool = True) -> None:
        self.save_tensors = save_tensors

        # Temp files: layer_idx -> {key: [NamedTemporaryFile, ...]}
        self._tmp: dict[int, dict[str, list]] = {}

        # Online mean of (gate_raw > 0) per neuron: (sum_positive, count)
        self._act_sum: dict[int, np.ndarray] = {}
        self._act_count: dict[int, int] = {}

    def _append(self, layer_idx: int, key: str, arr: np.ndarray) -> None:
        """Write arr to a new temp file, remembered for later concatenation."""
        f = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
        np.save(f, arr)
        f.close()
        self._tmp.setdefault(layer_idx, {}).setdefault(key, []).append(f.name)

    # -- hook factories ------------------------------------------------------

    def gate_up_pre_hook(self, layer_idx: int):
        def _hook(module, args, kwargs):
            if self.save_tensors:
                x = args[0].detach().float().cpu().numpy()
                self._append(layer_idx, "gate_up_input", x)

        return _hook

    def gate_up_post_hook(self, layer_idx: int):
        def _hook(module, args, output):
            # vLLM's MergedColumnParallelLinear returns (tensor, bias_or_None).
            # Unwrap to get the actual weight-matrix output.
            tensor = output[0] if isinstance(output, tuple) else output
            # tensor shape: [num_tokens, 2 * intermediate_size];
            # the gate half is the first intermediate_size columns.
            intermediate = tensor.shape[-1] // 2
            gate = tensor[..., :intermediate].detach().float().cpu().numpy()
            if self.save_tensors:
                self._append(layer_idx, "gate_raw", gate)
            # Track fraction of tokens where gate logit > 0 (SiLU > 0 iff x > 0)
            positive = (gate > 0).astype(np.float32)
            if layer_idx in self._act_sum:
                self._act_sum[layer_idx] += positive.sum(axis=0)
                self._act_count[layer_idx] += gate.shape[0]
            else:
                self._act_sum[layer_idx] = positive.sum(axis=0)
                self._act_count[layer_idx] = gate.shape[0]

        return _hook

    # -- results -------------------------------------------------------------

    def save(self, output_path: Path) -> dict[str, tuple]:
        """Concatenate temp files one layer at a time and write to a .npz.

        Processes one layer at a time so peak RAM is bounded to the largest
        single layer's tensors rather than the full dataset.  Returns a
        summary dict mapping key -> shape for reporting.
        """
        import io
        import zipfile

        summary: dict[str, tuple] = {}
        all_layers = sorted(self._act_count)

        # np.savez writes a zip archive; we build it ourselves so we can
        # flush each array immediately after writing rather than holding
        # all arrays in memory simultaneously.
        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as zf:
            for layer_idx in all_layers:
                prefix = f"layer{layer_idx}"

                if self.save_tensors and layer_idx in self._tmp:
                    for key, paths in self._tmp[layer_idx].items():
                        arr = np.concatenate([np.load(p) for p in paths], axis=0)
                        for p in paths:
                            os.unlink(p)
                        npz_key = f"{prefix}/{key}.npy"
                        buf = io.BytesIO()
                        np.save(buf, arr)
                        zf.writestr(npz_key, buf.getvalue())
                        summary[f"{prefix}/{key}"] = arr.shape
                        del arr, buf

                # neuron_activity: fraction of tokens where gate_raw > 0
                arr = self._act_sum[layer_idx] / self._act_count[layer_idx]
                npz_key = f"{prefix}/neuron_activity.npy"
                buf = io.BytesIO()
                np.save(buf, arr)
                zf.writestr(npz_key, buf.getvalue())
                summary[f"{prefix}/neuron_activity"] = arr.shape
                del arr, buf

        return summary


# ---------------------------------------------------------------------------
# Registration helpers
# ---------------------------------------------------------------------------


def _get_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Return the list of transformer layers for supported model families."""
    # Llama / Mistral / Qwen / Gemma style
    for attr in ("model",):
        inner = getattr(model, attr, None)
        if inner is not None:
            layers = getattr(inner, "layers", None)
            if layers is not None:
                return list(layers)
    # Fallback: GPT-2 / GPT-J style
    transformer = getattr(model, "transformer", None)
    if transformer is not None:
        return list(getattr(transformer, "h", []))
    raise RuntimeError(
        "Could not locate transformer layers. "
        "Inspect model.named_modules() and pass a custom accessor."
    )


def _get_mlp(layer: torch.nn.Module) -> torch.nn.Module:
    """Return the MLP sub-module from a transformer layer."""
    for attr in ("mlp", "ffn", "feed_forward"):
        m = getattr(layer, attr, None)
        if m is not None:
            return m
    raise RuntimeError(
        f"Could not find MLP in layer {type(layer).__name__}. "
        "Check layer.named_children() and update _get_mlp()."
    )


def _get_proj(mlp: torch.nn.Module, *candidates: str) -> torch.nn.Module:
    for name in candidates:
        m = getattr(mlp, name, None)
        if m is not None:
            return m
    raise RuntimeError(
        f"Could not find projection {candidates} in {type(mlp).__name__}."
    )


def register_ffn_hooks(
    model: torch.nn.Module,
    recorder: _FFNRecorder,
    layer_indices: list[int] | None = None,
) -> list:
    """Register recording hooks on gate_up_proj.

    Args:
        model: The loaded nn.Module (e.g. ``llm.llm_engine...model``).
        recorder: The ``_FFNRecorder`` instance to collect tensors into.
        layer_indices: Which layers to instrument. ``None`` means all layers.

    Returns:
        List of ``RemovableHook`` handles; call ``.remove()`` on each when done.
    """
    layers = _get_layers(model)
    handles = []

    for i, layer in enumerate(layers):
        if layer_indices is not None and i not in layer_indices:
            continue

        mlp = _get_mlp(layer)

        gate_up = _get_proj(
            mlp, "gate_up_proj", "gate_proj", "c_fc", "dense_h_to_4h", "fc1"
        )

        handles += [
            gate_up.register_forward_pre_hook(
                recorder.gate_up_pre_hook(i), with_kwargs=True
            ),
            gate_up.register_forward_hook(recorder.gate_up_post_hook(i)),
        ]

    if not handles:
        raise RuntimeError(
            "No hooks were registered. Check --layers values or the model architecture."
        )
    return handles


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_calibration_set(path: str) -> list[str]:
    """Return all non-empty lines from a calibration-set file as chunks."""
    chunks = [
        line.rstrip("\n")
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not chunks:
        raise ValueError(f"Calibration set {path!r} contains no non-empty lines.")
    return chunks


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Record FFN activation tensors during vLLM inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", required=True, help="Model name or path.")

    prompt_group = p.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--prompts",
        nargs="+",
        help="One or more prompt strings.",
    )
    prompt_group.add_argument(
        "--calibration-set",
        metavar="FILE",
        help=(
            "Path to a calibration-set text file. "
            "Every non-empty line is treated as one chunk (prompt). "
            "Use --num-chunks to sample a random subset."
        ),
    )

    p.add_argument(
        "--num-chunks",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of chunks to randomly sample from --calibration-set. "
            "Omit (or 0) to use all chunks. Ignored when --prompts is used."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="S",
        help="Random seed for --num-chunks sampling (default: no fixed seed).",
    )
    p.add_argument(
        "--layers",
        nargs="*",
        type=int,
        default=None,
        metavar="N",
        help="Layer indices to record. Omit to record all layers.",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=1,
        help="Max new tokens to generate per prompt (default: 1).",
    )
    p.add_argument(
        "--output",
        default="ffn_activations.npz",
        help="Output .npz file path (default: ffn_activations.npz).",
    )
    p.add_argument(
        "--dtype",
        default="bfloat16",
        help="Model dtype passed to LLM (default: bfloat16).",
    )
    p.add_argument(
        "--kv-cache-gb",
        type=float,
        default=1.0,
        metavar="GB",
        help="KV cache size in GiB (default: 1.0). "
        "Kept small on CPU/Mac to avoid OOM; increase for longer sequences.",
    )
    p.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        metavar="N",
        help="Maximum sequence length (default: 2048). "
        "Reducing this shrinks the KV cache and lowers RAM usage.",
    )
    p.add_argument(
        "--chunk-len",
        type=int,
        default=512,
        metavar="N",
        help="Maximum number of tokens per prompt/chunk (default: 512). "
        "Chunks that exceed this limit are truncated before inference. "
        "Set to 0 to disable truncation.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        metavar="N",
        help="Number of prompts to pass to generate() at once (default: 8). "
        "Larger values improve throughput but increase peak RAM usage.",
    )
    p.add_argument(
        "--no-save-tensors",
        dest="save_tensors",
        action="store_false",
        default=True,
        help="Skip saving full per-token tensors (gate_up_input, gate_raw); "
        "record only neuron_activity. Reduces RAM usage for large calibration sets.",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    # Import here so the module is importable without vllm installed
    from vllm import LLM, SamplingParams

    # Disable multiprocessing so the model lives in-process and is reachable
    # via llm.llm_engine.model_executor.  In multiprocess mode (the v1 default)
    # the model runs in a subprocess and cannot be accessed directly.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    # Resolve prompts from either --prompts or --calibration-set
    if args.prompts is not None:
        prompts = args.prompts
    else:
        chunks = _load_calibration_set(args.calibration_set)
        n = args.num_chunks or 0
        if n and n < len(chunks):
            rng = random.Random(args.seed)
            prompts = rng.sample(chunks, n)
            print(
                f"Sampled {n} of {len(chunks)} chunks from "
                f"{args.calibration_set!r} (seed={args.seed}).",
                file=sys.stderr,
            )
        else:
            prompts = chunks
            print(
                f"Using all {len(prompts)} chunks from {args.calibration_set!r}.",
                file=sys.stderr,
            )

    kv_bytes = int(args.kv_cache_gb * 1024**3)
    print(
        f"Loading model: {args.model}  "
        f"(KV cache: {args.kv_cache_gb:.1f} GiB, "
        f"max_model_len: {args.max_model_len})",
        file=sys.stderr,
    )

    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        enforce_eager=True,  # Required: CUDA graphs bypass Python hooks
        kv_cache_memory_bytes=kv_bytes,
        max_model_len=args.max_model_len,
    )

    # Reach the underlying nn.Module
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model

    # Truncate prompts to --chunk-len tokens so they fit within max_model_len.
    # The tokenizer is available once the LLM is loaded.
    if args.chunk_len > 0:
        tokenizer = llm.get_tokenizer()
        truncated = 0
        capped = []
        for p in prompts:
            ids = tokenizer.encode(p)
            if len(ids) > args.chunk_len:
                p = tokenizer.decode(ids[: args.chunk_len], skip_special_tokens=True)
                truncated += 1
            capped.append(p)
        if truncated:
            print(
                f"Truncated {truncated} of {len(capped)} prompts to "
                f"{args.chunk_len} tokens.",
                file=sys.stderr,
            )
        prompts = capped

    recorder = _FFNRecorder(save_tensors=args.save_tensors)
    handles = register_ffn_hooks(model, recorder, layer_indices=args.layers)

    layer_desc = f"layers {args.layers}" if args.layers is not None else "all layers"
    print(
        f"Registered hooks on {len(handles) // 2} layer(s) ({layer_desc}).",
        file=sys.stderr,
    )
    # Two handles per layer: pre-hook (gate_up_input) + post-hook (gate_raw)

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
    )

    # Process prompts in batches to bound peak RAM.  The recorder accumulates
    # statistics across all batches; neuron_activity is the mean over all tokens.
    batch_size = max(1, args.batch_size)
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        llm.generate(batch, sampling_params, use_tqdm=False)
        print(
            f"  {min(start + batch_size, len(prompts))}/{len(prompts)} prompts processed",
            end="\r",
            file=sys.stderr,
        )
    print(file=sys.stderr)  # newline after progress

    for h in handles:
        h.remove()

    # Delete the model before collecting results — frees the largest block
    # of RAM before the final concatenation and write step.
    del model, llm

    output_path = Path(args.output)
    summary = recorder.save(output_path)
    print(f"Saved {len(summary)} arrays to {output_path}", file=sys.stderr)

    for key, shape in sorted(summary.items()):
        print(f"  {key}: shape={shape}")


if __name__ == "__main__":
    main()
