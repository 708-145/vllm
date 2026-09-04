# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Record FFN (gate/up projection) activation tensors during vLLM inference.

For each transformer layer this script captures:

  * ``gate_up_proj`` **input**  – hidden states entering the MLP
    shape ``[num_tokens, hidden_size]``

  * ``gate_up_proj`` **raw output** – pre-activation concatenation
    ``[gate | up]``, shape ``[num_tokens, 2 * intermediate_size]``

  * ``down_proj`` **input** – post-SiluAndMul activations feeding the
    down-projection, shape ``[num_tokens, intermediate_size]``.
    This is the *hidden-dimension activity* vector: high values indicate
    neurons that fired strongly for the given input tokens.

  * Per-neuron **mean absolute activity** (scalar per intermediate neuron,
    averaged across all tokens) for quick analysis of neuron utilisation.

All tensors are moved to CPU before saving.  Results are written to a
NumPy ``.npz`` archive with keys of the form::

    layer<N>/<quantity>

where ``<quantity>`` is one of ``gate_up_input``, ``gate_raw``,
``up_raw``, ``down_input``, ``neuron_activity``.

Usage::

    python tools/profiler/record_ffn_activations.py \\
        --model meta-llama/Llama-3.2-1B \\
        --prompts "The capital of France is" "Once upon a time" \\
        --output ffn_activations.npz

    # Record only layers 0 and 15 to keep the file small:
    python tools/profiler/record_ffn_activations.py \\
        --model meta-llama/Llama-3.2-1B \\
        --prompts "Hello world" \\
        --layers 0 15 \\
        --output ffn_activations.npz

    # Load and inspect results:
    import numpy as np
    data = np.load("ffn_activations.npz")
    print(data["layer0/neuron_activity"].shape)  # (intermediate_size,)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Hook state
# ---------------------------------------------------------------------------


class _FFNRecorder:
    """Accumulates FFN activation tensors across all forward calls."""

    def __init__(self) -> None:
        # Maps layer_idx -> list of tensors collected across calls
        self._gate_up_inputs: dict[int, list[torch.Tensor]] = {}
        self._gate_up_outputs: dict[int, list[torch.Tensor]] = {}
        self._down_inputs: dict[int, list[torch.Tensor]] = {}

    # -- hook factories ------------------------------------------------------

    def gate_up_pre_hook(self, layer_idx: int):
        def _hook(module, args, kwargs):
            x = args[0].detach().cpu()
            self._gate_up_inputs.setdefault(layer_idx, []).append(x)

        return _hook

    def gate_up_post_hook(self, layer_idx: int):
        def _hook(module, args, output):
            # output is (tensor, bias_or_None) for vLLM parallel linears
            tensor = output[0] if isinstance(output, tuple) else output
            self._gate_up_outputs.setdefault(layer_idx, []).append(
                tensor.detach().cpu()
            )

        return _hook

    def down_pre_hook(self, layer_idx: int):
        def _hook(module, args, kwargs):
            x = args[0].detach().cpu()
            self._down_inputs.setdefault(layer_idx, []).append(x)

        return _hook

    # -- results -------------------------------------------------------------

    def to_numpy(self) -> dict[str, np.ndarray]:
        """Concatenate all collected tensors and return a flat key→array dict."""
        results: dict[str, np.ndarray] = {}
        all_layers = sorted(
            set(self._gate_up_inputs)
            | set(self._gate_up_outputs)
            | set(self._down_inputs)
        )

        for layer_idx in all_layers:
            prefix = f"layer{layer_idx}"

            if layer_idx in self._gate_up_inputs:
                cat = torch.cat(self._gate_up_inputs[layer_idx], dim=0)
                results[f"{prefix}/gate_up_input"] = cat.float().numpy()

            if layer_idx in self._gate_up_outputs:
                cat = torch.cat(self._gate_up_outputs[layer_idx], dim=0)
                arr = cat.float().numpy()
                mid = arr.shape[1] // 2
                results[f"{prefix}/gate_raw"] = arr[:, :mid]
                results[f"{prefix}/up_raw"] = arr[:, mid:]

            if layer_idx in self._down_inputs:
                cat = torch.cat(self._down_inputs[layer_idx], dim=0)
                arr = cat.float().numpy()
                results[f"{prefix}/down_input"] = arr
                # Mean absolute activation across tokens: shape (intermediate,)
                results[f"{prefix}/neuron_activity"] = np.abs(arr).mean(axis=0)

        return results


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
    """Register recording hooks on gate_up_proj and down_proj.

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
        down = _get_proj(mlp, "down_proj", "c_proj", "dense_4h_to_h", "fc2")

        handles += [
            gate_up.register_forward_pre_hook(
                recorder.gate_up_pre_hook(i), with_kwargs=True
            ),
            gate_up.register_forward_hook(recorder.gate_up_post_hook(i)),
            down.register_forward_pre_hook(recorder.down_pre_hook(i), with_kwargs=True),
        ]

    if not handles:
        raise RuntimeError(
            "No hooks were registered. Check --layers values or the model architecture."
        )
    return handles


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Record FFN activation tensors during vLLM inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", required=True, help="Model name or path.")
    p.add_argument(
        "--prompts",
        nargs="+",
        required=True,
        help="One or more prompt strings.",
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
        default="auto",
        help="Model dtype passed to LLM (default: auto).",
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
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    # Import here so the module is importable without vllm installed
    from vllm import LLM, SamplingParams

    # Disable multiprocessing so the model lives in-process and is reachable
    # via llm.llm_engine.model_executor.  In multiprocess mode (the v1 default)
    # the model runs in a subprocess and cannot be accessed directly.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

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

    recorder = _FFNRecorder()
    handles = register_ffn_hooks(model, recorder, layer_indices=args.layers)

    layer_desc = f"layers {args.layers}" if args.layers is not None else "all layers"
    print(
        f"Registered hooks on {len(handles) // 3} layer(s) ({layer_desc}).",
        file=sys.stderr,
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
    )
    llm.generate(args.prompts, sampling_params)

    for h in handles:
        h.remove()

    results = recorder.to_numpy()

    output_path = Path(args.output)
    np.savez(output_path, **results)
    print(f"Saved {len(results)} arrays to {output_path}", file=sys.stderr)

    # Print a brief summary
    for key, arr in sorted(results.items()):
        print(f"  {key}: shape={arr.shape} dtype={arr.dtype}")


if __name__ == "__main__":
    main()
