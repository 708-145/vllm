# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 14 – end-to-end top-1 perturbation rate.

Motivation
----------
Experiment 13 measured top-1 preservation per MLP layer in isolation.
Per-layer values were low (~0.06 at 10% hot) because a single-layer
perturbation rarely changes the final argmax.

This experiment measures the *end-to-end* top-1 perturbation rate: the
fraction of tokens where the final next-token prediction changes when the
hybrid MLP scheme runs across **all 40 layers simultaneously**.

Method
------
For each (config, hot-fraction) pair we run llm.generate twice:

  Pass 0 — baseline:
    Normal inference. A hook on model.model.norm captures final hidden
    states; W_U projection gives the per-token argmax (= generated token).

  Pass N — hybrid:
    Each layer's mlp.forward is monkey-patched to HybridMLP.
    The same norm hook captures the perturbed hidden states.

top1_match     = fraction of tokens where hybrid argmax == baseline argmax
top1_perturb   = 1 − top1_match  (tokens where prediction changes)

Configs tested (α_gate=0.75, full-prec up, full W_down):
  A. exp10_current : route on |gate_approx[t]|       (current token)
  B. exp11_proxy   : route on |gate_approx[t-1]|     (proxy prior)

HybridMLP:
  1. gate_approx = T(W_gate, τ) @ x               ternary gate
  2. up_full     = W_up @ x                        exact up
  3. hot = top-k by routing signal
  4. gate_hybrid[hot]  = W_gate[hot] @ x
     gate_hybrid[cold] = gate_approx[cold]
  5. swiglu = silu(gate_hybrid) * up_full
  6. out    = W_down @ swiglu                      exact down

Note: vLLM runs on the CPU backend on this machine (no Triton/CUDA).
Each generate pass over 8 prompts × 32 tokens takes ~35 s on CPU.
Default budget: 8 prompts, 32 tokens, 3 hot fractions → ~7 passes ≈ 4 min.

Usage::

    python tools/profiler/exp14_e2e_top1.py \\
        --model ibm-granite/granite-4.2-3b \\
        --calibration-set bartowski-imatrix-v5-semantic.txt
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def top_k_mask(scores: torch.Tensor, k: int) -> torch.BoolTensor:
    idx = torch.topk(scores, k, dim=1, largest=True, sorted=False).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Experiment 14: end-to-end top-1 perturbation rate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", default="ibm-granite/granite-4.2-3b")
    p.add_argument("--calibration-set", default="bartowski-imatrix-v5-semantic.txt")
    p.add_argument("--num-prompts", type=int, default=8,
                   help="Prompts per pass (CPU is slow; default 8).")
    p.add_argument("--max-tokens", type=int, default=32,
                   help="Max generated tokens per prompt (default 32).")
    p.add_argument("--tau-gate", type=float, default=0.75)
    p.add_argument(
        "--hot-fractions", nargs="+", type=float,
        default=[0.05, 0.10, 0.20],
        metavar="F",
        help="Hot-channel fractions to sweep (default: 5%% 10%% 20%%).",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Hybrid MLP
# ---------------------------------------------------------------------------

class HybridMLP:
    """Callable replacement for GraniteMLP.forward.

    Holds a reference to the original mlp plus a scaled bfloat16 T_gate.
    No full weight copies stored — weights are read from the model at call
    time (bf16 → float32) to avoid OOM.

    T_gate is the sign-ternary proxy SCALED by mean(|W_gate|) so that
    gate_approx = T_gate_scaled @ x has the same expected magnitude as the
    true gate, making it safe to use as the cold-channel gate value.
    """

    def __init__(
        self,
        mlp,                    # GraniteMLP — weights accessed via .weight
        tau_gate: float,        # raw threshold = α × mean|W_gate|
        mean_abs_gate: float,   # mean(|W_gate|) for scaling T_gate
        k_hot: int,
        routing: str,           # "current" | "proxy_prior"
    ):
        self._mlp          = mlp
        self._routing      = routing
        self._k_hot        = k_hot
        self._mean_abs     = mean_abs_gate
        W_fused = mlp.gate_up_proj.weight.detach()   # (2I, H) bf16
        I = W_fused.shape[0] // 2
        W_gate_f = W_fused[:I].float()
        # Scale ternary mask so cold-channel gate_approx ≈ true gate magnitude
        T_gate = (W_gate_f.sign() * (W_gate_f.abs() >= tau_gate) * mean_abs_gate
                  ).to(W_fused.dtype)
        self._T_gate = T_gate   # (I, H) bf16 — one copy
        self._I      = I
        self._prior: torch.Tensor | None = None

    def reset_prior(self) -> None:
        self._prior = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        xf = x.float()

        # Read weights from already-loaded model (bf16 → float32, no extra storage)
        W_fused = self._mlp.gate_up_proj.weight.detach().float()  # (2I, H)
        I = self._I
        W_gate = W_fused[:I]
        W_up   = W_fused[I:]
        W_down = self._mlp.down_proj.weight.detach().float()      # (H, I)

        T_gate_f    = self._T_gate.float()
        gate_approx = xf @ T_gate_f.T   # (T, I)  — scaled ternary approximation
        gate_full   = xf @ W_gate.T     # (T, I)  — exact
        up_full     = xf @ W_up.T       # (T, I)  — exact

        signal = gate_approx.abs() if (
            self._routing == "current" or self._prior is None
        ) else self._prior.abs()

        hot           = top_k_mask(signal, self._k_hot)
        gate_hybrid   = torch.where(hot, gate_full, gate_approx)
        swiglu_hybrid = F.silu(gate_hybrid) * up_full
        out           = (swiglu_hybrid @ W_down.T).to(orig_dtype)

        self._prior = gate_approx.detach()
        return out


# ---------------------------------------------------------------------------
# Norm-hook capture
# ---------------------------------------------------------------------------

class NormCapture:
    """Captures per-token argmax by hooking model.model.norm output."""

    def __init__(self, W_U: torch.Tensor):
        self._W_U = W_U.float()   # (vocab, H)
        self.ids: list[int] = []
        self._handle = None

    def attach(self, norm_module) -> None:
        self._handle = norm_module.register_forward_hook(self._hook)

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _hook(self, module, args, output) -> None:
        logits = output.float() @ self._W_U.T   # (T, vocab)
        self.ids.extend(logits.argmax(dim=-1).cpu().tolist())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_prompts(path: str, n: int) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip()][:n]


def _run(llm, prompts: list[str], W_U: torch.Tensor) -> np.ndarray:
    """Run with max_tokens=1 and return per-prompt-token argmax.

    With max_tokens=1 each prompt gets exactly one decode step, so the norm
    hook fires for N_prefill + N_prompts total tokens.  We discard the last
    N_prompts entries (decode steps) and keep only the prefill predictions.
    Token count is stable across passes regardless of chunked-prefill ordering
    because each run sees the same prompts.
    """
    from vllm import SamplingParams
    cap = NormCapture(W_U)
    cap.attach(
        llm.llm_engine.model_executor.driver_worker.model_runner.model.model.norm)
    llm.generate(prompts,
                 SamplingParams(max_tokens=1, temperature=0.0),
                 use_tqdm=False)
    cap.detach()
    # Drop the N_prompts decode-step entries at the end
    ids = cap.ids
    n_decode = len(prompts)
    prefill_ids = ids[:-n_decode] if n_decode < len(ids) else ids
    return np.array(prefill_ids, dtype=np.int32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    args = _parse_args(argv)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    prompts = _load_prompts(args.calibration_set, args.num_prompts)
    print(f"Loaded {len(prompts)} prompts.", file=sys.stderr)

    from vllm import LLM
    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              kv_cache_memory_bytes=int(0.5 * 1024**3), max_model_len=512,
              enable_prefix_caching=False)
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model

    W_U = model.lm_head.weight.detach().float()

    # Precompute per-layer tau thresholds and mean|W_gate| (no weight copies kept)
    tau_per_layer: list[float] = []
    mean_abs_per_layer: list[float] = []
    for layer in model.model.layers:
        W_gate = layer.mlp.gate_up_proj.weight.detach().float()
        I = W_gate.shape[0] // 2
        mean_abs = float(W_gate[:I].abs().mean())
        tau_per_layer.append(args.tau_gate * mean_abs)
        mean_abs_per_layer.append(mean_abs)
        del W_gate

    orig_forwards = [layer.mlp.forward for layer in model.model.layers]

    # Baseline
    print("Baseline pass...", file=sys.stderr)
    baseline = _run(llm, prompts, W_U)
    n_tok = len(baseline)
    print(f"  {n_tok} prefill-token predictions captured.", file=sys.stderr)

    configs = ["exp10_current", "exp11_proxy"]
    routing_map = {"exp10_current": "current", "exp11_proxy": "proxy_prior"}
    results: dict[str, dict[float, float]] = {c: {} for c in configs}

    n_passes = len(configs) * len(args.hot_fractions)
    pass_idx  = 0
    for cfg in configs:
        for frac in args.hot_fractions:
            pass_idx += 1
            hybrids: list[HybridMLP] = []
            I = model.model.layers[0].mlp.gate_up_proj.weight.shape[0] // 2
            for li, layer in enumerate(model.model.layers):
                h = HybridMLP(
                    mlp=layer.mlp,
                    tau_gate=tau_per_layer[li],
                    mean_abs_gate=mean_abs_per_layer[li],
                    k_hot=max(1, int(frac * I)),
                    routing=routing_map[cfg],
                )
                hybrids.append(h)
                layer.mlp.forward = h

            for h in hybrids:
                h.reset_prior()

            print(f"  [{pass_idx}/{n_passes}] {cfg}  hot={frac*100:.0f}%...",
                  end="  ", file=sys.stderr, flush=True)
            hybrid_ids = _run(llm, prompts, W_U)
            hybrid = hybrid_ids[:n_tok]
            match = float((hybrid == baseline).mean())
            results[cfg][frac] = match
            print(f"match={match:.4f}  perturb={1-match:.4f}", file=sys.stderr)

            for li, layer in enumerate(model.model.layers):
                layer.mlp.forward = orig_forwards[li]
            del hybrids

    # Summary
    print("\n--- Experiment 14 Results ---\n")
    hdr = (f"  {'config':<22}"
           + "".join(f"  {f*100:>5.0f}%" for f in args.hot_fractions))

    print("End-to-end top-1 match rate  (↑ better, 1.0 = identical to baseline):")
    print(hdr)
    for cfg in configs:
        row = f"  {cfg:<22}" + "".join(
            f"  {results[cfg][frac]:>6.4f}" for frac in args.hot_fractions)
        print(row)

    print("\nEnd-to-end top-1 perturbation rate  (↓ better, 0.0 = no change):")
    print(hdr)
    for cfg in configs:
        row = f"  {cfg:<22}" + "".join(
            f"  {1 - results[cfg][frac]:>6.4f}" for frac in args.hot_fractions)
        print(row)

    print("\nΔ match exp11_proxy − exp10_current  (+ = proxy better):")
    row = f"  {'':22}" + "".join(
        f"  {results['exp11_proxy'][f] - results['exp10_current'][f]:>+6.4f}"
        for f in args.hot_fractions)
    print(row)


if __name__ == "__main__":
    main()
