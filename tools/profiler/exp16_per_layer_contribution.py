# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 16 – per-layer perturbation contribution.

Motivation
----------
Experiment 14 showed that approximating all 40 layers simultaneously produces
a 48% top-1 perturbation rate at 10% hot channels.  This experiment identifies
**which layers are responsible** for the bulk of that perturbation by:

  1. Single-layer sweep: patch each layer independently and measure the
     end-to-end top-1 perturbation caused by that layer alone.
  2. Cumulative sweep: patch the top-k highest-contribution layers simultaneously
     (ranked by single-layer perturbation) and show how e2e perturbation scales.

Both sweeps use the same HybridMLP from exp14 (ternary gate α=0.75, full up,
full down, routing on |gate_approx[t]|, 10% hot channels).

Usage::

    python tools/profiler/exp16_per_layer_contribution.py \\
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
        description="Experiment 16: per-layer perturbation contribution.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", default="ibm-granite/granite-4.2-3b")
    p.add_argument("--calibration-set", default="bartowski-imatrix-v5-semantic.txt")
    p.add_argument("--num-prompts", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=1)
    p.add_argument("--tau-gate", type=float, default=0.75)
    p.add_argument("--hot-fraction", type=float, default=0.10,
                   help="Hot-channel fraction for the single-layer sweep (default 10%%).")
    p.add_argument(
        "--cumulative-ks", nargs="+", type=int,
        default=[1, 2, 4, 8, 16, 24, 32, 40],
        metavar="K",
        help="Layer counts for cumulative sweep (default: 1 2 4 8 16 24 32 40).",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# HybridMLP (identical to exp14, routing=current only)
# ---------------------------------------------------------------------------

class HybridMLP:
    def __init__(self, mlp, tau_gate: float, mean_abs_gate: float, k_hot: int):
        self._mlp     = mlp
        self._k_hot   = k_hot
        W_fused = mlp.gate_up_proj.weight.detach()
        I = W_fused.shape[0] // 2
        W_gate_f = W_fused[:I].float()
        T_gate = (W_gate_f.sign() * (W_gate_f.abs() >= tau_gate) * mean_abs_gate
                  ).to(W_fused.dtype)
        self._T_gate = T_gate
        self._I      = I

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        xf = x.float()
        W_fused = self._mlp.gate_up_proj.weight.detach().float()
        I = self._I
        W_gate = W_fused[:I]
        W_up   = W_fused[I:]
        W_down = self._mlp.down_proj.weight.detach().float()
        T_gate_f    = self._T_gate.float()
        gate_approx = xf @ T_gate_f.T
        gate_full   = xf @ W_gate.T
        up_full     = xf @ W_up.T
        hot           = top_k_mask(gate_approx.abs(), self._k_hot)
        gate_hybrid   = torch.where(hot, gate_full, gate_approx)
        swiglu_hybrid = F.silu(gate_hybrid) * up_full
        return (swiglu_hybrid @ W_down.T).to(orig_dtype)


# ---------------------------------------------------------------------------
# Norm-hook capture
# ---------------------------------------------------------------------------

class NormCapture:
    def __init__(self, W_U: torch.Tensor):
        self._W_U = W_U.float()
        self.ids: list[int] = []
        self._handle = None

    def attach(self, norm_module) -> None:
        self._handle = norm_module.register_forward_hook(self._hook)

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _hook(self, module, args, output) -> None:
        logits = output.float() @ self._W_U.T
        self.ids.extend(logits.argmax(dim=-1).cpu().tolist())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_prompts(path: str, n: int) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip()][:n]


def _run(llm, prompts: list[str], W_U: torch.Tensor) -> np.ndarray:
    from vllm import SamplingParams
    cap = NormCapture(W_U)
    cap.attach(
        llm.llm_engine.model_executor.driver_worker.model_runner.model.model.norm)
    llm.generate(prompts,
                 SamplingParams(max_tokens=1, temperature=0.0),
                 use_tqdm=False)
    cap.detach()
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

    W_U   = model.lm_head.weight.detach().float()
    layers = model.model.layers
    n_layers = len(layers)
    I = layers[0].mlp.gate_up_proj.weight.shape[0] // 2
    k_hot = max(1, int(args.hot_fraction * I))

    tau_per_layer:      list[float] = []
    mean_abs_per_layer: list[float] = []
    for layer in layers:
        W_gate = layer.mlp.gate_up_proj.weight.detach().float()
        mean_abs = float(W_gate[:I].abs().mean())
        tau_per_layer.append(args.tau_gate * mean_abs)
        mean_abs_per_layer.append(mean_abs)
        del W_gate

    orig_forwards = [layer.mlp.forward for layer in layers]

    def make_hybrid(li: int) -> HybridMLP:
        return HybridMLP(
            mlp=layers[li].mlp,
            tau_gate=tau_per_layer[li],
            mean_abs_gate=mean_abs_per_layer[li],
            k_hot=k_hot,
        )

    # --- Baseline ---
    print("Baseline pass...", file=sys.stderr)
    baseline = _run(llm, prompts, W_U)
    n_tok = len(baseline)
    print(f"  {n_tok} prefill-token predictions captured.", file=sys.stderr)

    # -----------------------------------------------------------------------
    # Part 1: single-layer sweep
    # -----------------------------------------------------------------------
    print(f"\nSingle-layer sweep ({n_layers} passes)...", file=sys.stderr)
    perturb_by_layer = np.zeros(n_layers, dtype=np.float32)
    for li in range(n_layers):
        layers[li].mlp.forward = make_hybrid(li)
        ids = _run(llm, prompts, W_U)[:n_tok]
        perturb_by_layer[li] = float((ids != baseline).mean())
        layers[li].mlp.forward = orig_forwards[li]
        print(f"  layer {li:2d}  perturb={perturb_by_layer[li]:.4f}",
              file=sys.stderr)

    ranked_layers = list(np.argsort(perturb_by_layer)[::-1])

    # -----------------------------------------------------------------------
    # Part 2: cumulative sweep (top-k layers by single-layer contribution)
    # -----------------------------------------------------------------------
    print(f"\nCumulative sweep...", file=sys.stderr)
    cum_ks     = [k for k in args.cumulative_ks if k <= n_layers]
    cum_perturb: dict[int, float] = {}
    for k in cum_ks:
        active = ranked_layers[:k]
        for li in active:
            layers[li].mlp.forward = make_hybrid(li)
        ids = _run(llm, prompts, W_U)[:n_tok]
        p = float((ids != baseline).mean())
        cum_perturb[k] = p
        for li in active:
            layers[li].mlp.forward = orig_forwards[li]
        print(f"  top-{k:2d} layers  perturb={p:.4f}", file=sys.stderr)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n--- Experiment 16 Results (hot={args.hot_fraction*100:.0f}%,"
          f" tau_gate={args.tau_gate}) ---\n")

    print("Single-layer perturbation (approximating one layer at a time):")
    print(f"  {'Layer':>5}  {'Perturb':>8}")
    for li in range(n_layers):
        print(f"  {li:5d}  {perturb_by_layer[li]:8.4f}")

    print("\nLayers ranked by single-layer contribution (highest first):")
    print("  " + ", ".join(str(li) for li in ranked_layers[:20]) + " ...")

    print("\nCumulative perturbation (top-k layers by contribution):")
    print(f"  {'k':>4}  {'Layers (top-k)':>40}  {'Perturb':>8}  {'Match':>7}")
    for k in cum_ks:
        top_k_str = str(ranked_layers[:k])
        print(f"  {k:4d}  {str(ranked_layers[:k]):<40}  "
              f"{cum_perturb[k]:8.4f}  {1-cum_perturb[k]:7.4f}")

    print(f"\n  (All 40 layers, exp14 reference: perturb=0.484, match=0.516)")


if __name__ == "__main__":
    main()
