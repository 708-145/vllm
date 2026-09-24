# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 24 – E5M3 threshold sweep T=0.2..0.8 with per-layer hot% monitoring.

Motivation
----------
Exp23 found T=0.5 as the best threshold (match=0.736) but only sampled 5 points.
This experiment sweeps T ∈ {0.20, 0.25, ..., 0.80} (13 points) to find the
optimum more precisely, and records the mean hot% per layer at each threshold to:

  1. Identify whether some layers have near-zero hot% (dead layers) at tight T.
  2. Determine whether a per-layer minimum hot fraction improves quality.

Two variants are compared at each T:
  A. Pure threshold  : hot = |gate_approx| > T * mean(|gate_approx|)
  B. Threshold+floor : hot = A ∪ top-floor% channels (guarantees minimum activity)

floor = 2% of channels (164 of 8192) — a conservative minimum to keep all
layers in the loop.

Usage::

    python tools/profiler/exp24_threshold_sweep.py \\
        --model ibm-granite/granite-4.2-3b \\
        --calibration-set bartowski-imatrix-v5-semantic.txt
"""

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEV = _device()
BLOCK_SIZE = 8
_M3_FRAC_LOG2 = torch.log2(1.0 + torch.arange(8).float() / 8.0)


# ---------------------------------------------------------------------------
# E5M3 encoding  (identical to exp22/23)
# ---------------------------------------------------------------------------

def _floor_eps(W: torch.Tensor, pct: float = 1.0) -> torch.Tensor:
    flat = W.abs().reshape(-1)
    k = max(1, int(len(flat) * pct / 100.0))
    return flat.kthvalue(k).values.clamp(min=1e-9)


def build_e5m3_encoded(W: torch.Tensor, B: int = BLOCK_SIZE) -> torch.Tensor:
    O, I = W.shape
    eps = _floor_eps(W)
    pad = (B - I % B) % B
    Wp  = F.pad(W, (0, pad)) if pad else W
    W_b = Wp.reshape(-1, B)
    wa     = W_b.abs().clamp(min=eps)
    tilt   = torch.log1p(wa / eps)
    log2_s = (tilt * torch.log2(wa)).sum(1) / tilt.sum(1).clamp(min=1e-9)
    e      = log2_s.floor().to(torch.int32)
    frac   = log2_s - e.float()
    m_lut  = _M3_FRAC_LOG2.to(frac.device)
    m_best = (frac.unsqueeze(1) - m_lut).abs().argmin(1)
    scales = (2.0 ** (e.float() + m_lut[m_best])).clamp(min=1e-9)
    n_blk  = Wp.shape[1] // B
    s_exp  = scales.reshape(O, n_blk).unsqueeze(2).expand(O, n_blk, B).reshape(O, Wp.shape[1])
    return (Wp.sign() * s_exp)[:, :I].contiguous()


# ---------------------------------------------------------------------------
# HybridMLP with hot-fraction tracking
# ---------------------------------------------------------------------------

class HybridMLPThresholdFloor:
    """E5M3 gate, threshold routing with optional minimum hot floor.

    Records per-call mean hot fraction for monitoring.
    """

    def __init__(self, mlp, threshold: float, floor_frac: float):
        """
        Args:
            threshold:  T — hot = |gate_approx| > T * mean(|gate_approx|)
            floor_frac: minimum hot fraction guaranteed per token (0 = pure threshold)
        """
        self._mlp       = mlp
        self._threshold = threshold
        self._floor_k   = 0   # set after knowing I

        W_fused = mlp.gate_up_proj.weight.detach().float()
        I = W_fused.shape[0] // 2
        self._I          = I
        self._floor_k    = max(0, int(floor_frac * I))
        self._W_gate_enc = build_e5m3_encoded(W_fused[:I])

        # Accumulated hot fraction stats across all calls
        self.hot_frac_sum   = 0.0
        self.hot_frac_count = 0

    def reset_stats(self) -> None:
        self.hot_frac_sum   = 0.0
        self.hot_frac_count = 0

    def mean_hot_frac(self) -> float:
        if self.hot_frac_count == 0:
            return float("nan")
        return self.hot_frac_sum / self.hot_frac_count

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        xf = x.float()
        I  = self._I

        gate_approx = xf @ self._W_gate_enc.T          # (T, I)

        # Threshold mask
        thresh = self._threshold * gate_approx.abs().mean(dim=-1, keepdim=True)
        hot    = gate_approx.abs() > thresh             # (T, I)

        # Floor: guarantee at least floor_k channels per token
        if self._floor_k > 0:
            # Union with top-floor_k channels (avoids sort by using kthvalue per token)
            # For small floor_k this is cheap; topk is O(I log floor_k)
            topk_idx = torch.topk(gate_approx.abs(), self._floor_k,
                                   dim=-1, sorted=False).indices
            floor_mask = torch.zeros_like(hot)
            floor_mask.scatter_(-1, topk_idx, True)
            hot = hot | floor_mask

        # Track mean hot fraction
        self.hot_frac_sum   += float(hot.float().mean())
        self.hot_frac_count += 1

        W_fused   = self._mlp.gate_up_proj.weight.detach().float()
        W_gate    = W_fused[:I]
        W_up      = W_fused[I:]
        W_down    = self._mlp.down_proj.weight.detach().float()

        gate_full     = xf @ W_gate.T
        up_full       = xf @ W_up.T
        gate_hybrid   = torch.where(hot, gate_full, gate_approx)
        swiglu        = F.silu(gate_hybrid) * up_full
        return (swiglu @ W_down.T).to(orig_dtype)


# ---------------------------------------------------------------------------
# NormCapture / infra
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
        self.ids.extend(
            (output.float() @ self._W_U.T).argmax(dim=-1).cpu().tolist())


def _load_prompts(path: str, n: int) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip()][:n]


def _get_internals(llm):
    e = llm.llm_engine
    try:
        mr = e.model_executor.driver_worker.worker.model_runner
    except AttributeError:
        mr = e.model_executor.driver_worker.model_runner
    return mr.model.model.norm, mr.model.model.layers, mr.model.lm_head.weight


def _run(llm, prompts, W_U, norm) -> np.ndarray:
    from vllm import SamplingParams
    cap = NormCapture(W_U.detach().float())
    cap.attach(norm)
    llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0.0),
                 use_tqdm=False)
    cap.detach()
    ids = cap.ids
    n_dec = len(prompts)
    return np.array(ids[:-n_dec] if n_dec < len(ids) else ids, dtype=np.int32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Exp24: E5M3 threshold sweep T=0.20..0.80 with hot% monitoring.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", default="ibm-granite/granite-4.2-3b")
    p.add_argument("--calibration-set", default="bartowski-imatrix-v5-semantic.txt")
    p.add_argument("--num-prompts", type=int, default=8)
    p.add_argument(
        "--t-min", type=float, default=0.20,
        help="Start of threshold sweep (default 0.20).")
    p.add_argument(
        "--t-max", type=float, default=0.80,
        help="End of threshold sweep (default 0.80).")
    p.add_argument(
        "--t-step", type=float, default=0.05,
        help="Threshold step size (default 0.05).")
    p.add_argument(
        "--floor-frac", type=float, default=0.02,
        help="Minimum hot fraction floor (default 0.02 = 2%%).")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    args = _parse_args(argv)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    # Build threshold list: round to 2 dp to avoid float noise
    thresholds = [
        round(args.t_min + i * args.t_step, 2)
        for i in range(round((args.t_max - args.t_min) / args.t_step) + 1)
    ]
    print(f"Device: {DEV}  B={BLOCK_SIZE} E5M3", file=sys.stderr)
    print(f"Thresholds: {thresholds}", file=sys.stderr)
    print(f"Floor: {args.floor_frac*100:.0f}%", file=sys.stderr)

    prompts = _load_prompts(args.calibration_set, args.num_prompts)
    print(f"Loaded {len(prompts)} prompts.", file=sys.stderr)

    from vllm import LLM
    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              kv_cache_memory_bytes=int(0.5 * 1024**3), max_model_len=512,
              enable_prefix_caching=False)

    norm, layers, W_U = _get_internals(llm)
    orig_forwards = [l.mlp.forward for l in layers]

    print("Baseline pass...", file=sys.stderr)
    baseline = _run(llm, prompts, W_U, norm)
    n_tok = len(baseline)
    print(f"  {n_tok} prefill-token predictions captured.", file=sys.stderr)

    # Results: variant -> T -> match
    variants = ["pure", "floored"]
    results:   dict[str, dict[float, float]] = {v: {} for v in variants}
    # Per-layer hot fractions: variant -> T -> list[float] (one per layer)
    hot_fracs: dict[str, dict[float, list[float]]] = {
        v: {t: [] for t in thresholds} for v in variants}

    n_passes  = len(thresholds) * len(variants)
    pass_idx  = 0

    for T in thresholds:
        for variant, floor in [("pure", 0.0), ("floored", args.floor_frac)]:
            pass_idx += 1
            hybrids = [
                HybridMLPThresholdFloor(l.mlp, T, floor) for l in layers]
            for l, h in zip(layers, hybrids):
                l.mlp.forward = h
                h.reset_stats()

            print(f"  [{pass_idx}/{n_passes}] T={T:.2f}  {variant}...",
                  end="  ", file=sys.stderr, flush=True)
            ids   = _run(llm, prompts, W_U, norm)[:n_tok]
            match = float((ids == baseline).mean())
            results[variant][T] = match

            # Collect per-layer hot fractions
            for h in hybrids:
                hot_fracs[variant][T].append(h.mean_hot_frac())

            mean_hot = float(np.mean(hot_fracs[variant][T]))
            print(f"match={match:.4f}  perturb={1-match:.4f}  "
                  f"mean_hot={mean_hot*100:.1f}%", file=sys.stderr)

            for l, fwd in zip(layers, orig_forwards):
                l.mlp.forward = fwd
            del hybrids

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n--- Experiment 24 Results ---\n")
    print(f"  B=8 E5M3 gate, full up+down, floor={args.floor_frac*100:.0f}%\n")

    # Main results table
    print(f"  {'T':>5}  {'pure match':>11}  {'pure hot%':>10}  "
          f"{'floored match':>14}  {'floor hot%':>11}  "
          f"{'Δ floor-pure':>13}")
    for T in thresholds:
        pm   = results["pure"][T]
        fm   = results["floored"][T]
        ph   = float(np.mean(hot_fracs["pure"][T])) * 100
        fh   = float(np.mean(hot_fracs["floored"][T])) * 100
        diff = fm - pm
        marker = "  ◄ best" if pm == max(results["pure"].values()) and variant == "pure" else ""
        print(f"  {T:>5.2f}  {pm:>11.4f}  {ph:>9.1f}%  "
              f"{fm:>14.4f}  {fh:>10.1f}%  {diff:>+13.4f}")

    # Best T per variant
    best_pure   = max(results["pure"],    key=lambda t: results["pure"][t])
    best_floor  = max(results["floored"], key=lambda t: results["floored"][t])
    print(f"\n  Best pure:    T={best_pure:.2f}  match={results['pure'][best_pure]:.4f}  "
          f"hot={np.mean(hot_fracs['pure'][best_pure])*100:.1f}%")
    print(f"  Best floored: T={best_floor:.2f}  match={results['floored'][best_floor]:.4f}  "
          f"hot={np.mean(hot_fracs['floored'][best_floor])*100:.1f}%")
    print(f"\n  Reference: exp23 T=0.50 match=0.7360  exp14 ternary @30% match=0.6320")

    # Per-layer hot fraction detail at best T (pure)
    print(f"\n  Per-layer hot% at best pure T={best_pure:.2f}:")
    hf = hot_fracs["pure"][best_pure]
    print(f"  {'layer':>6}  {'hot%':>7}    " * 5)
    row_vals = [(i, v*100) for i, v in enumerate(hf)]
    for row_start in range(0, len(row_vals), 5):
        chunk = row_vals[row_start:row_start+5]
        print("  " + "    ".join(f"{li:>4}  {v:>6.1f}%" for li, v in chunk))

    # Flag any layers with very low hot%
    low_layers = [(i, v*100) for i, v in enumerate(hf) if v < 5.0]
    if low_layers:
        print(f"\n  ⚠ Layers with hot% < 5% at T={best_pure:.2f}: "
              + ", ".join(f"layer {i} ({v:.1f}%)" for i, v in low_layers))
    else:
        print(f"\n  All layers have hot% ≥ 5% at T={best_pure:.2f}.")


if __name__ == "__main__":
    main()
