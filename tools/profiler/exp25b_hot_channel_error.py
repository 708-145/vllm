# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 25b – gate_approx error on hot channels only.

For each threshold T (hot = |gate_approx| > T * mean|gate_approx|), measures:
  - The activation error of gate_approx vs gate_full, restricted to the hot
    channels selected by that threshold.
  - Hot fraction (what % of channels are hot on average).

The key question: when E5M3 routing correctly identifies hot channels, how
accurate is gate_approx *on those hot channels* specifically?  If hot channels
are already well-approximated, recomputing them at full precision is low-value.
If they are poorly approximated, recomputing them is essential.

Also reports error on cold channels for reference.

Usage::

    python tools/profiler/exp25b_hot_channel_error.py \\
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
EPS = 1e-9


# ---------------------------------------------------------------------------
# E5M3 encoding (identical to exp22/23)
# ---------------------------------------------------------------------------

_M3_FRAC_LOG2 = torch.log2(1.0 + torch.arange(8).float() / 8.0)


def _floor_eps(W: torch.Tensor, pct: float = 1.0) -> torch.Tensor:
    flat = W.abs().reshape(-1)
    k = max(1, int(len(flat) * pct / 100.0))
    return flat.kthvalue(k).values.clamp(min=EPS)


def build_e5m3_encoded(W: torch.Tensor, B: int = BLOCK_SIZE) -> torch.Tensor:
    O, I = W.shape
    eps = _floor_eps(W)
    pad = (B - I % B) % B
    Wp  = F.pad(W, (0, pad)) if pad else W
    W_b = Wp.reshape(-1, B)

    wa     = W_b.abs().clamp(min=eps)
    tilt   = torch.log1p(wa / eps)
    log2_s = (tilt * torch.log2(wa)).sum(1) / tilt.sum(1).clamp(min=EPS)

    e      = log2_s.floor().to(torch.int32)
    frac   = log2_s - e.float()
    m_lut  = _M3_FRAC_LOG2.to(frac.device)
    m_best = (frac.unsqueeze(1) - m_lut).abs().argmin(1)
    log2_s_q = e.float() + m_lut[m_best]
    scales   = (2.0 ** log2_s_q).clamp(min=EPS)

    n_blk  = Wp.shape[1] // B
    s_exp  = scales.reshape(O, n_blk).unsqueeze(2).expand(O, n_blk, B).reshape(O, Wp.shape[1])
    return (Wp.sign() * s_exp)[:, :I].contiguous()


# ---------------------------------------------------------------------------
# Masked error stats
# ---------------------------------------------------------------------------

def masked_stats(full: torch.Tensor, approx: torch.Tensor,
                 mask: torch.Tensor) -> dict:
    """Error stats restricted to positions where mask is True.

    Args:
        full:   (T, I) full-precision activations.
        approx: (T, I) approximate activations.
        mask:   (T, I) bool — True for the positions to measure.

    Returns:
        dict with rel_err, snr_db, r2, n_elements.
    """
    if mask.sum() == 0:
        return dict(rel_err=float("nan"), snr_db=float("nan"),
                    r2=float("nan"), n_elements=0)
    f = full[mask]
    a = approx[mask]
    d = a - f

    rms_err  = d.pow(2).mean().sqrt().item()
    rms_full = f.pow(2).mean().sqrt().item()
    snr_db   = 20.0 * np.log10(rms_full / (rms_err + EPS))
    rel_err  = (d.abs() / (f.abs() + EPS)).mean().item()
    r2       = 1.0 - d.var().item() / (f.var().item() + EPS)
    return dict(rel_err=rel_err, snr_db=snr_db, r2=r2,
                n_elements=int(mask.sum().item()))


# ---------------------------------------------------------------------------
# MLP hook
# ---------------------------------------------------------------------------

class ThresholdErrorHook:
    """Hooks one MLP layer; accumulates error stats per threshold."""

    def __init__(self, mlp, thresholds: list[float]):
        self._mlp        = mlp
        self._thresholds = thresholds

        W_fused = mlp.gate_up_proj.weight.detach().float()
        I = W_fused.shape[0] // 2
        self._I          = I
        self._W_gate     = W_fused[:I]
        self._W_gate_enc = build_e5m3_encoded(W_fused[:I])

        # accumulators: list-of-lists indexed by threshold
        self._hot_stats:  list[list[dict]] = [[] for _ in thresholds]
        self._cold_stats: list[list[dict]] = [[] for _ in thresholds]
        self._hot_fracs:  list[list[float]] = [[] for _ in thresholds]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        xf          = x.float()
        gate_full   = xf @ self._W_gate.T      # (T, I)
        gate_approx = xf @ self._W_gate_enc.T  # (T, I)

        signal = gate_approx.abs()
        mean_signal = signal.mean(dim=-1, keepdim=True)

        for ti, T in enumerate(self._thresholds):
            hot  = signal > T * mean_signal      # (T, I)
            cold = ~hot

            self._hot_stats[ti].append(masked_stats(gate_full, gate_approx, hot))
            self._cold_stats[ti].append(masked_stats(gate_full, gate_approx, cold))
            self._hot_fracs[ti].append(hot.float().mean().item())

        # Forward: unmodified
        W_fused = self._mlp.gate_up_proj.weight.detach().float()
        W_up    = W_fused[self._I:]
        W_down  = self._mlp.down_proj.weight.detach().float()
        swiglu  = F.silu(gate_full) * (xf @ W_up.T)
        return (swiglu @ W_down.T).to(x.dtype)

    def summary(self) -> list[dict]:
        """Returns one dict per threshold with averaged stats."""
        results = []
        for ti, T in enumerate(self._thresholds):
            def _avg(stats_list, key):
                vals = [s[key] for s in stats_list if not np.isnan(s.get(key, float("nan")))]
                return float(np.mean(vals)) if vals else float("nan")

            results.append(dict(
                threshold=T,
                hot_frac=float(np.mean(self._hot_fracs[ti])),
                hot_rel_err=_avg(self._hot_stats[ti], "rel_err"),
                hot_snr_db=_avg(self._hot_stats[ti], "snr_db"),
                hot_r2=_avg(self._hot_stats[ti], "r2"),
                cold_rel_err=_avg(self._cold_stats[ti], "rel_err"),
                cold_snr_db=_avg(self._cold_stats[ti], "snr_db"),
                cold_r2=_avg(self._cold_stats[ti], "r2"),
            ))
        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_prompts(path: str, n: int) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip()][:n]


def _get_internals(llm):
    e = llm.llm_engine
    try:
        mr = e.model_executor.driver_worker.worker.model_runner
    except AttributeError:
        mr = e.model_executor.driver_worker.model_runner
    return mr.model.model.layers


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="Exp25b: gate_approx error on hot vs cold channels.")
    p.add_argument("--model", default="ibm-granite/granite-4.2-3b")
    p.add_argument("--calibration-set", default="bartowski-imatrix-v5-semantic.txt")
    p.add_argument("--num-prompts", type=int, default=8)
    p.add_argument(
        "--thresholds", nargs="+", type=float,
        default=[0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
        metavar="T",
        help="Threshold factors (hot = |gate_approx| > T * mean(|gate_approx|)).",
    )
    args = p.parse_args(argv)

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    print(f"Device: {DEV}  B={BLOCK_SIZE}", file=sys.stderr)

    prompts = _load_prompts(args.calibration_set, args.num_prompts)
    print(f"Loaded {len(prompts)} prompts.", file=sys.stderr)

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              kv_cache_memory_bytes=int(0.5 * 1024**3), max_model_len=512,
              enable_prefix_caching=False)

    layers = _get_internals(llm)
    orig_forwards = [l.mlp.forward for l in layers]

    hooks = [ThresholdErrorHook(l.mlp, args.thresholds) for l in layers]
    for l, h in zip(layers, hooks):
        l.mlp.forward = h

    print("Running calibration pass...", file=sys.stderr)
    llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0.0),
                 use_tqdm=False)

    for l, fwd in zip(layers, orig_forwards):
        l.mlp.forward = fwd

    # ----------------------------------------------------------------
    # Macro-average across layers for each threshold
    # ----------------------------------------------------------------
    # summaries[layer][threshold_idx]
    all_summaries = [h.summary() for h in hooks]

    # Transpose to summaries_by_T[threshold_idx] = list over layers
    n_T = len(args.thresholds)
    by_T = [[] for _ in range(n_T)]
    for layer_sums in all_summaries:
        for ti, s in enumerate(layer_sums):
            by_T[ti].append(s)

    def _macro(tier_list, key):
        vals = [s[key] for s in tier_list if not np.isnan(s.get(key, float("nan")))]
        return float(np.mean(vals)) if vals else float("nan")

    # ----------------------------------------------------------------
    # Print macro table (averaged across all 40 layers)
    # ----------------------------------------------------------------
    print("\n" + "="*95)
    print("gate_approx error — macro-averaged across all 40 layers (E5M3 B=8)")
    print("="*95)
    print(f"{'T':>5}  {'hot%':>6}  "
          f"{'hot rel%':>9}  {'hot SNR':>8}  {'hot R²':>7}  "
          f"{'cold rel%':>10}  {'cold SNR':>9}  {'cold R²':>8}")
    print("-"*95)
    for ti in range(n_T):
        s_list = by_T[ti]
        T      = args.thresholds[ti]
        hf     = _macro(s_list, "hot_frac") * 100
        print(f"{T:>5.2f}  {hf:>6.1f}%  "
              f"{_macro(s_list,'hot_rel_err')*100:>9.1f}  "
              f"{_macro(s_list,'hot_snr_db'):>8.1f}  "
              f"{_macro(s_list,'hot_r2'):>7.4f}  "
              f"{_macro(s_list,'cold_rel_err')*100:>10.1f}  "
              f"{_macro(s_list,'cold_snr_db'):>9.1f}  "
              f"{_macro(s_list,'cold_r2'):>8.4f}")
    print("="*95)

    print("\nNotes:")
    print("  T        = threshold factor (hot = |gate_approx| > T × mean|gate_approx|)")
    print("  hot%     = fraction of channels selected as hot (averaged per token per layer)")
    print("  rel%     = mean |gate_approx − gate_full| / (|gate_full| + ε) on that subset")
    print("  SNR dB   = 20·log10(RMS_full / RMS_err) on that subset")
    print("  R²       = 1 − var(err) / var(full)  on that subset")
    print()
    print("  Hot channels are large-|gate_approx| channels — these are recomputed at full")
    print("  precision in exp23/24.  Cold channels use gate_approx directly.")


if __name__ == "__main__":
    main()
