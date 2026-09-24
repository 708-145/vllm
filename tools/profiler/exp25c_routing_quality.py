# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 25c – routing quality: gate_approx hot decision vs gate_full.

For each threshold T, compares the hot/cold binary decision made by
gate_approx against the oracle decision made by gate_full (using the same
threshold applied to |gate_full|).

Metrics per threshold (macro-averaged across all layers and tokens):
  hot_frac_approx   fraction of channels declared hot by gate_approx
  hot_frac_full     fraction of channels that would be hot under gate_full oracle
  precision         of hot_approx set: fraction that are also hot_full
  recall            of hot_full set: fraction caught by hot_approx
  f1                harmonic mean of precision and recall
  iou               |hot_approx ∩ hot_full| / |hot_approx ∪ hot_full|

Usage::

    python tools/profiler/exp25c_routing_quality.py \\
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

    e        = log2_s.floor().to(torch.int32)
    frac     = log2_s - e.float()
    m_lut    = _M3_FRAC_LOG2.to(frac.device)
    m_best   = (frac.unsqueeze(1) - m_lut).abs().argmin(1)
    log2_s_q = e.float() + m_lut[m_best]
    scales   = (2.0 ** log2_s_q).clamp(min=EPS)

    n_blk  = Wp.shape[1] // B
    s_exp  = scales.reshape(O, n_blk).unsqueeze(2).expand(O, n_blk, B).reshape(O, Wp.shape[1])
    return (Wp.sign() * s_exp)[:, :I].contiguous()


# ---------------------------------------------------------------------------
# Routing quality hook
# ---------------------------------------------------------------------------

class RoutingQualityHook:
    """Measures hot/cold routing agreement between gate_approx and gate_full."""

    def __init__(self, mlp, thresholds: list[float]):
        self._mlp        = mlp
        self._thresholds = thresholds

        W_fused = mlp.gate_up_proj.weight.detach().float()
        I = W_fused.shape[0] // 2
        self._I          = I
        self._W_gate     = W_fused[:I]
        self._W_gate_enc = build_e5m3_encoded(W_fused[:I])

        # per-threshold accumulators
        n = len(thresholds)
        self._tp   = [0.0] * n   # true  positives (hot_approx ∩ hot_full)
        self._fp   = [0.0] * n   # false positives (hot_approx ∩ cold_full)
        self._fn   = [0.0] * n   # false negatives (cold_approx ∩ hot_full)
        self._tn   = [0.0] * n   # true  negatives
        self._n    = [0.0] * n   # total elements seen

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        xf          = x.float()
        gate_full   = xf @ self._W_gate.T      # (T, I)
        gate_approx = xf @ self._W_gate_enc.T  # (T, I)

        sig_approx = gate_approx.abs()
        sig_full   = gate_full.abs()
        mean_approx = sig_approx.mean(dim=-1, keepdim=True)
        mean_full   = sig_full.mean(dim=-1, keepdim=True)

        for ti, T in enumerate(self._thresholds):
            hot_a = sig_approx > T * mean_approx   # (T, I) approx decision
            hot_f = sig_full   > T * mean_full      # (T, I) oracle decision

            tp = ( hot_a &  hot_f).float().sum().item()
            fp = ( hot_a & ~hot_f).float().sum().item()
            fn = (~hot_a &  hot_f).float().sum().item()
            tn = (~hot_a & ~hot_f).float().sum().item()

            self._tp[ti] += tp
            self._fp[ti] += fp
            self._fn[ti] += fn
            self._tn[ti] += tn
            self._n[ti]  += hot_a.numel()

        # Unmodified forward
        W_fused = self._mlp.gate_up_proj.weight.detach().float()
        W_up    = W_fused[self._I:]
        W_down  = self._mlp.down_proj.weight.detach().float()
        swiglu  = F.silu(gate_full) * (xf @ W_up.T)
        return (swiglu @ W_down.T).to(x.dtype)

    def summary(self) -> list[dict]:
        results = []
        for ti, T in enumerate(self._thresholds):
            tp, fp, fn, tn, n = (self._tp[ti], self._fp[ti],
                                  self._fn[ti], self._tn[ti], self._n[ti])
            precision = tp / (tp + fp + EPS)
            recall    = tp / (tp + fn + EPS)
            f1        = 2 * precision * recall / (precision + recall + EPS)
            iou       = tp / (tp + fp + fn + EPS)
            acc       = (tp + tn) / (n + EPS)
            hot_frac_a = (tp + fp) / (n + EPS)
            hot_frac_f = (tp + fn) / (n + EPS)
            results.append(dict(
                threshold=T,
                hot_frac_approx=hot_frac_a,
                hot_frac_full=hot_frac_f,
                precision=precision,
                recall=recall,
                f1=f1,
                iou=iou,
                acc=acc,
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
        description="Exp25c: routing quality of gate_approx vs gate_full oracle.")
    p.add_argument("--model", default="ibm-granite/granite-4.2-3b")
    p.add_argument("--calibration-set", default="bartowski-imatrix-v5-semantic.txt")
    p.add_argument("--num-prompts", type=int, default=8)
    p.add_argument(
        "--thresholds", nargs="+", type=float,
        default=[0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
        metavar="T",
        help="Threshold factors (hot = |gate| > T * mean|gate|).",
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

    hooks = [RoutingQualityHook(l.mlp, args.thresholds) for l in layers]
    for l, h in zip(layers, hooks):
        l.mlp.forward = h

    print("Running calibration pass...", file=sys.stderr)
    llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0.0),
                 use_tqdm=False)

    for l, fwd in zip(layers, orig_forwards):
        l.mlp.forward = fwd

    # ----------------------------------------------------------------
    # Macro-average across all 40 layers
    # ----------------------------------------------------------------
    all_summaries = [h.summary() for h in hooks]
    n_T = len(args.thresholds)

    # Each hook already accumulates globally (not per-batch), so we can
    # just average the per-layer summary dicts
    by_T = [[layer_sums[ti] for layer_sums in all_summaries] for ti in range(n_T)]

    def _macro(tier_list, key):
        return float(np.mean([s[key] for s in tier_list]))

    print("\n" + "="*90)
    print("Routing quality: gate_approx hot decision vs gate_full oracle")
    print("Same threshold T applied to both |gate_approx| and |gate_full|")
    print("Macro-averaged across all 40 layers")
    print("="*90)
    print(f"{'T':>5}  {'hot%(A)':>8}  {'hot%(F)':>8}  "
          f"{'prec':>7}  {'recall':>7}  {'F1':>7}  {'IoU':>7}  {'acc':>7}")
    print("-"*90)
    for ti in range(n_T):
        s = by_T[ti]
        T = args.thresholds[ti]
        print(f"{T:>5.2f}  "
              f"{_macro(s,'hot_frac_approx')*100:>7.1f}%  "
              f"{_macro(s,'hot_frac_full')*100:>7.1f}%  "
              f"{_macro(s,'precision'):>7.4f}  "
              f"{_macro(s,'recall'):>7.4f}  "
              f"{_macro(s,'f1'):>7.4f}  "
              f"{_macro(s,'iou'):>7.4f}  "
              f"{_macro(s,'acc'):>7.4f}")
    print("="*90)
    print()
    print("Notes:")
    print("  hot%(A) = fraction of channels hot according to gate_approx")
    print("  hot%(F) = fraction of channels hot according to gate_full (oracle)")
    print("  prec    = precision:  of channels approx calls hot, how many truly are?")
    print("  recall  = recall:     of truly hot channels, how many did approx catch?")
    print("  F1      = harmonic mean of precision and recall")
    print("  IoU     = intersection-over-union of the two hot sets")
    print("  acc     = overall channel classification accuracy (hot/cold)")


if __name__ == "__main__":
    main()
