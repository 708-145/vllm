# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 25 – activation error of gate_approx and up_approx.

Measures the relative error of the E5M3-encoded gate approximation and the
ternary-encoded gate/up approximations against their full-precision counterparts,
as seen on real activations from calibration prompts.

For each MLP layer, hooks capture the actual hidden-state input x.  We then
compute for every token position t:

  gate_full[t, i]   = x[t] @ W_gate[i]       (full BF16→FP32)
  gate_approx[t, i] = x[t] @ W_gate_enc[i]   (E5M3 encoded)

  up_full[t, i]     = x[t] @ W_up[i]
  up_approx[t, i]   = x[t] @ W_up_enc[i]     (E5M3 encoded, hypothetical)

Reported metrics per layer (and macro-averaged):
  mean_rel_err    = mean(|approx - full| / (|full| + eps))
  rms_err         = rms(approx - full)
  rms_full        = rms(full)
  snr_db          = 20 * log10(rms_full / rms_err)   (higher = better)
  r2              = 1 - var(approx - full) / var(full)

Usage::

    python tools/profiler/exp25_activation_error.py \\
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

_M3_FRAC_LOG2 = torch.log2(1.0 + torch.arange(8).float() / 8.0)  # (8,)


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
# Ternary encoding (α-scaled, identical to exp14)
# ---------------------------------------------------------------------------

def build_ternary_encoded(W: torch.Tensor, alpha: float = 0.75) -> torch.Tensor:
    """sign(W) * (|W| >= alpha * mean|W|) * mean|W| — scaled ternary."""
    mean_abs = W.abs().mean().item()
    tau      = alpha * mean_abs
    T        = (W.sign() * (W.abs() >= tau).float()) * mean_abs
    return T.contiguous()


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def activation_errors(
    full: torch.Tensor,   # (T, I)
    approx: torch.Tensor, # (T, I)
) -> dict:
    diff     = approx - full
    abs_full = full.abs()

    rms_err  = diff.pow(2).mean().sqrt().item()
    rms_full = full.pow(2).mean().sqrt().item()
    snr_db   = 20.0 * np.log10(rms_full / (rms_err + EPS))

    rel_err  = (diff.abs() / (abs_full + EPS)).mean().item()

    var_full = full.var().item()
    var_err  = diff.var().item()
    r2       = 1.0 - var_err / (var_full + EPS)

    return dict(rel_err=rel_err, rms_err=rms_err, rms_full=rms_full,
                snr_db=snr_db, r2=r2)


# ---------------------------------------------------------------------------
# MLP hook that accumulates activations and computes errors in-place
# ---------------------------------------------------------------------------

class ErrorAccumulator:
    """Replaces mlp.forward; accumulates per-layer error stats over all tokens."""

    def __init__(self, mlp):
        self._mlp = mlp
        W_fused = mlp.gate_up_proj.weight.detach().float()
        I = W_fused.shape[0] // 2
        self._I = I
        self._W_gate     = W_fused[:I]
        self._W_up       = W_fused[I:]
        self._W_gate_e5  = build_e5m3_encoded(W_fused[:I])
        self._W_up_e5    = build_e5m3_encoded(W_fused[I:])
        self._W_gate_ter = build_ternary_encoded(W_fused[:I])
        self._W_up_ter   = build_ternary_encoded(W_fused[I:])

        # Accumulators — lists of per-batch tensors, concatenated at summary time
        self._gate_e5_stats:  list[dict] = []
        self._up_e5_stats:    list[dict] = []
        self._gate_ter_stats: list[dict] = []

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        gate_full   = xf @ self._W_gate.T
        up_full     = xf @ self._W_up.T
        gate_e5     = xf @ self._W_gate_e5.T
        up_e5       = xf @ self._W_up_e5.T
        gate_ter    = xf @ self._W_gate_ter.T

        self._gate_e5_stats.append(activation_errors(gate_full, gate_e5))
        self._up_e5_stats.append(activation_errors(up_full, up_e5))
        self._gate_ter_stats.append(activation_errors(gate_full, gate_ter))

        # Forward pass uses original MLP (we only measure, don't perturb)
        W_down = self._mlp.down_proj.weight.detach().float()
        swiglu = F.silu(gate_full) * up_full
        return (swiglu @ W_down.T).to(x.dtype)

    def summary(self) -> dict:
        def _avg(stats_list, key):
            return float(np.mean([s[key] for s in stats_list]))

        return {
            "gate_e5":  {k: _avg(self._gate_e5_stats, k)
                         for k in ("rel_err", "rms_err", "rms_full", "snr_db", "r2")},
            "up_e5":    {k: _avg(self._up_e5_stats, k)
                         for k in ("rel_err", "rms_err", "rms_full", "snr_db", "r2")},
            "gate_ter": {k: _avg(self._gate_ter_stats, k)
                         for k in ("rel_err", "rms_err", "rms_full", "snr_db", "r2")},
        }


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
    p = argparse.ArgumentParser(description="Exp25: gate/up activation error.")
    p.add_argument("--model", default="ibm-granite/granite-4.2-3b")
    p.add_argument("--calibration-set", default="bartowski-imatrix-v5-semantic.txt")
    p.add_argument("--num-prompts", type=int, default=8)
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

    accumulators = [ErrorAccumulator(l.mlp) for l in layers]
    for l, acc in zip(layers, accumulators):
        l.mlp.forward = acc

    print("Running calibration pass...", file=sys.stderr)
    llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0.0),
                 use_tqdm=False)

    for l, fwd in zip(layers, orig_forwards):
        l.mlp.forward = fwd

    # ----------------------------------------------------------------
    # Collect per-layer summaries
    # ----------------------------------------------------------------
    summaries = [acc.summary() for acc in accumulators]

    # ----------------------------------------------------------------
    # Print per-layer table
    # ----------------------------------------------------------------
    hdr = (f"{'Layer':>5}  {'gate_e5 rel%':>12}  {'gate_e5 SNR':>11}  "
           f"{'gate_e5 R²':>10}  {'up_e5 rel%':>10}  {'up_e5 SNR':>10}  "
           f"{'gate_ter rel%':>13}  {'gate_ter SNR':>12}")
    print("\n" + "="*110)
    print("Activation error: E5M3 gate, E5M3 up (hypothetical), ternary gate")
    print("="*110)
    print(hdr)
    print("-"*110)

    for i, s in enumerate(summaries):
        ge  = s["gate_e5"]
        ue  = s["up_e5"]
        gt  = s["gate_ter"]
        print(f"{i:>5}  "
              f"{ge['rel_err']*100:>12.2f}  "
              f"{ge['snr_db']:>11.1f}  "
              f"{ge['r2']:>10.4f}  "
              f"{ue['rel_err']*100:>10.2f}  "
              f"{ue['snr_db']:>10.1f}  "
              f"{gt['rel_err']*100:>13.2f}  "
              f"{gt['snr_db']:>12.1f}")

    # ----------------------------------------------------------------
    # Macro averages
    # ----------------------------------------------------------------
    def _macro(key1, key2):
        return float(np.mean([s[key1][key2] for s in summaries]))

    print("-"*110)
    print(f"{'MEAN':>5}  "
          f"{_macro('gate_e5','rel_err')*100:>12.2f}  "
          f"{_macro('gate_e5','snr_db'):>11.1f}  "
          f"{_macro('gate_e5','r2'):>10.4f}  "
          f"{_macro('up_e5','rel_err')*100:>10.2f}  "
          f"{_macro('up_e5','snr_db'):>10.1f}  "
          f"{_macro('gate_ter','rel_err')*100:>13.2f}  "
          f"{_macro('gate_ter','snr_db'):>12.1f}")
    print("="*110)

    print("\nNotes:")
    print("  gate_e5   = B=8 E5M3 encoded W_gate (used in exp22–24)")
    print("  up_e5     = B=8 E5M3 encoded W_up   (hypothetical; up is always full in exp20–24)")
    print("  gate_ter  = ternary α=0.75 encoded W_gate (used in exp14)")
    print("  rel_err   = mean |approx−full| / (|full| + ε)  over all tokens × channels")
    print("  SNR dB    = 20·log10(RMS_full / RMS_err)  — higher is better")
    print("  R²        = 1 − var(err) / var(full)      — closer to 1 is better")


if __name__ == "__main__":
    main()
