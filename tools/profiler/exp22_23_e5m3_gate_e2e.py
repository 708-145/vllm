# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiments 22 & 23 – E5M3 gate scale; top-k and threshold routing.

Exp 22: B=8 E5M3 gate encoding, top-k hot routing (same as exp21 but E5M3).
Exp 23: B=8 E5M3 gate encoding, threshold routing on |gate_approx| (exp2-style
        but with E5M3 precision).

Motivation
----------
Exp21 showed that B=8 E8M0 gate encoding is worse than ternary α=0.75 despite
lower TARE, because E8M0's mean scale error of 19.6% over-estimates cold gate
values, inflating cold SiLU contributions.

E5M3 (5 exponent bits, 3 mantissa bits, 1 byte/block) reduces mean scale error
from 19.6% to 2.5% — a 7.2× improvement at the same storage cost.  With E5M3:

  s_e5m3 = (1 + m/8) * 2^e,   m ∈ {0..7},  e chosen to minimise |log2(s*) - log2(s)|

Exp 22 tests whether this tighter scale makes top-k routing competitive with
the ternary baseline (exp14 match=0.516 @10%).

Exp 23 tests magnitude-threshold routing on gate_approx.  In exp2 this failed
catastrophically because sign-approx magnitude was a poor proxy for true gate
magnitude.  With E5M3 cold values having only 2.5% scale error, |gate_approx|
should now reliably reflect |gate_full|, potentially enabling a threshold-based
routing that avoids the top-k sort cost.

Usage::

    python tools/profiler/exp22_23_e5m3_gate_e2e.py \\
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


# ---------------------------------------------------------------------------
# E5M3 encoding
# ---------------------------------------------------------------------------

# Pre-build lookup: fractional log2 for each mantissa code 0..7
_M3_FRAC_LOG2 = torch.log2(1.0 + torch.arange(8).float() / 8.0)  # (8,)


def _floor_eps(W: torch.Tensor, pct: float = 1.0) -> torch.Tensor:
    flat = W.abs().reshape(-1)
    k = max(1, int(len(flat) * pct / 100.0))
    return flat.kthvalue(k).values.clamp(min=1e-9)


def build_e5m3_encoded(W: torch.Tensor, B: int = BLOCK_SIZE) -> torch.Tensor:
    """W_enc (O, I) float32 = sign(W) * E5M3_optimal_scale_per_block.

    E5M3: s = (1 + m/8) * 2^e,  m in 0..7
    Optimal s* = exp2(tilt-weighted mean log2(|w_b|)).
    Quantise to nearest E5M3 value: find e = floor(log2(s*)), pick m that
    minimises |frac(log2(s*)) - M3_FRAC_LOG2[m]|.

    Storage: 1 byte per block (5-bit exponent + 3-bit mantissa).
    """
    O, I = W.shape
    eps = _floor_eps(W)
    pad = (B - I % B) % B
    Wp  = F.pad(W, (0, pad)) if pad else W
    W_b = Wp.reshape(-1, B)                        # (n_blocks, B)

    wa     = W_b.abs().clamp(min=eps)
    tilt   = torch.log1p(wa / eps)
    log2_s = (tilt * torch.log2(wa)).sum(1) / tilt.sum(1).clamp(min=1e-9)

    # Quantise to E5M3
    e      = log2_s.floor().to(torch.int32)        # integer exponent
    frac   = log2_s - e.float()                    # fractional part in [0, 1)
    m_lut  = _M3_FRAC_LOG2.to(frac.device)
    m_best = (frac.unsqueeze(1) - m_lut).abs().argmin(1)   # (n_blocks,)
    log2_s_q = e.float() + m_lut[m_best]
    scales   = (2.0 ** log2_s_q).clamp(min=1e-9)  # (n_blocks,) E5M3 values

    n_blk  = Wp.shape[1] // B
    s_exp  = scales.reshape(O, n_blk).unsqueeze(2).expand(O, n_blk, B).reshape(O, Wp.shape[1])
    return (Wp.sign() * s_exp)[:, :I].contiguous()


# ---------------------------------------------------------------------------
# top-k mask
# ---------------------------------------------------------------------------

def top_k_mask(scores: torch.Tensor, k: int) -> torch.BoolTensor:
    idx  = torch.topk(scores, k, dim=-1, largest=True, sorted=False).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    return mask


# ---------------------------------------------------------------------------
# HybridMLP variants
# ---------------------------------------------------------------------------

class HybridMLPTopK:
    """Exp22: E5M3 gate encoding, top-k hot routing, full-precision up+down."""

    def __init__(self, mlp, k_hot: int, routing: str):
        self._mlp     = mlp
        self._k_hot   = k_hot
        self._routing = routing       # "current" | "proxy_prior"
        self._prior: torch.Tensor | None = None

        W_fused = mlp.gate_up_proj.weight.detach().float()
        I = W_fused.shape[0] // 2
        self._I          = I
        self._W_gate_enc = build_e5m3_encoded(W_fused[:I])

    def reset_prior(self) -> None:
        self._prior = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        xf = x.float()
        I  = self._I

        gate_approx = xf @ self._W_gate_enc.T      # (T, I)

        signal = (self._prior.abs()
                  if self._routing == "proxy_prior" and self._prior is not None
                  else gate_approx.abs())
        hot = top_k_mask(signal, self._k_hot)

        W_fused   = self._mlp.gate_up_proj.weight.detach().float()
        W_gate    = W_fused[:I]
        W_up      = W_fused[I:]
        W_down    = self._mlp.down_proj.weight.detach().float()

        gate_full     = xf @ W_gate.T
        up_full       = xf @ W_up.T
        gate_hybrid   = torch.where(hot, gate_full, gate_approx)
        swiglu        = F.silu(gate_hybrid) * up_full
        out           = (swiglu @ W_down.T).to(orig_dtype)

        self._prior = gate_approx.detach()
        return out


class HybridMLPThreshold:
    """Exp23: E5M3 gate encoding, threshold routing on |gate_approx|.

    hot = channels where |gate_approx| > threshold_factor * mean(|gate_approx|)

    threshold_factor is set once per layer at construction based on the desired
    mean sparsity.  At inference the hot set varies per token depending on the
    actual gate_approx values — no sort needed, just a comparison.
    """

    def __init__(self, mlp, threshold_factor: float, routing: str):
        self._mlp              = mlp
        self._threshold_factor = threshold_factor
        self._routing          = routing
        self._prior: torch.Tensor | None = None

        W_fused = mlp.gate_up_proj.weight.detach().float()
        I = W_fused.shape[0] // 2
        self._I          = I
        self._W_gate_enc = build_e5m3_encoded(W_fused[:I])

    def reset_prior(self) -> None:
        self._prior = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        xf = x.float()
        I  = self._I

        gate_approx = xf @ self._W_gate_enc.T      # (T, I)

        # Use prior or current as routing signal
        signal = (self._prior.abs()
                  if self._routing == "proxy_prior" and self._prior is not None
                  else gate_approx.abs())

        # Threshold: hot where |signal| > factor * mean(|signal|) per token
        thresh = self._threshold_factor * signal.mean(dim=-1, keepdim=True)
        hot    = signal > thresh                    # (T, I) — no sort

        W_fused   = self._mlp.gate_up_proj.weight.detach().float()
        W_gate    = W_fused[:I]
        W_up      = W_fused[I:]
        W_down    = self._mlp.down_proj.weight.detach().float()

        gate_full     = xf @ W_gate.T
        up_full       = xf @ W_up.T
        gate_hybrid   = torch.where(hot, gate_full, gate_approx)
        swiglu        = F.silu(gate_hybrid) * up_full
        out           = (swiglu @ W_down.T).to(orig_dtype)

        self._prior = gate_approx.detach()
        return out


# ---------------------------------------------------------------------------
# NormCapture / infra (shared with exp14/20/21)
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
        description="Exp22+23: E5M3 gate encoding, top-k and threshold routing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", default="ibm-granite/granite-4.2-3b")
    p.add_argument("--calibration-set", default="bartowski-imatrix-v5-semantic.txt")
    p.add_argument("--num-prompts", type=int, default=8)
    p.add_argument(
        "--hot-fractions", nargs="+", type=float,
        default=[0.05, 0.10, 0.20, 0.30],
        metavar="F",
        help="Hot fractions for exp22 top-k routing.",
    )
    p.add_argument(
        "--thresholds", nargs="+", type=float,
        default=[0.5, 1.0, 1.5, 2.0, 3.0],
        metavar="T",
        help="Threshold factors for exp23 (hot = |gate| > T * mean(|gate|)).",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    args = _parse_args(argv)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    print(f"Device: {DEV}  B={BLOCK_SIZE} E5M3", file=sys.stderr)

    prompts = _load_prompts(args.calibration_set, args.num_prompts)
    print(f"Loaded {len(prompts)} prompts.", file=sys.stderr)

    from vllm import LLM
    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              kv_cache_memory_bytes=int(0.5 * 1024**3), max_model_len=512,
              enable_prefix_caching=False)

    norm, layers, W_U = _get_internals(llm)
    I = layers[0].mlp.gate_up_proj.weight.shape[0] // 2
    orig_forwards = [l.mlp.forward for l in layers]

    print("Baseline pass...", file=sys.stderr)
    baseline = _run(llm, prompts, W_U, norm)
    n_tok = len(baseline)
    print(f"  {n_tok} prefill-token predictions captured.", file=sys.stderr)

    exp22_results: dict[str, dict[float, float]] = {
        "current": {}, "proxy_prior": {}}
    exp23_results: dict[str, dict[float, float]] = {
        "current": {}, "proxy_prior": {}}

    # exp14 references
    exp14 = {0.05: 0.456, 0.10: 0.516, 0.20: 0.588, 0.30: 0.632}

    # ----------------------------------------------------------------
    # Exp 22 — top-k routing
    # ----------------------------------------------------------------
    print("\n=== Exp22: E5M3 gate, top-k routing ===", file=sys.stderr)
    configs22 = ["current", "proxy_prior"]
    n22 = len(configs22) * len(args.hot_fractions)
    p22 = 0
    for cfg in configs22:
        for frac in args.hot_fractions:
            p22 += 1
            k_hot = max(1, int(frac * I))
            hybrids = [HybridMLPTopK(l.mlp, k_hot, cfg) for l in layers]
            for l, h in zip(layers, hybrids):
                l.mlp.forward = h
            for h in hybrids:
                h.reset_prior()

            print(f"  [{p22}/{n22}] {cfg}  hot={frac*100:.0f}%...",
                  end="  ", file=sys.stderr, flush=True)
            ids = _run(llm, prompts, W_U, norm)[:n_tok]
            match = float((ids == baseline).mean())
            exp22_results[cfg][frac] = match
            print(f"match={match:.4f}  perturb={1-match:.4f}", file=sys.stderr)

            for l, fwd in zip(layers, orig_forwards):
                l.mlp.forward = fwd
            del hybrids

    # ----------------------------------------------------------------
    # Exp 23 — threshold routing
    # ----------------------------------------------------------------
    print("\n=== Exp23: E5M3 gate, threshold routing ===", file=sys.stderr)
    configs23 = ["current", "proxy_prior"]
    n23 = len(configs23) * len(args.thresholds)
    p23 = 0
    for cfg in configs23:
        for tfac in args.thresholds:
            p23 += 1
            hybrids = [HybridMLPThreshold(l.mlp, tfac, cfg) for l in layers]
            for l, h in zip(layers, hybrids):
                l.mlp.forward = h
            for h in hybrids:
                h.reset_prior()

            print(f"  [{p23}/{n23}] {cfg}  T={tfac:.1f}×mean...",
                  end="  ", file=sys.stderr, flush=True)
            ids = _run(llm, prompts, W_U, norm)[:n_tok]
            match = float((ids == baseline).mean())
            exp23_results[cfg][tfac] = match
            print(f"match={match:.4f}  perturb={1-match:.4f}", file=sys.stderr)

            for l, fwd in zip(layers, orig_forwards):
                l.mlp.forward = fwd
            del hybrids

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("\n" + "="*60)
    print("--- Exp22: E5M3 gate, top-k routing ---\n")
    hdr = f"  {'config':<14}" + "".join(f"  {f*100:>5.0f}%" for f in args.hot_fractions)
    print("Top-1 match rate  (↑ better):")
    print(hdr)
    for cfg in configs22:
        row = f"  {cfg:<14}" + "".join(
            f"  {exp22_results[cfg][f]:>6.4f}" for f in args.hot_fractions)
        print(row)
    print(f"  {'exp14 ref':<14}" + "".join(
        f"  {exp14.get(f, float('nan')):>6.4f}" for f in args.hot_fractions))
    print(f"  {'exp21 E8M0':<14}" + "".join(
        f"  {v:>6.4f}" for v in [0.3780, 0.4140, 0.4560, 0.5200]))

    print("\nΔ vs exp14  (+ = exp22 better):")
    print(hdr)
    for cfg in configs22:
        row = f"  {cfg:<14}" + "".join(
            f"  {exp22_results[cfg][f] - exp14.get(f, float('nan')):>+6.4f}"
            for f in args.hot_fractions)
        print(row)

    print("\n" + "="*60)
    print("--- Exp23: E5M3 gate, threshold routing ---\n")
    print("Top-1 match rate  (↑ better):")
    thdr = f"  {'config':<14}" + "".join(f"  T={t:.1f}" for t in args.thresholds)
    print(thdr)
    for cfg in configs23:
        row = f"  {cfg:<14}" + "".join(
            f"  {exp23_results[cfg][t]:>6.4f}" for t in args.thresholds)
        print(row)
    print(f"\n  (exp14 ref @10% hot = 0.5160, exp22 proxy_prior @10% = "
          f"{exp22_results['proxy_prior'].get(0.10, float('nan')):.4f})")


if __name__ == "__main__":
    main()
