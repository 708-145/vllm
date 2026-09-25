# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 37 – 3bpw cold up: 2-bit/weight, 2 E5M3 scales per B=16 block.

Encoding scheme
---------------
Block size B=16 weights.  Each weight is encoded as one of 4 magnitude levels
selected by 2 bits: {−s_hi, −s_lo, +s_lo, +s_hi}.

  bit layout per weight:  [sign_bit | magnitude_bit]
    00 → −s_hi
    01 → −s_lo
    10 → +s_lo
    11 → +s_hi

Two E5M3 scales per block (1 byte each) → 2 bytes scale overhead.
Code storage: B×2 bits = 16×2 = 32 bits = 4 bytes.
Total: 6 bytes per 16 weights = **3 bits per weight**.

Compression vs BF16 (2 bytes/weight): 16 bytes / 6 bytes = **2.67×**.
Compared to E5M3 B=8 1bpw (6 bytes / 8 weights = 6/8 = 0.75 bytes/weight = 8×
compression): 3bpw is less compressed but should be significantly more accurate.

TARE-optimal scale selection
----------------------------
Given block |w|, find (s_lo, s_hi) that minimise the TARE loss:

  L = Σᵢ tiltᵢ × (log|w̃ᵢ| − log|wᵢ|)²

where each wᵢ is assigned to s_lo if |wᵢ| < √(s_lo·s_hi) else s_hi.

Solved by EM in log-space (1-D 2-centroid k-means with tilt weighting):
  1. Init: s_lo = geometric mean of lower half, s_hi = upper half (by |w|).
  2. E-step: assign each weight to nearest centroid in log-space.
  3. M-step: recompute each centroid as tilt-weighted geometric mean of its
             members.
  4. Repeat until convergence (≤10 steps; typically 3–4).

Both centroids are quantised to E5M3 (same as exp22–27).

Motivation
----------
Exp36 showed E5M3 B=8 1bpw cold up is too lossy: +25 pp strict at 20% hot.
The hypothesis: the 1bpw sign encoding cannot represent the range of |w_up|
values within a block.  Two magnitude levels per block should substantially
reduce the approximation error for cold up channels.

Scheme compared in this experiment (hot=20%/25%/30%/50%):
  A. sparse_gate + 3bpw_up   (this experiment)
  B. sparse_gate_only        (full precision up — upper bound, from exp35/36)
  C. exp24 E5M3 T=0.20       (reference)

Both strict and thermal match reported (T=0.7 and T=1.0, τ=ln2).

Usage::

    python tools/profiler/exp37_3bpw_cold_up.py \\
        --model ibm-granite/granite-4.2-3b \\
        --calibration-set bartowski-imatrix-v5-semantic.txt \\
        --weight-cache exp35_sparse_weights.pt
"""

import argparse
import math
import os
import sys
import time
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


DEV   = _device()
I_DIM = 8192
LN2   = math.log(2)
EPS   = 1e-9

# E5M3 lookup table for quantising scales
_M3_FRAC_LOG2 = torch.log2(1.0 + torch.arange(8).float() / 8.0)


# ---------------------------------------------------------------------------
# E5M3 scalar quantisation  (same rounding as exp22–27)
# ---------------------------------------------------------------------------

def _quantise_e5m3(s: torch.Tensor) -> torch.Tensor:
    """Quantise a tensor of positive scales to the nearest E5M3 value.

    E5M3: value = (1 + m/8) * 2^e,  m ∈ {0..7},  e ∈ Z
    Returns float32 tensor of same shape with E5M3-rounded values.
    """
    s      = s.clamp(min=EPS)
    log2_s = torch.log2(s)
    e      = log2_s.floor().to(torch.int32)
    frac   = log2_s - e.float()
    m_lut  = _M3_FRAC_LOG2.to(frac.device)
    m_best = (frac.unsqueeze(-1) - m_lut).abs().argmin(-1)
    return (2.0 ** (e.float() + m_lut[m_best])).clamp(min=EPS)


# ---------------------------------------------------------------------------
# 3bpw encoding: 2-level TARE-optimal E5M3 scales per B=16 block
# ---------------------------------------------------------------------------

def _floor_eps(W: torch.Tensor, pct: float = 1.0) -> torch.Tensor:
    flat = W.abs().reshape(-1)
    k    = max(1, int(len(flat) * pct / 100.0))
    return flat.kthvalue(k).values.clamp(min=EPS)


def build_3bpw_encoded(W: torch.Tensor, B: int = 16,
                        em_steps: int = 10) -> torch.Tensor:
    """Encode W (O, I) as a 2-level TARE-optimal float32 approximation.

    Each block of B weights gets two TARE-optimal E5M3 scales (s_lo, s_hi).
    Each weight is approximated as ±s_lo or ±s_hi (2 bits/weight).
    Returns W_enc (O, I) float32 — the reconstructed approximation — ready
    for use as a drop-in weight matrix in x @ W_enc.T.

    Storage cost (informational):
      codes: B × 2 bits = 4 bytes / block
      scales: 2 × 1 byte E5M3 = 2 bytes / block
      total: 6 bytes / 16 weights = 3 bits per weight = 2.67× vs BF16
    """
    eps  = _floor_eps(W)
    O, I = W.shape
    pad  = (B - I % B) % B
    Wp   = F.pad(W, (0, pad)) if pad else W
    n_row_blocks = Wp.shape[1] // B
    n_blocks     = O * n_row_blocks

    W_b  = Wp.reshape(n_blocks, B)          # (n_blocks, B)
    wa   = W_b.abs().clamp(min=EPS)         # (n_blocks, B)
    logw = torch.log(wa)                    # (n_blocks, B)
    tilt = torch.log1p(wa / eps)            # (n_blocks, B)  tilt weights

    # ---- TARE-optimal 2-centroid EM in log-space ----
    # Initialise: s_lo = tilt-weighted geomean of bottom half,
    #             s_hi = tilt-weighted geomean of top half (split by median |w|)
    median_logw = logw.median(dim=1, keepdim=True).values   # (n_blocks, 1)
    lo_mask     = logw <= median_logw                        # (n_blocks, B)
    hi_mask     = ~lo_mask

    def _tilt_geomean(lw, tm, mask):
        """Tilt-weighted geometric mean of log-weights, masked."""
        denom = (tm * mask.float()).sum(dim=1).clamp(min=EPS)
        numer = (tm * lw * mask.float()).sum(dim=1)
        return (numer / denom).exp().clamp(min=EPS)

    s_lo = _tilt_geomean(logw, tilt, lo_mask)   # (n_blocks,)
    s_hi = _tilt_geomean(logw, tilt, hi_mask)   # (n_blocks,)
    # Guard: ensure s_lo <= s_hi
    s_lo, s_hi = torch.minimum(s_lo, s_hi), torch.maximum(s_lo, s_hi)

    for _ in range(em_steps):
        # E-step: assign each weight to s_lo or s_hi by nearest in log-space
        log_mid  = (torch.log(s_lo) + torch.log(s_hi)).unsqueeze(1) / 2.0
        hi_mask  = logw > log_mid                            # (n_blocks, B)
        lo_mask  = ~hi_mask

        # M-step: recompute centroids
        s_lo_new = _tilt_geomean(logw, tilt, lo_mask)
        s_hi_new = _tilt_geomean(logw, tilt, hi_mask)

        # Handle degenerate case: all weights in one partition
        all_hi = lo_mask.sum(1) == 0
        all_lo = hi_mask.sum(1) == 0
        s_lo_new = torch.where(all_hi, s_lo, s_lo_new)
        s_hi_new = torch.where(all_lo, s_hi, s_hi_new)

        # Ensure ordering
        s_lo_new, s_hi_new = (torch.minimum(s_lo_new, s_hi_new),
                               torch.maximum(s_lo_new, s_hi_new))

        if (s_lo_new - s_lo).abs().max() < 1e-7 and \
           (s_hi_new - s_hi).abs().max() < 1e-7:
            break
        s_lo, s_hi = s_lo_new, s_hi_new

    # ---- Quantise both scales to E5M3 ----
    s_lo_q = _quantise_e5m3(s_lo)   # (n_blocks,)
    s_hi_q = _quantise_e5m3(s_hi)   # (n_blocks,)

    # ---- Final assignment and reconstruction ----
    log_mid = (torch.log(s_lo_q) + torch.log(s_hi_q)).unsqueeze(1) / 2.0
    use_hi  = logw > log_mid                              # (n_blocks, B) bool
    level   = torch.where(use_hi,
                          s_hi_q.unsqueeze(1).expand_as(W_b),
                          s_lo_q.unsqueeze(1).expand_as(W_b))
    W_enc   = W_b.sign() * level                          # (n_blocks, B)

    # Reshape back to (O, I), strip padding
    W_enc = W_enc.reshape(O, Wp.shape[1])[:, :I].contiguous()
    return W_enc


# ---------------------------------------------------------------------------
# E5M3 B=8 encoding  (for reference comparison, identical to exp22–27/36)
# ---------------------------------------------------------------------------

def build_e5m3_encoded(W: torch.Tensor, B: int = 8) -> torch.Tensor:
    """1bpw E5M3 B=8 encoding (sign + single E5M3 scale per 8 weights)."""
    O, I = W.shape
    eps  = _floor_eps(W)
    pad  = (B - I % B) % B
    Wp   = F.pad(W, (0, pad)) if pad else W
    W_b  = Wp.reshape(-1, B)
    wa     = W_b.abs().clamp(min=eps)
    tilt   = torch.log1p(wa / eps)
    log2_s = (tilt * torch.log2(wa)).sum(1) / tilt.sum(1).clamp(min=EPS)
    e      = log2_s.floor().to(torch.int32)
    frac   = log2_s - e.float()
    m_lut  = _M3_FRAC_LOG2.to(frac.device)
    m_best = (frac.unsqueeze(1) - m_lut).abs().argmin(1)
    scales = (2.0 ** (e.float() + m_lut[m_best])).clamp(min=EPS)
    n_blk  = Wp.shape[1] // B
    s_exp  = (scales.reshape(O, n_blk)
                    .unsqueeze(2).expand(O, n_blk, B)
                    .reshape(O, Wp.shape[1]))
    return (Wp.sign() * s_exp)[:, :I].contiguous()


# ---------------------------------------------------------------------------
# Thermal match metric  (identical to exp34–36)
# ---------------------------------------------------------------------------

def thermal_match(
    lf: torch.Tensor,
    lh: torch.Tensor,
    temperature: float = 0.7,
    tau: float = LN2,
) -> tuple[float, float, float]:
    baseline_top1 = lf.argmax(-1)
    hybrid_top1   = lh.argmax(-1)
    exact_match   = (baseline_top1 == hybrid_top1)
    h_rank1 = lh.gather(-1, hybrid_top1.unsqueeze(-1)).squeeze(-1)
    b_in_h  = lh.gather(-1, baseline_top1.unsqueeze(-1)).squeeze(-1)
    gap      = h_rank1 - b_in_h
    forgiven = (~exact_match) & (gap < temperature * tau)
    strict   = float(exact_match.float().mean())
    thermal  = float((exact_match | forgiven).float().mean())
    gaps     = gap[~exact_match]
    mean_gap = float(gaps.mean()) if gaps.numel() > 0 else 0.0
    return strict, thermal, mean_gap


# ---------------------------------------------------------------------------
# MLP wrappers
# ---------------------------------------------------------------------------

class SparseGateOnlyMLP:
    """Upper bound: sparse gate routing, cold gate sparse, up full precision."""
    def __init__(self, mlp, W_gate_sp, k_hot):
        self._mlp = mlp; self._W_g_sp = W_gate_sp
        self._k_hot = k_hot; self._I = mlp.gate_up_proj.weight.shape[0] // 2

    def __call__(self, x):
        orig = x.dtype; xf = x.float(); I = self._I
        gs = xf @ self._W_g_sp.T
        k  = min(self._k_hot, I)
        idx = torch.topk(gs.abs(), k, dim=-1, sorted=False).indices
        hot = torch.zeros(xf.shape[0], I, dtype=torch.bool, device=xf.device)
        hot.scatter_(-1, idx, True)
        W = self._mlp.gate_up_proj.weight.detach().float()
        gf = xf @ W[:I].T; uf = xf @ W[I:].T
        wd = self._mlp.down_proj.weight.detach().float()
        return (F.silu(torch.where(hot, gf, gs)) * uf @ wd.T).to(orig)


class SparseGateEncodedUpMLP:
    """Sparse gate routing + encoded cold up (any pre-built W_up_enc)."""
    def __init__(self, mlp, W_gate_sp, W_up_enc, k_hot):
        self._mlp = mlp; self._W_g_sp = W_gate_sp
        self._W_u_enc = W_up_enc; self._k_hot = k_hot
        self._I = mlp.gate_up_proj.weight.shape[0] // 2

    def __call__(self, x):
        orig = x.dtype; xf = x.float(); I = self._I
        gs = xf @ self._W_g_sp.T
        k  = min(self._k_hot, I)
        idx = torch.topk(gs.abs(), k, dim=-1, sorted=False).indices
        hot = torch.zeros(xf.shape[0], I, dtype=torch.bool, device=xf.device)
        hot.scatter_(-1, idx, True)
        W  = self._mlp.gate_up_proj.weight.detach().float()
        gf = xf @ W[:I].T; uf = xf @ W[I:].T
        ue = xf @ self._W_u_enc.T
        wd = self._mlp.down_proj.weight.detach().float()
        gate = torch.where(hot, gf, gs)
        up   = torch.where(hot, uf, ue)
        return (F.silu(gate) * up @ wd.T).to(orig)


class E5M3ThresholdMLP:
    """Exp24 reference: E5M3 threshold routing, up full."""
    def __init__(self, mlp, threshold):
        self._mlp = mlp; self._threshold = threshold
        W = mlp.gate_up_proj.weight.detach().float()
        I = W.shape[0] // 2; self._I = I
        self._W_gate_enc = build_e5m3_encoded(W[:I])

    def __call__(self, x):
        orig = x.dtype; xf = x.float(); I = self._I
        ga = xf @ self._W_gate_enc.T
        hot = ga.abs() > self._threshold * ga.abs().mean(-1, keepdim=True)
        W  = self._mlp.gate_up_proj.weight.detach().float()
        gf = xf @ W[:I].T; uf = xf @ W[I:].T
        wd = self._mlp.down_proj.weight.detach().float()
        return (F.silu(torch.where(hot, gf, ga)) * uf @ wd.T).to(orig)


# ---------------------------------------------------------------------------
# LogitCapture  (identical to exp34–36)
# ---------------------------------------------------------------------------

class LogitCapture:
    def __init__(self, W_U):
        self._W_U = W_U.float(); self.logits = []; self._handle = None

    def attach(self, m):
        self._handle = m.register_forward_hook(self._hook)

    def detach(self):
        if self._handle: self._handle.remove(); self._handle = None

    def _hook(self, m, a, out):
        self.logits.append((out.float() @ self._W_U.T).cpu())

    def all_logits(self, n_dec):
        cat = torch.cat(self.logits, 0)
        return cat[:-n_dec] if n_dec < cat.shape[0] else cat


def _load_prompts(path, n):
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip()][:n]


def _get_internals(llm):
    e = llm.llm_engine
    try:    mr = e.model_executor.driver_worker.worker.model_runner
    except: mr = e.model_executor.driver_worker.model_runner
    return mr.model.model.norm, mr.model.model.layers, mr.model.lm_head.weight


def _run_logits(llm, prompts, W_U, norm):
    from vllm import SamplingParams
    cap = LogitCapture(W_U.detach().float())
    cap.attach(norm)
    llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0.0),
                 use_tqdm=False)
    cap.detach()
    return cap.all_logits(len(prompts))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Exp37: 3bpw 2-level TARE-optimal E5M3 cold up.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--model",           default="ibm-granite/granite-4.2-3b")
    p.add_argument("--calibration-set", default="bartowski-imatrix-v5-semantic.txt")
    p.add_argument("--num-prompts",     type=int, default=8)
    p.add_argument("--keep-rate",       type=float, default=0.5)
    p.add_argument("--finetune-steps",  type=int, default=300)
    p.add_argument("--hot-fractions",   nargs="+", type=float,
                   default=[0.20, 0.25, 0.30, 0.50])
    p.add_argument("--temperatures",    nargs="+", type=float, default=[0.7, 1.0])
    p.add_argument("--tau",             type=float, default=LN2)
    p.add_argument("--em-steps",        type=int, default=10,
                   help="EM iterations for 2-centroid log-space k-means (default 10).")
    p.add_argument("--weight-cache",    default="exp35_sparse_weights.pt")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = _parse_args(argv)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    print(f"Device: {DEV}  tau={args.tau:.4f}  temps={args.temperatures}  "
          f"em_steps={args.em_steps}", file=sys.stderr)

    prompts = _load_prompts(args.calibration_set, args.num_prompts)
    print(f"Loaded {len(prompts)} prompts.", file=sys.stderr)

    from vllm import LLM
    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              kv_cache_memory_bytes=int(0.5 * 1024**3), max_model_len=512,
              enable_prefix_caching=False)

    norm, layers, W_U = _get_internals(llm)
    orig_forwards = [l.mlp.forward for l in layers]
    I  = layers[0].mlp.gate_up_proj.weight.shape[0] // 2
    kr = args.keep_rate; ft = args.finetune_steps

    # ------------------------------------------------------------------
    # Load sparse W_gate from cache
    # ------------------------------------------------------------------
    cache_path = Path(args.weight_cache)
    cache_key  = f"kr{kr}_ft{ft}"
    saved = torch.load(cache_path, weights_only=True)
    if cache_key not in saved:
        raise RuntimeError(f"Key {cache_key} not found in {cache_path}. "
                           "Run exp35 first.")
    sp_gate = saved[cache_key]["gate"]
    print(f"\nLoaded sparse W_gate ({len(sp_gate)} layers) from {cache_path}.",
          file=sys.stderr)

    # ------------------------------------------------------------------
    # Build encoded W_up variants for all layers
    # ------------------------------------------------------------------
    print(f"\nBuilding 3bpw (2-level E5M3, B=16) W_up ...", file=sys.stderr)
    t0 = time.time()
    enc_3bpw = []
    for li, layer in enumerate(layers):
        Wu = layer.mlp.gate_up_proj.weight.detach().float()[I:]
        enc_3bpw.append(build_3bpw_encoded(Wu, B=16, em_steps=args.em_steps))
        print(f"  3bpw layer {li:2d}", end="\r", file=sys.stderr, flush=True)
    print(f"  done ({time.time()-t0:.0f}s)", file=sys.stderr)

    # Also build 1bpw E5M3 B=8 for direct comparison with exp36
    print(f"\nBuilding 1bpw E5M3 B=8 W_up (exp36 baseline) ...", file=sys.stderr)
    t0 = time.time()
    enc_e5m3 = []
    for li, layer in enumerate(layers):
        Wu = layer.mlp.gate_up_proj.weight.detach().float()[I:]
        enc_e5m3.append(build_e5m3_encoded(Wu, B=8))
        print(f"  e5m3 layer {li:2d}", end="\r", file=sys.stderr, flush=True)
    print(f"  done ({time.time()-t0:.0f}s)", file=sys.stderr)

    # ------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------
    print("\nBaseline pass ...", file=sys.stderr)
    lf_base = _run_logits(llm, prompts, W_U, norm)
    n_tok   = lf_base.shape[0]
    print(f"  {n_tok} prefill tokens, vocab={lf_base.shape[1]}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Define runs
    # ------------------------------------------------------------------
    temps = args.temperatures
    runs: list[tuple[str, object]] = []

    for frac in args.hot_fractions:
        k   = max(1, int(frac * I))
        pct = f"{frac*100:.0f}%"
        runs.append((f"sparse+3bpw_up       hot={pct}", lambda _k=k: [
            SparseGateEncodedUpMLP(
                layers[li].mlp, sp_gate[li], enc_3bpw[li], _k)
            for li in range(len(layers))
        ]))
        runs.append((f"sparse+e5m3_up(1bpw) hot={pct}", lambda _k=k: [
            SparseGateEncodedUpMLP(
                layers[li].mlp, sp_gate[li], enc_e5m3[li], _k)
            for li in range(len(layers))
        ]))
        runs.append((f"sparse_gate_only     hot={pct}", lambda _k=k: [
            SparseGateOnlyMLP(layers[li].mlp, sp_gate[li], _k)
            for li in range(len(layers))
        ]))

    runs.append(("exp24 E5M3 T=0.20   (~88% hot)", lambda: [
        E5M3ThresholdMLP(layers[li].mlp, 0.20) for li in range(len(layers))
    ]))

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    results: dict[str, dict] = {}

    for idx, (label, build_fn) in enumerate(runs):
        hybrids = build_fn()
        for l, h in zip(layers, hybrids):
            l.mlp.forward = h

        print(f"\n[{idx+1:2d}/{len(runs)}] {label} ...",
              end="  ", file=sys.stderr, flush=True)
        lh = _run_logits(llm, prompts, W_U, norm)[:n_tok]

        rec: dict = {}
        s, _, _ = thermal_match(lf_base, lh, 1.0, 0.0)
        rec["strict"] = s
        for T in temps:
            _, tm, mg = thermal_match(lf_base, lh, T, args.tau)
            rec[f"thermal_{T}"] = tm; rec[f"gap_{T}"] = mg
        results[label] = rec

        print("strict={:.4f}  ".format(rec["strict"]) +
              "  ".join("th@{T}={v:.4f}".format(T=T, v=rec[f"thermal_{T}"])
                        for T in temps), file=sys.stderr)
        for l, fwd in zip(layers, orig_forwards):
            l.mlp.forward = fwd
        del hybrids

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    cw = 11
    cols = ["strict"] + [f"th@T={T}" for T in temps] + [f"gap@{T}" for T in temps]
    hdr  = f"  {'scheme':<38}" + "".join(f"  {c:>{cw}}" for c in cols)

    def _row(label, rec):
        r = f"  {label:<38}  {rec['strict']:>{cw}.4f}"
        for T in temps: r += f"  {rec[f'thermal_{T}']:>{cw}.4f}"
        for T in temps: r += f"  {rec[f'gap_{T}']:>{cw}.4f}"
        return r

    print("\n" + "=" * 92)
    print(f"Exp37: 3bpw cold up  (2-level E5M3 B=16 TARE-optimal, sparse gate kr={kr}+ft)")
    print("cold gate = x@W_gate_sparse.T  |  cold up = x@W_up_enc.T  |  down = full")
    print("=" * 92)
    print(hdr)
    print("  " + "-" * 88)
    for label, rec in results.items():
        print(_row(label, rec))
    print()
    print("  gap = mean logit gap among perturbed tokens")
    print("  FP8-equiv: strict ~3-5%, thermal ~2-4%  |  NVFP4-equiv: strict ~10-20%, thermal ~8-16%")
    print("=" * 92)

    # Perturbation % table
    print("\nPerturbation rates (%):")
    hdr2 = (f"  {'scheme':<38}  {'strict%':>8}" +
            "".join(f"  {'th%@'+str(T):>9}" for T in temps) +
            "".join(f"  {'Δ@'+str(T):>8}" for T in temps))
    print(hdr2)
    for label, rec in results.items():
        sp  = (1 - rec["strict"]) * 100
        row = f"  {label:<38}  {sp:>7.1f}%"
        for T in temps:
            tp = (1 - rec[f"thermal_{T}"]) * 100; row += f"  {tp:>8.1f}%"
        for T in temps:
            tp = (1 - rec[f"thermal_{T}"]) * 100; row += f"  {sp-tp:>7.1f}pp"
        print(row)

    # 3bpw vs full-up delta table
    print("\nΔ encoded cold up vs full-precision cold up (+ = encoded worse):")
    hdr3 = f"  {'hot%':<6}  {'enc':>14}  {'Δ strict':>10}  {'Δ th@0.7':>10}  {'Δ th@1.0':>10}  {'Δ gap':>8}"
    print(hdr3)
    for frac in args.hot_fractions:
        pct = f"{frac*100:.0f}%"
        k_full = f"sparse_gate_only     hot={pct}"
        for enc_label in [f"sparse+3bpw_up       hot={pct}",
                          f"sparse+e5m3_up(1bpw) hot={pct}"]:
            if enc_label not in results or k_full not in results:
                continue
            re, rf = results[enc_label], results[k_full]
            ds  = (1 - re["strict"])     - (1 - rf["strict"])
            d07 = (1 - re.get("thermal_0.7", 0)) - (1 - rf.get("thermal_0.7", 0))
            d10 = (1 - re.get("thermal_1.0", 0)) - (1 - rf.get("thermal_1.0", 0))
            dg  = re.get("gap_0.7", 0)  - rf.get("gap_0.7", 0)
            short = "3bpw " if "3bpw" in enc_label else "1bpw "
            print(f"  {pct:<6}  {short:>14}  {ds*100:>+9.1f}pp  "
                  f"{d07*100:>+9.1f}pp  {d10*100:>+9.1f}pp  {dg:>+7.3f}L")
        print()


if __name__ == "__main__":
    main()
