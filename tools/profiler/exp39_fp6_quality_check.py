# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 39 – FP6 (S1E2M3) static encoding quality check (no routing).

Encoding scheme
---------------
Block size B=16 weights.  One TARE-optimal E5M3 block scale per block.
Each weight is encoded as FP6 S1E2M3 relative to the block scale:

  FP6 S1E2M3 format:
    1 sign bit | 2 exponent bits | 3 mantissa bits
    Normal  (e∈{1,2,3}): value = (1 + m/8) * 2^(e - bias),  bias = 1
      → e=1: [1.000, 1.875]  (×1)
      → e=2: [2.000, 3.750]  (×2)
      → e=3: [4.000, 7.500]  (×4)
    Subnormal (e=0):      value = (m/8) * 2^(1 - bias) = m/8
      → {0, 1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8}

  All 32 non-negative codes:
    subnormal: 0.000, 0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875
    normal e=1: 1.000, 1.125, 1.250, 1.375, 1.500, 1.625, 1.750, 1.875
    normal e=2: 2.000, 2.250, 2.500, 2.750, 3.000, 3.250, 3.500, 3.750
    normal e=3: 4.000, 4.500, 5.000, 5.500, 6.000, 6.500, 7.000, 7.500

  Block scale s (E5M3, TARE-optimal): found by minimising TARE loss over the
  block after dividing by s.  The quantised FP6 value of weight w is:
    fp6_quantise(w / s) * s

Storage:
  codes:  B × 6 bits = 96 bits = 12 bytes per block
  scale:  1 × E5M3  =  8 bits =  1 byte  per block
  total:  13 bytes / 16 weights = 6.5 bits per weight
  vs BF16 (2 bytes/weight, 16 weights = 32 bytes): 32/13 ≈ 2.46× compression

Comparison with previous schemes:
  1bpw  E5M3 B=8:  8× compression   (exp36)
  3bpw  E5M3 B=16: 2.67× compression (exp37/38)
  6.5bpw FP6 B=16: 2.46× compression (this experiment)

Conditions evaluated (all 40 layers, prefill, 8 calibration prompts):
  A. gate only:        W_gate=FP6, W_up=full, W_down=full
  B. gate + up:        W_gate=FP6, W_up=FP6, W_down=full
  C. gate + down:      W_gate=FP6, W_up=full, W_down=FP6
  D. gate + up + down: W_gate=FP6, W_up=FP6, W_down=FP6

Same structure as exp38 for direct comparison.

Metrics: strict match, thermal@0.7, thermal@1.0, mean gap.

Usage::

    python tools/profiler/exp39_fp6_quality_check.py \\
        --model ibm-granite/granite-4.2-3b \\
        --calibration-set bartowski-imatrix-v5-semantic.txt
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

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
LN2 = math.log(2)
EPS = 1e-9

# E5M3 LUT for block scale quantisation (reused from exp22–38)
_M3_FRAC_LOG2 = torch.log2(1.0 + torch.arange(8).float() / 8.0)

# ---------------------------------------------------------------------------
# FP6 S1E2M3 LUT — all 32 non-negative representable values
# ---------------------------------------------------------------------------
# Built once at module load; shape (32,) float32, sorted ascending.
# Index encodes (e[1:0] | m[2:0]) — i.e. upper 2 bits = exponent, lower 3 = mantissa.
def _build_fp6_lut() -> torch.Tensor:
    vals = []
    bias = 1
    for e in range(4):       # 2-bit exponent: 0..3
        for m in range(8):   # 3-bit mantissa: 0..7
            if e == 0:       # subnormal
                v = (m / 8.0) * (2.0 ** (1 - bias))   # = m/8
            else:            # normal
                v = (1.0 + m / 8.0) * (2.0 ** (e - bias))
            vals.append(v)
    return torch.tensor(vals, dtype=torch.float32)   # (32,) in code order


_FP6_LUT = _build_fp6_lut()   # non-negative values, indexed by (e<<3)|m


# ---------------------------------------------------------------------------
# E5M3 scalar quantisation  (identical to exp22–38)
# ---------------------------------------------------------------------------

def _quantise_e5m3(s: torch.Tensor) -> torch.Tensor:
    s      = s.clamp(min=EPS)
    log2_s = torch.log2(s)
    e      = log2_s.floor().to(torch.int32)
    frac   = log2_s - e.float()
    m_lut  = _M3_FRAC_LOG2.to(frac.device)
    m_best = (frac.unsqueeze(-1) - m_lut).abs().argmin(-1)
    return (2.0 ** (e.float() + m_lut[m_best])).clamp(min=EPS)


# ---------------------------------------------------------------------------
# FP6 S1E2M3 encoding with TARE-optimal E5M3 block scale
# ---------------------------------------------------------------------------

def _floor_eps(W: torch.Tensor, pct: float = 1.0) -> torch.Tensor:
    flat = W.abs().reshape(-1)
    k    = max(1, int(len(flat) * pct / 100.0))
    return flat.kthvalue(k).values.clamp(min=EPS)


def _fp6_quantise_blocks(wa_scaled: torch.Tensor) -> torch.Tensor:
    """Round (n_blocks, B) normalised magnitudes to nearest FP6 value.

    wa_scaled: (n_blocks, B) float32, non-negative, already divided by scale.
    Returns (n_blocks, B) float32 quantised magnitudes (still in scale units).

    Uses a vectorised nearest-neighbour lookup against the 32-entry FP6 LUT.
    The LUT is in sorted order, so we use searchsorted + neighbour comparison
    instead of a full (n_blocks, B, 32) broadcast to keep memory bounded.
    """
    lut = _FP6_LUT.to(wa_scaled.device)     # (32,) sorted ascending
    flat = wa_scaled.reshape(-1)             # (N,)
    # searchsorted gives insertion index; compare left and right neighbours
    idx_r = torch.searchsorted(lut, flat)           # (N,) in [0, 32]
    idx_r = idx_r.clamp(0, len(lut) - 1)
    idx_l = (idx_r - 1).clamp(0, len(lut) - 1)
    vl = lut[idx_l]; vr = lut[idx_r]
    # Pick nearest; ties go to the right (larger value)
    use_r = (flat - vl) >= (vr - flat)
    q_flat = torch.where(use_r, vr, vl)
    return q_flat.reshape(wa_scaled.shape)


def build_fp6_encoded(W: torch.Tensor, B: int = 16) -> torch.Tensor:
    """Encode W (O, I) with FP6 S1E2M3 + one TARE-optimal E5M3 scale per B-block.

    Algorithm:
      1. Compute TARE-optimal block scale s as the tilt-weighted geometric mean
         of |w| shifted to align with the FP6 grid's geometric mean (no
         iteration — single closed-form pass, identical structure to E5M3 1bpw).
      2. Quantise scale to E5M3.
      3. Quantise each |w|/s to nearest FP6 S1E2M3 magnitude.
      4. Reconstruct: W_enc = sign(w) * fp6_round(|w|/s) * s

    Storage (informational): 13 bytes / 16 weights = 6.5 bpw = 2.46× vs BF16.

    Args:
        W:  (O, I) float32 weight matrix.
        B:  block size (default 16).

    Returns:
        W_enc: (O, I) float32 reconstructed approximation, drop-in for W.
    """
    lut  = _FP6_LUT.to(W.device)
    eps  = _floor_eps(W)
    O, I = W.shape
    pad  = (B - I % B) % B
    Wp   = F.pad(W, (0, pad)) if pad else W
    n_row_blocks = Wp.shape[1] // B
    n_blocks     = O * n_row_blocks

    W_b   = Wp.reshape(n_blocks, B)              # (n_blocks, B)
    signs = W_b.sign()                            # (n_blocks, B)
    wa    = W_b.abs().clamp(min=EPS)              # (n_blocks, B)
    tilt  = torch.log1p(wa / eps)                 # (n_blocks, B)
    log_wa = torch.log(wa)                        # (n_blocks, B)

    # ------------------------------------------------------------------
    # TARE-optimal scale: single closed-form pass (no iteration).
    # Compute the tilt-weighted geometric mean of |w|, then shift it to
    # align with the FP6 grid's geometric mean.
    # fp6_logmean = mean(log(lut[lut > 0])) — the "centre" of the FP6 grid
    # in log-space.  Dividing the block's geometric mean by fp6_logmean gives
    # the scale that centres the block on the FP6 grid.
    # ------------------------------------------------------------------
    fp6_nz    = lut[lut > 0]
    fp6_logmu = fp6_nz.log().mean()                          # scalar
    log_s     = ((tilt * log_wa).sum(1)
                 / tilt.sum(1).clamp(min=EPS)) - fp6_logmu  # (n_blocks,)
    s_opt     = log_s.exp().clamp(min=EPS)                   # (n_blocks,)
    s_q       = _quantise_e5m3(s_opt)                        # (n_blocks,) E5M3

    # ------------------------------------------------------------------
    # Quantise each weight to FP6 relative to its block scale
    # ------------------------------------------------------------------
    s_exp   = s_q.unsqueeze(1)                               # (n_blocks, 1)
    scaled  = wa / s_exp                                     # (n_blocks, B) |w|/s
    q_mag   = _fp6_quantise_blocks(scaled) * s_exp           # (n_blocks, B)
    W_enc   = signs * q_mag                                  # (n_blocks, B)

    return W_enc.reshape(O, Wp.shape[1])[:, :I].contiguous()


# ---------------------------------------------------------------------------
# Thermal match metric  (identical to exp34–38)
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
# MLP wrapper — uniform FP6 encoding (no routing)
# ---------------------------------------------------------------------------

class EncodedFP6MLP:
    """Apply FP6 S1E2M3 + E5M3 block-scale encoding to selected matrices.

    All channels are encoded; no hot/cold routing.

    Args:
        mlp:       vLLM MLP module (fused gate_up_proj + down_proj)
        enc_gate:  encoded W_gate (I, H) float32, or None → full precision
        enc_up:    encoded W_up   (I, H) float32, or None → full precision
        enc_down:  encoded W_down (H, I) float32, or None → full precision
    """
    def __init__(self, mlp, enc_gate, enc_up, enc_down):
        self._mlp      = mlp
        self._enc_gate = enc_gate
        self._enc_up   = enc_up
        self._enc_down = enc_down
        self._I        = mlp.gate_up_proj.weight.shape[0] // 2

    def __call__(self, x):
        orig = x.dtype
        xf   = x.float()
        I    = self._I
        W    = self._mlp.gate_up_proj.weight.detach().float()
        Wg   = self._enc_gate if self._enc_gate is not None else W[:I]
        Wu   = self._enc_up   if self._enc_up   is not None else W[I:]
        gate = F.silu(xf @ Wg.T)
        up   = xf @ Wu.T
        act  = gate * up
        if self._enc_down is not None:
            out = act @ self._enc_down.T
        else:
            wd  = self._mlp.down_proj.weight.detach().float()
            out = act @ wd.T
        return out.to(orig)


# ---------------------------------------------------------------------------
# LogitCapture  (identical to exp34–38)
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
        description="Exp39: FP6 S1E2M3 + E5M3 block-scale encoding quality check.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--model",           default="ibm-granite/granite-4.2-3b")
    p.add_argument("--calibration-set", default="bartowski-imatrix-v5-semantic.txt")
    p.add_argument("--num-prompts",     type=int, default=8)
    p.add_argument("--temperatures",    nargs="+", type=float, default=[0.7, 1.0])
    p.add_argument("--tau",             type=float, default=LN2)
    p.add_argument("--block-size",      type=int, default=16)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = _parse_args(argv)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    print(f"Device: {DEV}  tau={args.tau:.4f}  temps={args.temperatures}  "
          f"B={args.block_size}", file=sys.stderr)
    print(f"FP6 LUT ({len(_FP6_LUT)} values): {_FP6_LUT.tolist()}", file=sys.stderr)

    prompts = _load_prompts(args.calibration_set, args.num_prompts)
    print(f"Loaded {len(prompts)} prompts.", file=sys.stderr)

    from vllm import LLM
    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              kv_cache_memory_bytes=int(0.5 * 1024**3), max_model_len=512,
              enable_prefix_caching=False)

    norm, layers, W_U = _get_internals(llm)
    orig_forwards = [l.mlp.forward for l in layers]
    I = layers[0].mlp.gate_up_proj.weight.shape[0] // 2
    B = args.block_size

    # ------------------------------------------------------------------
    # Build FP6-encoded matrices for all layers
    # ------------------------------------------------------------------
    enc_gate = []; enc_up = []; enc_down = []
    for name, store, extractor in [
        ("gate", enc_gate, lambda l: l.mlp.gate_up_proj.weight.detach().float()[:I]),
        ("up",   enc_up,   lambda l: l.mlp.gate_up_proj.weight.detach().float()[I:]),
        ("down", enc_down, lambda l: l.mlp.down_proj.weight.detach().float()),
    ]:
        print(f"\nEncoding W_{name} (FP6 S1E2M3 B={B}) ...", file=sys.stderr)
        t0 = time.time()
        for li, layer in enumerate(layers):
            W = extractor(layer)
            store.append(build_fp6_encoded(W, B=B).cpu())
            print(f"  layer {li:2d}", end="\r", file=sys.stderr, flush=True)
        print(f"  done ({time.time()-t0:.0f}s)", file=sys.stderr)

    # ------------------------------------------------------------------
    # Baseline (full precision)
    # ------------------------------------------------------------------
    print("\nBaseline pass ...", file=sys.stderr)
    lf_base = _run_logits(llm, prompts, W_U, norm)
    n_tok   = lf_base.shape[0]
    print(f"  {n_tok} prefill tokens, vocab={lf_base.shape[1]}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Conditions: (label, gate_enc?, up_enc?, down_enc?)
    # ------------------------------------------------------------------
    temps = args.temperatures
    conditions: list[tuple[str, bool, bool, bool]] = [
        ("gate_only     (gate=FP6, up=full, down=full)", True,  False, False),
        ("gate+up       (gate=FP6, up=FP6,  down=full)", True,  True,  False),
        ("gate+down     (gate=FP6, up=full, down=FP6)",  True,  False, True),
        ("gate+up+down  (all=FP6)",                      True,  True,  True),
    ]

    results: dict[str, dict] = {}

    for idx, (label, use_gate, use_up, use_down) in enumerate(conditions):
        hybrids = [
            EncodedFP6MLP(
                layers[li].mlp,
                enc_gate[li] if use_gate else None,
                enc_up[li]   if use_up   else None,
                enc_down[li] if use_down else None,
            )
            for li in range(len(layers))
        ]
        for l, h in zip(layers, hybrids):
            l.mlp.forward = h

        print(f"\n[{idx+1}/{len(conditions)}] {label} ...",
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
    cw   = 11
    cols = ["strict"] + [f"th@T={T}" for T in temps] + [f"gap@{T}" for T in temps]
    hdr  = f"  {'condition':<48}" + "".join(f"  {c:>{cw}}" for c in cols)

    def _row(lbl, rec):
        r = f"  {lbl:<48}  {rec['strict']:>{cw}.4f}"
        for T in temps: r += f"  {rec[f'thermal_{T}']:>{cw}.4f}"
        for T in temps: r += f"  {rec[f'gap_{T}']:>{cw}.4f}"
        return r

    print("\n" + "=" * 115)
    print(f"Exp39: FP6 S1E2M3 encoding quality check  (B={B}, 1× E5M3 block scale, 6.5 bpw)")
    print("No routing — FP6 encoding applied uniformly to all channels of the listed matrices.")
    print("=" * 115)
    print(hdr)
    print("  " + "-" * 111)
    for label, rec in results.items():
        print(_row(label, rec))
    print()
    print("  gap = mean logit gap among perturbed tokens (L = logits)")
    print("  FP8-equiv: strict ~3-5%, thermal ~2-4%  |  NVFP4-equiv: strict ~10-20%")
    print("=" * 115)

    print("\nPerturbation rates (%):")
    hdr2 = (f"  {'condition':<48}  {'strict%':>8}" +
            "".join(f"  {'th%@'+str(T):>9}" for T in temps) +
            "".join(f"  {'Δ@'+str(T):>8}" for T in temps))
    print(hdr2)
    for label, rec in results.items():
        sp  = (1 - rec["strict"]) * 100
        row = f"  {label:<48}  {sp:>7.1f}%"
        for T in temps:
            tp = (1 - rec[f"thermal_{T}"]) * 100; row += f"  {tp:>8.1f}%"
        for T in temps:
            tp = (1 - rec[f"thermal_{T}"]) * 100; row += f"  {sp-tp:>7.1f}pp"
        print(row)

    # ------------------------------------------------------------------
    # Comparison table vs exp38 3bpw results
    # ------------------------------------------------------------------
    exp38 = {
        "gate_only":    {"strict": 0.490, "thermal_0.7": 0.538, "gap_0.7": 3.532},
        "gate+up":      {"strict": 0.036, "thermal_0.7": 0.054, "gap_0.7": 8.477},
        "gate+down":    {"strict": 0.116, "thermal_0.7": 0.140, "gap_0.7": 8.095},
        "gate+up+down": {"strict": 0.002, "thermal_0.7": 0.008, "gap_0.7": 10.595},
    }
    key_map = {
        "gate_only     (gate=FP6, up=full, down=full)": "gate_only",
        "gate+up       (gate=FP6, up=FP6,  down=full)": "gate+up",
        "gate+down     (gate=FP6, up=full, down=FP6)":  "gate+down",
        "gate+up+down  (all=FP6)":                      "gate+up+down",
    }
    print("\nΔ FP6 vs 3bpw (exp38)  (negative = FP6 better):")
    hdr3 = (f"  {'condition':<48}  {'Δ strict':>10}  "
            f"{'Δ th@0.7':>10}  {'Δ gap@0.7':>10}")
    print(hdr3)
    for label, rec in results.items():
        k = key_map.get(label)
        if k not in exp38:
            continue
        ref = exp38[k]
        ds  = ((1 - rec["strict"]) - (1 - ref["strict"])) * 100
        dt  = ((1 - rec.get("thermal_0.7", 0)) - (1 - ref["thermal_0.7"])) * 100
        dg  = rec.get("gap_0.7", 0) - ref["gap_0.7"]
        print(f"  {label:<48}  {ds:>+9.1f}pp  {dt:>+9.1f}pp  {dg:>+9.3f}L")


if __name__ == "__main__":
    main()
