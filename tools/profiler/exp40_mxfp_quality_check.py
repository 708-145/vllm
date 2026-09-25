# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 40 – OCP MX (MXFP8 / MXFP6) static encoding quality check.

OCP Microscaling (MX) formats, B=32, E8M0 shared block scale.

E8M0 scale
----------
Pure power-of-two: scale = 2^e, e stored as uint8 with bias=127.
  byte=0 → scale=2^-127 (min)
  byte=127 → scale=1
  byte=254 → scale=2^127 (max)
  byte=255 → NaN (unused here)

Scale selection per block:
  block_max = max(|w_i|) over the 32 weights in the block
  e = ceil(log2(block_max / fp_max))        ← aligns fp_max to block_max
  scale = 2^e

Element formats (values are w / scale, then quantised to fp_format):
  All MX element values are clipped to [-fp_max, +fp_max] before encoding.

MXFP8-E4M3  (OCP)
  bias=7, max_normal=448, 256 codes (NaN=S.1111.111 excluded → 255 usable)
  smallest normal: 2^(1-7) = 1/64
  smallest subnormal: 2^(1-7) / 8 = 1/512

MXFP8-E5M2  (OCP)
  bias=15, max_normal=57344, NaN=S.11111.10 and S.11111.11 excluded
  smallest normal: 2^(1-15) = 2^-14
  smallest subnormal: 2^-14 / 4 = 2^-16

MXFP6-E3M2  (OCP, no NaN/Inf)
  bias=3, max_normal=28.0
  normal: (1 + m/4) * 2^(e-3), e in {1..7}, m in {0..3}
  subnormal (e=0): (m/4) * 2^(1-3) = m/4 * 0.25

MXFP6-E2M3  (OCP, no NaN/Inf)
  bias=1, max_normal=7.5
  normal: (1 + m/8) * 2^(e-1), e in {1..3}, m in {0..7}
  subnormal (e=0): (m/8) * 2^(1-1) = m/8

Storage cost (B=32):
  MXFP8: 32 * 8 bits + 8 bits scale = 264 bits = 33 bytes → 8.25 bpw, 1.94× vs BF16
  MXFP6: 32 * 6 bits + 8 bits scale = 200 bits = 25 bytes → 6.25 bpw, 2.56× vs BF16

Conditions evaluated (all 40 layers, prefill, 8 calibration prompts):
  A. gate only        W_gate=MX, W_up=full, W_down=full
  B. gate + up        W_gate=MX, W_up=MX,  W_down=full
  C. gate + down      W_gate=MX, W_up=full, W_down=MX
  D. gate + up + down W_gate=MX, W_up=MX,  W_down=MX

Four formats: MXFP8-E4M3, MXFP8-E5M2, MXFP6-E3M2, MXFP6-E2M3.

Usage::

    python tools/profiler/exp40_mxfp_quality_check.py \\
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

# ---------------------------------------------------------------------------
# OCP MX format descriptors
# ---------------------------------------------------------------------------

# Each format: (name, e_bits, m_bits, bias, fp_max, has_nan)
# fp_max = max representable normal magnitude
_MX_FORMATS = {
    "MXFP8-E4M3": dict(e_bits=4, m_bits=3, bias=7,
                        fp_max=448.0,   has_nan=True),
    "MXFP8-E5M2": dict(e_bits=5, m_bits=2, bias=15,
                        fp_max=57344.0, has_nan=True),
    "MXFP6-E3M2": dict(e_bits=3, m_bits=2, bias=3,
                        fp_max=28.0,    has_nan=False),
    "MXFP6-E2M3": dict(e_bits=2, m_bits=3, bias=1,
                        fp_max=7.5,     has_nan=False),
}


def _build_mx_lut(fmt: dict) -> torch.Tensor:
    """Build sorted non-negative LUT for an MX element format.

    Returns float32 tensor of all representable non-negative magnitudes,
    sorted ascending.  NaN codes are excluded.
    """
    e_bits = fmt["e_bits"]; m_bits = fmt["m_bits"]
    bias   = fmt["bias"];   fp_max = fmt["fp_max"]
    n_exp  = 1 << e_bits    # 2^e_bits exponent codes
    n_man  = 1 << m_bits    # 2^m_bits mantissa codes

    vals = set()
    for e in range(n_exp):
        for m in range(n_man):
            if e == 0:
                # subnormal: (m / n_man) * 2^(1 - bias)
                v = (m / n_man) * (2.0 ** (1 - bias))
            else:
                # normal: (1 + m / n_man) * 2^(e - bias)
                v = (1.0 + m / n_man) * (2.0 ** (e - bias))
            if v <= fp_max:          # exclude NaN / Inf codes implicitly
                vals.add(round(v, 12))

    lut = torch.tensor(sorted(vals), dtype=torch.float32)
    return lut


# Pre-build LUTs at module load
_LUTS: dict[str, torch.Tensor] = {
    name: _build_mx_lut(fmt) for name, fmt in _MX_FORMATS.items()
}


# ---------------------------------------------------------------------------
# E8M0 scale selection  (OCP spec §3.1)
# ---------------------------------------------------------------------------

def _e8m0_scale(block_max: torch.Tensor, fp_max: float) -> torch.Tensor:
    """OCP E8M0 scale for each block.

    scale = 2^e where e = ceil(log2(block_max / fp_max))
    Clamps to E8M0 representable range [2^-127, 2^127].

    Args:
        block_max: (n_blocks,) maximum absolute value per block.
        fp_max:    maximum representable magnitude of the element format.

    Returns:
        (n_blocks,) float32 power-of-two scales.
    """
    block_max = block_max.clamp(min=EPS)
    log2_ratio = torch.log2(block_max / fp_max)
    e = torch.ceil(log2_ratio).clamp(-127, 127).to(torch.int32)
    return (2.0 ** e.float()).clamp(min=2.0**-127)


# ---------------------------------------------------------------------------
# MX nearest-neighbour quantisation (vectorised)
# ---------------------------------------------------------------------------

def _mx_quantise_blocks(wa_scaled: torch.Tensor,
                         lut: torch.Tensor) -> torch.Tensor:
    """Round (n_blocks, B) scaled magnitudes to nearest MX element value.

    Uses searchsorted on the sorted LUT — O(log N_lut) per element, no
    full broadcast.

    Args:
        wa_scaled: (n_blocks, B) non-negative magnitudes after dividing by scale.
                   Values > lut.max() are clipped to lut.max() (fp_max).
        lut:       sorted non-negative LUT for the target format.

    Returns:
        (n_blocks, B) quantised magnitudes in the same units as wa_scaled.
    """
    flat   = wa_scaled.reshape(-1).clamp(max=lut[-1])  # clip to fp_max
    idx_r  = torch.searchsorted(lut, flat).clamp(0, len(lut) - 1)
    idx_l  = (idx_r - 1).clamp(0, len(lut) - 1)
    vl     = lut[idx_l];  vr = lut[idx_r]
    q_flat = torch.where((flat - vl) >= (vr - flat), vr, vl)
    return q_flat.reshape(wa_scaled.shape)


# ---------------------------------------------------------------------------
# MX block encoding: build reconstructed float32 weight matrix
# ---------------------------------------------------------------------------

def build_mx_encoded(W: torch.Tensor, fmt_name: str,
                      B: int = 32) -> torch.Tensor:
    """Encode W (O, I) with OCP MX format and E8M0 block scale.

    Algorithm:
      1. Partition W into blocks of B consecutive weights along dim 1.
      2. Compute E8M0 scale per block from block max.
      3. Divide by scale, quantise each element to the MX element format.
      4. Reconstruct: W_enc = sign(w) * quant(|w|/scale) * scale

    Args:
        W:        (O, I) float32 weight matrix.
        fmt_name: one of the keys in _MX_FORMATS.
        B:        block size (default 32 per OCP spec).

    Returns:
        W_enc: (O, I) float32 reconstructed approximation.
    """
    fmt    = _MX_FORMATS[fmt_name]
    fp_max = fmt["fp_max"]
    lut    = _LUTS[fmt_name].to(W.device)

    O, I = W.shape
    pad  = (B - I % B) % B
    Wp   = F.pad(W, (0, pad)) if pad else W
    n_row_blocks = Wp.shape[1] // B
    n_blocks     = O * n_row_blocks

    W_b    = Wp.reshape(n_blocks, B)                  # (n_blocks, B)
    signs  = W_b.sign()                               # (n_blocks, B)
    wa     = W_b.abs()                                # (n_blocks, B)

    # E8M0 scale from block max
    block_max = wa.max(dim=1).values                  # (n_blocks,)
    scale     = _e8m0_scale(block_max, fp_max)        # (n_blocks,)
    s_exp     = scale.unsqueeze(1)                    # (n_blocks, 1)

    # Quantise
    wa_scaled = wa / s_exp                            # (n_blocks, B)
    q_mag     = _mx_quantise_blocks(wa_scaled, lut)   # (n_blocks, B)
    W_enc     = signs * q_mag * s_exp                 # (n_blocks, B)

    return W_enc.reshape(O, Wp.shape[1])[:, :I].contiguous()


# ---------------------------------------------------------------------------
# Thermal match  (identical to exp34–39)
# ---------------------------------------------------------------------------

def thermal_match(lf, lh, temperature=0.7, tau=LN2):
    baseline_top1 = lf.argmax(-1);  hybrid_top1 = lh.argmax(-1)
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
# MLP wrapper  (no routing — uniform encoding)
# ---------------------------------------------------------------------------

class EncodedMxMLP:
    def __init__(self, mlp, enc_gate, enc_up, enc_down):
        self._mlp = mlp; self._enc_gate = enc_gate
        self._enc_up = enc_up; self._enc_down = enc_down
        self._I = mlp.gate_up_proj.weight.shape[0] // 2

    def __call__(self, x):
        orig = x.dtype; xf = x.float(); I = self._I
        W  = self._mlp.gate_up_proj.weight.detach().float()
        Wg = self._enc_gate if self._enc_gate is not None else W[:I]
        Wu = self._enc_up   if self._enc_up   is not None else W[I:]
        gate = F.silu(xf @ Wg.T)
        up   = xf @ Wu.T
        act  = gate * up
        if self._enc_down is not None:
            out = act @ self._enc_down.T
        else:
            out = act @ self._mlp.down_proj.weight.detach().float().T
        return out.to(orig)


# ---------------------------------------------------------------------------
# LogitCapture / helpers  (identical to exp34–39)
# ---------------------------------------------------------------------------

class LogitCapture:
    def __init__(self, W_U):
        self._W_U = W_U.float(); self.logits = []; self._handle = None
    def attach(self, m):   self._handle = m.register_forward_hook(self._hook)
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
    llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0.0), use_tqdm=False)
    cap.detach()
    return cap.all_logits(len(prompts))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Exp40: OCP MXFP8/MXFP6 static encoding quality check.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--model",           default="ibm-granite/granite-4.2-3b")
    p.add_argument("--calibration-set", default="bartowski-imatrix-v5-semantic.txt")
    p.add_argument("--num-prompts",     type=int, default=8)
    p.add_argument("--temperatures",    nargs="+", type=float, default=[0.7, 1.0])
    p.add_argument("--tau",             type=float, default=LN2)
    p.add_argument("--block-size",      type=int, default=32,
                   help="MX block size (default 32 per OCP spec).")
    p.add_argument("--formats",         nargs="+",
                   default=["MXFP8-E4M3","MXFP8-E5M2","MXFP6-E3M2","MXFP6-E2M3"],
                   choices=list(_MX_FORMATS.keys()))
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = _parse_args(argv)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    print(f"Device: {DEV}  B={args.block_size}  formats={args.formats}", file=sys.stderr)
    for name in args.formats:
        lut = _LUTS[name]
        print(f"  {name}: {len(lut)} codes, max={lut[-1]:.1f}, "
              f"min_nz={lut[lut>0].min():.6f}", file=sys.stderr)

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
    temps = args.temperatures

    # Conditions (same as exp38/39)
    conditions: list[tuple[str, bool, bool, bool]] = [
        ("gate_only     (gate=MX, up=full, down=full)", True,  False, False),
        ("gate+up       (gate=MX, up=MX,  down=full)",  True,  True,  False),
        ("gate+down     (gate=MX, up=full, down=MX)",   True,  False, True),
        ("gate+up+down  (all=MX)",                      True,  True,  True),
    ]

    # Baseline
    print("\nBaseline pass ...", file=sys.stderr)
    lf_base = _run_logits(llm, prompts, W_U, norm)
    n_tok   = lf_base.shape[0]
    print(f"  {n_tok} prefill tokens, vocab={lf_base.shape[1]}", file=sys.stderr)

    all_results: dict[str, dict[str, dict]] = {}   # fmt → label → rec

    for fmt_name in args.formats:
        print(f"\n{'='*70}", file=sys.stderr)
        print(f"Format: {fmt_name}", file=sys.stderr)

        # Build encoded matrices for all layers
        enc_gate = []; enc_up = []; enc_down = []
        for mname, store, extractor in [
            ("gate", enc_gate, lambda l: l.mlp.gate_up_proj.weight.detach().float()[:I]),
            ("up",   enc_up,   lambda l: l.mlp.gate_up_proj.weight.detach().float()[I:]),
            ("down", enc_down, lambda l: l.mlp.down_proj.weight.detach().float()),
        ]:
            print(f"  Encoding W_{mname} ...", end=" ", file=sys.stderr, flush=True)
            t0 = time.time()
            for layer in layers:
                store.append(build_mx_encoded(extractor(layer), fmt_name, B).cpu())
            print(f"{time.time()-t0:.0f}s", file=sys.stderr)

        results: dict[str, dict] = {}

        for idx, (label, use_g, use_u, use_d) in enumerate(conditions):
            hybrids = [
                EncodedMxMLP(
                    layers[li].mlp,
                    enc_gate[li] if use_g else None,
                    enc_up[li]   if use_u else None,
                    enc_down[li] if use_d else None,
                )
                for li in range(len(layers))
            ]
            for l, h in zip(layers, hybrids): l.mlp.forward = h

            print(f"  [{idx+1}/{len(conditions)}] {label} ...",
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
                  "  ".join(f"th@{T}={rec[f'thermal_{T}']:.4f}" for T in temps),
                  file=sys.stderr)
            for l, fwd in zip(layers, orig_forwards): l.mlp.forward = fwd
            del hybrids

        all_results[fmt_name] = results

    # ------------------------------------------------------------------
    # Summary tables
    # ------------------------------------------------------------------
    cw = 11
    cols = ["strict"] + [f"th@T={T}" for T in temps] + [f"gap@{T}" for T in temps]

    def _row(lbl, rec, cw=cw):
        r = f"  {lbl:<46}  {rec['strict']:>{cw}.4f}"
        for T in temps: r += f"  {rec[f'thermal_{T}']:>{cw}.4f}"
        for T in temps: r += f"  {rec[f'gap_{T}']:>{cw}.4f}"
        return r

    for fmt_name, results in all_results.items():
        bpw = 8.25 if "FP8" in fmt_name else 6.25
        print(f"\n{'='*105}")
        print(f"Exp40: {fmt_name}  (B={B}, E8M0 block scale, {bpw} bpw)")
        print(f"No routing — MX encoding applied uniformly.")
        print(f"{'='*105}")
        hdr = f"  {'condition':<46}" + "".join(f"  {c:>{cw}}" for c in cols)
        print(hdr); print("  " + "-"*101)
        for label, rec in results.items():
            print(_row(label, rec))
        print()
        print("  FP8-equiv: strict ~3-5%, thermal ~2-4%  |  NVFP4-equiv: strict ~10-20%")
        print(f"{'='*105}")

    # Cross-format summary: gate_only perturbation for quick comparison
    print("\n\nCross-format summary — perturbation rates (%):")
    col_w = 14
    print(f"  {'condition':<46}" +
          "".join(f"  {f:>{col_w}}" for f in args.formats))
    print("  " + "-"*105)
    for label, _, _, _ in conditions:
        row = f"  {label:<46}"
        for fmt_name in args.formats:
            rec = all_results[fmt_name][label]
            perturb = (1 - rec["strict"]) * 100
            row += f"  {perturb:>{col_w}.1f}%"
        print(row)

    print("\n  (thermal@0.7):")
    for label, _, _, _ in conditions:
        row = f"  {label:<46}"
        for fmt_name in args.formats:
            rec = all_results[fmt_name][label]
            perturb = (1 - rec["thermal_0.7"]) * 100
            row += f"  {perturb:>{col_w}.1f}%"
        print(row)


if __name__ == "__main__":
    main()
