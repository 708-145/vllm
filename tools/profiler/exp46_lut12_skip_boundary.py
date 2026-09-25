# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 46 – LUT12 per-channel with boundary layers at full precision.

Same as exp45 (LUT12, 12 non-neg magnitudes, Lloyd-Max per channel) but layers
0–(fp_first-1) and (N-fp_last)–(N-1) are left at full BF16 precision.

Default: first 5 and last 3 layers full precision, middle 32 encoded.

Comparison points
-----------------
  exp45  LUT12 all 40 layers  :  strict=9.2%   thermal@0.7=3.4%   5.25 bpw
  exp46  LUT12 layers 5–36    :  ?              ?                  < 5.25 bpw*

  * bpw is a weighted average; full-precision layers cost 16 bpw.
    Effective bpw = (n_enc × 5.25 + n_full × 16) / 40

Usage::

    python tools/profiler/exp46_lut12_skip_boundary.py \\
        --model ibm-granite/granite-4.2-3b \\
        --calibration-set bartowski-imatrix-v5-semantic.txt \\
        --fp-first 5 --fp-last 3
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch

# Reuse all encoding / inference helpers from exp45
sys.path.insert(0, str(Path(__file__).parent))
from exp45_lut12_per_channel import (
    ENC_DEV, LN2,
    build_mxfp4, build_lut12_per_channel,
    thermal_match, EncodedMLP, LogitCapture,
    _load_prompts, _get_internals, _run_logits,
)


def _effective_bpw(n_total: int, n_encoded: int,
                   enc_bpw: float = 5.25, full_bpw: float = 16.0) -> float:
    n_full = n_total - n_encoded
    return (n_encoded * enc_bpw + n_full * full_bpw) / n_total


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",           default="ibm-granite/granite-4.2-3b")
    p.add_argument("--calibration-set", default="bartowski-imatrix-v5-semantic.txt")
    p.add_argument("--num-prompts",     type=int, default=8)
    p.add_argument("--n-levels",        type=int, default=12)
    p.add_argument("--lloyd-iters",     type=int, default=30)
    p.add_argument("--chunk",           type=int, default=512)
    p.add_argument("--fp-first",        type=int, default=5,
                   help="Keep first N layers at full precision (default 5).")
    p.add_argument("--fp-last",         type=int, default=3,
                   help="Keep last N layers at full precision (default 3).")
    args = p.parse_args(argv)

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    print(f"Encoding device: {ENC_DEV}", file=sys.stderr)

    from vllm import LLM
    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              kv_cache_memory_bytes=int(0.5 * 1024**3), max_model_len=512,
              enable_prefix_caching=False)
    norm, layers, W_U = _get_internals(llm)
    orig_forwards = [l.mlp.forward for l in layers]
    n_layers = len(layers)
    I_dim    = layers[0].mlp.gate_up_proj.weight.shape[0] // 2

    # Which layers to encode vs leave full precision
    fp_first  = args.fp_first
    fp_last   = args.fp_last
    enc_start = fp_first
    enc_end   = n_layers - fp_last          # exclusive
    enc_idxs  = list(range(enc_start, enc_end))
    n_encoded = len(enc_idxs)
    print(f"\nLayers {enc_start}–{enc_end-1} encoded ({n_encoded}/{n_layers}); "
          f"layers 0–{fp_first-1} and {enc_end}–{n_layers-1} full precision.",
          file=sys.stderr)

    prompts = _load_prompts(args.calibration_set, args.num_prompts)
    print(f"Loaded {len(prompts)} prompts.", file=sys.stderr)

    # ------------------------------------------------------------------
    # Encode only the middle layers (full-precision layers get None)
    # ------------------------------------------------------------------
    print(f"\nLUT{args.n_levels} encoding layers {enc_start}–{enc_end-1} "
          f"({ENC_DEV}) ...", file=sys.stderr)
    t0 = time.time()
    lut_gate  = [None] * n_layers
    lut_up    = [None] * n_layers
    lut_down  = [None] * n_layers
    for li in enc_idxs:
        layer = layers[li]
        Wgu   = layer.mlp.gate_up_proj.weight.detach().float()
        Wd    = layer.mlp.down_proj.weight.detach().float()
        t1    = time.time()
        eg, _ = build_lut12_per_channel(Wgu[:I_dim], args.n_levels,
                                        args.lloyd_iters, chunk=args.chunk)
        eu, _ = build_lut12_per_channel(Wgu[I_dim:], args.n_levels,
                                        args.lloyd_iters, chunk=args.chunk)
        ed, _ = build_lut12_per_channel(Wd,           args.n_levels,
                                        args.lloyd_iters, chunk=args.chunk)
        lut_gate[li] = eg; lut_up[li] = eu; lut_down[li] = ed
        print(f"  layer {li:2d}  {time.time()-t1:.1f}s", file=sys.stderr)
    print(f"  total {time.time()-t0:.0f}s", file=sys.stderr)

    # ------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------
    print("\nBaseline pass ...", file=sys.stderr)
    lf_base = _run_logits(llm, prompts, W_U, norm)
    n_tok   = lf_base.shape[0]
    print(f"  {n_tok} tokens", file=sys.stderr)

    # ------------------------------------------------------------------
    # Quality runs
    # ------------------------------------------------------------------
    results = {}

    # (a) LUT12 middle layers only (boundary = full precision)
    label_partial = (f"LUT{args.n_levels} layers {enc_start}–{enc_end-1} "
                     f"(fp: 0–{fp_first-1}, {enc_end}–{n_layers-1})")
    hybrids = [EncodedMLP(layers[li].mlp,
                          lut_gate[li], lut_up[li], lut_down[li])
               for li in range(n_layers)]
    for l, h in zip(layers, hybrids): l.mlp.forward = h
    lh = _run_logits(llm, prompts, W_U, norm)[:n_tok]
    s07, t07 = thermal_match(lf_base, lh, 0.7)
    _,   t10 = thermal_match(lf_base, lh, 1.0)
    results[label_partial] = (s07, t07, t10)
    print(f"  {label_partial}: strict={1-s07:.4f}  "
          f"th@0.7={1-t07:.4f}  th@1.0={1-t10:.4f}", file=sys.stderr)
    for l, fwd in zip(layers, orig_forwards): l.mlp.forward = fwd
    del hybrids

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    eff_bpw      = _effective_bpw(n_layers, n_encoded)
    eff_bpw_all  = 5.25   # exp45 LUT12 all layers

    ref_rows = [
        # label, strict_match, th07_match, th10_match, bpw_str
        ("MXFP4-E2M1 all layers (exp43 ref)",      0.828, 0.914, 0.924, "4.25"),
        ("LUT12 all 40 layers (exp45 ref)",         0.908, 0.966, 0.972, f"{eff_bpw_all:.2f}"),
        ("MXFP6-E2M3 all layers (exp43 ref)",       0.958, 0.992, 0.994, "6.25"),
    ]

    print(f"\n\n{'='*80}")
    print(f"Exp46: LUT{args.n_levels} boundary-skip  "
          f"(fp-first={fp_first}, fp-last={fp_last}, "
          f"encoded={n_encoded}/{n_layers} layers)")
    print(f"{'='*80}")
    print(f"  {'Scheme':<52}  {'strict%':>7}  {'th@0.7%':>8}  {'th@1.0%':>8}  bpw")
    print(f"  {'-'*76}")
    for label, (s, t07, t10) in results.items():
        print(f"  {label:<52}  {(1-s)*100:>6.1f}%  {(1-t07)*100:>7.1f}%"
              f"  {(1-t10)*100:>7.1f}%  {eff_bpw:.3f}")
    for label, s, t07, t10, bpw in ref_rows:
        print(f"  {label:<52}  {(1-s)*100:>6.1f}%  {(1-t07)*100:>7.1f}%"
              f"  {(1-t10)*100:>7.1f}%  {bpw}")
    print(f"\n  Effective bpw (MLP weights only): {eff_bpw:.3f}  "
          f"({n_encoded} × 5.25 + {n_layers-n_encoded} × 16) / {n_layers}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
