# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 43 – MXFP4-E2M1 and hypothetical MXFP5-E2M2 quality check.

Formats
-------
MXFP4-E2M1  (OCP standard):
  1 sign, 2 exponent bits, 1 mantissa bit.  bias=1, no NaN/Inf in MX variant.
  Normal  (e=1..3): (1 + m/2) * 2^(e-1),  m in {0,1}
  Subnormal (e=0):  (m/2) * 2^(1-1) = m/2, m in {0,1} → {0, 0.5}
  Non-negative codes (8 total): 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
  fp_max = 6.0
  Storage (B=32): 32*4 + 8 = 136 bits = 17 bytes → 4.25 bpw, 3.76× vs BF16

MXFP5-E2M2  (hypothetical, non-standard):
  1 sign, 2 exponent bits, 2 mantissa bits.  bias=1, no NaN/Inf.
  Normal  (e=1..3): (1 + m/4) * 2^(e-1),  m in {0,1,2,3}
  Subnormal (e=0):  (m/4) * 2^(1-1) = m/4, m in {0,1,2,3} → {0, 0.25, 0.5, 0.75}
  Non-negative codes (16 total):
    0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
    2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0
  fp_max = 7.0
  Storage (B=32): 32*5 + 8 = 168 bits = 21 bytes → 5.25 bpw, 3.05× vs BF16

Both use E8M0 B=32 block scale (OCP MX convention), same as exp40.

Same 4 matrix combinations as exp38–42.

Usage::

    python tools/profiler/exp43_mxfp4_mxfp5_quality_check.py \\
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


def _device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():         return torch.device("cuda")
    return torch.device("cpu")

DEV = _device()
LN2 = math.log(2)
EPS = 1e-9

# ---------------------------------------------------------------------------
# Format descriptors  (same structure as exp40)
# ---------------------------------------------------------------------------

_MX_FORMATS = {
    "MXFP4-E2M1": dict(e_bits=2, m_bits=1, bias=1, has_nan=False),
    "MXFP5-E2M2": dict(e_bits=2, m_bits=2, bias=1, has_nan=False),
    # Include MXFP6-E2M3 from exp40 as inline reference
    "MXFP6-E2M3": dict(e_bits=2, m_bits=3, bias=1, has_nan=False),
}


def _build_lut(fmt: dict) -> torch.Tensor:
    e_bits = fmt["e_bits"]; m_bits = fmt["m_bits"]; bias = fmt["bias"]
    n_exp = 1 << e_bits;    n_man = 1 << m_bits
    vals = set()
    for e in range(n_exp):
        for m in range(n_man):
            v = ((m / n_man) * (2.0 ** (1 - bias)) if e == 0
                 else (1.0 + m / n_man) * (2.0 ** (e - bias)))
            vals.add(round(v, 12))
    return torch.tensor(sorted(vals), dtype=torch.float32)


_LUTS = {name: _build_lut(fmt) for name, fmt in _MX_FORMATS.items()}

# ---------------------------------------------------------------------------
# E8M0 scale  (OCP standard — identical to exp40/41/42)
# ---------------------------------------------------------------------------

def _e8m0_scale(block_max: torch.Tensor, fp_max: float) -> torch.Tensor:
    e = torch.ceil(torch.log2(block_max.clamp(min=EPS) / fp_max)).clamp(-127, 127)
    return (2.0 ** e).clamp(min=2.0 ** -127)


def _fp_max(fmt_name: str) -> float:
    lut = _LUTS[fmt_name]
    return float(lut[-1])


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def _quantise_blocks(wa_scaled: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    flat  = wa_scaled.reshape(-1).clamp(max=lut[-1])
    idx_r = torch.searchsorted(lut, flat).clamp(0, len(lut) - 1)
    idx_l = (idx_r - 1).clamp(0, len(lut) - 1)
    vl = lut[idx_l]; vr = lut[idx_r]
    q  = torch.where((flat - vl) >= (vr - flat), vr, vl)
    return q.reshape(wa_scaled.shape)


def build_mx_encoded(W: torch.Tensor, fmt_name: str, B: int = 32) -> torch.Tensor:
    fmt    = _MX_FORMATS[fmt_name]
    fp_max = _fp_max(fmt_name)
    lut    = _LUTS[fmt_name].to(W.device)
    O, I   = W.shape
    pad    = (B - I % B) % B
    Wp     = F.pad(W, (0, pad)) if pad else W
    W_b    = Wp.reshape(-1, B)
    signs  = W_b.sign()
    wa     = W_b.abs()
    scale  = _e8m0_scale(wa.max(1).values, fp_max).unsqueeze(1)
    W_enc  = signs * _quantise_blocks(wa / scale, lut) * scale
    return W_enc.reshape(O, Wp.shape[1])[:, :I].contiguous()


# ---------------------------------------------------------------------------
# Thermal match / MLP wrapper / infra  (identical to exp40–42)
# ---------------------------------------------------------------------------

def thermal_match(lf, lh, temperature=0.7, tau=LN2):
    bt = lf.argmax(-1); ht = lh.argmax(-1); exact = (bt == ht)
    gap = (lh.gather(-1, ht.unsqueeze(-1)).squeeze(-1) -
           lh.gather(-1, bt.unsqueeze(-1)).squeeze(-1))
    forgiven = (~exact) & (gap < temperature * tau)
    strict   = float(exact.float().mean())
    thermal  = float((exact | forgiven).float().mean())
    gaps     = gap[~exact]
    return strict, thermal, float(gaps.mean()) if gaps.numel() > 0 else 0.0


class EncodedMLP:
    def __init__(self, mlp, enc_gate, enc_up, enc_down):
        self._mlp = mlp; self._enc_gate = enc_gate
        self._enc_up = enc_up; self._enc_down = enc_down
        self._I = mlp.gate_up_proj.weight.shape[0] // 2

    def __call__(self, x):
        orig = x.dtype; xf = x.float(); I = self._I
        W  = self._mlp.gate_up_proj.weight.detach().float()
        Wg = self._enc_gate if self._enc_gate is not None else W[:I]
        Wu = self._enc_up   if self._enc_up   is not None else W[I:]
        act = F.silu(xf @ Wg.T) * (xf @ Wu.T)
        wd  = (self._enc_down if self._enc_down is not None
               else self._mlp.down_proj.weight.detach().float())
        return (act @ wd.T).to(orig)


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
        description="Exp43: MXFP4-E2M1 and MXFP5-E2M2 quality check.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--model",           default="ibm-granite/granite-4.2-3b")
    p.add_argument("--calibration-set", default="bartowski-imatrix-v5-semantic.txt")
    p.add_argument("--num-prompts",     type=int, default=8)
    p.add_argument("--temperatures",    nargs="+", type=float, default=[0.7, 1.0])
    p.add_argument("--tau",             type=float, default=LN2)
    p.add_argument("--block-size",      type=int, default=32)
    p.add_argument("--formats",         nargs="+",
                   default=["MXFP4-E2M1", "MXFP5-E2M2", "MXFP6-E2M3"],
                   choices=list(_MX_FORMATS.keys()))
    return p.parse_args(argv)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = _parse_args(argv)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    print(f"Device: {DEV}  B={args.block_size}", file=sys.stderr)
    for name in args.formats:
        lut = _LUTS[name]
        bpw = int(name[4]) + 0.25   # e.g. MXFP4 → 4.25 bpw
        print(f"  {name}: {len(lut)} codes, max={lut[-1]:.2f}, "
              f"min_nz={lut[lut>0].min():.4f}, ~{bpw:.2f} bpw", file=sys.stderr)

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

    conditions: list[tuple[str, bool, bool, bool]] = [
        ("gate_only     (gate=MX, up=full, down=full)", True,  False, False),
        ("gate+up       (gate=MX, up=MX,  down=full)",  True,  True,  False),
        ("gate+down     (gate=MX, up=full, down=MX)",   True,  False, True),
        ("gate+up+down  (all=MX)",                      True,  True,  True),
    ]

    print("\nBaseline pass ...", file=sys.stderr)
    lf_base = _run_logits(llm, prompts, W_U, norm)
    n_tok   = lf_base.shape[0]
    print(f"  {n_tok} tokens, vocab={lf_base.shape[1]}", file=sys.stderr)

    all_results: dict[str, dict[str, dict]] = {}

    for fmt_name in args.formats:
        print(f"\n{'='*60}\nFormat: {fmt_name}\n{'='*60}", file=sys.stderr)

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
                EncodedMLP(layers[li].mlp,
                           enc_gate[li] if use_g else None,
                           enc_up[li]   if use_u else None,
                           enc_down[li] if use_d else None)
                for li in range(len(layers))
            ]
            for l, h in zip(layers, hybrids): l.mlp.forward = h
            print(f"  [{idx+1}/{len(conditions)}] {label} ...",
                  end="  ", file=sys.stderr, flush=True)
            lh = _run_logits(llm, prompts, W_U, norm)[:n_tok]

            rec: dict = {}
            s, _, _ = thermal_match(lf_base, lh, 1.0, 0.0); rec["strict"] = s
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
    # Summary
    # ------------------------------------------------------------------
    cw = 11
    print(f"\n\n{'='*115}")
    print("Exp43: MXFP4-E2M1 / MXFP5-E2M2 / MXFP6-E2M3  (B=32, E8M0 scale)")
    print(f"{'='*115}")
    hdr = (f"  {'condition':<46}" +
           "".join(f"  {f:>20}" for f in args.formats))
    print(hdr + "   (strict% / th@0.7%)")
    print("  " + "-"*111)
    for label, _, _, _ in conditions:
        row = f"  {label:<46}"
        for fmt_name in args.formats:
            rec = all_results[fmt_name][label]
            sp = (1 - rec["strict"]) * 100
            tp = (1 - rec["thermal_0.7"]) * 100
            row += f"  {sp:>8.1f}% /{tp:>6.1f}%"
        print(row)

    # Mean gap table
    print(f"\n  Mean logit gap (perturbed tokens):")
    for label, _, _, _ in conditions:
        row = f"  {label:<46}"
        for fmt_name in args.formats:
            row += f"  {all_results[fmt_name][label]['gap_0.7']:>19.3f}L"
        print(row)

    # Full cross-format progression table incl. exp40 MXFP8 reference
    exp40_ref = {
        "MXFP8-E4M3": {
            "gate_only":   (0.984, 1.000, 0.038),
            "gate+up":     (0.972, 1.000, 0.101),
            "gate+down":   (0.974, 0.998, 0.116),
            "gate+up+down":(0.964, 0.996, 0.172),
        },
    }
    lbl_map = {
        "gate_only     (gate=MX, up=full, down=full)": "gate_only",
        "gate+up       (gate=MX, up=MX,  down=full)":  "gate+up",
        "gate+down     (gate=MX, up=full, down=MX)":   "gate+down",
        "gate+up+down  (all=MX)":                      "gate+up+down",
    }
    bpw_map = {"MXFP4-E2M1": "4.25", "MXFP5-E2M2": "5.25",
               "MXFP6-E2M3": "6.25", "MXFP8-E4M3": "8.25"}

    print(f"\n\nFull progression: strict% / thermal@0.7%  (E8M0 B=32 throughout)")
    cols = ["MXFP4-E2M1", "MXFP5-E2M2", "MXFP6-E2M3", "MXFP8-E4M3"]
    hdr2 = f"  {'condition':<28}" + "".join(
        f"  {c+' ('+bpw_map.get(c,'?')+'bpw)':>22}" for c in cols)
    print(hdr2)
    print("  " + "-"*118)
    for label, _, _, _ in conditions:
        k = lbl_map[label]
        short = label.split("(")[0].strip()
        row = f"  {short:<28}"
        for fmt_name in cols:
            if fmt_name in all_results:
                rec = all_results[fmt_name][label]
                sp = (1 - rec["strict"]) * 100
                tp = (1 - rec["thermal_0.7"]) * 100
            else:
                s, t, _ = exp40_ref["MXFP8-E4M3"][k]
                sp = (1 - s) * 100; tp = (1 - t) * 100
            row += f"  {sp:>10.1f}% /{tp:>6.1f}%"
        print(row)

    print(f"\n  FP8-equiv: strict ~3-5%  |  NVFP4-equiv: strict ~10-20%")
    print(f"{'='*115}")


if __name__ == "__main__":
    main()
