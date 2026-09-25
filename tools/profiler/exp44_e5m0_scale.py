# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 44 – E5M0 vs E8M0 block scale: does the range matter?

Isolates the scale exponent width.  Element formats fixed to the E2Mx family
(MXFP4-E2M1, MXFP5-E2M2, MXFP6-E2M3); B=32 throughout.

E8M0 scale  (OCP standard, exp40/43):
  8-bit, value = 2^(byte − 127),  e ∈ [−127, 127]
  Round: e = ceil(log2(block_max / fp_max)), clamped to [−127, 127]

E5M0 scale  (this experiment):
  5-bit, value = 2^(byte − 15),   e ∈ [−15, 15]  (analogous to FP16 exponent)
  Round: same ceil(log2(…)) rule, clamped to [−15, 15]

Storage cost difference (B=32):
  E8M0: 8 bits / block  →  4.25 / 5.25 / 6.25 bpw (as in exp40/43)
  E5M0: 5 bits / block  →  4.16 / 5.16 / 6.16 bpw  (~0.1 bpw cheaper)

If E5M0 produces the same results as E8M0, the extra 3 bits of scale range
are wasted for typical LLM weight distributions.  If it degrades, the extended
range is necessary (e.g. for outlier blocks with very large or very small
weights that fall outside [2^−15, 2^15]).

Usage::

    python tools/profiler/exp44_e5m0_scale.py \\
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
# E2Mx LUTs  (identical to exp43)
# ---------------------------------------------------------------------------

_MX_FORMATS = {
    "MXFP4-E2M1": dict(e_bits=2, m_bits=1, bias=1, fp_max=6.0),
    "MXFP5-E2M2": dict(e_bits=2, m_bits=2, bias=1, fp_max=7.0),
    "MXFP6-E2M3": dict(e_bits=2, m_bits=3, bias=1, fp_max=7.5),
}

def _build_lut(fmt):
    n_exp = 1 << fmt["e_bits"]; n_man = 1 << fmt["m_bits"]; bias = fmt["bias"]
    vals = set()
    for e in range(n_exp):
        for m in range(n_man):
            v = ((m / n_man) * (2.0 ** (1 - bias)) if e == 0
                 else (1.0 + m / n_man) * (2.0 ** (e - bias)))
            vals.add(round(v, 12))
    return torch.tensor(sorted(vals), dtype=torch.float32)

_LUTS = {name: _build_lut(fmt) for name, fmt in _MX_FORMATS.items()}

# ---------------------------------------------------------------------------
# Scale variants
# ---------------------------------------------------------------------------

def _e8m0_scale(block_max: torch.Tensor, fp_max: float) -> torch.Tensor:
    """OCP E8M0: 2^ceil(log2(block_max/fp_max)), e clamped to [−127, 127]."""
    e = torch.ceil(torch.log2(block_max.clamp(min=EPS) / fp_max)).clamp(-127, 127)
    return 2.0 ** e

def _e5m0_scale(block_max: torch.Tensor, fp_max: float) -> torch.Tensor:
    """E5M0: same rule, e clamped to [−15, 15] (5-bit unsigned with bias 15)."""
    e = torch.ceil(torch.log2(block_max.clamp(min=EPS) / fp_max)).clamp(-15, 15)
    return 2.0 ** e

_SCALE_FNS = {
    "E8M0": _e8m0_scale,
    "E5M0": _e5m0_scale,
}

# ---------------------------------------------------------------------------
# Encoding / thermal match / MLP / infra  (identical to exp40–43)
# ---------------------------------------------------------------------------

def _quantise_blocks(wa_scaled, lut):
    flat  = wa_scaled.reshape(-1).clamp(max=lut[-1])
    idx_r = torch.searchsorted(lut, flat).clamp(0, len(lut) - 1)
    idx_l = (idx_r - 1).clamp(0, len(lut) - 1)
    vl = lut[idx_l]; vr = lut[idx_r]
    return torch.where((flat - vl) >= (vr - flat), vr, vl).reshape(wa_scaled.shape)

def build_mx_encoded(W, fmt_name, scale_name, B=32):
    fmt    = _MX_FORMATS[fmt_name]
    lut    = _LUTS[fmt_name].to(W.device)
    scale_fn = _SCALE_FNS[scale_name]
    O, I   = W.shape
    pad    = (B - I % B) % B
    Wp     = F.pad(W, (0, pad)) if pad else W
    W_b    = Wp.reshape(-1, B)
    signs  = W_b.sign()
    wa     = W_b.abs()
    scale  = scale_fn(wa.max(1).values, fmt["fp_max"]).unsqueeze(1)
    W_enc  = signs * _quantise_blocks(wa / scale, lut) * scale
    return W_enc.reshape(O, Wp.shape[1])[:, :I].contiguous()

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
    p = argparse.ArgumentParser(description="Exp44: E5M0 vs E8M0 block scale.")
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

    conditions = [
        ("gate_only     (gate=MX, up=full, down=full)", True,  False, False),
        ("gate+up       (gate=MX, up=MX,  down=full)",  True,  True,  False),
        ("gate+down     (gate=MX, up=full, down=MX)",   True,  False, True),
        ("gate+up+down  (all=MX)",                      True,  True,  True),
    ]

    print("\nBaseline pass ...", file=sys.stderr)
    lf_base = _run_logits(llm, prompts, W_U, norm)
    n_tok   = lf_base.shape[0]
    print(f"  {n_tok} tokens, vocab={lf_base.shape[1]}", file=sys.stderr)

    # First, diagnose how many blocks actually need exponents outside [−15,15]
    print("\nScale range diagnostics (fraction of blocks with |e| > 15):",
          file=sys.stderr)
    for fmt_name in args.formats:
        fp_max = _MX_FORMATS[fmt_name]["fp_max"]
        clipped = 0; total = 0
        for layer in layers:
            for W in [layer.mlp.gate_up_proj.weight.detach().float()[:I],
                      layer.mlp.gate_up_proj.weight.detach().float()[I:],
                      layer.mlp.down_proj.weight.detach().float()]:
                O2, I2 = W.shape
                pad = (B - I2 % B) % B
                Wp  = F.pad(W, (0, pad)) if pad else W
                wa  = Wp.reshape(-1, B).abs()
                bmax = wa.max(1).values.clamp(min=EPS)
                e    = torch.ceil(torch.log2(bmax / fp_max))
                clipped += int((e.abs() > 15).sum())
                total   += e.numel()
        print(f"  {fmt_name}: {clipped}/{total} blocks clipped "
              f"({100*clipped/total:.3f}%)", file=sys.stderr)

    all_results: dict[str, dict[str, dict[str, dict]]] = {}  # scale→fmt→label→rec

    for scale_name in ["E8M0", "E5M0"]:
        all_results[scale_name] = {}
        for fmt_name in args.formats:
            print(f"\n{'='*55}\n{fmt_name}  scale={scale_name}\n{'='*55}",
                  file=sys.stderr)

            enc_gate = []; enc_up = []; enc_down = []
            for mname, store, extractor in [
                ("gate", enc_gate, lambda l: l.mlp.gate_up_proj.weight.detach().float()[:I]),
                ("up",   enc_up,   lambda l: l.mlp.gate_up_proj.weight.detach().float()[I:]),
                ("down", enc_down, lambda l: l.mlp.down_proj.weight.detach().float()),
            ]:
                print(f"  W_{mname} ...", end=" ", file=sys.stderr, flush=True)
                t0 = time.time()
                for layer in layers:
                    store.append(
                        build_mx_encoded(extractor(layer), fmt_name, scale_name, B).cpu())
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

            all_results[scale_name][fmt_name] = results

    # ------------------------------------------------------------------
    # Summary: Δ (E5M0 − E8M0) per format × condition
    # ------------------------------------------------------------------
    lbl_map = {
        "gate_only     (gate=MX, up=full, down=full)": "gate_only",
        "gate+up       (gate=MX, up=MX,  down=full)":  "gate+up",
        "gate+down     (gate=MX, up=full, down=MX)":   "gate+down",
        "gate+up+down  (all=MX)":                      "gate+up+down",
    }

    print(f"\n\n{'='*120}")
    print("Exp44: E5M0 vs E8M0 block scale  —  strict% / thermal@0.7%  (B=32)")
    print(f"{'='*120}")

    for fmt_name in args.formats:
        print(f"\n  {fmt_name}")
        print(f"  {'condition':<46}  {'E8M0':>16}  {'E5M0':>16}  {'Δ strict':>10}  {'Δ th@0.7':>10}")
        print("  " + "-"*100)
        for label, _, _, _ in conditions:
            r8 = all_results["E8M0"][fmt_name][label]
            r5 = all_results["E5M0"][fmt_name][label]
            sp8 = (1 - r8["strict"]) * 100; tp8 = (1 - r8["thermal_0.7"]) * 100
            sp5 = (1 - r5["strict"]) * 100; tp5 = (1 - r5["thermal_0.7"]) * 100
            ds  = sp5 - sp8; dt = tp5 - tp8
            print(f"  {label:<46}  {sp8:>6.1f}%/{tp8:>5.1f}%"
                  f"  {sp5:>6.1f}%/{tp5:>5.1f}%"
                  f"  {ds:>+9.1f}pp  {dt:>+9.1f}pp")

    # Compact all-matrix cross table
    print(f"\n\n  All-matrix (gate+up+down) — strict% / thermal@0.7%:")
    print(f"  {'format':<14}" +
          "".join(f"  {'E8M0':>14}  {'E5M0':>14}  {'Δ strict':>10}" for _ in [1]))
    print("  " + "-"*60)
    for fmt_name in args.formats:
        lbl = "gate+up+down  (all=MX)"
        r8 = all_results["E8M0"][fmt_name][lbl]
        r5 = all_results["E5M0"][fmt_name][lbl]
        sp8 = (1-r8["strict"])*100; tp8 = (1-r8["thermal_0.7"])*100
        sp5 = (1-r5["strict"])*100; tp5 = (1-r5["thermal_0.7"])*100
        print(f"  {fmt_name:<14}  {sp8:>6.1f}%/{tp8:>4.1f}%"
              f"  {sp5:>6.1f}%/{tp5:>4.1f}%"
              f"  {sp5-sp8:>+9.1f}pp")

    print(f"\n  FP8-equiv: strict ~3-5%  |  NVFP4-equiv: strict ~10-20%")
    print(f"{'='*120}")


if __name__ == "__main__":
    main()
