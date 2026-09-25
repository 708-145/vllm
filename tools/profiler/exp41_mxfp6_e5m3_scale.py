# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 41 – MXFP6-E2M3 with E5M3 block scale vs E8M0 block scale.

Isolates the scale format as the variable: MXFP6-E2M3 element format (same
as exp40's best bpw result) with two scale variants:

  A. E8M0 scale  (OCP standard, power-of-two, same as exp40)
  B. E5M3 scale  (fine-grained non-power-of-two, same style as exp19/22/37/39)

E5M3 scale selection: TARE-weighted geometric mean of |w| in the block,
shifted to align with the MXFP6-E2M3 grid geometric mean — same closed-form
as exp39.

E8M0 scale selection: 2^ceil(log2(block_max / fp_max)) — OCP spec §3.1.

All other parameters identical: B=32 block size, MXFP6-E2M3 element format,
same 4 matrix combinations, same 8 calibration prompts.

Usage::

    python tools/profiler/exp41_mxfp6_e5m3_scale.py \\
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
# MXFP6-E2M3 LUT  (32 non-negative codes, same as exp40)
# ---------------------------------------------------------------------------
# bias=1, fp_max=7.5
# subnormal (e=0): m/8 * 2^(1-1) = m/8  for m in 0..7
# normal (e=1..3): (1+m/8) * 2^(e-1)    for m in 0..7

def _build_e2m3_lut() -> torch.Tensor:
    vals = []
    for e in range(4):
        for m in range(8):
            v = (m / 8.0) if e == 0 else (1.0 + m / 8.0) * (2.0 ** (e - 1))
            vals.append(v)
    return torch.tensor(sorted(set(round(v, 12) for v in vals)), dtype=torch.float32)

_LUT_E2M3 = _build_e2m3_lut()   # (32,) sorted ascending, max=7.5

# ---------------------------------------------------------------------------
# E8M0 scale  (OCP standard)
# ---------------------------------------------------------------------------

def _e8m0_scale(block_max: torch.Tensor, fp_max: float = 7.5) -> torch.Tensor:
    """scale = 2^ceil(log2(block_max / fp_max)), clamped to [2^-127, 2^127]."""
    e = torch.ceil(torch.log2(block_max.clamp(min=EPS) / fp_max)).clamp(-127, 127)
    return (2.0 ** e).clamp(min=2.0 ** -127)

# ---------------------------------------------------------------------------
# E5M3 scale  (TARE-optimal, same style as exp19/22/37/39)
# ---------------------------------------------------------------------------

_M3_FRAC_LOG2 = torch.log2(1.0 + torch.arange(8).float() / 8.0)

def _quantise_e5m3(s: torch.Tensor) -> torch.Tensor:
    s = s.clamp(min=EPS); log2_s = torch.log2(s)
    e = log2_s.floor().to(torch.int32); frac = log2_s - e.float()
    m_lut = _M3_FRAC_LOG2.to(s.device)
    m_best = (frac.unsqueeze(-1) - m_lut).abs().argmin(-1)
    return (2.0 ** (e.float() + m_lut[m_best])).clamp(min=EPS)

def _e5m3_scale(wa: torch.Tensor) -> torch.Tensor:
    """TARE-weighted geometric mean aligned to MXFP6-E2M3 grid centre."""
    eps    = wa.reshape(-1).kthvalue(max(1, int(wa.numel() * 0.01))).values.clamp(min=EPS)
    tilt   = torch.log1p(wa / eps)                    # (n_blocks, B)
    log_wa = torch.log(wa.clamp(min=EPS))
    lut_nz = _LUT_E2M3[_LUT_E2M3 > 0].to(wa.device)
    fp_logmu = lut_nz.log().mean()                    # scalar: grid centre in log-space
    log_s  = ((tilt * log_wa).sum(1) / tilt.sum(1).clamp(min=EPS)) - fp_logmu
    return _quantise_e5m3(log_s.exp().clamp(min=EPS))

# ---------------------------------------------------------------------------
# Shared quantisation kernel
# ---------------------------------------------------------------------------

def _quantise_blocks(wa_scaled: torch.Tensor,
                     lut: torch.Tensor) -> torch.Tensor:
    """Nearest-neighbour quantise (n_blocks, B) to lut values via searchsorted."""
    flat  = wa_scaled.reshape(-1).clamp(max=lut[-1])
    idx_r = torch.searchsorted(lut, flat).clamp(0, len(lut) - 1)
    idx_l = (idx_r - 1).clamp(0, len(lut) - 1)
    vl = lut[idx_l]; vr = lut[idx_r]
    q   = torch.where((flat - vl) >= (vr - flat), vr, vl)
    return q.reshape(wa_scaled.shape)

# ---------------------------------------------------------------------------
# Encoding builders
# ---------------------------------------------------------------------------

def _encode(W: torch.Tensor, scale_fn, B: int = 32) -> torch.Tensor:
    lut  = _LUT_E2M3.to(W.device)
    O, I = W.shape
    pad  = (B - I % B) % B
    Wp   = F.pad(W, (0, pad)) if pad else W
    W_b  = Wp.reshape(-1, B)
    signs = W_b.sign()
    wa    = W_b.abs()
    s     = scale_fn(wa).unsqueeze(1)          # (n_blocks, 1)
    q     = _quantise_blocks(wa / s, lut) * s  # (n_blocks, B)
    return (signs * q).reshape(O, Wp.shape[1])[:, :I].contiguous()

def build_e8m0(W: torch.Tensor, B: int = 32) -> torch.Tensor:
    return _encode(W, lambda wa: _e8m0_scale(wa.max(1).values), B)

def build_e5m3(W: torch.Tensor, B: int = 32) -> torch.Tensor:
    return _encode(W, _e5m3_scale, B)

# ---------------------------------------------------------------------------
# Thermal match  (identical to exp34–40)
# ---------------------------------------------------------------------------

def thermal_match(lf, lh, temperature=0.7, tau=LN2):
    bt = lf.argmax(-1); ht = lh.argmax(-1)
    exact = (bt == ht)
    gap   = lh.gather(-1, ht.unsqueeze(-1)).squeeze(-1) - \
            lh.gather(-1, bt.unsqueeze(-1)).squeeze(-1)
    forgiven = (~exact) & (gap < temperature * tau)
    strict   = float(exact.float().mean())
    thermal  = float((exact | forgiven).float().mean())
    gaps     = gap[~exact]
    return strict, thermal, float(gaps.mean()) if gaps.numel() > 0 else 0.0

# ---------------------------------------------------------------------------
# MLP wrapper
# ---------------------------------------------------------------------------

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
        if self._enc_down is not None:
            out = act @ self._enc_down.T
        else:
            out = act @ self._mlp.down_proj.weight.detach().float().T
        return out.to(orig)

# ---------------------------------------------------------------------------
# Infra
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
    p = argparse.ArgumentParser(description="Exp41: MXFP6-E2M3 E8M0 vs E5M3 scale.")
    p.add_argument("--model",           default="ibm-granite/granite-4.2-3b")
    p.add_argument("--calibration-set", default="bartowski-imatrix-v5-semantic.txt")
    p.add_argument("--num-prompts",     type=int, default=8)
    p.add_argument("--temperatures",    nargs="+", type=float, default=[0.7, 1.0])
    p.add_argument("--tau",             type=float, default=LN2)
    p.add_argument("--block-size",      type=int, default=32)
    return p.parse_args(argv)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = _parse_args(argv)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    print(f"Device: {DEV}  B={args.block_size}", file=sys.stderr)
    print(f"MXFP6-E2M3 LUT ({len(_LUT_E2M3)} values, max={_LUT_E2M3[-1]})", file=sys.stderr)

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
        ("gate_only     (gate=E2M3, up=full, down=full)", True,  False, False),
        ("gate+up       (gate=E2M3, up=E2M3, down=full)", True,  True,  False),
        ("gate+down     (gate=E2M3, up=full, down=E2M3)", True,  False, True),
        ("gate+up+down  (all=E2M3)",                      True,  True,  True),
    ]

    print("\nBaseline pass ...", file=sys.stderr)
    lf_base = _run_logits(llm, prompts, W_U, norm)
    n_tok   = lf_base.shape[0]
    print(f"  {n_tok} tokens, vocab={lf_base.shape[1]}", file=sys.stderr)

    all_results: dict[str, dict[str, dict]] = {}

    for scale_name, builder in [("E8M0 (OCP)", build_e8m0), ("E5M3 (TARE)", build_e5m3)]:
        print(f"\n{'='*60}\nScale: {scale_name}\n{'='*60}", file=sys.stderr)

        enc_gate = []; enc_up = []; enc_down = []
        for mname, store, extractor in [
            ("gate", enc_gate, lambda l: l.mlp.gate_up_proj.weight.detach().float()[:I]),
            ("up",   enc_up,   lambda l: l.mlp.gate_up_proj.weight.detach().float()[I:]),
            ("down", enc_down, lambda l: l.mlp.down_proj.weight.detach().float()),
        ]:
            print(f"  Encoding W_{mname} ...", end=" ", file=sys.stderr, flush=True)
            t0 = time.time()
            for layer in layers:
                store.append(builder(extractor(layer), B).cpu())
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

        all_results[scale_name] = results

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    cw = 11
    print(f"\n\n{'='*105}")
    print("Exp41: MXFP6-E2M3  —  E8M0 scale vs E5M3 scale  (B=32)")
    print(f"{'='*105}")
    hdr = (f"  {'condition':<48}  {'scale':>8}" +
           "".join(f"  {'strict':>{cw}}" for _ in [1]) +
           "".join(f"  {'th@'+str(T):>{cw}}" for T in temps) +
           "".join(f"  {'gap@'+str(T):>{cw}}" for T in temps))
    print(hdr)
    print("  " + "-"*101)
    for label, _, _, _ in conditions:
        for scale_name in all_results:
            rec = all_results[scale_name][label]
            row = (f"  {label:<48}  {scale_name:>8}"
                   f"  {rec['strict']:>{cw}.4f}")
            for T in temps: row += f"  {rec[f'thermal_{T}']:>{cw}.4f}"
            for T in temps: row += f"  {rec[f'gap_{T}']:>{cw}.4f}"
            print(row)
        print()

    # Delta table
    print("\nΔ (E5M3 − E8M0) perturbation, positive = E5M3 worse:")
    print(f"  {'condition':<48}  {'Δ strict':>10}  {'Δ th@0.7':>10}  {'Δ gap@0.7':>10}")
    print("  " + "-"*85)
    for label, _, _, _ in conditions:
        r_e8 = all_results["E8M0 (OCP)"][label]
        r_e5 = all_results["E5M3 (TARE)"][label]
        ds  = ((1 - r_e5["strict"])       - (1 - r_e8["strict"]))       * 100
        dt  = ((1 - r_e5["thermal_0.7"])  - (1 - r_e8["thermal_0.7"])) * 100
        dg  = r_e5["gap_0.7"] - r_e8["gap_0.7"]
        print(f"  {label:<48}  {ds:>+9.1f}pp  {dt:>+9.1f}pp  {dg:>+9.3f}L")

    print(f"\n  FP8-equiv: strict ~3-5%  |  NVFP4-equiv: strict ~10-20%")
    print(f"{'='*105}")


if __name__ == "__main__":
    main()
