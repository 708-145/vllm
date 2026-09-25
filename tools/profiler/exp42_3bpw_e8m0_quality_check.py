# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 42 – 3bpw with E8M0 scales: static encoding quality check.

Encoding scheme
---------------
Block size B=16.  Each weight is one of 4 signed levels: {−s_hi, −s_lo, +s_lo, +s_hi}
(2 bits per weight).  Two scales per block, both E8M0 (exact powers of two).

  bit layout:  [sign_bit | magnitude_bit]
    0x → −s_lo   (magnitude_bit=0)
    1x → +s_lo
    x0 → s_lo    (lower magnitude)
    x1 → s_hi    (higher magnitude)

Storage:
  codes: B×2 bits = 32 bits = 4 bytes per block
  scales: 2×1 byte E8M0 = 2 bytes per block
  total: 6 bytes / 16 weights = **3 bits per weight** = 2.67× vs BF16

E8M0 scale quantisation:
  s_E8M0 = 2^round(log2(s))   — nearest power of two

TARE-optimal scale selection (2-centroid, same EM as exp37):
  EM in log₂-space, M-step centroids rounded to nearest power of two (E8M0).
  This replaces exp37's E5M3 rounding of centroids with E8M0 rounding,
  following the lesson from exp41 that power-of-two scales are structurally
  required for floating-point element formats.

  Note: the 2-level scheme does NOT have an explicit floating-point exponent
  field per weight — weights are just ±s_lo or ±s_hi.  However, since s_lo
  and s_hi are powers of two, the reconstructed values are exact powers of two
  (times ±1), which removes the fractional exponent misalignment problem.

Conditions (all 40 layers, 8 prompts, same as exp38–41):
  A. gate only    B. gate+up    C. gate+down    D. all

Direct comparisons:
  exp37 / exp38: same 3bpw scheme with E5M3 scales
  exp40 MXFP6-E2M3: 6.25 bpw with E8M0 scales (the gold standard)

Usage::

    python tools/profiler/exp42_3bpw_e8m0_quality_check.py \\
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
# E8M0 scale: nearest power of two
# ---------------------------------------------------------------------------

def _e8m0_round(s: torch.Tensor) -> torch.Tensor:
    """Round positive scale tensor to nearest power of two (E8M0)."""
    return (2.0 ** torch.round(torch.log2(s.clamp(min=EPS)))).clamp(min=2.0 ** -127)


# ---------------------------------------------------------------------------
# 3bpw encoding with E8M0 scales
# ---------------------------------------------------------------------------

def _floor_eps(W: torch.Tensor, pct: float = 1.0) -> torch.Tensor:
    flat = W.abs().reshape(-1)
    k    = max(1, int(len(flat) * pct / 100.0))
    return flat.kthvalue(k).values.clamp(min=EPS)


def build_3bpw_e8m0(W: torch.Tensor, B: int = 16,
                     em_steps: int = 20) -> torch.Tensor:
    """Encode W (O, I) as 2-level ±{s_lo, s_hi} with E8M0 power-of-two scales.

    Algorithm:
      1. EM in log₂-space to find TARE-optimal (s_lo, s_hi).
      2. Round each centroid to nearest power of two (E8M0).
      3. Re-assign weights to nearest E8M0 level.
      4. Reconstruct: W_enc = sign(w) × assigned_level.

    Args:
        W:        (O, I) float32 weight matrix.
        B:        block size (default 16).
        em_steps: maximum EM iterations (default 20; typically converges in 4–6).

    Returns:
        W_enc: (O, I) float32 reconstructed approximation.
    """
    eps  = _floor_eps(W)
    O, I = W.shape
    pad  = (B - I % B) % B
    Wp   = F.pad(W, (0, pad)) if pad else W
    n_row_blocks = Wp.shape[1] // B
    n_blocks     = O * n_row_blocks

    W_b   = Wp.reshape(n_blocks, B)          # (n_blocks, B)
    signs = W_b.sign()                        # (n_blocks, B)
    wa    = W_b.abs().clamp(min=EPS)          # (n_blocks, B)
    log2w = torch.log2(wa)                    # (n_blocks, B)
    tilt  = torch.log1p(wa / eps)             # (n_blocks, B)  TARE weights

    # ---- Initialise: s_lo = tilt-geomean of lower half, s_hi = upper half ----
    med   = log2w.median(dim=1, keepdim=True).values   # (n_blocks, 1)
    lo_mask = log2w <= med
    hi_mask = ~lo_mask

    def _tilt_geomean_log2(mask):
        denom = (tilt * mask.float()).sum(1).clamp(min=EPS)
        numer = (tilt * log2w * mask.float()).sum(1)
        return numer / denom   # (n_blocks,) log2 centroid

    log2_s_lo = _tilt_geomean_log2(lo_mask)   # (n_blocks,)
    log2_s_hi = _tilt_geomean_log2(hi_mask)
    # Ensure ordering
    log2_s_lo, log2_s_hi = (torch.minimum(log2_s_lo, log2_s_hi),
                             torch.maximum(log2_s_lo, log2_s_hi))

    for _ in range(em_steps):
        # E-step: assign to nearest centroid in log₂-space
        mid     = (log2_s_lo + log2_s_hi).unsqueeze(1) / 2.0   # (n_blocks, 1)
        hi_mask = log2w > mid
        lo_mask = ~hi_mask

        # M-step: recompute centroids
        lo_new = _tilt_geomean_log2(lo_mask)
        hi_new = _tilt_geomean_log2(hi_mask)

        # Guard degenerate partitions (all weights in one bin)
        all_hi = lo_mask.sum(1) == 0
        all_lo = hi_mask.sum(1) == 0
        lo_new = torch.where(all_hi, log2_s_lo, lo_new)
        hi_new = torch.where(all_lo, log2_s_hi, hi_new)

        lo_new, hi_new = (torch.minimum(lo_new, hi_new),
                          torch.maximum(lo_new, hi_new))

        if ((lo_new - log2_s_lo).abs().max() < 1e-6 and
                (hi_new - log2_s_hi).abs().max() < 1e-6):
            break
        log2_s_lo, log2_s_hi = lo_new, hi_new

    # ---- Round centroids to nearest power of two (E8M0) ----
    # E8M0: round log2 to nearest integer
    log2_s_lo_q = torch.round(log2_s_lo).clamp(-127, 127)
    log2_s_hi_q = torch.round(log2_s_hi).clamp(-127, 127)
    # Ensure s_lo_q <= s_hi_q after rounding (they may collide)
    log2_s_lo_q, log2_s_hi_q = (torch.minimum(log2_s_lo_q, log2_s_hi_q),
                                  torch.maximum(log2_s_lo_q, log2_s_hi_q))

    s_lo_q = (2.0 ** log2_s_lo_q)   # (n_blocks,)
    s_hi_q = (2.0 ** log2_s_hi_q)   # (n_blocks,)

    # ---- Final assignment using quantised E8M0 scales ----
    mid    = (log2_s_lo_q + log2_s_hi_q).unsqueeze(1) / 2.0   # (n_blocks, 1)
    use_hi = log2w > mid                                        # (n_blocks, B)
    level  = torch.where(use_hi,
                         s_hi_q.unsqueeze(1).expand_as(W_b),
                         s_lo_q.unsqueeze(1).expand_as(W_b))
    W_enc  = signs * level

    return W_enc.reshape(O, Wp.shape[1])[:, :I].contiguous()


# ---------------------------------------------------------------------------
# Exp37/38 3bpw with E5M3 scales (for inline comparison)
# ---------------------------------------------------------------------------

_M3_FRAC_LOG2 = torch.log2(1.0 + torch.arange(8).float() / 8.0)

def _quantise_e5m3(s: torch.Tensor) -> torch.Tensor:
    s      = s.clamp(min=EPS); log2_s = torch.log2(s)
    e      = log2_s.floor().to(torch.int32); frac = log2_s - e.float()
    m_lut  = _M3_FRAC_LOG2.to(s.device)
    m_best = (frac.unsqueeze(-1) - m_lut).abs().argmin(-1)
    return (2.0 ** (e.float() + m_lut[m_best])).clamp(min=EPS)

def build_3bpw_e5m3(W: torch.Tensor, B: int = 16,
                     em_steps: int = 20) -> torch.Tensor:
    """Same 2-level EM as above but rounds centroids to E5M3 (exp37 baseline)."""
    eps  = _floor_eps(W)
    O, I = W.shape
    pad  = (B - I % B) % B
    Wp   = F.pad(W, (0, pad)) if pad else W
    n_blocks = O * (Wp.shape[1] // B)
    W_b   = Wp.reshape(n_blocks, B)
    signs = W_b.sign()
    wa    = W_b.abs().clamp(min=EPS)
    logw  = torch.log(wa); tilt = torch.log1p(wa / eps)

    def _tgm(mask):
        d = (tilt * mask.float()).sum(1).clamp(min=EPS)
        return ((tilt * logw * mask.float()).sum(1) / d).exp().clamp(min=EPS)

    med     = logw.median(dim=1, keepdim=True).values
    s_lo    = _tgm(logw <= med); s_hi = _tgm(logw > med)
    s_lo, s_hi = torch.minimum(s_lo, s_hi), torch.maximum(s_lo, s_hi)

    for _ in range(em_steps):
        mid     = (torch.log(s_lo) + torch.log(s_hi)).unsqueeze(1) / 2
        hi_mask = logw > mid; lo_mask = ~hi_mask
        s_lo_n  = _tgm(lo_mask); s_hi_n = _tgm(hi_mask)
        all_hi  = lo_mask.sum(1) == 0; all_lo = hi_mask.sum(1) == 0
        s_lo_n  = torch.where(all_hi, s_lo, s_lo_n)
        s_hi_n  = torch.where(all_lo, s_hi, s_hi_n)
        s_lo_n, s_hi_n = torch.minimum(s_lo_n, s_hi_n), torch.maximum(s_lo_n, s_hi_n)
        if (s_lo_n - s_lo).abs().max() < 1e-7 and (s_hi_n - s_hi).abs().max() < 1e-7:
            break
        s_lo, s_hi = s_lo_n, s_hi_n

    s_lo_q = _quantise_e5m3(s_lo); s_hi_q = _quantise_e5m3(s_hi)
    mid    = (torch.log(s_lo_q) + torch.log(s_hi_q)).unsqueeze(1) / 2
    use_hi = logw > mid
    level  = torch.where(use_hi,
                         s_hi_q.unsqueeze(1).expand_as(W_b),
                         s_lo_q.unsqueeze(1).expand_as(W_b))
    W_enc  = signs * level
    return W_enc.reshape(O, Wp.shape[1])[:, :I].contiguous()


# ---------------------------------------------------------------------------
# Thermal match  (identical to exp34–41)
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
        wd  = (self._enc_down if self._enc_down is not None
               else self._mlp.down_proj.weight.detach().float())
        return (act @ wd.T).to(orig)


# ---------------------------------------------------------------------------
# Infra  (identical to exp38–41)
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
        description="Exp42: 3bpw with E8M0 scales vs E5M3 scales (B=16).",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--model",           default="ibm-granite/granite-4.2-3b")
    p.add_argument("--calibration-set", default="bartowski-imatrix-v5-semantic.txt")
    p.add_argument("--num-prompts",     type=int, default=8)
    p.add_argument("--temperatures",    nargs="+", type=float, default=[0.7, 1.0])
    p.add_argument("--tau",             type=float, default=LN2)
    p.add_argument("--block-size",      type=int, default=16)
    p.add_argument("--em-steps",        type=int, default=20)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = _parse_args(argv)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    print(f"Device: {DEV}  B={args.block_size}  em_steps={args.em_steps}",
          file=sys.stderr)

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
        ("gate_only     (gate=3bpw, up=full, down=full)", True,  False, False),
        ("gate+up       (gate=3bpw, up=3bpw, down=full)", True,  True,  False),
        ("gate+down     (gate=3bpw, up=full, down=3bpw)", True,  False, True),
        ("gate+up+down  (all=3bpw)",                      True,  True,  True),
    ]

    print("\nBaseline pass ...", file=sys.stderr)
    lf_base = _run_logits(llm, prompts, W_U, norm)
    n_tok   = lf_base.shape[0]
    print(f"  {n_tok} tokens, vocab={lf_base.shape[1]}", file=sys.stderr)

    variants = [
        ("E8M0 (this exp)", build_3bpw_e8m0),
        ("E5M3 (exp38 ref)", build_3bpw_e5m3),
    ]
    all_results: dict[str, dict[str, dict]] = {}

    for scale_name, builder in variants:
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
                store.append(builder(extractor(layer), B, args.em_steps).cpu())
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
    print(f"\n\n{'='*110}")
    print("Exp42: 3bpw (2-level ±{{s_lo,s_hi}}, B=16)  —  E8M0 scale vs E5M3 scale")
    print(f"{'='*110}")
    hdr = (f"  {'condition':<48}  {'scale':>16}" +
           f"  {'strict':>{cw}}  {'th@0.7':>{cw}}  {'th@1.0':>{cw}}"
           f"  {'gap@0.7':>{cw}}")
    print(hdr); print("  " + "-"*106)
    for label, _, _, _ in conditions:
        for sname in all_results:
            rec = all_results[sname][label]
            print(f"  {label:<48}  {sname:>16}"
                  f"  {rec['strict']:>{cw}.4f}"
                  f"  {rec['thermal_0.7']:>{cw}.4f}"
                  f"  {rec['thermal_1.0']:>{cw}.4f}"
                  f"  {rec['gap_0.7']:>{cw}.4f}")
        print()

    # Delta table
    e8_name = "E8M0 (this exp)"; e5_name = "E5M3 (exp38 ref)"
    print("\nΔ (E8M0 − E5M3) perturbation, negative = E8M0 better:")
    print(f"  {'condition':<48}  {'Δ strict':>10}  {'Δ th@0.7':>10}  {'Δ gap@0.7':>10}")
    print("  " + "-"*88)
    for label, _, _, _ in conditions:
        r8 = all_results[e8_name][label]; r5 = all_results[e5_name][label]
        ds = ((1-r8["strict"])      - (1-r5["strict"]))      * 100
        dt = ((1-r8["thermal_0.7"]) - (1-r5["thermal_0.7"])) * 100
        dg = r8["gap_0.7"] - r5["gap_0.7"]
        print(f"  {label:<48}  {ds:>+9.1f}pp  {dt:>+9.1f}pp  {dg:>+9.3f}L")

    # Reference comparison table
    ref = {
        "3bpw E5M3 (exp38)": {
            "gate_only":    (0.490, 0.538), "gate+up": (0.036, 0.054),
            "gate+down":    (0.116, 0.140), "gate+up+down": (0.002, 0.008),
        },
        "MXFP6-E2M3 E8M0 (exp40)": {
            "gate_only":    (0.984, 1.000), "gate+up": (0.968, 0.996),
            "gate+down":    (0.966, 1.000), "gate+up+down": (0.958, 0.992),
        },
    }
    lbl_map = {
        "gate_only     (gate=3bpw, up=full, down=full)": "gate_only",
        "gate+up       (gate=3bpw, up=3bpw, down=full)": "gate+up",
        "gate+down     (gate=3bpw, up=full, down=3bpw)": "gate+down",
        "gate+up+down  (all=3bpw)":                      "gate+up+down",
    }
    print(f"\nContext: perturbation % (strict / thermal@0.7)")
    print(f"  {'condition':<32}  {'3bpw E5M3':>14}  {'3bpw E8M0':>14}  "
          f"{'MXFP6-E2M3':>14}")
    print("  " + "-"*80)
    for label, _, _, _ in conditions:
        k = lbl_map[label]
        r8 = all_results[e8_name][label]
        e5s, e5t = ref["3bpw E5M3 (exp38)"][k]
        mxs, mxt = ref["MXFP6-E2M3 E8M0 (exp40)"][k]
        short = label.split("(")[0].strip()
        print(f"  {short:<32}"
              f"  {(1-e5s)*100:>6.1f}%/{(1-e5t)*100:>5.1f}%"
              f"  {(1-r8['strict'])*100:>6.1f}%/{(1-r8['thermal_0.7'])*100:>5.1f}%"
              f"  {(1-mxs)*100:>6.1f}%/{(1-mxt)*100:>5.1f}%")

    print(f"\n  FP8-equiv: strict ~3-5%  |  NVFP4-equiv: strict ~10-20%")
    print(f"{'='*110}")


if __name__ == "__main__":
    main()
