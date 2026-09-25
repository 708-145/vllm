# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 38 – 3bpw static encoding quality check (no prediction).

Applies the same 3bpw encoding from exp37 (2-bit/weight, 2 E5M3 scales per
B=16 block, TARE-optimal EM) to different combinations of the MLP weight
matrices, with no hot/cold routing — the approximation is applied uniformly to
all channels.  This gives a clean upper-bound quality baseline for the encoding
scheme itself, decoupled from routing error.

Conditions evaluated (all 40 layers, prefill, 8 calibration prompts):
  A. gate only:       W_gate=3bpw, W_up=full, W_down=full
  B. gate + up:       W_gate=3bpw, W_up=3bpw, W_down=full
  C. gate + down:     W_gate=3bpw, W_up=full, W_down=3bpw
  D. gate + up + down: W_gate=3bpw, W_up=3bpw, W_down=3bpw

Note: down projection has shape (H, I) in HF convention; in vLLM it is stored
as down_proj.weight shape (H, I) — rows are output dimensions, cols are input
(intermediate) dimensions.  The encoding is applied the same way (TARE on rows).

Metrics (from exp34 onwards): strict match, thermal@0.7, thermal@1.0, mean gap.

This experiment has NO routing — all channels are processed by the encoded
weight matrices.  The result answers: "what does 3bpw do to output quality if
applied to each matrix independently or jointly?"

Usage::

    python tools/profiler/exp38_3bpw_quality_check.py \\
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

_M3_FRAC_LOG2 = torch.log2(1.0 + torch.arange(8).float() / 8.0)


# ---------------------------------------------------------------------------
# E5M3 scalar quantisation  (same as exp22–37)
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
# 3bpw encoding  (identical to exp37)
# ---------------------------------------------------------------------------

def _floor_eps(W: torch.Tensor, pct: float = 1.0) -> torch.Tensor:
    flat = W.abs().reshape(-1)
    k    = max(1, int(len(flat) * pct / 100.0))
    return flat.kthvalue(k).values.clamp(min=EPS)


def build_3bpw_encoded(W: torch.Tensor, B: int = 16,
                        em_steps: int = 10) -> torch.Tensor:
    """Encode W (O, I) as 2-level TARE-optimal E5M3 approximation.

    Each B-weight block gets two E5M3 scales (s_lo, s_hi) via EM in log-space.
    Returns float32 reconstructed matrix, same shape as W.
    """
    eps  = _floor_eps(W)
    O, I = W.shape
    pad  = (B - I % B) % B
    Wp   = F.pad(W, (0, pad)) if pad else W
    n_row_blocks = Wp.shape[1] // B
    n_blocks     = O * n_row_blocks

    W_b  = Wp.reshape(n_blocks, B)
    wa   = W_b.abs().clamp(min=EPS)
    logw = torch.log(wa)
    tilt = torch.log1p(wa / eps)

    median_logw = logw.median(dim=1, keepdim=True).values
    lo_mask     = logw <= median_logw
    hi_mask     = ~lo_mask

    def _tilt_geomean(lw, tm, mask):
        denom = (tm * mask.float()).sum(dim=1).clamp(min=EPS)
        numer = (tm * lw * mask.float()).sum(dim=1)
        return (numer / denom).exp().clamp(min=EPS)

    s_lo = _tilt_geomean(logw, tilt, lo_mask)
    s_hi = _tilt_geomean(logw, tilt, hi_mask)
    s_lo, s_hi = torch.minimum(s_lo, s_hi), torch.maximum(s_lo, s_hi)

    for _ in range(em_steps):
        log_mid = (torch.log(s_lo) + torch.log(s_hi)).unsqueeze(1) / 2.0
        hi_mask = logw > log_mid
        lo_mask = ~hi_mask
        s_lo_new = _tilt_geomean(logw, tilt, lo_mask)
        s_hi_new = _tilt_geomean(logw, tilt, hi_mask)
        all_hi   = lo_mask.sum(1) == 0
        all_lo   = hi_mask.sum(1) == 0
        s_lo_new = torch.where(all_hi, s_lo, s_lo_new)
        s_hi_new = torch.where(all_lo, s_hi, s_hi_new)
        s_lo_new, s_hi_new = (torch.minimum(s_lo_new, s_hi_new),
                               torch.maximum(s_lo_new, s_hi_new))
        if ((s_lo_new - s_lo).abs().max() < 1e-7 and
                (s_hi_new - s_hi).abs().max() < 1e-7):
            break
        s_lo, s_hi = s_lo_new, s_hi_new

    s_lo_q = _quantise_e5m3(s_lo)
    s_hi_q = _quantise_e5m3(s_hi)
    log_mid = (torch.log(s_lo_q) + torch.log(s_hi_q)).unsqueeze(1) / 2.0
    use_hi  = logw > log_mid
    level   = torch.where(use_hi,
                          s_hi_q.unsqueeze(1).expand_as(W_b),
                          s_lo_q.unsqueeze(1).expand_as(W_b))
    W_enc   = W_b.sign() * level
    return W_enc.reshape(O, Wp.shape[1])[:, :I].contiguous()


# ---------------------------------------------------------------------------
# Thermal match metric  (identical to exp34–37)
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
# MLP wrappers — uniform 3bpw (no routing)
# ---------------------------------------------------------------------------

class Encoded3bpwMLP:
    """Apply 3bpw encoding to selected matrices; no routing, all channels encoded.

    Args:
        mlp:         the layer's MLP module (vLLM fused gate_up_proj + down_proj)
        enc_gate:    encoded W_gate (I, H) float32, or None → use full precision
        enc_up:      encoded W_up   (I, H) float32, or None → use full precision
        enc_down:    encoded W_down (H, I) float32, or None → use full precision
    """
    def __init__(self, mlp, enc_gate, enc_up, enc_down):
        self._mlp      = mlp
        self._enc_gate = enc_gate   # (I, H) or None
        self._enc_up   = enc_up     # (I, H) or None
        self._enc_down = enc_down   # (H, I) or None
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
# LogitCapture  (identical to exp34–37)
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
        description="Exp38: 3bpw static encoding quality check (no routing).",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--model",           default="ibm-granite/granite-4.2-3b")
    p.add_argument("--calibration-set", default="bartowski-imatrix-v5-semantic.txt")
    p.add_argument("--num-prompts",     type=int, default=8)
    p.add_argument("--temperatures",    nargs="+", type=float, default=[0.7, 1.0])
    p.add_argument("--tau",             type=float, default=LN2)
    p.add_argument("--em-steps",        type=int, default=10)
    p.add_argument("--block-size",      type=int, default=16)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = _parse_args(argv)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    print(f"Device: {DEV}  tau={args.tau:.4f}  temps={args.temperatures}  "
          f"em_steps={args.em_steps}  B={args.block_size}", file=sys.stderr)

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
    # Build 3bpw encoded matrices for all layers
    # ------------------------------------------------------------------
    enc_gate = []; enc_up = []; enc_down = []
    for name, store, extractor in [
        ("gate", enc_gate, lambda l: l.mlp.gate_up_proj.weight.detach().float()[:I]),
        ("up",   enc_up,   lambda l: l.mlp.gate_up_proj.weight.detach().float()[I:]),
        ("down", enc_down, lambda l: l.mlp.down_proj.weight.detach().float()),
    ]:
        print(f"\nEncoding W_{name} (3bpw B={B}) ...", file=sys.stderr)
        t0 = time.time()
        for li, layer in enumerate(layers):
            W = extractor(layer)
            store.append(build_3bpw_encoded(W, B=B, em_steps=args.em_steps).cpu())
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
    # Define runs: (label, gate_enc?, up_enc?, down_enc?)
    # ------------------------------------------------------------------
    temps = args.temperatures
    conditions: list[tuple[str, bool, bool, bool]] = [
        ("gate_only     (gate=3bpw, up=full, down=full)", True,  False, False),
        ("gate+up       (gate=3bpw, up=3bpw, down=full)", True,  True,  False),
        ("gate+down     (gate=3bpw, up=full, down=3bpw)", True,  False, True),
        ("gate+up+down  (all=3bpw)",                      True,  True,  True),
    ]

    results: dict[str, dict] = {}

    for idx, (label, use_gate, use_up, use_down) in enumerate(conditions):
        hybrids = [
            Encoded3bpwMLP(
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
    hdr  = f"  {'condition':<46}" + "".join(f"  {c:>{cw}}" for c in cols)

    def _row(lbl, rec):
        r = f"  {lbl:<46}  {rec['strict']:>{cw}.4f}"
        for T in temps: r += f"  {rec[f'thermal_{T}']:>{cw}.4f}"
        for T in temps: r += f"  {rec[f'gap_{T}']:>{cw}.4f}"
        return r

    print("\n" + "=" * 110)
    print(f"Exp38: 3bpw static encoding quality check  (B={B}, em_steps={args.em_steps})")
    print("No routing — 3bpw encoding applied uniformly to all channels of the listed matrices.")
    print("=" * 110)
    print(hdr)
    print("  " + "-" * 106)
    for label, rec in results.items():
        print(_row(label, rec))
    print()
    print("  gap = mean logit gap among perturbed tokens (L = logits)")
    print("  FP8-equiv: strict ~3-5%, thermal ~2-4%  |  NVFP4-equiv: strict ~10-20%")
    print("=" * 110)

    print("\nPerturbation rates (%):")
    hdr2 = (f"  {'condition':<46}  {'strict%':>8}" +
            "".join(f"  {'th%@'+str(T):>9}" for T in temps) +
            "".join(f"  {'Δ@'+str(T):>8}" for T in temps))
    print(hdr2)
    for label, rec in results.items():
        sp  = (1 - rec["strict"]) * 100
        row = f"  {label:<46}  {sp:>7.1f}%"
        for T in temps:
            tp = (1 - rec[f"thermal_{T}"]) * 100; row += f"  {tp:>8.1f}%"
        for T in temps:
            tp = (1 - rec[f"thermal_{T}"]) * 100; row += f"  {sp-tp:>7.1f}pp"
        print(row)


if __name__ == "__main__":
    main()
