# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 44d – MXFP4-E2M1 code usage and LUT12 feasibility.

Questions:
  1. Are all 16 signed codes used evenly, or are some rare?
  2. If the 4 least-used magnitude codes are pruned (replaced by nearest
     neighbours → "LUT12"), how much does all-matrix perturbation change?

No vLLM inference is needed for (1).  For (2) we reuse the forward-hook
infrastructure from exp43 to run the calibration prompts through the
patched model.

MXFP4-E2M1 non-negative codes (8 magnitudes):
  idx 0 → 0.0   (subnormal m=0)
  idx 1 → 0.5   (subnormal m=1)
  idx 2 → 1.0   (normal e=1 m=0)
  idx 3 → 1.5   (normal e=1 m=1)
  idx 4 → 2.0   (normal e=2 m=0)
  idx 5 → 3.0   (normal e=2 m=1)
  idx 6 → 4.0   (normal e=3 m=0)
  idx 7 → 6.0   (normal e=3 m=1)   ← fp_max

Signed codes: each magnitude appears twice (positive and negative),
except 0.0 which maps to both +0 and −0 → effectively 15 distinct values
but we count the 8 unsigned buckets (sign tracked separately).

Usage::

    python tools/profiler/exp44d_mxfp4_code_usage.py \\
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
# MXFP4-E2M1 codebook
# ---------------------------------------------------------------------------

# Non-negative magnitudes in ascending order (unsigned LUT)
MXFP4_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                          dtype=torch.float32)
FP_MAX = 6.0  # MXFP4 fp_max


def _e8m0_scale(block_max: torch.Tensor) -> torch.Tensor:
    e = torch.ceil(torch.log2(block_max.clamp(min=EPS) / FP_MAX)).clamp(-127, 127)
    return (2.0 ** e).clamp(min=2.0 ** -127)


def _quantise_to_lut(wa_scaled: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """Nearest-neighbour quantise (abs values already scaled to [0, fp_max])."""
    flat = wa_scaled.reshape(-1).clamp(max=lut[-1])
    idx_r = torch.searchsorted(lut, flat).clamp(0, len(lut) - 1)
    idx_l = (idx_r - 1).clamp(0, len(lut) - 1)
    vl = lut[idx_l]; vr = lut[idx_r]
    # Return indices (not values) for counting
    idx = torch.where((flat - vl) >= (vr - flat), idx_r, idx_l)
    return idx.reshape(wa_scaled.shape)


def encode_and_count(W: torch.Tensor, lut: torch.Tensor, B: int = 32):
    """Encode W with MXFP4, return (W_enc float32, per-code counts tensor [8])."""
    lut = lut.to(W.device)
    O, I  = W.shape
    pad   = (B - I % B) % B
    Wp    = F.pad(W, (0, pad)) if pad else W
    W_b   = Wp.reshape(-1, B)
    signs = W_b.sign()
    wa    = W_b.abs()
    scale = _e8m0_scale(wa.max(1).values).unsqueeze(1)
    idx   = _quantise_to_lut(wa / scale, lut)          # shape (nblocks, B)
    counts = torch.bincount(idx.reshape(-1), minlength=len(lut))
    W_enc = signs * lut[idx] * scale
    return W_enc.reshape(O, Wp.shape[1])[:, :I].contiguous(), counts


# ---------------------------------------------------------------------------
# LUT12: prune the 4 least-used codes (remap to nearest neighbour)
# ---------------------------------------------------------------------------

def make_lut12(lut: torch.Tensor, drop_indices: list[int]) -> torch.Tensor:
    """Return a new 8-entry LUT where the 4 dropped codes are replaced by
    their nearest remaining neighbour (so encoding still returns 8 entries
    but dropped slots point to a kept value)."""
    lut12 = lut.clone()
    kept = [i for i in range(len(lut)) if i not in drop_indices]
    for d in drop_indices:
        # nearest kept neighbour by value
        nearest = min(kept, key=lambda k: abs(lut[k] - lut[d]))
        lut12[d] = lut[nearest]
    return lut12


# ---------------------------------------------------------------------------
# Perturbation measurement (no vLLM — pure forward-hook, reuses exp43 helpers)
# ---------------------------------------------------------------------------

def thermal_match(lf, lh, temperature=0.7, tau=LN2):
    bt = lf.argmax(-1); ht = lh.argmax(-1); exact = (bt == ht)
    gap = (lh.gather(-1, ht.unsqueeze(-1)).squeeze(-1) -
           lh.gather(-1, bt.unsqueeze(-1)).squeeze(-1))
    forgiven = (~exact) & (gap < temperature * tau)
    return float(exact.float().mean()), float((exact | forgiven).float().mean())


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
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",           default="ibm-granite/granite-4.2-3b")
    p.add_argument("--calibration-set", default="bartowski-imatrix-v5-semantic.txt")
    p.add_argument("--num-prompts",     type=int, default=8)
    p.add_argument("--block-size",      type=int, default=32)
    p.add_argument("--skip-quality",    action="store_true",
                   help="Only print code usage; skip the LUT12 quality run.")
    args = p.parse_args(argv)

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    B = args.block_size
    lut = MXFP4_LUT

    # -----------------------------------------------------------------------
    # Phase 1: code usage — weight-only, no inference
    # -----------------------------------------------------------------------
    print("\n" + "="*70, file=sys.stderr)
    print("Phase 1: MXFP4-E2M1 code usage across all MLP weights", file=sys.stderr)
    print("="*70, file=sys.stderr)

    from vllm import LLM
    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              kv_cache_memory_bytes=int(0.5 * 1024**3), max_model_len=512,
              enable_prefix_caching=False)
    norm, layers, W_U = _get_internals(llm)
    orig_forwards = [l.mlp.forward for l in layers]
    I_dim = layers[0].mlp.gate_up_proj.weight.shape[0] // 2

    total_counts = torch.zeros(8, dtype=torch.int64)
    per_matrix   = {"gate": torch.zeros(8, dtype=torch.int64),
                    "up":   torch.zeros(8, dtype=torch.int64),
                    "down": torch.zeros(8, dtype=torch.int64)}

    enc_gate, enc_up, enc_down = [], [], []

    t0 = time.time()
    for li, layer in enumerate(layers):
        Wgu = layer.mlp.gate_up_proj.weight.detach().float()
        Wd  = layer.mlp.down_proj.weight.detach().float()
        eg, cg = encode_and_count(Wgu[:I_dim], lut, B)
        eu, cu = encode_and_count(Wgu[I_dim:], lut, B)
        ed, cd = encode_and_count(Wd,           lut, B)
        enc_gate.append(eg.cpu()); enc_up.append(eu.cpu()); enc_down.append(ed.cpu())
        per_matrix["gate"] += cg.cpu()
        per_matrix["up"]   += cu.cpu()
        per_matrix["down"] += cd.cpu()
        total_counts += (cg + cu + cd).cpu()
    print(f"  Encoded all layers in {time.time()-t0:.0f}s", file=sys.stderr)

    total = total_counts.sum().item()
    print(f"\nMXFP4-E2M1 unsigned code usage  ({total:,} weight slots total)")
    print(f"{'idx':>4}  {'value':>6}  {'count':>14}  {'%':>7}  {'cumulative':>11}  per-matrix (gate/up/down %)")
    print("-" * 85)
    cum = 0.0
    for i, v in enumerate(lut.tolist()):
        c = int(total_counts[i])
        pct = 100.0 * c / total
        cum += pct
        cg = 100.0 * int(per_matrix["gate"][i]) / int(per_matrix["gate"].sum())
        cu = 100.0 * int(per_matrix["up"][i])   / int(per_matrix["up"].sum())
        cd = 100.0 * int(per_matrix["down"][i]) / int(per_matrix["down"].sum())
        print(f"  {i:>2}  {v:>6.2f}  {c:>14,}  {pct:>7.3f}%  {cum:>10.3f}%"
              f"  {cg:>6.2f}% / {cu:>6.2f}% / {cd:>6.2f}%")

    # identify 4 least-used (by unsigned count)
    sorted_idx = total_counts.argsort().tolist()   # ascending usage
    drop4 = sorted_idx[:4]
    print(f"\n4 least-used codes (indices): {drop4} "
          f"→ values {[lut[i].item() for i in drop4]}")
    print(f"  Combined weight: {100.0*total_counts[drop4].sum().item()/total:.3f}%")

    if args.skip_quality:
        return

    # -----------------------------------------------------------------------
    # Phase 2: LUT12 quality — replace dropped codes with nearest neighbour
    # -----------------------------------------------------------------------
    print("\n" + "="*70, file=sys.stderr)
    print("Phase 2: LUT12 quality check (all-matrix encoding)", file=sys.stderr)
    print("="*70, file=sys.stderr)

    lut12 = make_lut12(lut, drop4)
    print(f"LUT12 remapping: {list(zip(lut.tolist(), lut12.tolist()))}", file=sys.stderr)

    # Re-encode with lut12
    enc_gate12, enc_up12, enc_down12 = [], [], []
    t0 = time.time()
    for li, layer in enumerate(layers):
        Wgu = layer.mlp.gate_up_proj.weight.detach().float()
        Wd  = layer.mlp.down_proj.weight.detach().float()
        eg12, _ = encode_and_count(Wgu[:I_dim], lut12, B)
        eu12, _ = encode_and_count(Wgu[I_dim:], lut12, B)
        ed12, _ = encode_and_count(Wd,           lut12, B)
        enc_gate12.append(eg12.cpu())
        enc_up12.append(eu12.cpu())
        enc_down12.append(ed12.cpu())
    print(f"  LUT12 re-encode done in {time.time()-t0:.0f}s", file=sys.stderr)

    prompts = _load_prompts(args.calibration_set, args.num_prompts)
    lf_base = _run_logits(llm, prompts, W_U, norm)
    n_tok   = lf_base.shape[0]
    print(f"  Baseline: {n_tok} tokens", file=sys.stderr)

    results = {}
    for label, eg_list, eu_list, ed_list in [
        ("MXFP4 (baseline)",    enc_gate,   enc_up,   enc_down),
        ("MXFP4-LUT12 (4 pruned)", enc_gate12, enc_up12, enc_down12),
    ]:
        hybrids = [EncodedMLP(layers[li].mlp, eg_list[li], eu_list[li], ed_list[li])
                   for li in range(len(layers))]
        for l, h in zip(layers, hybrids): l.mlp.forward = h
        lh = _run_logits(llm, prompts, W_U, norm)[:n_tok]
        strict, th07 = thermal_match(lf_base, lh, 0.7)
        _, th10       = thermal_match(lf_base, lh, 1.0)
        results[label] = (strict, th07, th10)
        print(f"  {label}: strict={1-strict:.4f}  th@0.7={1-th07:.4f}  th@1.0={1-th10:.4f}",
              file=sys.stderr)
        for l, fwd in zip(layers, orig_forwards): l.mlp.forward = fwd
        del hybrids

    print(f"\n\n{'='*70}")
    print("Exp44d: MXFP4-E2M1 code usage + LUT12 feasibility")
    print(f"{'='*70}")
    print(f"{'Scheme':<30}  {'strict%':>8}  {'thermal@0.7%':>13}  {'thermal@1.0%':>13}")
    print("-" * 70)
    for label, (s, t07, t10) in results.items():
        print(f"  {label:<28}  {(1-s)*100:>7.1f}%  {(1-t07)*100:>12.1f}%  {(1-t10)*100:>12.1f}%")
    print(f"\n  Drop4 codes: {drop4} → values {[lut[i].item() for i in drop4]}")
    drop_pct = 100.0 * total_counts[drop4].sum().item() / total
    print(f"  Combined usage of dropped codes: {drop_pct:.3f}% of all weight slots")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
