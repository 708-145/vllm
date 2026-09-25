# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 45 – Per-channel LUT12 quantisation vs MXFP4-E2M1.

Idea
----
Instead of the fixed 8-entry OCP MXFP4 codebook, fit an **optimal 12-entry
non-negative LUT per output channel** using Lloyd-Max iteration (k-means on
absolute scaled weights).  Sign is handled separately as in all MX formats.

Each output channel of each MLP matrix (gate, up, down) gets its own 12
non-negative reconstruction levels, fitted to the E8M0-scaled absolute weights
of that channel's B=32 blocks.  The E8M0 per-block scale is retained unchanged.

Storage cost
------------
- Per-weight index:  ceil(log2(12)) = 4 bits  → same as MXFP4 (4.25 bpw)
- Per-channel LUT:   12 × BF16 = 24 bytes/channel
  Total LUT overhead: 757,760 channels × 24 B ≈ 18 MB over ~1.27 GB weights
  → negligible (~0.001 bpw extra)

Comparison points
-----------------
  MXFP4-E2M1 (8 fixed codes)   4.25 bpw   (exp43/44d baseline)
  LUT12 per-channel (12 codes)  4.25 bpw   + 18 MB LUT sidecar
  MXFP6-E2M3 (32 fixed codes)  6.25 bpw   (upper reference, exp43)

Questions answered
------------------
  1. How close does per-channel LUT12 get to MXFP4 (8 fixed codes)?
  2. How close does it get to MXFP6 (32 fixed codes)?
  3. Is the gap driven by "optimal level placement" or "more levels"?

All heavy tensors are moved to MPS (Metal) for encoding; encoded results are
returned to CPU for vLLM inference.

Usage::

    python tools/profiler/exp45_lut12_per_channel.py \\
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
# Device selection — prefer MPS (Metal), then CUDA, then CPU
# ---------------------------------------------------------------------------

def _best_device() -> torch.device:
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():         return torch.device("cuda")
    return torch.device("cpu")

ENC_DEV = _best_device()   # Metal/CUDA for encoding kernels
LN2     = math.log(2)
EPS     = 1e-9
B       = 32               # MX block size

# ---------------------------------------------------------------------------
# E8M0 scale  (OCP standard — identical to all previous experiments)
# ---------------------------------------------------------------------------

def _e8m0_scale(block_max: torch.Tensor, fp_max: float) -> torch.Tensor:
    e = torch.ceil(torch.log2(block_max.clamp(min=EPS) / fp_max)).clamp(-127, 127)
    return (2.0 ** e).clamp(min=2.0 ** -127)


# ---------------------------------------------------------------------------
# MXFP4-E2M1 reference encoder  (from exp43/44d)
# ---------------------------------------------------------------------------

MXFP4_LUT  = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                           dtype=torch.float32)
MXFP4_FPMAX = 6.0


def _nn_quantise(wa_sc: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """Nearest-neighbour quantise; wa_sc and lut must be on the same device."""
    flat = wa_sc.reshape(-1).clamp(max=lut[-1])
    ir   = torch.searchsorted(lut.contiguous(), flat.contiguous()).clamp(0, len(lut)-1)
    il   = (ir - 1).clamp(0)
    q    = torch.where((flat - lut[il]) >= (lut[ir] - flat), lut[ir], lut[il])
    return q.reshape(wa_sc.shape)


def build_mxfp4(W: torch.Tensor) -> torch.Tensor:
    """MXFP4-E2M1 encoder; runs on ENC_DEV, returns CPU tensor."""
    W   = W.to(ENC_DEV)
    lut = MXFP4_LUT.to(ENC_DEV)
    O, I  = W.shape
    pad   = (B - I % B) % B
    Wp    = F.pad(W, (0, pad)) if pad else W
    Wb    = Wp.reshape(-1, B)
    signs = Wb.sign()
    wa    = Wb.abs()
    scale = _e8m0_scale(wa.max(1).values, MXFP4_FPMAX).unsqueeze(1)
    Wenc  = signs * _nn_quantise(wa / scale, lut) * scale
    return Wenc.reshape(O, Wp.shape[1])[:, :I].contiguous().cpu()


# ---------------------------------------------------------------------------
# Per-channel Lloyd-Max fitting — fully vectorised, chunked for large matrices
# ---------------------------------------------------------------------------

def _lloyd_max_batch(wa_sc: torch.Tensor, n_levels: int,
                     n_iter: int = 30,
                     chunk:  int = 512) -> torch.Tensor:
    """Vectorised Lloyd-Max over all output channels.

    Parameters
    ----------
    wa_sc   : (O, N) float32 on ENC_DEV — abs-scaled weights per channel
    n_levels: number of non-negative reconstruction levels
    chunk   : channels processed together (keeps (chunk, N, K) below ~1 GB)

    Returns
    -------
    centroids : (O, n_levels) float32, sorted ascending, level-0 = 0.0
    """
    O, N   = wa_sc.shape
    device = wa_sc.device

    # Initialise from per-channel quantiles
    qs        = torch.linspace(0.0, 1.0, n_levels, device=device)
    centroids = torch.quantile(wa_sc, qs, dim=1).T.contiguous()   # (O, K)

    for _ in range(n_iter):
        new_c = centroids.clone()
        for c0 in range(0, O, chunk):
            c1     = min(c0 + chunk, O)
            C      = c1 - c0
            wa_ch  = wa_sc[c0:c1]                                  # (C, N)
            ce_ch  = centroids[c0:c1]                              # (C, K)
            # assignment: argmin over K dimension
            dists  = (wa_ch.unsqueeze(2) - ce_ch.unsqueeze(1)).abs()  # (C, N, K)
            assign = dists.argmin(2)                               # (C, N)
            # centroid update via scatter_add — no Python loop over K
            # flatten (C, N) → (C*N,) with channel offset for scatter
            ch_off = torch.arange(C, device=device).unsqueeze(1) * n_levels
            flat_idx = (assign + ch_off).reshape(-1)               # (C*N,)
            flat_wa  = wa_ch.reshape(-1)                           # (C*N,)
            sums     = torch.zeros(C * n_levels, device=device)
            cnts     = torch.zeros(C * n_levels, device=device)
            sums.scatter_add_(0, flat_idx, flat_wa)
            cnts.scatter_add_(0, flat_idx, torch.ones_like(flat_wa))
            cnts.clamp_(min=1)
            new_c[c0:c1] = (sums / cnts).reshape(C, n_levels)

        if (new_c - centroids).abs().max() < 1e-6:
            break
        centroids = new_c

    centroids[:, 0] = 0.0                           # pin exact-zero level
    return centroids.sort(dim=1).values             # (O, K)


def _quantise_with_luts(wa_flat: torch.Tensor,
                        luts: torch.Tensor,
                        chunk: int = 512) -> torch.Tensor:
    """Nearest-neighbour quantise wa_flat with per-channel luts, chunked.

    Parameters
    ----------
    wa_flat : (O, N) abs-scaled weights
    luts    : (O, K) per-channel codebooks

    Returns
    -------
    q_flat : (O, N) quantised values
    """
    O, N = wa_flat.shape
    q_flat = torch.empty_like(wa_flat)
    for c0 in range(0, O, chunk):
        c1     = min(c0 + chunk, O)
        wa_ch  = wa_flat[c0:c1]                                    # (C, N)
        lu_ch  = luts[c0:c1]                                       # (C, K)
        dists  = (wa_ch.unsqueeze(2) - lu_ch.unsqueeze(1)).abs()   # (C, N, K)
        idx    = dists.argmin(2)                                    # (C, N)
        q_flat[c0:c1] = lu_ch.gather(1, idx)                       # (C, N)
    return q_flat


def build_lut12_per_channel(W: torch.Tensor,
                             n_levels: int = 12,
                             n_iter:   int = 50,
                             fp_max_ref: float = MXFP4_FPMAX,
                             chunk: int = 512,
                             ) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode W with per-output-channel optimal LUT (n_levels non-neg codes).

    Runs entirely on ENC_DEV (Metal/CUDA/CPU); returns CPU tensors.

    Returns
    -------
    W_enc : float32 encoded weight tensor, same shape as W  (CPU)
    luts  : float32 tensor (O, n_levels) per-channel codebooks  (CPU)
    """
    W = W.to(ENC_DEV)
    O, I    = W.shape
    pad     = (B - I % B) % B
    Wp      = F.pad(W, (0, pad)) if pad else W
    Wb      = Wp.reshape(O, -1, B)                  # (O, nblocks, B)
    nblocks = Wb.shape[1]

    signs = Wb.sign()
    wa    = Wb.abs()                                 # (O, nblocks, B)

    # E8M0 per-block scale (same fp_max reference as MXFP4)
    scale = _e8m0_scale(wa.reshape(-1, B).max(1).values,
                        fp_max_ref).reshape(O, nblocks, 1)  # (O, nblocks, 1)
    wa_sc = (wa / scale).clamp(max=fp_max_ref)              # (O, nblocks, B)

    # Fit one LUT per output channel
    wa_flat = wa_sc.reshape(O, -1)                          # (O, nblocks*B)
    luts    = _lloyd_max_batch(wa_flat, n_levels, n_iter, chunk)  # (O, K)

    # Quantise using per-channel LUTs
    q_flat = _quantise_with_luts(wa_flat, luts, chunk)      # (O, nblocks*B)

    q_sc  = q_flat.reshape(O, nblocks, B)
    Wenc  = (signs * q_sc * scale).reshape(O, -1)

    return Wenc[:, :I].contiguous().cpu(), luts.cpu()


# ---------------------------------------------------------------------------
# Inference helpers  (identical structure to exp43/44d)
# ---------------------------------------------------------------------------

def thermal_match(lf, lh, temperature=0.7, tau=LN2):
    bt = lf.argmax(-1); ht = lh.argmax(-1); exact = (bt == ht)
    gap = (lh.gather(-1, ht.unsqueeze(-1)).squeeze(-1) -
           lh.gather(-1, bt.unsqueeze(-1)).squeeze(-1))
    forgiven = (~exact) & (gap < temperature * tau)
    return (float(exact.float().mean()),
            float((exact | forgiven).float().mean()))


class EncodedMLP:
    def __init__(self, mlp, enc_gate, enc_up, enc_down):
        self._mlp = mlp
        self._enc_gate = enc_gate; self._enc_up = enc_up; self._enc_down = enc_down
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
    def attach(self, m): self._handle = m.register_forward_hook(self._hook)
    def detach(self):
        if self._handle: self._handle.remove(); self._handle = None
    def _hook(self, m, a, out):
        self.logits.append((out.float() @ self._W_U.T).cpu())
    def all_logits(self, n_dec):
        cat = torch.cat(self.logits, 0)
        return cat[:-n_dec] if n_dec < cat.shape[0] else cat


def _load_prompts(path, n):
    return [l.strip() for l in Path(path).read_text().splitlines()
            if l.strip()][:n]

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
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",           default="ibm-granite/granite-4.2-3b")
    p.add_argument("--calibration-set", default="bartowski-imatrix-v5-semantic.txt")
    p.add_argument("--num-prompts",     type=int, default=8)
    p.add_argument("--n-levels",        type=int, default=12,
                   help="Non-negative LUT entries per channel (default 12).")
    p.add_argument("--lloyd-iters",     type=int, default=30)
    p.add_argument("--chunk",           type=int, default=512,
                   help="Channels per Metal/CUDA chunk in Lloyd-Max (default 512).")
    args = p.parse_args(argv)

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    print(f"Encoding device: {ENC_DEV}", file=sys.stderr)

    from vllm import LLM
    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              kv_cache_memory_bytes=int(0.5 * 1024**3), max_model_len=512,
              enable_prefix_caching=False)
    norm, layers, W_U = _get_internals(llm)
    orig_forwards = [l.mlp.forward for l in layers]
    I_dim = layers[0].mlp.gate_up_proj.weight.shape[0] // 2

    prompts = _load_prompts(args.calibration_set, args.num_prompts)
    print(f"Loaded {len(prompts)} prompts.", file=sys.stderr)

    # ------------------------------------------------------------------
    # Encode: MXFP4 (reference)
    # ------------------------------------------------------------------
    print(f"\nMXFP4-E2M1 encoding (fixed 8 codes, {ENC_DEV}) ...", file=sys.stderr)
    t0 = time.time()
    mxfp4_gate, mxfp4_up, mxfp4_down = [], [], []
    for layer in layers:
        Wgu = layer.mlp.gate_up_proj.weight.detach().float()
        Wd  = layer.mlp.down_proj.weight.detach().float()
        mxfp4_gate.append(build_mxfp4(Wgu[:I_dim]))
        mxfp4_up.append(  build_mxfp4(Wgu[I_dim:]))
        mxfp4_down.append(build_mxfp4(Wd))
    print(f"  done in {time.time()-t0:.0f}s", file=sys.stderr)

    # ------------------------------------------------------------------
    # Encode: LUT12 per-channel (Lloyd-Max on Metal)
    # ------------------------------------------------------------------
    print(f"\nLUT{args.n_levels} per-channel encoding "
          f"(Lloyd-Max {args.lloyd_iters} iters, chunk={args.chunk}, {ENC_DEV}) ...",
          file=sys.stderr)
    t0 = time.time()
    lut12_gate, lut12_up, lut12_down = [], [], []
    for li, layer in enumerate(layers):
        Wgu = layer.mlp.gate_up_proj.weight.detach().float()
        Wd  = layer.mlp.down_proj.weight.detach().float()
        t1  = time.time()
        eg, _ = build_lut12_per_channel(Wgu[:I_dim], args.n_levels,
                                        args.lloyd_iters, chunk=args.chunk)
        eu, _ = build_lut12_per_channel(Wgu[I_dim:], args.n_levels,
                                        args.lloyd_iters, chunk=args.chunk)
        ed, _ = build_lut12_per_channel(Wd,           args.n_levels,
                                        args.lloyd_iters, chunk=args.chunk)
        lut12_gate.append(eg); lut12_up.append(eu); lut12_down.append(ed)
        print(f"  layer {li:2d}  {time.time()-t1:.1f}s", file=sys.stderr)
    print(f"  total {time.time()-t0:.0f}s", file=sys.stderr)

    # ------------------------------------------------------------------
    # Baseline logits
    # ------------------------------------------------------------------
    print("\nBaseline pass ...", file=sys.stderr)
    lf_base = _run_logits(llm, prompts, W_U, norm)
    n_tok   = lf_base.shape[0]
    print(f"  {n_tok} tokens", file=sys.stderr)

    # ------------------------------------------------------------------
    # Quality: all-matrix for both schemes
    # ------------------------------------------------------------------
    results = {}
    schemes = [
        ("MXFP4-E2M1 (8 fixed codes)",
         mxfp4_gate, mxfp4_up, mxfp4_down),
        (f"LUT{args.n_levels} per-channel (optimal)",
         lut12_gate, lut12_up, lut12_down),
    ]

    for label, eg_list, eu_list, ed_list in schemes:
        hybrids = [EncodedMLP(layers[li].mlp,
                              eg_list[li], eu_list[li], ed_list[li])
                   for li in range(len(layers))]
        for l, h in zip(layers, hybrids): l.mlp.forward = h
        lh = _run_logits(llm, prompts, W_U, norm)[:n_tok]
        s07, t07 = thermal_match(lf_base, lh, 0.7)
        _,   t10 = thermal_match(lf_base, lh, 1.0)
        results[label] = (s07, t07, t10)
        print(f"  {label}: strict={1-s07:.4f}  th@0.7={1-t07:.4f}  th@1.0={1-t10:.4f}",
              file=sys.stderr)
        for l, fwd in zip(layers, orig_forwards): l.mlp.forward = fwd
        del hybrids

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    # MXFP6-E2M3 all-matrix from exp43
    ref_rows = [
        ("MXFP6-E2M3 (32 fixed codes, exp43 ref)", 0.958, 0.992, 0.994, "6.25"),
    ]
    bpw_map = {
        "MXFP4-E2M1 (8 fixed codes)":                   "4.25",
        f"LUT{args.n_levels} per-channel (optimal)":     "4.25+ε",
    }

    print(f"\n\n{'='*74}")
    print(f"Exp45: LUT{args.n_levels} per-channel vs MXFP4  "
          f"(all-matrix gate+up+down, E8M0 B=32)")
    print(f"{'='*74}")
    print(f"  {'Scheme':<40}  {'strict%':>8}  {'th@0.7%':>8}  {'th@1.0%':>8}  bpw")
    print(f"  {'-'*70}")
    for label, (s, t07, t10) in results.items():
        bpw = bpw_map.get(label, "?")
        print(f"  {label:<40}  {(1-s)*100:>7.1f}%  {(1-t07)*100:>7.1f}%"
              f"  {(1-t10)*100:>7.1f}%  {bpw}")
    for label, s, t07, t10, bpw in ref_rows:
        print(f"  {label:<40}  {(1-s)*100:>7.1f}%  {(1-t07)*100:>7.1f}%"
              f"  {(1-t10)*100:>7.1f}%  {bpw}")
    print(f"{'='*74}")


if __name__ == "__main__":
    main()
