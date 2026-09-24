# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 28 – binary E5M3 SVD routing at rank 2048.

Motivation
----------
Exp27 showed that rank=1024 full-precision SVD routing achieves 0.834 match
at 50% hot, beating exp24's 0.736.  The limitation was that rank=1024 was the
maximum cached rank.

This experiment explores rank=2048 (the maximum useful rank for Vt, since
H=2560 limits the SVD), with the SVD factor matrices themselves encoded in
the same B=8 E5M3 binary format used for the cold-channel weights.

Encoding the routing predictor:
  Vt_g_enc[r, H]  = E5M3(Vt_g)   row-wise B=8 blocks along H dimension
  UT_g_enc[r, I]  = E5M3(UT_g)   row-wise B=8 blocks along I dimension

  routing:  hg = (xf @ Vt_g_enc.T) * s_g   →  (T, r)
            gate_lr = hg @ UT_g_enc          →  (T, I)

The singular values s_g are kept at full precision (one float per singular
value, negligible storage and no loss worth caring about).

Since Vt rows are unit-norm orthonormal vectors and UT rows are scaled by s,
the E5M3 block encoding imposes per-8-element scale quantisation on the
direction vectors.  The question is whether this preserves enough directional
information to route correctly.

Rank 2048 requires recomputing SVD beyond the cached rank=1024.  The new
factors are cached to svd_factors_r2048.pt.

Hot fractions swept: 0.20, 0.30, 0.50 — same as exp27 for direct comparison.

Usage::

    python tools/profiler/exp28_e5m3_svd_routing.py \\
        --model ibm-granite/granite-4.2-3b \\
        --calibration-set bartowski-imatrix-v5-semantic.txt
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
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
BLOCK_SIZE = 8
EPS = 1e-9


# ---------------------------------------------------------------------------
# E5M3 encoding (identical to exp22/23/26/27)
# ---------------------------------------------------------------------------

_M3_FRAC_LOG2 = torch.log2(1.0 + torch.arange(8).float() / 8.0)


def _floor_eps(W: torch.Tensor, pct: float = 1.0) -> torch.Tensor:
    flat = W.abs().reshape(-1)
    k = max(1, int(len(flat) * pct / 100.0))
    return flat.kthvalue(k).values.clamp(min=EPS)


def build_e5m3_encoded(W: torch.Tensor, B: int = BLOCK_SIZE) -> torch.Tensor:
    """Encode (O, I) weight matrix with B=8 E5M3 block scales."""
    O, I = W.shape
    eps = _floor_eps(W)
    pad = (B - I % B) % B
    Wp  = F.pad(W, (0, pad)) if pad else W
    W_b = Wp.reshape(-1, B)

    wa     = W_b.abs().clamp(min=eps)
    tilt   = torch.log1p(wa / eps)
    log2_s = (tilt * torch.log2(wa)).sum(1) / tilt.sum(1).clamp(min=EPS)

    e        = log2_s.floor().to(torch.int32)
    frac     = log2_s - e.float()
    m_lut    = _M3_FRAC_LOG2.to(frac.device)
    m_best   = (frac.unsqueeze(1) - m_lut).abs().argmin(1)
    log2_s_q = e.float() + m_lut[m_best]
    scales   = (2.0 ** log2_s_q).clamp(min=EPS)

    n_blk  = Wp.shape[1] // B
    s_exp  = scales.reshape(O, n_blk).unsqueeze(2).expand(O, n_blk, B).reshape(O, Wp.shape[1])
    return (Wp.sign() * s_exp)[:, :I].contiguous()


# ---------------------------------------------------------------------------
# SVD factor loading / computation
# ---------------------------------------------------------------------------

def _load_or_compute_svd(layers, max_rank: int,
                          cache_path: Path) -> list[tuple]:
    """Returns list of (U_g, s_g, Vt_g, U_u, s_u, Vt_u) per layer, float32 cpu."""
    if cache_path.exists():
        print(f"Loading SVD factors from {cache_path} ...", file=sys.stderr)
        factors = torch.load(cache_path, weights_only=True)
        # Ensure float32 regardless of storage dtype
        return [tuple(t.float() for t in f) for f in factors]

    print(f"Computing SVD (max rank {max_rank}) for {len(layers)} layers ...",
          file=sys.stderr)
    t0 = time.time()
    factors = []
    for li, layer in enumerate(layers):
        W = layer.mlp.gate_up_proj.weight.detach().float().cpu()
        I = W.shape[0] // 2
        W_gate = W[:I]
        W_up   = W[I:]

        U_g, s_g, Vt_g = torch.linalg.svd(W_gate, full_matrices=False)
        U_u, s_u, Vt_u = torch.linalg.svd(W_up,   full_matrices=False)

        r = min(max_rank, s_g.shape[0])
        factors.append((
            U_g[:, :r].contiguous().float(),   # (I, r)
            s_g[  :r].float(),                 # (r,)
            Vt_g[ :r].contiguous().float(),    # (r, H)
            U_u[:, :r].contiguous().float(),
            s_u[  :r].float(),
            Vt_u[ :r].contiguous().float(),
        ))
        if (li + 1) % 10 == 0:
            print(f"  {li+1}/{len(layers)} layers ({time.time()-t0:.0f}s)",
                  file=sys.stderr)

    torch.save(factors, cache_path)
    print(f"SVD done in {time.time()-t0:.1f}s, saved to {cache_path}",
          file=sys.stderr)
    return factors


# ---------------------------------------------------------------------------
# Hybrid MLP: E5M3-encoded SVD routing + full-prec hot + E5M3 cold
# ---------------------------------------------------------------------------

class E5M3SvdRoutingMLP:
    """Routing via E5M3-encoded rank-r SVD factors; hot full-precision; cold E5M3.

    Routing:
      Vt_g_enc, UT_g_enc = E5M3(Vt_g), E5M3(UT_g.T).T  (rows encoded)
      hg       = xf @ Vt_g_enc.T * s_g          (T, r)
      gate_lr  = hg @ UT_g_enc                   (T, I)  routing signal

    Computation:
      hot  = union top-k(|gate_lr|, k) ∪ top-k(|up_lr|, k)
      gate = where(hot, gate_full, gate_cold_e5m3)
      up   = where(hot, up_full,   up_cold_e5m3)
      out  = (silu(gate) * up) @ W_down.T
    """

    def __init__(
        self,
        mlp,
        U_g:  torch.Tensor,   # (I, r) float32 cpu
        s_g:  torch.Tensor,   # (r,)
        Vt_g: torch.Tensor,   # (r, H)
        U_u:  torch.Tensor,
        s_u:  torch.Tensor,
        Vt_u: torch.Tensor,
        rank: int,
        k_hot: int,
    ):
        self._mlp   = mlp
        self._k_hot = k_hot

        r = min(rank, s_g.shape[0])

        # Singular values: full precision (tiny storage)
        self._s_g = s_g[:r].float()
        self._s_u = s_u[:r].float()

        # Encode Vt rows (r, H): each row is a right singular vector
        Vt_g_r = Vt_g[:r].float()   # (r, H)
        Vt_u_r = Vt_u[:r].float()

        # Encode UT rows (r, I): U columns transposed → rows of UT
        UT_g_r = U_g[:, :r].T.float().contiguous()   # (r, I)
        UT_u_r = U_u[:, :r].T.float().contiguous()

        self._Vt_g_enc = build_e5m3_encoded(Vt_g_r)   # (r, H)
        self._Vt_u_enc = build_e5m3_encoded(Vt_u_r)
        self._UT_g_enc = build_e5m3_encoded(UT_g_r)   # (r, I)
        self._UT_u_enc = build_e5m3_encoded(UT_u_r)

        # Cold-channel encoding for gate and up
        W_fused = mlp.gate_up_proj.weight.detach().float()
        I_dim = W_fused.shape[0] // 2
        self._I          = I_dim
        self._W_gate_enc = build_e5m3_encoded(W_fused[:I_dim])
        self._W_up_enc   = build_e5m3_encoded(W_fused[I_dim:])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        xf = x.float()
        I  = self._I

        # Routing via E5M3-encoded SVD factors
        hg      = (xf @ self._Vt_g_enc.T) * self._s_g   # (T, r)
        gate_lr = hg @ self._UT_g_enc                    # (T, I)

        hu     = (xf @ self._Vt_u_enc.T) * self._s_u
        up_lr  = hu @ self._UT_u_enc                     # (T, I)

        k = min(self._k_hot, I)
        idx_g = torch.topk(gate_lr.abs(), k, dim=-1, sorted=False).indices
        idx_u = torch.topk(up_lr.abs(),   k, dim=-1, sorted=False).indices
        hot = torch.zeros(xf.shape[0], I, dtype=torch.bool, device=xf.device)
        hot.scatter_(-1, idx_g, True)
        hot.scatter_(-1, idx_u, True)

        # Full-precision weights
        W_fused  = self._mlp.gate_up_proj.weight.detach().float()
        W_gate   = W_fused[:I]
        W_up     = W_fused[I:]
        W_down   = self._mlp.down_proj.weight.detach().float()

        gate_full = xf @ W_gate.T
        up_full   = xf @ W_up.T
        gate_cold = xf @ self._W_gate_enc.T
        up_cold   = xf @ self._W_up_enc.T

        gate   = torch.where(hot, gate_full, gate_cold)
        up     = torch.where(hot, up_full,   up_cold)
        swiglu = F.silu(gate) * up
        return (swiglu @ W_down.T).to(orig_dtype)


# ---------------------------------------------------------------------------
# NormCapture / run helper
# ---------------------------------------------------------------------------

class NormCapture:
    def __init__(self, W_U: torch.Tensor):
        self._W_U = W_U.float()
        self.ids: list[int] = []
        self._handle = None

    def attach(self, norm_module) -> None:
        self._handle = norm_module.register_forward_hook(self._hook)

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _hook(self, module, args, output) -> None:
        self.ids.extend(
            (output.float() @ self._W_U.T).argmax(dim=-1).cpu().tolist())


def _load_prompts(path: str, n: int) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip()][:n]


def _get_internals(llm):
    e = llm.llm_engine
    try:
        mr = e.model_executor.driver_worker.worker.model_runner
    except AttributeError:
        mr = e.model_executor.driver_worker.model_runner
    return mr.model.model.norm, mr.model.model.layers, mr.model.lm_head.weight


def _run(llm, prompts, W_U, norm) -> np.ndarray:
    from vllm import SamplingParams
    cap = NormCapture(W_U.detach().float())
    cap.attach(norm)
    llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0.0),
                 use_tqdm=False)
    cap.detach()
    ids = cap.ids
    n_dec = len(prompts)
    return np.array(ids[:-n_dec] if n_dec < len(ids) else ids, dtype=np.int32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Exp28: E5M3-encoded SVD routing at rank 2048.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", default="ibm-granite/granite-4.2-3b")
    p.add_argument("--calibration-set", default="bartowski-imatrix-v5-semantic.txt")
    p.add_argument("--num-prompts", type=int, default=8)
    p.add_argument(
        "--ranks", nargs="+", type=int, default=[1024, 2048],
        metavar="R",
        help="SVD ranks to test (2048 is max useful for H=2560).",
    )
    p.add_argument(
        "--hot-fractions", nargs="+", type=float, default=[0.20, 0.30, 0.50],
        metavar="F",
    )
    p.add_argument(
        "--svd-cache", default="svd_factors_r2048.pt",
        help="Path to SVD cache (computed fresh if absent).",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    args = _parse_args(argv)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    print(f"Device: {DEV}  B={BLOCK_SIZE} E5M3", file=sys.stderr)

    prompts = _load_prompts(args.calibration_set, args.num_prompts)
    print(f"Loaded {len(prompts)} prompts.", file=sys.stderr)

    from vllm import LLM
    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              kv_cache_memory_bytes=int(0.5 * 1024**3), max_model_len=512,
              enable_prefix_caching=False)

    norm, layers, W_U = _get_internals(llm)
    I = layers[0].mlp.gate_up_proj.weight.shape[0] // 2
    orig_forwards = [l.mlp.forward for l in layers]

    max_rank = max(args.ranks)
    svd_factors = _load_or_compute_svd(layers, max_rank, Path(args.svd_cache))

    print("Baseline pass ...", file=sys.stderr)
    baseline = _run(llm, prompts, W_U, norm)
    n_tok = len(baseline)
    print(f"  {n_tok} token predictions captured.", file=sys.stderr)

    # Reference points
    exp24_ref = {0.20: 0.818, 0.30: 0.798, 0.50: 0.736}
    exp27_ref = {  # rank=1024, full-precision SVD factors
        (1024, 0.20): 0.612, (1024, 0.30): 0.702, (1024, 0.50): 0.834,
    }

    results: dict[tuple, float] = {}

    n_total = len(args.ranks) * len(args.hot_fractions)
    idx = 0
    for rank in args.ranks:
        for frac in args.hot_fractions:
            idx += 1
            k_hot = max(1, int(frac * I))

            mlps = [
                E5M3SvdRoutingMLP(
                    layers[li].mlp,
                    U_g =svd_factors[li][0],
                    s_g =svd_factors[li][1],
                    Vt_g=svd_factors[li][2],
                    U_u =svd_factors[li][3],
                    s_u =svd_factors[li][4],
                    Vt_u=svd_factors[li][5],
                    rank=rank,
                    k_hot=k_hot,
                )
                for li in range(len(layers))
            ]
            for l, m in zip(layers, mlps):
                l.mlp.forward = m

            print(f"  [{idx}/{n_total}] rank={rank:4d}  hot={frac*100:.0f}% ...",
                  end="  ", file=sys.stderr, flush=True)
            ids = _run(llm, prompts, W_U, norm)[:n_tok]
            match = float((ids == baseline).mean())
            results[(rank, frac)] = match
            print(f"match={match:.4f}  perturb={1-match:.4f}", file=sys.stderr)

            for l, fwd in zip(layers, orig_forwards):
                l.mlp.forward = fwd
            del mlps

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    fracs = args.hot_fractions
    H_dim = 2560

    print("\n" + "="*80)
    print("Exp28: E5M3-encoded SVD routing (binary factor matrices)")
    print("hot = union top-k(|gate_lr|) ∪ top-k(|up_lr|)  [actual hot% ≤ 2×frac]")
    print("cold: E5M3 B=8 for both gate and up")
    print("="*80)

    hdr = f"  {'rank':>6}" + "".join(f"  {f*100:>5.0f}%" for f in fracs)
    print("\nTop-1 match (↑ better):")
    print(hdr)
    for rank in args.ranks:
        row = f"  {rank:>6}" + "".join(
            f"  {results[(rank,f)]:>6.4f}" for f in fracs)
        print(row)
    print(f"  {'exp24':>6}" + "".join(
        f"  {exp24_ref.get(f, float('nan')):>6.4f}" for f in fracs))
    print(f"  {'e27r1k':>6}" + "".join(       # exp27 rank=1024 full-prec SVD
        f"  {exp27_ref.get((1024,f), float('nan')):>6.4f}" for f in fracs))

    print("\nΔ vs exp27 rank=1024 full-prec (+ = exp28 better):")
    print(hdr)
    for rank in args.ranks:
        row = f"  {rank:>6}" + "".join(
            f"  {results[(rank,f)] - exp27_ref.get((1024,f), float('nan')):>+6.4f}"
            for f in fracs)
        print(row)

    print("\nΔ vs exp24 E5M3-threshold (+ = exp28 better):")
    print(hdr)
    for rank in args.ranks:
        row = f"  {rank:>6}" + "".join(
            f"  {results[(rank,f)] - exp24_ref.get(f, float('nan')):>+6.4f}"
            for f in fracs)
        print(row)
    print("="*80)

    print("\nRouting cost (E5M3-encoded SVD factors, 1-bit sign + E5M3 scale):")
    for rank in args.ranks:
        r_eff = min(rank, H_dim)   # rank capped at H=2560
        cost_lr   = 2 * r_eff * (H_dim + I)
        cost_full = 2 * H_dim * I
        print(f"  rank={rank} (eff {r_eff}): {cost_lr/1e6:.1f}M FLOPs "
              f"= {100*cost_lr/cost_full:.0f}% of one full GEMM "
              f"  storage: {r_eff*(H_dim+I)*1/1e6:.1f}M bytes (1-bit+scale)")


if __name__ == "__main__":
    main()
