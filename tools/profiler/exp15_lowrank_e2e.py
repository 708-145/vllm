# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 15 – low-rank gate+up approximation, end-to-end top-1.

Motivation
----------
Experiment 3 evaluated low-rank SVD as a *gate proxy* (routing signal only).
This experiment applies low-rank approximation to **both gate and up** as the
actual computation (no hot/cold routing split), while keeping the down
projection full-precision.  It then measures the end-to-end top-1 perturbation
rate using the same approach as experiment 14.

Scheme (all channels, no routing split):
  W_gate ≈ U_r S_r Vt_r   (truncated SVD, rank r)
  W_up   ≈ U_r S_r Vt_r   (separate SVD per projection)

  gate_lr = W_gate_r @ x = (U_r * s_r) @ (Vt_r @ x)
  up_lr   = W_up_r   @ x
  swiglu  = silu(gate_lr) * up_lr
  out     = W_down @ swiglu          (full precision always)

FLOP cost per token vs full GEMM (H=2560, I=8192):
  Full gate:      2 * H * I  = 41.9 M
  Rank-r gate:    2 * H * r  + 2 * r * I  = 2r(H+I)
  Rank-128:  2.74 M  →  6.5%  of full
  Rank-512:  10.9 M  →  26%   of full
  Rank-1024: 21.8 M  →  52%   of full

Memory: SVD factors stored as bfloat16, max rank 1024 only.
  Per layer per projection: U (8192×1024) + s (1024) + Vt (1024×2560) = 22 MB bf16
  40 layers × 2 projections × 22 MB = 1.76 GB additional

Usage::

    python tools/profiler/exp15_lowrank_e2e.py \\
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


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Experiment 15: low-rank gate+up, end-to-end top-1.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", default="ibm-granite/granite-4.2-3b")
    p.add_argument("--calibration-set", default="bartowski-imatrix-v5-semantic.txt")
    p.add_argument("--num-prompts", type=int, default=8)
    p.add_argument(
        "--ranks", nargs="+", type=int, default=[128, 512, 1024],
        metavar="R",
    )
    p.add_argument("--max-svd-rank", type=int, default=1024,
                   help="Maximum rank stored; must be >= max(--ranks).")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Low-rank MLP replacement
# ---------------------------------------------------------------------------

class LowRankMLP:
    """GraniteMLP.forward replacement using rank-r gate and up approximations.

    Stores SVD factors as bfloat16 (same dtype as model weights) and casts to
    float32 at call time.  No full weight matrix copies.

    W_gate ≈ (U_r * s_r) @ Vt_r   →   gate = ((x @ Vt_r.T) * s_r) @ U_r.T
    Same for up.
    """

    def __init__(
        self,
        mlp,
        # SVD factors from torch.linalg.svd(W, full_matrices=False):
        #   U  : (I, K)   left singular vectors
        #   s  : (K,)     singular values
        #   Vt : (K, H)   right singular vectors (transposed)
        # stored truncated to max_rank as bfloat16
        U_gate:  torch.Tensor,   # (I, max_rank) bf16
        s_gate:  torch.Tensor,   # (max_rank,)   bf16
        Vt_gate: torch.Tensor,   # (max_rank, H) bf16
        U_up:    torch.Tensor,
        s_up:    torch.Tensor,
        Vt_up:   torch.Tensor,
        rank: int,
    ):
        self._mlp = mlp
        r = rank
        # W_gate ≈ U[:,:r] @ diag(s[:r]) @ Vt[:r,:]
        # gate(x) = W_gate @ x.T ≡ U[:,:r] @ (diag(s[:r]) @ (Vt[:r,:] @ x.T))
        # In row-vector convention (x: T×H):
        #   step1: x @ Vt[:r,:].T  →  (T, r)
        #   step2: * s[:r]          →  (T, r)
        #   step3: @ U[:,:r].T      →  (T, I)   because (U[:,:r].T).T = U[:,:r]
        self._Vt_gate = Vt_gate[:r].contiguous()   # (r, H)
        self._s_gate  = s_gate [:r]                # (r,)
        self._UT_gate = U_gate [:, :r].T.contiguous()  # (r, I) — U transposed
        self._Vt_up   = Vt_up  [:r].contiguous()
        self._s_up    = s_up   [:r]
        self._UT_up   = U_up   [:, :r].T.contiguous()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        xf = x.float()

        # Step: (T,H) @ (H,r) → (T,r) → *(r,) → (T,r) @ (r,I) → (T,I)
        # _Vt: (r,H), _UT: (r,I) stored as U[:,r].T
        hid_gate = xf @ self._Vt_gate.float().T   # (T,H)@(H,r) = (T,r)
        hid_gate = hid_gate * self._s_gate.float()
        gate_lr  = hid_gate @ self._UT_gate.float()  # (T,r)@(r,I) = (T,I)

        hid_up = xf @ self._Vt_up.float().T
        hid_up = hid_up * self._s_up.float()
        up_lr  = hid_up @ self._UT_up.float()        # (T,I)

        swiglu = F.silu(gate_lr) * up_lr                  # (T, I)

        W_down = self._mlp.down_proj.weight.detach().float()   # (H, I)
        out = (swiglu @ W_down.T).to(orig_dtype)
        return out


# ---------------------------------------------------------------------------
# Norm-hook capture (same as exp14)
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
        logits = output.float() @ self._W_U.T
        self.ids.extend(logits.argmax(dim=-1).cpu().tolist())


def _load_prompts(path: str, n: int) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip()][:n]


def _run(llm, prompts: list[str], W_U: torch.Tensor) -> np.ndarray:
    from vllm import SamplingParams
    cap = NormCapture(W_U)
    cap.attach(
        llm.llm_engine.model_executor.driver_worker.model_runner.model.model.norm)
    llm.generate(prompts,
                 SamplingParams(max_tokens=1, temperature=0.0),
                 use_tqdm=False)
    cap.detach()
    # Drop the N_prompts decode-step entries at the end (one per prompt)
    ids = cap.ids
    n_decode = len(prompts)
    return np.array(ids[:-n_decode] if n_decode < len(ids) else ids,
                    dtype=np.int32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    args = _parse_args(argv)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    max_rank = max(max(args.ranks), args.max_svd_rank)

    prompts = _load_prompts(args.calibration_set, args.num_prompts)
    print(f"Loaded {len(prompts)} prompts.", file=sys.stderr)
    print(f"Ranks: {args.ranks}  max_svd_rank: {max_rank}", file=sys.stderr)

    from vllm import LLM
    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              kv_cache_memory_bytes=int(0.5 * 1024**3), max_model_len=512,
              enable_prefix_caching=False)
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    W_U = model.lm_head.weight.detach().float()

    # --- Precompute SVD factors (cached to disk to avoid recomputation) ---
    cache_path = Path(f"svd_factors_r{max_rank}.pt")
    if cache_path.exists():
        print(f"Loading SVD factors from {cache_path}...", file=sys.stderr)
        svd_factors = torch.load(cache_path, weights_only=True)
    else:
        print(f"Computing SVD (max rank {max_rank}) for all layers...",
              file=sys.stderr)
        t0 = time.time()
        svd_factors: list[tuple] = []
        for li, layer in enumerate(model.model.layers):
            W = layer.mlp.gate_up_proj.weight.detach().float().cpu()
            I = W.shape[0] // 2
            W_gate = W[:I]   # (I, H) = (8192, 2560)
            W_up   = W[I:]

            U_g, s_g, Vt_g = torch.linalg.svd(W_gate, full_matrices=False)
            U_u, s_u, Vt_u = torch.linalg.svd(W_up,   full_matrices=False)

            r = min(max_rank, s_g.shape[0])
            # U_g[:,  :r]: (I, r) — constructor slices columns and transposes
            # Vt_g[: r]: (r, H)
            svd_factors.append((
                U_g[:, :r].to(torch.bfloat16),
                s_g[  :r].to(torch.bfloat16),
                Vt_g[ :r].to(torch.bfloat16),
                U_u[:, :r].to(torch.bfloat16),
                s_u[  :r].to(torch.bfloat16),
                Vt_u[ :r].to(torch.bfloat16),
            ))
            del W, W_gate, W_up, U_g, s_g, Vt_g, U_u, s_u, Vt_u

            if (li + 1) % 10 == 0:
                print(f"  {li+1}/{len(model.model.layers)} layers done "
                      f"({time.time()-t0:.0f}s)", file=sys.stderr)

        print(f"SVD complete in {time.time()-t0:.0f}s.", file=sys.stderr)
        torch.save(svd_factors, cache_path)
        print(f"Saved SVD factors to {cache_path}.", file=sys.stderr)

    orig_forwards = [layer.mlp.forward for layer in model.model.layers]

    # --- Baseline ---
    print("Baseline pass...", file=sys.stderr)
    baseline = _run(llm, prompts, W_U)
    n_tok = len(baseline)
    print(f"  {n_tok} prefill-token predictions captured.", file=sys.stderr)

    results: dict[int, float] = {}

    n_passes = len(args.ranks)
    for pass_idx, rank in enumerate(args.ranks, 1):
        # Install low-rank MLP for all layers
        for li, layer in enumerate(model.model.layers):
            U_g, s_g, Vt_g, U_u, s_u, Vt_u = svd_factors[li]
            h = LowRankMLP(
                mlp=layer.mlp,
                U_gate=U_g, s_gate=s_g, Vt_gate=Vt_g,
                U_up=U_u,   s_up=s_u,   Vt_up=Vt_u,
                rank=rank,
            )
            layer.mlp.forward = h

        print(f"  [{pass_idx}/{n_passes}] rank={rank}...",
              end="  ", file=sys.stderr, flush=True)
        hybrid_ids = _run(llm, prompts, W_U)
        hybrid = hybrid_ids[:n_tok]
        match = float((hybrid == baseline).mean())
        results[rank] = match
        print(f"match={match:.4f}  perturb={1-match:.4f}", file=sys.stderr)

        for li, layer in enumerate(model.model.layers):
            layer.mlp.forward = orig_forwards[li]

    # --- Summary ---
    print("\n--- Experiment 15 Results ---\n")

    H, I = 2560, 8192
    full_flops = 2 * H * I

    print("End-to-end top-1 results vs low-rank gate+up approximation:\n")
    print(f"  {'rank':>6}  {'FLOP%':>7}  {'energy%':>9}  "
          f"{'match':>8}  {'perturb':>9}")

    for rank in args.ranks:
        flop_frac = 2 * rank * (H + I) / full_flops * 100
        # mean energy fraction across layers at this rank
        mean_energy = float(np.mean([
            float((svd_factors[li][1][:rank].float()**2).sum() /
                  (svd_factors[li][1].float()**2).sum())
            for li in range(len(model.model.layers))
        ])) * 100
        match = results[rank]
        print(f"  {rank:>6}  {flop_frac:>6.1f}%  {mean_energy:>8.1f}%  "
              f"{match:>8.4f}  {1-match:>9.4f}")

    # Reference: ternary gate best (exp14 at 10% hot)
    print(f"\n  {'ternary(10%)':>6}  {'~52%':>7}  {'n/a':>9}  "
          f"{'0.5160':>8}  {'0.4840':>9}  (exp14 reference)")
    print(f"  {'full prec':>6}  {'100%':>7}  {'100%':>9}  "
          f"{'1.0000':>8}  {'0.0000':>9}  (baseline)")


if __name__ == "__main__":
    main()
