# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Nonlinear output predictor: σ(Px) for activity / hot-channel routing.

Trains a predictor of shape (I, H) — same as W_gate — with an optional
scalar output nonlinearity (none / ReLU / SiLU / Abs).

Targets:
  A. Regression:  predict down_input[i] = SiLU(gate[i]) * up[i]  (MSE loss)
  B. Binary:      predict hot[i] = (|down_input[i]| > threshold)  (BCE loss)

For each (target, nonlinearity, rank) combination we:
  1. Solve / train the predictor on 80% of tokens per layer.
  2. Evaluate routing quality on the held-out 20%.

Full-rank (H=2560 columns) is the maximum — same inference cost as one GEMM.
Low-rank variants (r << H) reduce cost: P = A B^T, A∈R^{I×r}, B∈R^{H×r}.

Key question: does a scalar output nonlinearity help over linear cross-cov SVD?

Training:
  Full-rank optimal solution is the closed-form least-squares / logistic
  regression solution, solved layer-by-layer offline from the NPZ activations.
  For the nonlinear case we use a few steps of gradient descent (Adam).

Usage::

    python tools/profiler/exp_nonlinear_predictor.py \\
        --activations ffn_activations128.npz
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


EPS = 1e-9
H, I_DIM = 2560, 8192


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEV = _device()

NONLINEARITIES = {
    "none":  lambda x: x,
    "relu":  torch.relu,
    "silu":  F.silu,
    "abs":   torch.abs,
}


# ---------------------------------------------------------------------------
# Routing quality helpers
# ---------------------------------------------------------------------------

def topk_hot(scores: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, scores.shape[-1])
    idx = torch.topk(scores.abs(), k, dim=-1, sorted=False).indices
    m   = torch.zeros_like(scores, dtype=torch.bool)
    m.scatter_(-1, idx, True)
    return m


def routing_metrics(pred: torch.Tensor, oracle: torch.Tensor,
                    k: int) -> dict:
    hot_p = topk_hot(pred,   k)
    hot_o = topk_hot(oracle, k)
    tp = ( hot_p &  hot_o).float().sum().item()
    fp = ( hot_p & ~hot_o).float().sum().item()
    fn = (~hot_p &  hot_o).float().sum().item()
    prec   = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1     = 2 * prec * recall / (prec + recall + EPS)
    iou    = tp / (tp + fp + fn + EPS)
    return dict(prec=prec, recall=recall, f1=f1, iou=iou)


# ---------------------------------------------------------------------------
# Closed-form full-rank solvers
# ---------------------------------------------------------------------------

def solve_linear_regression(X_tr: torch.Tensor,
                             Y_tr: torch.Tensor,
                             ridge: float = 1e-4) -> torch.Tensor:
    """Closed-form ridge regression: P = (X^T X + λI)^{-1} X^T Y.

    Returns P of shape (I, H) such that P @ x ≈ y.
    Solved as: P^T = (X^T X + λI)^{-1} X^T Y  →  P = result.T
    """
    N, H_ = X_tr.shape
    A = X_tr.T @ X_tr + ridge * torch.eye(H_, device=X_tr.device, dtype=X_tr.dtype)
    B = X_tr.T @ Y_tr   # (H, I)
    # Solve A @ P^T = B  →  P^T = A^{-1} B
    Pt = torch.linalg.solve(A, B)   # (H, I)
    return Pt.T.contiguous()        # (I, H)


# ---------------------------------------------------------------------------
# Gradient-based nonlinear solver
# ---------------------------------------------------------------------------

def train_nonlinear(X_tr: torch.Tensor, Y_tr: torch.Tensor,
                    rank: int, nonlin_name: str,
                    target: str,
                    n_steps: int = 300, lr: float = 3e-3,
                    batch_size: int = 512) -> torch.Tensor:
    """Train P (or A,B for low-rank) with output nonlinearity.

    For full-rank (rank == H): P ∈ R^{I×H}.
    For low-rank:              P = A @ B.T, A∈R^{I×r}, B∈R^{H×r}.

    Returns P_full (I, H) for inference.
    """
    nonlin = NONLINEARITIES[nonlin_name]
    N = X_tr.shape[0]
    device = X_tr.device

    if rank >= H:
        # Full-rank: initialise from linear solution as warm start
        P_lin = solve_linear_regression(X_tr, Y_tr if target == "reg"
                                        else (Y_tr > 0).float())
        P = P_lin.clone().requires_grad_(True)
        params = [P]
        def forward(x):
            return nonlin(x @ P.T)
    else:
        # Low-rank: P = A @ B.T
        A = torch.randn(I_DIM, rank, device=device) * 0.01
        B = torch.randn(H,     rank, device=device) * 0.01
        A.requires_grad_(True)
        B.requires_grad_(True)
        params = [A, B]
        def forward(x):
            return nonlin(x @ B @ A.T)

    opt = torch.optim.Adam(params, lr=lr)

    for step in range(n_steps):
        idx = torch.randint(0, N, (batch_size,), device=device)
        xb  = X_tr[idx]   # (bs, H)
        yb  = Y_tr[idx]   # (bs, I)

        pred = forward(xb)

        if target == "reg":
            loss = F.mse_loss(pred, yb)
        else:
            # Binary target: use sigmoid + BCE on the raw (pre-nonlin) output
            # For binary we don't apply nonlin to the loss input
            if rank >= H:
                logits = xb @ P.T
            else:
                logits = xb @ B @ A.T
            loss = F.binary_cross_entropy_with_logits(
                logits, (yb.abs() > yb.abs().mean(dim=-1, keepdim=True)).float())

        opt.zero_grad()
        loss.backward()
        opt.step()

    if rank >= H:
        return P.detach()
    else:
        return (A @ B.T).detach()   # materialise full (I, H) matrix


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_layer(X: torch.Tensor, Y: torch.Tensor,
                   ranks: list[int],
                   hot_fracs: list[float],
                   n_steps: int,
                   train_frac: float = 0.8) -> dict:
    """Train and evaluate all (rank, nonlin, target) combos for one layer.

    Returns nested dict: results[rank][nonlin][target][frac] = metrics dict.
    """
    N = X.shape[0]
    n_tr = int(N * train_frac)
    X_tr, X_te = X[:n_tr], X[n_tr:]
    Y_tr, Y_te = Y[:n_tr], Y[n_tr:]

    results = {}

    for rank in ranks:
        results[rank] = {}
        for nonlin_name in NONLINEARITIES:
            results[rank][nonlin_name] = {}
            for target in ["reg", "bin"]:
                key = f"{nonlin_name}/{target}"

                if nonlin_name == "none" and rank >= H:
                    # Closed-form solution (fast, exact)
                    Y_fit = Y_tr if target == "reg" else (Y_tr.abs() > Y_tr.abs().mean(dim=-1, keepdim=True)).float()
                    P = solve_linear_regression(X_tr, Y_fit)
                    pred_te = X_te @ P.T
                else:
                    P = train_nonlinear(X_tr, Y_tr, rank=rank,
                                        nonlin_name=nonlin_name,
                                        target=target,
                                        n_steps=n_steps)
                    nonlin = NONLINEARITIES[nonlin_name]
                    pred_te = nonlin(X_te @ P.T)

                results[rank][nonlin_name][target] = {}
                for frac in hot_fracs:
                    k = max(1, int(frac * I_DIM))
                    m = routing_metrics(pred_te, Y_te, k)
                    results[rank][nonlin_name][target][frac] = m

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="Nonlinear output predictor routing quality.")
    p.add_argument("--activations", default="ffn_activations128.npz")
    p.add_argument(
        "--ranks", nargs="+", type=int,
        default=[256, H],   # low-rank and full-rank
        metavar="R",
        help=f"Predictor ranks. Use {H} for full-rank (same as one GEMM).",
    )
    p.add_argument(
        "--hot-fractions", nargs="+", type=float, default=[0.20, 0.50])
    p.add_argument(
        "--layers", nargs="+", type=int,
        default=list(range(0, 40, 8)),   # 5 representative layers
    )
    p.add_argument("--n-steps", type=int, default=400,
                   help="Gradient steps for nonlinear training.")
    args = p.parse_args(argv)

    data  = np.load(args.activations)
    N_tot = data["layer0/gate_up_input"].shape[0]
    print(f"Dataset: N={N_tot} tokens/layer, H={H}, I={I_DIM}")
    print(f"Ranks: {args.ranks}  nonlinearities: {list(NONLINEARITIES)}")
    print(f"Layers: {args.layers}  n_steps (nonlinear): {args.n_steps}")

    # Accumulate macro-average results
    # acc[rank][nonlin][target][frac] = list of f1 values
    from collections import defaultdict
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: defaultdict(list))))

    for li in args.layers:
        X = torch.from_numpy(data[f"layer{li}/gate_up_input"]).float().to(DEV)
        Y = torch.from_numpy(data[f"layer{li}/down_input"]).float().to(DEV)
        print(f"\nLayer {li} ...", end=" ", flush=True)
        t0 = time.time()
        res = evaluate_layer(X, Y,
                             ranks=args.ranks,
                             hot_fracs=args.hot_fractions,
                             n_steps=args.n_steps)
        print(f"{time.time()-t0:.0f}s", flush=True)

        for rank in args.ranks:
            for nonlin in NONLINEARITIES:
                for target in ["reg", "bin"]:
                    for frac in args.hot_fractions:
                        m = res[rank][nonlin][target][frac]
                        acc[rank][nonlin][target][frac].append(m["f1"])

    # ----------------------------------------------------------------
    # Print summary tables per hot fraction
    # ----------------------------------------------------------------
    for frac in args.hot_fractions:
        k = max(1, int(frac * I_DIM))
        print(f"\n{'='*80}")
        print(f"Routing F1  —  hot={frac*100:.0f}%  k={k}  "
              f"(avg layers {args.layers})")
        print(f"{'='*80}")
        print(f"  {'nonlin':<8}  {'target':<5}" +
              "".join(f"  rank={r:>5}" for r in args.ranks))
        print(f"  {'-'*60}")
        for nonlin in NONLINEARITIES:
            for target in ["reg", "bin"]:
                row = f"  {nonlin:<8}  {target:<5}"
                for rank in args.ranks:
                    vals = acc[rank][nonlin][target][frac]
                    f1   = float(np.mean(vals)) if vals else float("nan")
                    row += f"  {f1:>9.4f}"
                print(row)
        print()
        # Reference lines
        print(f"  Reference (from exp25c/25d, using weight matrices):")
        if abs(frac - 0.20) < 0.01:
            print(f"    Linear cross-cov (exp25e):  F1≈0.32  (all ranks)")
            print(f"    SVD W_gate rank=1024:        F1=0.745")
            print(f"    E5M3 W_gate (exp25c):        F1=0.784")
        elif abs(frac - 0.50) < 0.01:
            print(f"    Linear cross-cov (exp25e):  F1≈0.54  (all ranks)")
            print(f"    SVD W_gate rank=1024:        F1=0.873")
        print(f"{'='*80}")

    # ----------------------------------------------------------------
    # Compute cost summary
    # ----------------------------------------------------------------
    print("\nInference cost (per token, per layer):")
    for rank in args.ranks:
        r_eff = min(rank, H)
        cost  = 2 * r_eff * I_DIM  # P @ x  (one GEMM: (I, r) @ (r,) per token)
        full  = 2 * H * I_DIM
        print(f"  rank={rank:>5}: {cost/1e6:.1f}M FLOPs  "
              f"= {100*cost/full:.0f}% of full gate GEMM")


if __name__ == "__main__":
    main()
