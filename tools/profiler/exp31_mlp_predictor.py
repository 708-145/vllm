# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 31 – two-layer MLP predictor for hot-channel routing.

Motivation
----------
Exp30 showed that scalar output nonlinearities σ(Px) do not help over linear
regression because SwiGLU activity a[i] = SiLU(W_gate[i]·x) * W_up[i]·x is
bilinear — it requires knowing two independent scalar projections simultaneously.
A scalar nonlinearity applied after a single linear map cannot express this.

A two-layer MLP can break the bilinear barrier:
  h = SiLU(x @ W1.T)    hidden (N, r)     W1 ∈ R^{r×H}
  y = h @ W2.T           output (N, I)     W2 ∈ R^{I×r}

With hidden width r, W1 can encode a mix of W_gate and W_up directions into the
hidden state, and W2 can then form products.  If r ≥ 2 the MLP can in principle
represent SiLU(a) * b for any pair (a, b) = (W_gate[i]·x, W_up[i]·x) by
allocating pairs of hidden units.

Inference cost vs full gate GEMM (H=2560, I=8192, one GEMM = 2*H*I = 41.9M FLOPs):
  r=256:  2*(H*r + r*I) = 2*(655k + 2.1M) = 5.5M FLOPs  = 13% of one GEMM
  r=1024: 2*(2.6M + 8.4M)                 = 22.1M FLOPs  = 53% of one GEMM
  r=2560: 2*(6.6M + 20.9M)               = 55.0M FLOPs  = 131% of one GEMM

Targets:
  reg: MSE on down_input (SwiGLU activity)
  bin: BCE on hot labels

Training: Adam, 500 steps, batch=512, lr=3e-3.
          W1 initialised with top-r left singular vectors of W_gate (heuristic
          warm start that encodes relevant directions from the start).
          W2 initialised with random N(0, 1/r).
Split: 80% train, 20% test per layer.
Device: MPS if available.

Usage::

    python tools/profiler/exp31_mlp_predictor.py \\
        --activations ffn_activations128.npz \\
        --svd-cache svd_factors_r1024.pt
"""

import argparse
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


EPS   = 1e-9
H_DIM = 2560
I_DIM = 8192


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEV = _device()


# ---------------------------------------------------------------------------
# Routing quality helpers
# ---------------------------------------------------------------------------

def topk_hot(scores: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, scores.shape[-1])
    idx = torch.topk(scores.abs(), k, dim=-1, sorted=False).indices
    m   = torch.zeros_like(scores, dtype=torch.bool)
    m.scatter_(-1, idx, True)
    return m


def routing_f1(pred: torch.Tensor, Y: torch.Tensor, k: int) -> float:
    hot_p = topk_hot(pred, k)
    hot_o = topk_hot(Y,    k)
    tp = ( hot_p &  hot_o).float().sum().item()
    fp = ( hot_p & ~hot_o).float().sum().item()
    fn = (~hot_p &  hot_o).float().sum().item()
    prec   = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    return 2 * prec * recall / (prec + recall + EPS)


# ---------------------------------------------------------------------------
# Two-layer MLP module
# ---------------------------------------------------------------------------

HIDDEN_ACTS = {
    "silu": F.silu,
    "relu": F.relu,
    "gelu": F.gelu,
}


class TwoLayerMLP(nn.Module):
    """h = act(x @ W1.T);  out = h @ W2.T"""
    def __init__(self, h_in: int, hidden: int, h_out: int, act: str = "silu"):
        super().__init__()
        self.W1  = nn.Linear(h_in,  hidden, bias=False)
        self.W2  = nn.Linear(hidden, h_out,  bias=False)
        self.act = HIDDEN_ACTS[act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(self.act(self.W1(x)))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_mlp(X_tr: torch.Tensor, Y_tr: torch.Tensor,
              hidden: int, target: str, hidden_act: str,
              n_steps: int, lr: float, batch_size: int,
              svd_W1_init: torch.Tensor | None = None) -> TwoLayerMLP:
    """Train two-layer MLP predictor.

    Args:
        svd_W1_init: if given, initialise W1 with top-hidden rows of this
                     (r, H) matrix (e.g. Vt from SVD of W_gate).
    """
    device = X_tr.device
    model  = TwoLayerMLP(H_DIM, hidden, I_DIM, act=hidden_act).to(device)

    # W1 warm start: use SVD directions of W_gate if available
    if svd_W1_init is not None:
        r = min(hidden, svd_W1_init.shape[0])
        with torch.no_grad():
            model.W1.weight[:r].copy_(svd_W1_init[:r].to(device))
    # W2: small random
    nn.init.normal_(model.W2.weight, std=1.0 / hidden**0.5)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    N   = X_tr.shape[0]

    for _ in range(n_steps):
        idx = torch.randint(0, N, (batch_size,), device=device)
        xb, yb = X_tr[idx], Y_tr[idx]

        pred = model(xb)

        if target == "reg":
            loss = F.mse_loss(pred, yb)
        else:
            hot_labels = (yb.abs() > yb.abs().mean(dim=-1, keepdim=True)).float()
            loss = F.binary_cross_entropy_with_logits(pred, hot_labels)

        opt.zero_grad()
        loss.backward()
        opt.step()

    return model


# ---------------------------------------------------------------------------
# Per-layer evaluation
# ---------------------------------------------------------------------------

def evaluate_layer(X: torch.Tensor, Y: torch.Tensor,
                   hidden_widths: list[int],
                   hidden_acts: list[str],
                   hot_fracs: list[float],
                   n_steps: int,
                   svd_Vt_gate: torch.Tensor | None,
                   train_frac: float = 0.8,
                   lr: float = 3e-3,
                   batch_size: int = 512) -> dict:
    """Returns results[hidden][act][target][frac] = f1."""
    N    = X.shape[0]
    n_tr = int(N * train_frac)
    X_tr, X_te = X[:n_tr], X[n_tr:]
    Y_tr, Y_te = Y[:n_tr], Y[n_tr:]

    results = {}
    for hidden in hidden_widths:
        results[hidden] = {}
        for act in hidden_acts:
            results[hidden][act] = {}
            for target in ["reg", "bin"]:
                model = train_mlp(X_tr, Y_tr,
                                  hidden=hidden, target=target,
                                  hidden_act=act,
                                  n_steps=n_steps, lr=lr, batch_size=batch_size,
                                  svd_W1_init=svd_Vt_gate)
                model.eval()
                with torch.no_grad():
                    pred_te = model(X_te)

                results[hidden][act][target] = {}
                for frac in hot_fracs:
                    k = max(1, int(frac * I_DIM))
                    results[hidden][act][target][frac] = routing_f1(pred_te, Y_te, k)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="Exp31: two-layer MLP predictor for hot-channel routing.")
    p.add_argument("--activations",  default="ffn_activations128.npz")
    p.add_argument("--svd-cache",    default="svd_factors_r1024.pt",
                   help="SVD factors for W1 warm-start (optional).")
    p.add_argument("--hidden-widths", nargs="+", type=int,
                   default=[256, 1024, 2560])
    p.add_argument("--hidden-acts", nargs="+", type=str,
                   default=["silu", "relu", "gelu"],
                   choices=list(HIDDEN_ACTS),
                   help="Hidden layer activations to sweep.")
    p.add_argument("--hot-fractions", nargs="+", type=float,
                   default=[0.20, 0.50])
    p.add_argument("--layers", nargs="+", type=int,
                   default=list(range(0, 40, 8)))   # 0,8,16,24,32
    p.add_argument("--n-steps",  type=int, default=500)
    p.add_argument("--lr",       type=float, default=3e-3)
    p.add_argument("--batch",    type=int, default=512)
    args = p.parse_args(argv)

    print(f"Device: {DEV}")
    print(f"Hidden widths: {args.hidden_widths}  acts: {args.hidden_acts}  "
          f"layers: {args.layers}  steps: {args.n_steps}")

    data = np.load(args.activations)

    # Load SVD for W1 warm-start
    svd = None
    svd_path = Path(args.svd_cache)
    if svd_path.exists():
        raw = torch.load(svd_path, weights_only=True)
        # raw[li] = (U_g, s_g, Vt_g, ...)  Vt_g shape (r, H) — right sing. vecs
        svd = [f[2].float() for f in raw]   # list of (r, H) per layer
        print(f"Loaded SVD W1 warm-start from {svd_path}")
    else:
        print("No SVD cache found — W1 randomly initialised.")

    # Accumulators: acc[hidden][act][target][frac] = list of f1 per layer
    acc = defaultdict(lambda: defaultdict(
              lambda: defaultdict(lambda: defaultdict(list))))

    for li in args.layers:
        X = torch.from_numpy(data[f"layer{li}/gate_up_input"]).float().to(DEV)
        Y = torch.from_numpy(data[f"layer{li}/down_input"]).float().to(DEV)
        print(f"\nLayer {li:2d} ...", end=" ", flush=True)
        t0 = time.time()

        res = evaluate_layer(
            X, Y,
            hidden_widths=args.hidden_widths,
            hidden_acts=args.hidden_acts,
            hot_fracs=args.hot_fractions,
            n_steps=args.n_steps,
            svd_Vt_gate=svd[li].to(DEV) if svd else None,
            lr=args.lr,
            batch_size=args.batch,
        )
        print(f"{time.time()-t0:.0f}s", flush=True)

        for hidden in args.hidden_widths:
            for act in args.hidden_acts:
                for target in ["reg", "bin"]:
                    for frac in args.hot_fractions:
                        acc[hidden][act][target][frac].append(
                            res[hidden][act][target][frac])

    # ----------------------------------------------------------------
    # Summary tables
    # ----------------------------------------------------------------
    for frac in args.hot_fractions:
        k = max(1, int(frac * I_DIM))
        print(f"\n{'='*75}")
        print(f"Routing F1  —  hot={frac*100:.0f}%  k={k}"
              f"  (avg layers {args.layers})")
        print(f"{'='*75}")
        hdr = f"  {'hidden':>8}  {'act':<5}  {'target':<5}" + \
              "".join(f"  layer{li:02d}" for li in args.layers) + \
              "    mean"
        print(hdr)
        print(f"  {'-'*70}")
        for hidden in args.hidden_widths:
            for act in args.hidden_acts:
                for target in ["reg", "bin"]:
                    vals = acc[hidden][act][target][frac]
                    row  = f"  {hidden:>8}  {act:<5}  {target:<5}"
                    row += "".join(f"  {v:6.4f}" for v in vals)
                    row += f"  {np.mean(vals):6.4f}"
                    print(row)
            print()

        print()
        print("  References:")
        if abs(frac - 0.20) < 0.01:
            print("    Exp30 best (linear reg, full-rank):  F1=0.486")
            print("    SVD W_gate rank=1024 (exp25d):       F1=0.745")
            print("    E5M3 W_gate (exp25c):                F1=0.784")
        else:
            print("    Exp30 best (linear reg, full-rank):  F1=0.629")
            print("    SVD W_gate rank=1024 (exp25d):       F1=0.873")
        print(f"{'='*65}")

    # ----------------------------------------------------------------
    # FLOP cost
    # ----------------------------------------------------------------
    print("\nInference cost (two-layer MLP per token per layer):")
    full_gemm = 2 * H_DIM * I_DIM
    for hidden in args.hidden_widths:
        cost = 2 * (H_DIM * hidden + hidden * I_DIM)
        print(f"  hidden={hidden:>5}: {cost/1e6:.1f}M FLOPs"
              f"  = {100*cost/full_gemm:.0f}% of one gate GEMM")


if __name__ == "__main__":
    main()
