# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 32 – sparse W_gate predictor for hot-channel routing.

Motivation
----------
Exp25–31 established that the optimal routing signal is x @ W_gate.T — using
the actual weight matrix directly.  Learned predictors (exp25e, exp30, exp31)
are all worse.

This experiment asks: can we sparsify W_gate itself while preserving routing
quality?  A sparse W_gate (most weights zero) enables cheaper encoding and
faster matrix-vector products.  The goal is to find the keep-rate below which
routing quality degrades significantly.

Sparsification schemes:
  A. Magnitude pruning (no training): zero out rows with smallest L2 norm,
     or zero out individual weights with smallest |w| (unstructured).
  B. Row-wise pruning: zero entire output rows (channels) — structured,
     hardware-friendly.
  C. Fine-tuned sparse: start from magnitude-pruned W_gate, run a few hundred
     gradient steps to recover routing quality on the training split.

Keep rates swept: 0.5, 0.3, 0.2, 0.1
  0.5 → 50% of weights kept  (50% sparsity)
  0.3 → 30% kept             (70% sparsity)
  0.2 → 20% kept             (80% sparsity)
  0.1 → 10% kept             (90% sparsity)

Routing quality measured as F1 of top-k(|x @ W_sparse.T|) vs oracle
top-k(|x @ W_gate.T|), same as exp25c/25d.

Device: MPS.

Usage::

    python tools/profiler/exp32_sparse_gate_predictor.py \\
        --activations ffn_activations128.npz \\
        --model ibm-granite/granite-4.2-3b
"""

import argparse
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
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
# Routing quality
# ---------------------------------------------------------------------------

def topk_hot(scores: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, scores.shape[-1])
    idx = torch.topk(scores.abs(), k, dim=-1, sorted=False).indices
    m   = torch.zeros_like(scores, dtype=torch.bool)
    m.scatter_(-1, idx, True)
    return m


def routing_f1(pred: torch.Tensor, oracle: torch.Tensor, k: int) -> float:
    hot_p = topk_hot(pred,   k)
    hot_o = topk_hot(oracle, k)
    tp = ( hot_p &  hot_o).float().sum().item()
    fp = ( hot_p & ~hot_o).float().sum().item()
    fn = (~hot_p &  hot_o).float().sum().item()
    prec   = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    return 2 * prec * recall / (prec + recall + EPS)


# ---------------------------------------------------------------------------
# Sparsification schemes
# ---------------------------------------------------------------------------

def magnitude_prune_unstructured(W: torch.Tensor, keep_rate: float) -> torch.Tensor:
    """Zero out (1-keep_rate) fraction of weights with smallest |w|.

    Unstructured: mask applied element-wise across the full weight matrix.
    Returns a new (I, H) tensor with the same values at kept positions, 0 elsewhere.
    """
    flat   = W.abs().reshape(-1)
    k      = max(1, int(keep_rate * flat.numel()))
    thresh = flat.kthvalue(flat.numel() - k + 1).values   # (k+1)-th smallest = threshold
    mask   = W.abs() >= thresh
    return W * mask


def magnitude_prune_rows(W: torch.Tensor, keep_rate: float) -> torch.Tensor:
    """Zero out rows (output channels) with smallest L2 norm.

    Structured: entire rows zeroed.  Each row W[i] corresponds to one gate
    channel; zeroing it means that channel always gets gate_approx=0 as
    its routing signal.
    """
    norms  = W.norm(dim=1)                              # (I,)
    k      = max(1, int(keep_rate * W.shape[0]))
    thresh = norms.kthvalue(norms.numel() - k + 1).values
    mask   = (norms >= thresh).unsqueeze(1).float()     # (I, 1)
    return W * mask


def finetune_sparse(W_init: torch.Tensor,
                    mask: torch.Tensor,
                    X_tr: torch.Tensor,
                    Y_tr: torch.Tensor,
                    n_steps: int = 300,
                    lr: float = 1e-3,
                    batch_size: int = 512) -> torch.Tensor:
    """Fine-tune a masked W_gate to maximise routing quality.

    Optimises MSE(x @ W.T, x @ W_gate_full.T) subject to mask.
    W_gate_full responses are pre-computed as Y_tr = X_tr @ W_gate_full.T.
    The mask is held fixed; only non-zero weights are updated.
    """
    W = W_init.clone().requires_grad_(True)
    opt = torch.optim.Adam([W], lr=lr)
    N   = X_tr.shape[0]

    for _ in range(n_steps):
        idx  = torch.randint(0, N, (batch_size,), device=X_tr.device)
        xb   = X_tr[idx]
        yb   = Y_tr[idx]
        pred = xb @ (W * mask).T
        loss = F.mse_loss(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        # Re-apply mask after each step so zeroed weights stay zero
        with torch.no_grad():
            W.mul_(mask)

    return (W * mask).detach()


# ---------------------------------------------------------------------------
# Per-layer evaluation
# ---------------------------------------------------------------------------

def evaluate_layer(W_gate: torch.Tensor,
                   X: torch.Tensor,
                   hot_fracs: list[float],
                   keep_rates: list[float],
                   finetune_steps: int,
                   train_frac: float = 0.8) -> dict:
    """Evaluate all (scheme, keep_rate) combinations for one layer.

    Returns results[scheme][keep_rate][frac] = f1.
    """
    N    = X.shape[0]
    n_tr = int(N * train_frac)
    X_tr, X_te = X[:n_tr], X[n_tr:]

    # Oracle: full W_gate on test set
    with torch.no_grad():
        gate_full_te = X_te @ W_gate.T   # (N_te, I)
        gate_full_tr = X_tr @ W_gate.T   # (N_tr, I) — target for fine-tuning

    results = {s: {} for s in ["unstructured", "row", "unstructured_ft", "row_ft"]}

    for keep_rate in keep_rates:
        # ---- Magnitude pruning (unstructured) ----
        W_u  = magnitude_prune_unstructured(W_gate, keep_rate)
        pred = X_te @ W_u.T
        results["unstructured"][keep_rate] = {
            frac: routing_f1(pred, gate_full_te, max(1, int(frac * I_DIM)))
            for frac in hot_fracs
        }

        # ---- Row pruning (structured) ----
        W_r  = magnitude_prune_rows(W_gate, keep_rate)
        pred = X_te @ W_r.T
        results["row"][keep_rate] = {
            frac: routing_f1(pred, gate_full_te, max(1, int(frac * I_DIM)))
            for frac in hot_fracs
        }

        if finetune_steps > 0:
            # ---- Fine-tuned unstructured ----
            mask_u = (W_u != 0).float()
            W_uft  = finetune_sparse(W_u, mask_u, X_tr, gate_full_tr,
                                     n_steps=finetune_steps)
            pred   = X_te @ W_uft.T
            results["unstructured_ft"][keep_rate] = {
                frac: routing_f1(pred, gate_full_te, max(1, int(frac * I_DIM)))
                for frac in hot_fracs
            }

            # ---- Fine-tuned row ----
            mask_r = (W_r != 0).float()
            W_rft  = finetune_sparse(W_r, mask_r, X_tr, gate_full_tr,
                                     n_steps=finetune_steps)
            pred   = X_te @ W_rft.T
            results["row_ft"][keep_rate] = {
                frac: routing_f1(pred, gate_full_te, max(1, int(frac * I_DIM)))
                for frac in hot_fracs
            }

    # Also record keep_rate=1.0 baseline (full W_gate)
    pred = gate_full_te
    for scheme in results:
        results[scheme][1.0] = {
            frac: routing_f1(pred, gate_full_te, max(1, int(frac * I_DIM)))
            for frac in hot_fracs
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="Exp32: sparse W_gate predictor routing quality.")
    p.add_argument("--activations",  default="ffn_activations128.npz")
    p.add_argument("--model",        default="ibm-granite/granite-4.2-3b")
    p.add_argument("--keep-rates", nargs="+", type=float,
                   default=[0.5, 0.3, 0.2, 0.1],
                   help="Fraction of weights kept (1-sparsity).")
    p.add_argument("--hot-fractions", nargs="+", type=float,
                   default=[0.20, 0.50])
    p.add_argument("--layers", nargs="+", type=int,
                   default=list(range(0, 40, 8)))
    p.add_argument("--finetune-steps", type=int, default=300,
                   help="Adam steps after pruning (0 to skip fine-tuning).")
    args = p.parse_args(argv)

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    print(f"Device: {DEV}")
    print(f"Keep rates: {args.keep_rates}  layers: {args.layers}  "
          f"finetune_steps: {args.finetune_steps}")

    # Load W_gate weights from model
    print("Loading model weights...", file=sys.stderr)
    from vllm import LLM
    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              kv_cache_memory_bytes=int(0.5 * 1024**3), max_model_len=512,
              enable_prefix_caching=False)
    e = llm.llm_engine
    try:
        mr = e.model_executor.driver_worker.worker.model_runner
    except AttributeError:
        mr = e.model_executor.driver_worker.model_runner
    layers_model = mr.model.model.layers

    data = np.load(args.activations)

    # keep_rates extended with baseline
    keep_rates_full = args.keep_rates + [1.0]
    schemes = ["unstructured", "row"]
    if args.finetune_steps > 0:
        schemes += ["unstructured_ft", "row_ft"]

    # Accumulators: acc[scheme][keep_rate][frac] = list of f1
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for li in args.layers:
        W_fused = layers_model[li].mlp.gate_up_proj.weight.detach().float()
        W_gate  = W_fused[:I_DIM].to(DEV)   # (I, H)

        X = torch.from_numpy(data[f"layer{li}/gate_up_input"]).float().to(DEV)

        print(f"\nLayer {li:2d} ...", end=" ", flush=True)
        t0 = time.time()

        res = evaluate_layer(
            W_gate, X,
            hot_fracs=args.hot_fractions,
            keep_rates=args.keep_rates,
            finetune_steps=args.finetune_steps,
        )
        print(f"{time.time()-t0:.0f}s", flush=True)

        for scheme in schemes:
            for kr in keep_rates_full:
                for frac in args.hot_fractions:
                    acc[scheme][kr][frac].append(res[scheme][kr][frac])

    # ----------------------------------------------------------------
    # Summary tables
    # ----------------------------------------------------------------
    for frac in args.hot_fractions:
        k = max(1, int(frac * I_DIM))
        print(f"\n{'='*72}")
        print(f"Routing F1  —  hot={frac*100:.0f}%  k={k}"
              f"  (avg layers {args.layers})")
        print(f"oracle = top-k(|x @ W_gate_full.T|)")
        print(f"{'='*72}")
        hdr = (f"  {'scheme':<20}  " +
               "".join(f"  kr={kr:.1f}" for kr in keep_rates_full))
        print(hdr)
        print(f"  {'-'*60}")
        for scheme in schemes:
            row = f"  {scheme:<20}"
            for kr in keep_rates_full:
                vals = acc[scheme][kr][frac]
                row += f"  {np.mean(vals):6.4f}"
            print(row)

        print()
        print("  References (weight-derived, no sparsity):")
        if abs(frac - 0.20) < 0.01:
            print("    E5M3 W_gate vs oracle gate_full (exp25c): F1=0.784")
            print("    Full W_gate vs oracle gate_full:          F1=1.000 (by def)")
        else:
            print("    SVD W_gate rank=1024 vs oracle (exp25d):  F1=0.873")
            print("    Full W_gate vs oracle gate_full:          F1=1.000 (by def)")
        print(f"{'='*72}")

    # ----------------------------------------------------------------
    # Storage / compute savings
    # ----------------------------------------------------------------
    print("\nStorage and compute savings (W_gate shape 8192×2560 = 20971520 weights):")
    total_w = I_DIM * H_DIM
    for kr in args.keep_rates:
        nnz    = int(kr * total_w)
        saving = (1.0 - kr) * 100
        print(f"  keep={kr:.1f}: {nnz:>8d} non-zeros  ({saving:.0f}% zeroed)"
              f"  sparse GEMM saving ≈ {saving:.0f}% FLOPs (unstructured upper bound)")


if __name__ == "__main__":
    main()
