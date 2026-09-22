# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 5 – prior-token hotlist for out-of-critical-path refinement.

Scheme
------
At decode time, token t is on the critical path.  The hypothesis is that
hot channels are stable across adjacent tokens, so the hotlist H_{t-1}
(derived from the full gate computation of token t-1, which is already done)
can be reused to route token t without an extra predictor pass.

Concurrently, a refinement pass can run the gate projection for a subset of
channels in full precision to update the hotlist for token t+1.  This
refinement is *out of the critical path* — it only needs to finish before
layer l of token t+1 is computed.

This script evaluates:

  1. **Hotlist stability:** IoU (Jaccard) between the true hot set at token t
     and the true hot set at token t-1, across all adjacent token pairs.
     High IoU → prior hotlist is accurate; low IoU → scheme will fail.

  2. **Hybrid quality with prior hotlist:** SwiGLU and down cosine similarity
     when the hot/cold routing for token t is derived from |gate_full[t-1]|
     (oracle prior) at various hot-fraction thresholds.

  3. **Refinement budget:** given N channels recomputed in full precision
     during the inter-token window (budget expressed as fraction of I), how
     many mis-routed channels from the prior hotlist are corrected?  What is
     the resulting SwiGLU/down cosine similarity improvement?

All three metrics are swept over hot-channel fractions [10%, 20%, 30%, 50%]
and refinement budgets [0%, 5%, 10%, 20%, 30%] of I.

Adjacent-token pairs are formed from consecutive rows in the recorded
activation tensors.  The ~1% of pairs that cross sequence boundaries
(128 prompts × ~82 tokens each) add negligible noise.

Usage::

    python tools/profiler/exp5_prior_hotlist.py \\
        --model ibm-granite/granite-4.2-3b \\
        --act-file ffn_activations128_gate.npz
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEV = _device()


def t(a: np.ndarray) -> torch.Tensor:
    """numpy → torch on accelerator, float32."""
    return torch.from_numpy(a).to(DEV, dtype=torch.float32)


def cosine_sim_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a, b, dim=-1).mean())


def silu_t(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


def get_weights(model, layer_idx: int):
    """Return (W_gate, W_up, W_down) as float32 numpy arrays."""
    mlp = model.model.layers[layer_idx].mlp
    W = mlp.gate_up_proj.weight.detach().float().numpy()
    I = W.shape[0] // 2
    return W[:I], W[I:], mlp.down_proj.weight.detach().float().numpy()


def top_k_mask(scores: torch.Tensor, k: int) -> torch.BoolTensor:
    """Return bool (T, I) with the top-k positions per row set to True."""
    idx = torch.topk(scores, k, dim=1, largest=True, sorted=False).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask


def hybrid_pass(
    gate_approx: torch.Tensor,   # (T, I)
    gate_full:   torch.Tensor,   # (T, I)
    up_approx:   torch.Tensor,   # (T, I)
    up_full:     torch.Tensor,   # (T, I)
    S_down:      torch.Tensor,   # (H, I)
    residual_W:  torch.Tensor,   # (H, I)
    gate_raw:    torch.Tensor,   # (T, I)
    hot:         torch.BoolTensor,  # (T, I)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (swiglu_full, swiglu_hybrid, down_hybrid)."""
    gate_h = torch.where(hot, gate_full,  gate_approx)
    up_h   = torch.where(hot, up_full,    up_approx)
    swiglu_full   = silu_t(gate_raw) * up_full
    swiglu_hybrid = silu_t(gate_h)   * up_h
    # down_hybrid = sign(W_down) @ swiglu_hybrid
    #             + (W_down - sign(W_down))[:, hot_cols] @ swiglu_hybrid[:, hot_cols]
    # Vectorised: mask cold channels to zero, then one matmul for the correction.
    down_hybrid = (swiglu_hybrid @ S_down.T
                   + (swiglu_hybrid * hot.float()) @ residual_W.T)
    return swiglu_full, swiglu_hybrid, down_hybrid


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Experiment 5: prior-token hotlist.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", default="ibm-granite/granite-4.2-3b")
    p.add_argument("--act-file", default="ffn_activations128_gate.npz")
    p.add_argument("--layers", nargs="*", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=2000)
    p.add_argument(
        "--hot-fractions", nargs="+", type=float,
        default=[0.50, 0.30, 0.20, 0.10],
        metavar="F",
    )
    p.add_argument(
        "--refine-budgets", nargs="+", type=float,
        default=[0.0, 0.05, 0.10, 0.20, 0.30],
        metavar="B",
        help="Fraction of I channels recomputed to update the hotlist.",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    print(f"Using device: {DEV}", file=sys.stderr)

    act_data = np.load(args.act_file)
    layers = sorted(
        int(k.split("/")[0].replace("layer", ""))
        for k in act_data.files if k.endswith("/gate_raw")
    )
    if args.layers is not None:
        layers = [l for l in layers if l in args.layers]
    print(f"Evaluating {len(layers)} layers.", file=sys.stderr)

    from vllm import LLM
    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              kv_cache_memory_bytes=int(0.5 * 1024**3), max_model_len=512)
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model

    iou_by_frac: dict[float, list[float]] = {f: [] for f in args.hot_fractions}
    results: dict[float, dict[float, dict[str, list]]] = {
        f: {b: {"swiglu": [], "down": []} for b in args.refine_budgets}
        for f in args.hot_fractions
    }

    for layer_idx in layers:
        pfx = f"layer{layer_idx}"
        x_np       = act_data[f"{pfx}/gate_up_input"][: args.max_tokens].astype(np.float32)
        graw_np    = act_data[f"{pfx}/gate_raw"][: args.max_tokens].astype(np.float32)

        W_gate_np, W_up_np, W_down_np = get_weights(model, layer_idx)

        # Move everything to accelerator once
        x          = t(x_np)
        gate_raw   = t(graw_np)
        W_gate     = t(W_gate_np)
        W_up       = t(W_up_np)
        W_down     = t(W_down_np)
        S_gate     = W_gate.sign()
        S_up       = W_up.sign()
        S_down     = W_down.sign()
        residual_W = W_down - S_down

        T, H = x.shape
        I    = W_gate.shape[0]

        # All four GEMMs once per layer
        gate_full   = x @ W_gate.T      # (T, I)
        up_full     = x @ W_up.T        # (T, I)
        gate_approx = x @ S_gate.T      # (T, I)
        up_approx   = x @ S_up.T        # (T, I)

        # Adjacent-token index pairs
        # prior=0..T-2, current=1..T-1
        T_pairs = T - 1

        gate_full_prior   = gate_full  [:T_pairs]   # (T-1, I)
        gate_full_cur     = gate_full  [1:]
        up_full_cur       = up_full    [1:]
        gate_approx_cur   = gate_approx[1:]
        up_approx_cur     = up_approx  [1:]
        gate_raw_cur      = gate_raw   [1:]

        abs_gate_full_prior = gate_full_prior.abs()   # (T-1, I)
        abs_gate_full_cur   = gate_full_cur.abs()     # (T-1, I)

        for frac in args.hot_fractions:
            k_hot = max(1, int(frac * I))

            # True hot masks (top-k by |gate_full|)
            true_hot_prior = top_k_mask(abs_gate_full_prior, k_hot)   # (T-1, I)
            true_hot_cur   = top_k_mask(abs_gate_full_cur,   k_hot)   # (T-1, I)

            # IoU between adjacent tokens
            inter = (true_hot_prior & true_hot_cur).float().sum(1)    # (T-1,)
            union = (true_hot_prior | true_hot_cur).float().sum(1)
            iou_by_frac[frac].append(float((inter / union.clamp(min=1)).mean()))

            # Threshold for "hot" at current token (k-th largest |gate_full_cur|)
            kth_cur = torch.kthvalue(abs_gate_full_cur, I - k_hot + 1, dim=1).values
            # shape (T-1,), broadcast to (T-1, I)

            for budget in args.refine_budgets:
                if budget == 0.0:
                    hot_to_use = true_hot_prior
                else:
                    # Refinement: identify boundary channels in the prior hotlist
                    # (those ranked nearest to the k_hot cutoff) and re-evaluate
                    # them at the current token with full-precision gate values.
                    k_refine = max(1, int(budget * I))

                    # Boundary channels: those with rank closest to k_hot in prior
                    # = channels ranked [I-k_hot-k_refine//2 .. I-k_hot+k_refine//2]
                    # We select them as the k_refine channels nearest to the
                    # k_hot-th largest value in |gate_full_prior|.
                    kth_prior = torch.kthvalue(
                        abs_gate_full_prior, I - k_hot + 1, dim=1).values  # (T-1,)
                    dist_to_boundary = (abs_gate_full_prior
                                        - kth_prior.unsqueeze(1)).abs()    # (T-1, I)
                    # top-k_refine closest to boundary
                    boundary_mask = top_k_mask(-dist_to_boundary, k_refine) # (T-1, I)

                    # For boundary channels: use true gate_full_cur to decide hot/cold
                    true_decision = (
                        abs_gate_full_cur >= kth_cur.unsqueeze(1)          # (T-1, I)
                    )
                    # Start from prior hotlist, overwrite boundary channels
                    hot_to_use = torch.where(boundary_mask, true_decision, true_hot_prior)

                swiglu_full, swiglu_hybrid, down_hybrid = hybrid_pass(
                    gate_approx_cur, gate_full_cur,
                    up_approx_cur,   up_full_cur,
                    S_down, residual_W,
                    gate_raw_cur,
                    hot_to_use,
                )
                down_full = swiglu_full @ W_down.T

                results[frac][budget]["swiglu"].append(
                    cosine_sim_mean(swiglu_full, swiglu_hybrid))
                results[frac][budget]["down"].append(
                    cosine_sim_mean(down_full, down_hybrid))

            # free GPU memory for this frac iteration
            del true_hot_prior, true_hot_cur

        best_frac = args.hot_fractions[0]
        print(
            f"  layer {layer_idx:3d}: "
            f"IoU@{best_frac:.0%}={iou_by_frac[best_frac][-1]:.3f}  "
            f"swiglu(prior,no-refine)="
            f"{results[best_frac][0.0]['swiglu'][-1]:.4f}  "
            f"down={results[best_frac][0.0]['down'][-1]:.4f}",
            file=sys.stderr,
        )

        # Free layer tensors from GPU
        del (x, gate_raw, W_gate, W_up, W_down, S_gate, S_up, S_down,
             residual_W, gate_full, up_full, gate_approx, up_approx)
        if DEV.type == "mps":
            torch.mps.empty_cache()
        elif DEV.type == "cuda":
            torch.cuda.empty_cache()

    # --- Summary ---
    print("\n--- Experiment 5 Results ---")

    print("\nHotlist temporal stability (Jaccard IoU between H_{t-1} and H_t):")
    print(f"  {'hot%':>5}  {'mean IoU':>10}  {'min':>8}  {'max':>8}")
    for frac in args.hot_fractions:
        vals = iou_by_frac[frac]
        print(f"  {frac*100:5.0f}%  {np.mean(vals):10.4f}  "
              f"{np.min(vals):8.4f}  {np.max(vals):8.4f}")

    print("\nMean SwiGLU cosine similarity (prior-token hotlist + refinement):")
    hdr = f"  {'hot%':>5}" + "".join(
        f"  refine={b:.0%}" for b in args.refine_budgets)
    print(hdr)
    for frac in args.hot_fractions:
        row = f"  {frac*100:5.0f}%"
        for budget in args.refine_budgets:
            row += f"  {np.mean(results[frac][budget]['swiglu']):9.4f}"
        print(row)

    print("\nMean down-projection cosine similarity:")
    print(hdr)
    for frac in args.hot_fractions:
        row = f"  {frac*100:5.0f}%"
        for budget in args.refine_budgets:
            row += f"  {np.mean(results[frac][budget]['down']):9.4f}"
        print(row)

    ref_frac = args.hot_fractions[0]
    print(f"\nPer-layer detail (hot={ref_frac:.0%}):")
    hdr2 = f"  {'layer':>5}  {'IoU':>6}" + "".join(
        f"  swi@{b:.0%}  dn@{b:.0%}" for b in args.refine_budgets)
    print(hdr2)
    for i, li in enumerate(layers):
        row = f"  {li:5d}  {iou_by_frac[ref_frac][i]:6.4f}"
        for budget in args.refine_budgets:
            row += (f"  {results[ref_frac][budget]['swiglu'][i]:8.4f}"
                    f"  {results[ref_frac][budget]['down'][i]:7.4f}")
        print(row)


if __name__ == "__main__":
    main()
