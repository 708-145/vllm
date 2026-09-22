# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 6 – magnitude-confidence refinement of the prior-token hotlist.

Hypothesis (from experiment 5 post-analysis)
--------------------------------------------
Experiment 5 showed that refining the prior hotlist by recomputing *boundary*
channels (those ranked nearest the hot/cold cutoff in the prior token) slightly
hurts quality, because those channels are inherently unstable and flipping them
increases churn.

The better strategy: use the **magnitude of the prior gate value as a confidence
score**.  A channel with large |gate_full[t-1]| is almost certainly still hot at
token t — protect it.  A channel with small |gate_full[t-1]| is near-zero and
ambiguous — this is where the refinement budget should be spent.

Refinement strategy for experiment 6
--------------------------------------
Given a budget of k_refine channels to recompute at token t (out of the critical
path, with the true gate values):

  1. Among the k_hot prior-hot channels, identify the k_refine/2 with the
     *smallest* |gate_full[t-1]| (least confident hot).
  2. Among the (I - k_hot) prior-cold channels, identify the k_refine/2 with
     the *largest* |gate_full[t-1]| (least confident cold — most likely to have
     become hot).
  3. For those k_refine channels, re-evaluate using the true gate_full[t] and
     correct the hotlist.  All other channels keep the prior decision.

This protects high-confidence hot channels from being flipped, and focuses
corrections where the prior is least trustworthy.

Comparison
----------
For each (hot_frac, refine_budget) we report:
  - SwiGLU cosine similarity
  - Down projection cosine similarity

and compare directly against experiment 5's boundary-rank refinement at the
same budget.

Usage::

    python tools/profiler/exp6_magnitude_confidence_refinement.py \\
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
    return torch.from_numpy(a).to(DEV, dtype=torch.float32)


def cosine_sim_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a, b, dim=-1).mean())


def silu_t(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


def get_weights(model, layer_idx: int):
    mlp = model.model.layers[layer_idx].mlp
    W = mlp.gate_up_proj.weight.detach().float().numpy()
    I = W.shape[0] // 2
    return W[:I], W[I:], mlp.down_proj.weight.detach().float().numpy()


def top_k_mask(scores: torch.Tensor, k: int) -> torch.BoolTensor:
    """Bool (T, I) with the top-k positions per row True."""
    idx = torch.topk(scores, k, dim=1, largest=True, sorted=False).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask


def bottom_k_mask(scores: torch.Tensor, k: int) -> torch.BoolTensor:
    """Bool (T, I) with the bottom-k positions per row True."""
    idx = torch.topk(scores, k, dim=1, largest=False, sorted=False).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask


def hybrid_pass(
    gate_approx: torch.Tensor,
    gate_full:   torch.Tensor,
    up_approx:   torch.Tensor,
    up_full:     torch.Tensor,
    S_down:      torch.Tensor,
    residual_W:  torch.Tensor,
    gate_raw:    torch.Tensor,
    hot:         torch.BoolTensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gate_h        = torch.where(hot, gate_full,  gate_approx)
    up_h          = torch.where(hot, up_full,    up_approx)
    swiglu_full   = silu_t(gate_raw) * up_full
    swiglu_hybrid = silu_t(gate_h)   * up_h
    down_hybrid   = (swiglu_hybrid @ S_down.T
                     + (swiglu_hybrid * hot.float()) @ residual_W.T)
    return swiglu_full, swiglu_hybrid, down_hybrid


def refine_boundary_rank(
    prior_hot:          torch.BoolTensor,   # (T-1, I)
    abs_prior:          torch.Tensor,       # (T-1, I)
    abs_cur:            torch.Tensor,       # (T-1, I)
    kth_cur:            torch.Tensor,       # (T-1,)
    k_refine:           int,
) -> torch.BoolTensor:
    """Experiment 5 strategy: refine k_refine channels nearest rank boundary."""
    kth_prior        = torch.kthvalue(abs_prior,
                                      abs_prior.shape[1] - prior_hot.shape[1] // 2,
                                      dim=1).values
    # Recompute: use the proper hot count
    # kth_prior = the k_hot-th largest value in abs_prior (i.e. the boundary value)
    # We infer k_hot from prior_hot
    k_hot = int(prior_hot.float().sum(1).float().mean().round().item())
    kth_prior        = torch.kthvalue(abs_prior,
                                      abs_prior.shape[1] - k_hot + 1,
                                      dim=1).values
    dist_to_boundary = (abs_prior - kth_prior.unsqueeze(1)).abs()
    boundary_mask    = top_k_mask(-dist_to_boundary, k_refine)
    true_decision    = abs_cur >= kth_cur.unsqueeze(1)
    return torch.where(boundary_mask, true_decision, prior_hot)


def refine_magnitude_confidence(
    prior_hot:  torch.BoolTensor,   # (T-1, I)
    abs_prior:  torch.Tensor,       # (T-1, I)
    abs_cur:    torch.Tensor,       # (T-1, I)
    kth_cur:    torch.Tensor,       # (T-1,)
    k_refine:   int,
) -> torch.BoolTensor:
    """Experiment 6 strategy: refine channels where prior magnitude is least confident.

    Split the budget evenly:
      - k_refine//2 least-confident prior-hot channels (smallest |prior| among hot)
      - k_refine//2 least-confident prior-cold channels (largest |prior| among cold)

    These are the channels where the prior gate magnitude is closest to zero
    (hot) or largest among cold (most likely to have crossed into hot territory).
    All high-magnitude hot channels are protected.
    """
    k_half = max(1, k_refine // 2)

    # Least-confident hot: prior-hot channels with smallest |gate_full[t-1]|
    # Mask non-hot positions with +inf so they don't get selected as "smallest hot"
    hot_scores  = abs_prior.masked_fill(~prior_hot, float("inf"))
    weak_hot    = bottom_k_mask(hot_scores,  k_half)   # smallest |prior| among hot

    # Least-confident cold: prior-cold channels with largest |gate_full[t-1]|
    # Mask hot positions with -inf so they don't get selected as "largest cold"
    cold_scores = abs_prior.masked_fill(prior_hot, float("-inf"))
    strong_cold = top_k_mask(cold_scores, k_half)      # largest |prior| among cold

    refine_mask   = weak_hot | strong_cold              # (T-1, I)
    true_decision = abs_cur >= kth_cur.unsqueeze(1)     # true hot/cold at current token
    return torch.where(refine_mask, true_decision, prior_hot)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Experiment 6: magnitude-confidence hotlist refinement.",
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
        help="Fraction of I channels recomputed per token.",
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

    # Results indexed by strategy, hot_frac, budget
    strategies = ("no_refine", "boundary_rank", "magnitude_conf")
    results: dict[str, dict[float, dict[float, dict[str, list]]]] = {
        s: {f: {b: {"swiglu": [], "down": []}
                for b in args.refine_budgets}
            for f in args.hot_fractions}
        for s in strategies
    }

    for layer_idx in layers:
        pfx = f"layer{layer_idx}"
        x_np     = act_data[f"{pfx}/gate_up_input"][: args.max_tokens].astype(np.float32)
        graw_np  = act_data[f"{pfx}/gate_raw"][: args.max_tokens].astype(np.float32)

        W_gate_np, W_up_np, W_down_np = get_weights(model, layer_idx)

        x          = t(x_np)
        gate_raw   = t(graw_np)
        W_gate     = t(W_gate_np)
        W_up       = t(W_up_np)
        W_down     = t(W_down_np)
        S_gate     = W_gate.sign()
        S_up       = W_up.sign()
        S_down     = W_down.sign()
        residual_W = W_down - S_down

        T, H  = x.shape
        I     = W_gate.shape[0]
        T_pairs = T - 1

        gate_full   = x @ W_gate.T
        up_full     = x @ W_up.T
        gate_approx = x @ S_gate.T
        up_approx   = x @ S_up.T

        gate_full_prior   = gate_full[:T_pairs]
        gate_full_cur     = gate_full[1:]
        up_full_cur       = up_full[1:]
        gate_approx_cur   = gate_approx[1:]
        up_approx_cur     = up_approx[1:]
        gate_raw_cur      = gate_raw[1:]

        abs_prior = gate_full_prior.abs()   # (T-1, I)
        abs_cur   = gate_full_cur.abs()     # (T-1, I)

        for frac in args.hot_fractions:
            k_hot   = max(1, int(frac * I))
            kth_cur = torch.kthvalue(abs_cur, I - k_hot + 1, dim=1).values  # (T-1,)

            prior_hot = top_k_mask(abs_prior, k_hot)   # (T-1, I)

            for budget in args.refine_budgets:
                k_refine = max(1, int(budget * I))

                # --- no_refine (budget=0 baseline, same for all budgets) ---
                hot_no_refine = prior_hot

                # --- boundary_rank (exp5 strategy) ---
                if budget == 0.0:
                    hot_boundary = prior_hot
                    hot_magconf  = prior_hot
                else:
                    hot_boundary = refine_boundary_rank(
                        prior_hot, abs_prior, abs_cur, kth_cur, k_refine)
                    hot_magconf  = refine_magnitude_confidence(
                        prior_hot, abs_prior, abs_cur, kth_cur, k_refine)

                for strat, hot in (
                    ("no_refine",      hot_no_refine),
                    ("boundary_rank",  hot_boundary),
                    ("magnitude_conf", hot_magconf),
                ):
                    swiglu_full, swiglu_hybrid, down_hybrid = hybrid_pass(
                        gate_approx_cur, gate_full_cur,
                        up_approx_cur,   up_full_cur,
                        S_down, residual_W, gate_raw_cur, hot,
                    )
                    down_full = swiglu_full @ W_down.T
                    results[strat][frac][budget]["swiglu"].append(
                        cosine_sim_mean(swiglu_full, swiglu_hybrid))
                    results[strat][frac][budget]["down"].append(
                        cosine_sim_mean(down_full, down_hybrid))

            del prior_hot

        print(
            f"  layer {layer_idx:3d}: "
            f"no-refine@10%={results['no_refine'][0.10][0.0]['swiglu'][-1]:.4f}  "
            f"mag-conf@10%,20%budget="
            f"{results['magnitude_conf'][0.10][0.20]['swiglu'][-1]:.4f}",
            file=sys.stderr,
        )

        del (x, gate_raw, W_gate, W_up, W_down, S_gate, S_up, S_down, residual_W,
             gate_full, up_full, gate_approx, up_approx)
        if DEV.type == "mps":
            torch.mps.empty_cache()
        elif DEV.type == "cuda":
            torch.cuda.empty_cache()

    # --- Summary ---
    print("\n--- Experiment 6 Results ---")
    print("Mean SwiGLU cosine similarity across all layers")
    print("(no_refine = exp5 baseline; boundary_rank = exp5 strategy; "
          "magnitude_conf = exp6 strategy)\n")

    for frac in args.hot_fractions:
        print(f"  hot={frac:.0%}")
        print(f"  {'budget':>8}  {'no_refine':>10}  {'boundary_rank':>14}  "
              f"{'magnitude_conf':>15}  {'Δ vs boundary':>14}  {'Δ vs no-refine':>15}")
        for budget in args.refine_budgets:
            nr  = np.mean(results["no_refine"]     [frac][budget]["swiglu"])
            br  = np.mean(results["boundary_rank"] [frac][budget]["swiglu"])
            mc  = np.mean(results["magnitude_conf"][frac][budget]["swiglu"])
            nr0 = np.mean(results["no_refine"]     [frac][0.0   ]["swiglu"])
            print(f"  {budget:>8.0%}  {nr:>10.4f}  {br:>14.4f}  "
                  f"{mc:>15.4f}  {mc-br:>+14.4f}  {mc-nr0:>+15.4f}")
        print()

    print("Mean down-projection cosine similarity\n")
    for frac in args.hot_fractions:
        print(f"  hot={frac:.0%}")
        print(f"  {'budget':>8}  {'no_refine':>10}  {'boundary_rank':>14}  "
              f"{'magnitude_conf':>15}  {'Δ vs boundary':>14}")
        for budget in args.refine_budgets:
            nr = np.mean(results["no_refine"]     [frac][budget]["down"])
            br = np.mean(results["boundary_rank"] [frac][budget]["down"])
            mc = np.mean(results["magnitude_conf"][frac][budget]["down"])
            print(f"  {budget:>8.0%}  {nr:>10.4f}  {br:>14.4f}  "
                  f"{mc:>15.4f}  {mc-br:>+14.4f}")
        print()

    # Per-layer detail for the most interesting configuration
    ref_frac, ref_budget = 0.10, 0.20
    print(f"Per-layer SwiGLU cosine similarity (hot={ref_frac:.0%}, "
          f"refine budget={ref_budget:.0%}):")
    print(f"  {'layer':>5}  {'no_refine':>10}  {'boundary_rank':>14}  "
          f"{'magnitude_conf':>15}  {'Δ mag-conf vs boundary':>22}")
    for i, li in enumerate(layers):
        nr = results["no_refine"]     [ref_frac][ref_budget]["swiglu"][i]
        br = results["boundary_rank"] [ref_frac][ref_budget]["swiglu"][i]
        mc = results["magnitude_conf"][ref_frac][ref_budget]["swiglu"][i]
        print(f"  {li:5d}  {nr:10.4f}  {br:14.4f}  {mc:15.4f}  {mc-br:+22.4f}")


if __name__ == "__main__":
    main()
