# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 2 – hybrid sign+full-precision MLP predictor vs full precision.

Scheme (from pred1.md Core Idea)
---------------------------------
1. Compute gate and up projections cheaply using sign(W) as the low-precision
   proxy: gate_approx = sign(W_gate) @ x,  up_approx = sign(W_up) @ x.

2. Threshold |gate_approx| > gate_thresh to identify hot channels per token.
   Hot channels are recomputed with the true weights:
       gate_hybrid[hot]  = W_gate[hot] @ x
       gate_hybrid[cold] = gate_approx[cold]

3. Same hot mask applied to up projection:
       up_hybrid[hot]  = W_up[hot] @ x
       up_hybrid[cold] = up_approx[cold]

4. SwiGLU_hybrid = SiLU(gate_hybrid) * up_hybrid

5. Down projection: full sign-weight pass + sparse full-precision correction
   for hot input channels (channels i where gate was hot):
       down_approx = sign(W_down) @ swiglu_hybrid
       For each hot channel i: add (W_down[:, i] - sign(W_down[:, i])) * swiglu_hybrid[:, i]
   This is equivalent to:
       down_hybrid = sign(W_down) @ swiglu_hybrid
                   + (W_down - sign(W_down))[:, hot] @ swiglu_hybrid[:, hot]

The script sweeps gate_thresh over a range and reports cosine similarity at
each point, plus the fraction of hot channels (compute cost proxy).

Usage::

    python tools/profiler/exp2_hybrid_predictor.py \\
        --model ibm-granite/granite-4.2-3b \\
        --act-file ffn_activations128_gate.npz
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def cosine_sim_rows(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-row cosine similarity between two (T, D) arrays."""
    a_t = torch.from_numpy(a).float()
    b_t = torch.from_numpy(b).float()
    return float(F.cosine_similarity(a_t, b_t, dim=-1).mean())


def silu(x: np.ndarray) -> np.ndarray:
    return F.silu(torch.from_numpy(x)).numpy()


# ---------------------------------------------------------------------------
# Weight extraction
# ---------------------------------------------------------------------------


def get_layer_weights(model, layer_idx: int):
    """Return (W_gate, W_up, W_down) as float32 numpy arrays."""
    layer = model.model.layers[layer_idx]
    mlp = layer.mlp
    gate_up_w = mlp.gate_up_proj.weight.detach().float().numpy()  # (2I, H)
    intermediate = gate_up_w.shape[0] // 2
    W_gate = gate_up_w[:intermediate]   # (I, H)
    W_up   = gate_up_w[intermediate:]   # (I, H)
    W_down = mlp.down_proj.weight.detach().float().numpy()         # (H, I)
    return W_gate, W_up, W_down


# ---------------------------------------------------------------------------
# Hybrid evaluation at a single threshold
# ---------------------------------------------------------------------------


def evaluate_layer_hybrid(
    W_gate: np.ndarray,      # (I, H)
    W_up:   np.ndarray,      # (I, H)
    W_down: np.ndarray,      # (H, I)
    gate_up_input: np.ndarray,  # (T, H)
    gate_raw: np.ndarray,        # (T, I)  full-precision reference
    gate_thresh: float,
) -> dict[str, float]:
    """Run one threshold value and return cosine similarities + hot fraction.

    Hot channels are those where |gate_approx| > gate_thresh.  The mask is
    per-token so the set of recomputed channels varies across tokens.
    """
    S_gate = np.sign(W_gate)   # (I, H)
    S_up   = np.sign(W_up)     # (I, H)
    S_down = np.sign(W_down)   # (H, I)

    # --- Low-precision pass ---
    gate_approx = gate_up_input @ S_gate.T   # (T, I)
    up_approx   = gate_up_input @ S_up.T     # (T, I)

    # --- Hot mask: per-token, per-channel ---
    hot = np.abs(gate_approx) > gate_thresh   # (T, I) bool
    hot_fraction = float(hot.mean())

    # --- Hybrid gate ---
    # Correction: (W_gate - S_gate)[hot_channels] @ x[token]
    # Computed as: gate_full = gate_up_input @ W_gate.T, then blend
    gate_full  = gate_up_input @ W_gate.T    # (T, I)
    gate_hybrid = gate_approx.copy()
    gate_hybrid[hot] = gate_full[hot]

    # --- Hybrid up ---
    up_full  = gate_up_input @ W_up.T        # (T, I)
    up_hybrid = up_approx.copy()
    up_hybrid[hot] = up_full[hot]

    # --- SwiGLU hybrid ---
    swiglu_full   = silu(gate_raw)   * (gate_up_input @ W_up.T)
    swiglu_hybrid = silu(gate_hybrid) * up_hybrid

    cos_gate   = cosine_sim_rows(gate_raw,    gate_hybrid)
    cos_swiglu = cosine_sim_rows(swiglu_full, swiglu_hybrid)

    # --- Hybrid down projection ---
    # Full low-prec pass + sparse correction for hot input channels.
    # down_hybrid = sign(W_down) @ swiglu_hybrid
    #             + (W_down - sign(W_down))[:, hot_cols] @ swiglu_hybrid[:, hot_cols]
    # To avoid iterating over tokens we use the residual weight matrix:
    #   residual_W = W_down - S_down  (H, I)
    # Then: correction[t] = residual_W[:, hot[t]] @ swiglu_hybrid[t, hot[t]]
    # We broadcast this as a masked einsum equivalent:
    residual_W = W_down - S_down                         # (H, I)
    down_sign   = swiglu_hybrid @ S_down.T               # (T, H)
    # Correction: zero out cold channels' residual contribution per token
    masked_swiglu = swiglu_hybrid * hot                  # (T, I) – cold zeroed
    correction    = masked_swiglu @ residual_W.T         # (T, H)
    down_hybrid   = down_sign + correction               # (T, H)

    down_full = swiglu_full @ W_down.T                   # (T, H)
    cos_down  = cosine_sim_rows(down_full, down_hybrid)

    return {
        "gate":         cos_gate,
        "swiglu":       cos_swiglu,
        "down":         cos_down,
        "hot_fraction": hot_fraction,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Experiment 2: hybrid sign+full-precision predictor.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", default="ibm-granite/granite-4.2-3b",
                   help="Model name or path.")
    p.add_argument("--act-file", default="ffn_activations128_gate.npz",
                   metavar="FILE",
                   help="NPZ with gate_up_input and gate_raw.")
    p.add_argument("--layers", nargs="*", type=int, default=None, metavar="N",
                   help="Layer indices to evaluate (default: all).")
    p.add_argument("--max-tokens", type=int, default=None, metavar="N",
                   help="Cap token count per layer (for faster runs).")
    p.add_argument(
        "--thresholds", nargs="+", type=float,
        default=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
        metavar="T",
        help="gate_thresh values to sweep (default: 0 0.5 1 2 4 8 16 32).",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    print(f"Loading activations from {args.act_file!r} …", file=sys.stderr)
    act_data = np.load(args.act_file)

    available_layers = sorted(
        int(k.split("/")[0].replace("layer", ""))
        for k in act_data.files if k.endswith("/gate_raw")
    )
    layers = available_layers
    if args.layers is not None:
        layers = [l for l in layers if l in args.layers]
    print(f"Evaluating {len(layers)} layer(s), "
          f"{len(args.thresholds)} threshold(s).", file=sys.stderr)

    print(f"Loading model {args.model!r} …", file=sys.stderr)
    from vllm import LLM
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        enforce_eager=True,
        kv_cache_memory_bytes=int(0.5 * 1024**3),
        max_model_len=512,
    )
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model

    # thresh -> {metric: [per-layer values]}
    thresh_results: dict[float, dict[str, list]] = {
        t: {"gate": [], "swiglu": [], "down": [], "hot_fraction": []}
        for t in args.thresholds
    }

    for layer_idx in layers:
        prefix = f"layer{layer_idx}"
        gate_up_input = act_data[f"{prefix}/gate_up_input"]
        gate_raw      = act_data[f"{prefix}/gate_raw"]
        if args.max_tokens is not None:
            gate_up_input = gate_up_input[: args.max_tokens]
            gate_raw      = gate_raw[: args.max_tokens]

        W_gate, W_up, W_down = get_layer_weights(model, layer_idx)

        # Pre-compute all expensive matmuls once; reuse across all thresholds.
        S_gate     = np.sign(W_gate)
        S_up       = np.sign(W_up)
        S_down     = np.sign(W_down)
        residual_W = W_down - S_down                        # (H, I)

        gate_approx = gate_up_input @ S_gate.T              # (T, I)
        up_approx   = gate_up_input @ S_up.T                # (T, I)
        gate_full   = gate_up_input @ W_gate.T              # (T, I)
        up_full_p   = gate_up_input @ W_up.T                # (T, I)

        swiglu_full = silu(gate_raw) * up_full_p            # (T, I)
        down_full   = swiglu_full @ W_down.T                # (T, H)

        for thresh in args.thresholds:
            hot          = np.abs(gate_approx) > thresh     # (T, I)
            hot_fraction = float(hot.mean())

            gate_hybrid       = gate_approx.copy()
            gate_hybrid[hot]  = gate_full[hot]
            up_hybrid         = up_approx.copy()
            up_hybrid[hot]    = up_full_p[hot]

            swiglu_hybrid  = silu(gate_hybrid) * up_hybrid
            down_sign      = swiglu_hybrid @ S_down.T
            correction     = (swiglu_hybrid * hot) @ residual_W.T
            down_hybrid    = down_sign + correction

            r = {
                "gate":         cosine_sim_rows(gate_raw,    gate_hybrid),
                "swiglu":       cosine_sim_rows(swiglu_full, swiglu_hybrid),
                "down":         cosine_sim_rows(down_full,   down_hybrid),
                "hot_fraction": hot_fraction,
            }
            for k in r:
                thresh_results[thresh][k].append(r[k])

        last = args.thresholds[-1]
        lr   = thresh_results[last]
        print(
            f"  layer {layer_idx:3d} done  "
            f"(thresh={last}: gate={lr['gate'][-1]:.4f}  "
            f"swiglu={lr['swiglu'][-1]:.4f}  "
            f"down={lr['down'][-1]:.4f}  "
            f"hot={lr['hot_fraction'][-1]*100:.1f}%)",
            file=sys.stderr,
        )

    # --- Summary table ---
    print("\n--- Results (mean cosine similarity across all evaluated layers) ---")
    print(f"{'thresh':>8}  {'hot%':>6}  {'gate':>7}  {'swiglu':>7}  {'down':>7}")
    print("-" * 46)
    for thresh in args.thresholds:
        r = thresh_results[thresh]
        hot_pct   = np.mean(r["hot_fraction"]) * 100
        gate_mean = np.mean(r["gate"])
        swi_mean  = np.mean(r["swiglu"])
        down_mean = np.mean(r["down"])
        print(
            f"{thresh:8.1f}  {hot_pct:5.1f}%  "
            f"{gate_mean:7.4f}  {swi_mean:7.4f}  {down_mean:7.4f}"
        )

    # --- Per-layer detail for each threshold ---
    print("\n--- Per-layer detail ---")
    header = f"{'layer':>5}" + "".join(
        f"  gate@{t:<5.1f}  swi@{t:<5.1f}  dn@{t:<5.1f}  hot@{t:<5.1f}"
        for t in args.thresholds
    )
    print(header)
    for i, layer_idx in enumerate(layers):
        row = f"{layer_idx:5d}"
        for thresh in args.thresholds:
            r = thresh_results[thresh]
            row += (
                f"  {r['gate'][i]:9.4f}"
                f"  {r['swiglu'][i]:8.4f}"
                f"  {r['down'][i]:7.4f}"
                f"  {r['hot_fraction'][i]*100:6.1f}%"
            )
        print(row)


if __name__ == "__main__":
    main()
