# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 1 – sign-based low-precision MLP predictor vs full precision.

Scheme
------
For gate and up projections the "low-precision" approximation replaces each
weight value by its sign (i.e. +1 / -1), making the matmul a simple sum of
signed inputs.  Down projection follows the same idea on its weight matrix.
We then compare:

  * gate output (before SiLU):   W_gate @ x  vs  sign(W_gate) @ x
  * SwiGLU output:               SiLU(gate) * up  vs  SiLU(gate_sign) * up_sign
  * down projection output:      W_down @ swiglu  vs  sign(W_down) @ swiglu_sign

Metric: per-token cosine similarity, averaged across tokens and layers.

Usage::

    python tools/profiler/exp1_sign_predictor.py \\
        --model ibm-granite/granite-4.2-3b \\
        --gate-act-file ffn_activations128_gate.npz \\
        --down-act-file ffn_activations128.npz

The script loads model weights once, then streams activations layer by layer.
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
    """Mean per-row cosine similarity between two 2-D arrays."""
    a_t = torch.from_numpy(a).float()
    b_t = torch.from_numpy(b).float()
    sim = F.cosine_similarity(a_t, b_t, dim=-1)  # (T,)
    return float(sim.mean())


def silu(x: np.ndarray) -> np.ndarray:
    # Use torch for numerically stable sigmoid (avoids float32 overflow in exp)
    t = torch.from_numpy(x)
    return F.silu(t).numpy()


def swiglu(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    return silu(gate) * up


# ---------------------------------------------------------------------------
# Weight extraction
# ---------------------------------------------------------------------------


def get_layer_weights(model, layer_idx: int):
    """Return (W_gate, W_up, W_down) as float32 numpy arrays for one layer.

    W_gate_up is stored as a single merged matrix of shape
    [2 * intermediate_size, hidden_size]; the gate half comes first.
    W_down has shape [hidden_size, intermediate_size].
    """
    layer = model.model.layers[layer_idx]
    mlp = layer.mlp

    # gate_up_proj weight: [2*intermediate, hidden]
    gate_up_w = mlp.gate_up_proj.weight.detach().float().numpy()  # (2I, H)
    intermediate = gate_up_w.shape[0] // 2
    W_gate = gate_up_w[:intermediate]   # (I, H)
    W_up   = gate_up_w[intermediate:]   # (I, H)

    W_down = mlp.down_proj.weight.detach().float().numpy()  # (H, I)

    return W_gate, W_up, W_down


# ---------------------------------------------------------------------------
# Per-layer evaluation
# ---------------------------------------------------------------------------


def evaluate_layer(
    W_gate: np.ndarray,  # (I, H)
    W_up:   np.ndarray,  # (I, H)
    W_down: np.ndarray,  # (H, I)
    gate_up_input: np.ndarray,  # (T, H)  – input to gate_up_proj
    gate_raw: np.ndarray,        # (T, I)  – full-precision gate before SiLU
) -> dict[str, float]:
    """Compute cosine similarities for all three stages.

    All quantities are derived from the same (gate_up_input, gate_raw) pair so
    token counts are always consistent.
    """
    # Sign weights
    S_gate = np.sign(W_gate)  # (I, H)
    S_up   = np.sign(W_up)    # (I, H)
    S_down = np.sign(W_down)  # (H, I)

    # --- Stage 1: gate output (before SiLU) ---
    gate_approx = gate_up_input @ S_gate.T   # (T, I)
    cos_gate = cosine_sim_rows(gate_raw, gate_approx)

    # --- Stage 2: SwiGLU output ---
    # Full precision: SiLU(gate_raw) * (gate_up_input @ W_up.T)
    up_full     = gate_up_input @ W_up.T         # (T, I)
    swiglu_full = swiglu(gate_raw, up_full)       # (T, I)

    # Approximation: SiLU(gate_approx) * (gate_up_input @ S_up.T)
    up_approx     = gate_up_input @ S_up.T        # (T, I)
    swiglu_approx = swiglu(gate_approx, up_approx)  # (T, I)

    cos_swiglu = cosine_sim_rows(swiglu_full, swiglu_approx)

    # --- Stage 3: down projection output ---
    down_full   = swiglu_full   @ W_down.T   # (T, H)
    down_approx = swiglu_approx @ S_down.T   # (T, H)

    cos_down = cosine_sim_rows(down_full, down_approx)

    return {
        "gate":   cos_gate,
        "swiglu": cos_swiglu,
        "down":   cos_down,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Experiment 1: sign-predictor cosine similarity vs full precision.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", default="ibm-granite/granite-4.2-3b",
                   help="Model name or path (default: ibm-granite/granite-4.2-3b).")
    p.add_argument("--act-file", default="ffn_activations128_gate.npz",
                   metavar="FILE",
                   help="NPZ with gate_up_input and gate_raw "
                        "(default: ffn_activations128_gate.npz).")
    p.add_argument("--layers", nargs="*", type=int, default=None, metavar="N",
                   help="Layer indices to evaluate. Omit for all layers present in the files.")
    p.add_argument("--max-tokens", type=int, default=None, metavar="N",
                   help="Cap token count per layer (for faster runs).")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    # --- Load activations ---
    print(f"Loading activations from {args.act_file!r} …", file=sys.stderr)
    act_data = np.load(args.act_file)

    available_layers = sorted(
        int(k.split("/")[0].replace("layer", ""))
        for k in act_data.files if k.endswith("/gate_raw")
    )
    common_layers = available_layers
    if args.layers is not None:
        common_layers = [l for l in common_layers if l in args.layers]
    print(f"Evaluating {len(common_layers)} layer(s): {common_layers}", file=sys.stderr)

    # --- Load model weights ---
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

    # --- Evaluate ---
    results: dict[int, dict[str, float]] = {}
    for layer_idx in common_layers:
        prefix = f"layer{layer_idx}"
        gate_up_input = act_data[f"{prefix}/gate_up_input"]
        gate_raw      = act_data[f"{prefix}/gate_raw"]

        if args.max_tokens is not None:
            n = args.max_tokens
            gate_up_input = gate_up_input[:n]
            gate_raw      = gate_raw[:n]

        W_gate, W_up, W_down = get_layer_weights(model, layer_idx)
        results[layer_idx] = evaluate_layer(
            W_gate, W_up, W_down,
            gate_up_input, gate_raw,
        )
        print(
            f"  layer {layer_idx:3d}: "
            f"gate={results[layer_idx]['gate']:.4f}  "
            f"swiglu={results[layer_idx]['swiglu']:.4f}  "
            f"down={results[layer_idx]['down']:.4f}",
            file=sys.stderr,
        )

    # --- Summary ---
    gate_vals   = [v["gate"]   for v in results.values()]
    swiglu_vals = [v["swiglu"] for v in results.values()]
    down_vals   = [v["down"]   for v in results.values()]

    print("\n--- Results (mean cosine similarity across all evaluated layers) ---")
    print(f"  gate output (pre-SiLU):      {np.mean(gate_vals):.4f}  "
          f"[min {np.min(gate_vals):.4f}, max {np.max(gate_vals):.4f}]")
    print(f"  SwiGLU output (down input):  {np.mean(swiglu_vals):.4f}  "
          f"[min {np.min(swiglu_vals):.4f}, max {np.max(swiglu_vals):.4f}]")
    print(f"  down projection output:      {np.mean(down_vals):.4f}  "
          f"[min {np.min(down_vals):.4f}, max {np.max(down_vals):.4f}]")


if __name__ == "__main__":
    main()
