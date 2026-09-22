# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 10 – ternary gate proxy with pre-SiLU routing + full-precision up.

Motivation
----------
Experiment 9 showed that routing on |SwiGLU_approx| with exact up projection
eliminates the U-shaped cosine similarity curve seen in exp7/8.  However output
cosine similarity at 10% hot (0.181) still lags exp4's pre-SiLU routing
(0.245) and exp5's prior-token hotlist (0.310).

The hypothesis: the remaining gap is due to the routing signal itself.
|SwiGLU_approx| = |SiLU(gate_approx) * up_full| is dominated by neurons where
gate_approx pushes the SiLU output away from zero, which may not rank the
truly important neurons as well as the raw magnitude |gate_approx| used in
exp4.

This experiment replaces the routing signal with |gate_approx| (pre-SiLU, like
exp4) while keeping the rest of exp9's scheme:

  1. gate_approx  = T(W_gate, τ) @ x       ternary gate, all channels
  2. up_full      = W_up @ x               full-precision up, all channels
  3. hot = top-F channels by |gate_approx|  ← pre-SiLU routing (exp4 signal)
  4. gate_hybrid[hot]  = W_gate[hot] @ x   full-precision gate for hot
     gate_hybrid[cold] = gate_approx[cold] ternary for cold
  5. swiglu_hybrid = SiLU(gate_hybrid) * up_full
  6. out = W_down @ swiglu_hybrid

vs exp9:
  step 3 used |SiLU(gate_approx) * up_full| (post-SiLU) as the routing signal.

vs exp4:
  gate_approx uses ternary weights (not just sign), up is always exact, down is
  always full-precision W_down (no sparse correction term).

Expected outcome: better routing than exp9 (inheriting exp4's Spearman
correlation advantage) with better cold-channel quality than exp4 (ternary gate
vs sign-only gate for cold channels, full up).

Sweeps:
  - τ_factor α ∈ {0.0 (sign), 0.25, 0.5, 0.75, 1.0, 1.5}
  - hot fractions ∈ {0.5%, 1%, 2%, 5%, 10%, 20%, 30%}

Usage::

    python tools/profiler/exp10_ternary_gate_preSiLU_routing.py \\
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


def top_k_mask(scores: torch.Tensor, k: int) -> torch.BoolTensor:
    idx = torch.topk(scores, k, dim=1, largest=True, sorted=False).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask


def ternary(W: torch.Tensor, tau: float) -> torch.Tensor:
    """T(W, τ) = sign(W) where |W| >= τ, else 0."""
    return W.sign() * (W.abs() >= tau).float()


def get_weights(model, layer_idx: int):
    mlp = model.model.layers[layer_idx].mlp
    W = mlp.gate_up_proj.weight.detach().float().numpy()
    I = W.shape[0] // 2
    return W[:I], W[I:], mlp.down_proj.weight.detach().float().numpy()


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Experiment 10: ternary gate + pre-SiLU routing + full-precision up.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", default="ibm-granite/granite-4.2-3b")
    p.add_argument("--act-file", default="ffn_activations128_gate.npz")
    p.add_argument("--layers", nargs="*", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=2000)
    p.add_argument(
        "--tau-factors", nargs="+", type=float,
        default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5],
        metavar="A",
        help="τ = α × mean(|W_gate|). 0.0 = sign baseline.",
    )
    p.add_argument(
        "--hot-fractions", nargs="+", type=float,
        default=[0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30],
        metavar="F",
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
    print(
        f"Evaluating {len(layers)} layers, "
        f"{len(args.tau_factors)} τ-factors, "
        f"{len(args.hot_fractions)} hot-fractions.",
        file=sys.stderr,
    )

    from vllm import LLM
    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              kv_cache_memory_bytes=int(0.5 * 1024**3), max_model_len=512)
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model

    # Results: alpha -> hot_frac -> {gate_cos, neuron_cos, out_cos, zero_frac}
    results: dict[float, dict[float, dict[str, list]]] = {
        a: {f: {"gate_cos": [], "neuron_cos": [], "out_cos": [], "zero_frac": []}
            for f in args.hot_fractions}
        for a in args.tau_factors
    }

    for layer_idx in layers:
        pfx = f"layer{layer_idx}"
        x_np    = act_data[f"{pfx}/gate_up_input"][: args.max_tokens].astype(np.float32)
        graw_np = act_data[f"{pfx}/gate_raw"][: args.max_tokens].astype(np.float32)

        W_gate_np, W_up_np, W_down_np = get_weights(model, layer_idx)

        x        = t(x_np)
        gate_raw = t(graw_np)
        W_gate   = t(W_gate_np)
        W_up     = t(W_up_np)
        W_down   = t(W_down_np)
        I        = W_gate.shape[0]

        mean_abs_gate = float(W_gate.abs().mean())

        # Full-precision reference
        gate_full   = x @ W_gate.T               # (T, I)
        up_full     = x @ W_up.T                 # (T, I) — exact, used everywhere
        swiglu_full = F.silu(gate_raw) * up_full  # (T, I)
        out_full    = swiglu_full @ W_down.T      # (T, H)

        for alpha in args.tau_factors:
            tau_gate  = alpha * mean_abs_gate
            T_gate    = ternary(W_gate, tau_gate)               # (I, H)
            zero_frac = float((T_gate == 0).float().mean())

            gate_approx = x @ T_gate.T                          # (T, I)

            # Pre-SiLU routing: top-k by |gate_approx| (exp4 signal)
            gate_cos_val = cosine_sim_mean(gate_full, gate_approx)

            for frac in args.hot_fractions:
                k_hot = max(1, int(frac * I))
                hot   = top_k_mask(gate_approx.abs(), k_hot)    # (T, I)

                # Hot: full-precision gate; cold: ternary gate.  Up always exact.
                gate_hybrid   = torch.where(hot, gate_full, gate_approx)
                swiglu_hybrid = F.silu(gate_hybrid) * up_full   # (T, I)
                out_hybrid    = swiglu_hybrid @ W_down.T        # (T, H)

                results[alpha][frac]["gate_cos"].append(gate_cos_val)
                results[alpha][frac]["neuron_cos"].append(
                    cosine_sim_mean(swiglu_full, swiglu_hybrid))
                results[alpha][frac]["out_cos"].append(
                    cosine_sim_mean(out_full, out_hybrid))
                results[alpha][frac]["zero_frac"].append(zero_frac)

        best_alpha = 0.75
        print(
            f"  layer {layer_idx:3d}: "
            f"gate_cos(α=0)={results[0.0][0.10]['gate_cos'][-1]:.4f}  "
            f"gate_cos(α={best_alpha})={results[best_alpha][0.10]['gate_cos'][-1]:.4f}  "
            f"out@10%(α=0)={results[0.0][0.10]['out_cos'][-1]:.4f}  "
            f"out@10%(α={best_alpha})={results[best_alpha][0.10]['out_cos'][-1]:.4f}",
            file=sys.stderr,
        )

        del x, gate_raw, W_gate, W_up, W_down
        del gate_full, up_full, swiglu_full, out_full
        if DEV.type == "mps":
            torch.mps.empty_cache()
        elif DEV.type == "cuda":
            torch.cuda.empty_cache()

    # --- Summary ---
    print("\n--- Experiment 10 Results ---\n")

    print("Gate cosine similarity vs full-precision gate (mean across layers):")
    print(f"  {'α':>5}  {'zero%':>7}  {'gate_cos':>10}")
    for alpha in args.tau_factors:
        f0 = args.hot_fractions[0]
        gc = np.mean(results[alpha][f0]["gate_cos"])
        zf = np.mean(results[alpha][f0]["zero_frac"]) * 100
        print(f"  {alpha:>5.2f}  {zf:>6.1f}%  {gc:>10.4f}")

    hdr = f"  {'α':>5}  {'zero%':>6}" + "".join(
        f"  {f*100:>5.1f}%" for f in args.hot_fractions)

    print(f"\nOutput cosine similarity vs full precision:")
    print(hdr)
    for alpha in args.tau_factors:
        zf  = np.mean(results[alpha][args.hot_fractions[0]]["zero_frac"]) * 100
        row = f"  {alpha:>5.2f}  {zf:>5.1f}%"
        for frac in args.hot_fractions:
            row += f"  {np.mean(results[alpha][frac]['out_cos']):>6.4f}"
        print(row)

    print(f"\nNeuron (SwiGLU) cosine similarity vs full precision:")
    print(hdr)
    for alpha in args.tau_factors:
        zf  = np.mean(results[alpha][args.hot_fractions[0]]["zero_frac"]) * 100
        row = f"  {alpha:>5.2f}  {zf:>5.1f}%"
        for frac in args.hot_fractions:
            row += f"  {np.mean(results[alpha][frac]['neuron_cos']):>6.4f}"
        print(row)

    # Best alpha per hot fraction vs prior experiments
    print(f"\nBest α per hot fraction (output cosine similarity):")
    print(f"  {'hot%':>6}  {'best_α':>7}  {'out_cos':>9}  "
          f"{'vs exp9':>9}  {'vs exp4':>9}  {'vs exp5':>9}")
    exp9_ref = {0.005: 0.442, 0.01: 0.399, 0.02: 0.337, 0.05: 0.238,
                0.10:  0.181, 0.20: 0.387, 0.30: 0.699}
    exp4_ref = {0.10: 0.245, 0.20: 0.197, 0.30: 0.157}
    exp5_ref = {0.10: 0.310}
    for frac in args.hot_fractions:
        best_a = max(args.tau_factors,
                     key=lambda a: np.mean(results[a][frac]["out_cos"]))
        best_v = np.mean(results[best_a][frac]["out_cos"])
        e9 = exp9_ref.get(frac)
        e4 = exp4_ref.get(frac)
        e5 = exp5_ref.get(frac)
        d9 = f"{best_v - e9:+.4f}" if e9 is not None else "    n/a"
        d4 = f"{best_v - e4:+.4f}" if e4 is not None else "    n/a"
        d5 = f"{best_v - e5:+.4f}" if e5 is not None else "    n/a"
        print(f"  {frac*100:>6.1f}%  {best_a:>7.2f}  {best_v:>9.4f}  "
              f"{d9:>9}  {d4:>9}  {d5:>9}")

    # Per-layer detail at best overall config
    best_alpha_10 = max(args.tau_factors,
                        key=lambda a: np.mean(results[a][0.10]["out_cos"]))
    print(f"\nPer-layer detail: α={best_alpha_10} vs α=0.0 (sign), hot=10%:")
    print(f"  {'layer':>5}  {'gate_cos(0)':>12}  {'gate_cos(α)':>12}  "
          f"{'out_cos(0)':>11}  {'out_cos(α)':>11}  {'Δout':>7}")
    for i, li in enumerate(layers):
        gc0 = results[0.0]          [0.10]["gate_cos"][i]
        gca = results[best_alpha_10][0.10]["gate_cos"][i]
        oc0 = results[0.0]          [0.10]["out_cos"][i]
        oca = results[best_alpha_10][0.10]["out_cos"][i]
        print(f"  {li:5d}  {gc0:12.4f}  {gca:12.4f}  "
              f"{oc0:11.4f}  {oca:11.4f}  {oca-oc0:>+7.4f}")


if __name__ == "__main__":
    main()
