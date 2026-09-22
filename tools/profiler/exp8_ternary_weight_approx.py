# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 8 – ternary weight approximation for MLP sparsity routing.

Instead of using sign(W) ∈ {±1} as the weight proxy, use a ternary
approximation T(W, τ) ∈ {−1, 0, +1}:

    T(w, τ) = sign(w)  if |w| >= τ,  else 0

where τ = α × mean(|W_gate|) is a per-layer adaptive threshold.

Zeroing small weights removes the noise contribution of near-zero weights that
flip the sign of gate outputs without carrying meaningful directional signal.

The full scheme (identical to experiment 7, only the weight proxy changes):

  1. gate_approx  = T(W_gate, τ) @ x
  2. up_approx    = T(W_up,   τ) @ x
  3. swiglu_approx = SiLU(gate_approx) * up_approx     ← routing signal
  4. hot = top-F neurons by |swiglu_approx|
  5. swiglu_hybrid[hot]  = SiLU(W_gate[hot,:] @ x) * (W_up[hot,:] @ x)
     swiglu_hybrid[cold] = swiglu_approx[cold]
  6. out = W_down @ swiglu_hybrid

Sweeps:
  - τ_factor α ∈ {0.0 (sign baseline), 0.25, 0.5, 0.75, 1.0, 1.5}
  - hot fractions ∈ {0.5%, 1%, 2%, 5%, 10%, 20%, 30%}

Per-layer τ = α × mean(|W_gate|)  (computed once, applied to both gate and up).

Metrics vs full-precision reference:
  - Gate cosine similarity: cos(W_gate @ x, T(W_gate,τ) @ x)
  - Neuron cosine similarity: cos(swiglu_full, swiglu_hybrid)
  - Output cosine similarity: cos(W_down @ swiglu_full, W_down @ swiglu_hybrid)
  - Fraction of weights zeroed (sparsity of T(W))

Usage::

    python tools/profiler/exp8_ternary_weight_approx.py \\
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
    """Return T(W, τ) = sign(W) where |W| >= τ, else 0."""
    return W.sign() * (W.abs() >= tau).float()


def get_weights(model, layer_idx: int):
    mlp = model.model.layers[layer_idx].mlp
    W = mlp.gate_up_proj.weight.detach().float().numpy()
    I = W.shape[0] // 2
    return W[:I], W[I:], mlp.down_proj.weight.detach().float().numpy()


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Experiment 8: ternary weight approximation.",
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
        help="τ = α × mean(|W_gate|). 0.0 = sign baseline (exp7).",
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
    print(f"Evaluating {len(layers)} layers, "
          f"{len(args.tau_factors)} τ-factors, "
          f"{len(args.hot_fractions)} hot-fractions.", file=sys.stderr)

    from vllm import LLM
    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              kv_cache_memory_bytes=int(0.5 * 1024**3), max_model_len=512)
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model

    # Results: tau_factor -> hot_frac -> {gate_cos, neuron_cos, out_cos, zero_frac}
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

        # Per-layer adaptive threshold base
        mean_abs_gate = float(W_gate.abs().mean())
        mean_abs_up   = float(W_up.abs().mean())

        # Full-precision reference (computed once)
        gate_full   = x @ W_gate.T                        # (T, I)
        up_full     = x @ W_up.T                          # (T, I)
        swiglu_full = F.silu(gate_raw) * up_full          # (T, I)
        out_full    = swiglu_full @ W_down.T              # (T, H)

        for alpha in args.tau_factors:
            tau_gate = alpha * mean_abs_gate
            tau_up   = alpha * mean_abs_up

            T_gate = ternary(W_gate, tau_gate)   # (I, H)  values in {-1, 0, +1}
            T_up   = ternary(W_up,   tau_up)     # (I, H)

            zero_frac = float((T_gate == 0).float().mean())

            gate_approx   = x @ T_gate.T         # (T, I)
            up_approx     = x @ T_up.T           # (T, I)
            swiglu_approx = F.silu(gate_approx) * up_approx  # (T, I)

            gate_cos_alpha = cosine_sim_mean(gate_full, gate_approx)

            for frac in args.hot_fractions:
                k_hot = max(1, int(frac * I))
                hot   = top_k_mask(swiglu_approx.abs(), k_hot)   # (T, I)

                gate_hybrid   = torch.where(hot, gate_full,  gate_approx)
                up_hybrid     = torch.where(hot, up_full,    up_approx)
                swiglu_hybrid = F.silu(gate_hybrid) * up_hybrid
                out_hybrid    = swiglu_hybrid @ W_down.T

                results[alpha][frac]["gate_cos"].append(gate_cos_alpha)
                results[alpha][frac]["neuron_cos"].append(
                    cosine_sim_mean(swiglu_full, swiglu_hybrid))
                results[alpha][frac]["out_cos"].append(
                    cosine_sim_mean(out_full, out_hybrid))
                results[alpha][frac]["zero_frac"].append(zero_frac)

        best_alpha = args.tau_factors[-1]
        best_frac  = 0.10
        print(
            f"  layer {layer_idx:3d}: "
            f"gate_cos(α=0)={results[0.0][best_frac]['gate_cos'][-1]:.4f}  "
            f"gate_cos(α={best_alpha})={results[best_alpha][best_frac]['gate_cos'][-1]:.4f}  "
            f"out@10%(α=0)={results[0.0][best_frac]['out_cos'][-1]:.4f}  "
            f"out@10%(α={best_alpha})={results[best_alpha][best_frac]['out_cos'][-1]:.4f}",
            file=sys.stderr,
        )

        del x, gate_raw, W_gate, W_up, W_down
        del gate_full, up_full, swiglu_full, out_full
        if DEV.type == "mps":
            torch.mps.empty_cache()
        elif DEV.type == "cuda":
            torch.cuda.empty_cache()

    # --- Summary ---
    print("\n--- Experiment 8 Results ---\n")

    # Gate cosine similarity by tau factor (independent of hot fraction)
    print("Gate cosine similarity vs full-precision gate (mean across layers):")
    print(f"  {'α':>5}  {'zero%':>7}  {'gate_cos':>10}")
    for alpha in args.tau_factors:
        frac0 = args.hot_fractions[0]
        gc   = np.mean(results[alpha][frac0]["gate_cos"])
        zf   = np.mean(results[alpha][frac0]["zero_frac"]) * 100
        label = "(sign baseline)" if alpha == 0.0 else ""
        print(f"  {alpha:>5.2f}  {zf:>6.1f}%  {gc:>10.4f}  {label}")

    # Output cosine similarity table: rows=alpha, cols=hot_frac
    print(f"\nOutput cosine similarity vs full precision:")
    hdr = f"  {'α':>5}  {'zero%':>6}" + "".join(
        f"  {f*100:>5.1f}%" for f in args.hot_fractions)
    print(hdr)
    for alpha in args.tau_factors:
        zf  = np.mean(results[alpha][args.hot_fractions[0]]["zero_frac"]) * 100
        row = f"  {alpha:>5.2f}  {zf:>5.1f}%"
        for frac in args.hot_fractions:
            row += f"  {np.mean(results[alpha][frac]['out_cos']):>6.4f}"
        print(row)

    # Neuron cosine similarity table
    print(f"\nNeuron (SwiGLU) cosine similarity vs full precision:")
    print(hdr)
    for alpha in args.tau_factors:
        zf  = np.mean(results[alpha][args.hot_fractions[0]]["zero_frac"]) * 100
        row = f"  {alpha:>5.2f}  {zf:>5.1f}%"
        for frac in args.hot_fractions:
            row += f"  {np.mean(results[alpha][frac]['neuron_cos']):>6.4f}"
        print(row)

    # Best configuration per hot fraction
    print(f"\nBest α per hot fraction (by output cosine similarity):")
    print(f"  {'hot%':>6}  {'best_α':>7}  {'out_cos':>9}  {'vs sign (α=0)':>15}")
    for frac in args.hot_fractions:
        best_a = max(args.tau_factors,
                     key=lambda a: np.mean(results[a][frac]["out_cos"]))
        best_v = np.mean(results[best_a][frac]["out_cos"])
        base_v = np.mean(results[0.0][frac]["out_cos"])
        print(f"  {frac*100:>6.1f}%  {best_a:>7.2f}  {best_v:>9.4f}  "
              f"{best_v - base_v:>+15.4f}")

    # Per-layer detail for best alpha at 10% hot
    best_alpha_10 = max(args.tau_factors,
                        key=lambda a: np.mean(results[a][0.10]["out_cos"]))
    print(f"\nPer-layer detail: α={best_alpha_10} vs α=0.0 (sign), hot=10%:")
    print(f"  {'layer':>5}  {'gate_cos(0)':>12}  {'gate_cos(α)':>12}  "
          f"{'out_cos(0)':>11}  {'out_cos(α)':>11}  {'Δout':>7}")
    for i, li in enumerate(layers):
        gc0 = results[0.0]        [0.10]["gate_cos"][i]
        gca = results[best_alpha_10][0.10]["gate_cos"][i]
        oc0 = results[0.0]        [0.10]["out_cos"][i]
        oca = results[best_alpha_10][0.10]["out_cos"][i]
        print(f"  {li:5d}  {gc0:12.4f}  {gca:12.4f}  "
              f"{oc0:11.4f}  {oca:11.4f}  {oca-oc0:>+7.4f}")


if __name__ == "__main__":
    main()
