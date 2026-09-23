# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 12 – ternary W_down for cold SwiGLU channels.

Motivation
----------
Experiments 10–11 use full-precision W_down for all I intermediate channels.
The down projection is the most expensive single GEMM in the MLP block
(H × I = 2560 × 8192 for granite-4.2-3b), and with 10% hot channels 90% of
input columns still require full-precision multiply-accumulate.

Hypothesis: cold SwiGLU entries are small (near-zero) because either the gate
is near zero (suppressed by SiLU) or the up value is small.  The contribution
of cold columns to the output is therefore dominated by the sign of W_down
rather than its magnitude.  Replacing W_down[:, cold] with T(W_down, τ_down)
— keeping W_down[:, hot] exact — may recover most of the quality at lower
compute cost.

Scheme
------
Anchored on exp11's best configuration (α_gate=0.75, proxy-prior routing):

  1.  gate_approx[t]   = T(W_gate, τ_g) @ x[t]
  2.  up_full[t]       = W_up @ x[t]                         full-precision
  3.  hot = top-F by |gate_approx[t-1]|                      proxy-prior routing
  4.  gate_hybrid[hot]  = W_gate[hot] @ x[t]
      gate_hybrid[cold] = gate_approx[t][cold]
  5.  swiglu = SiLU(gate_hybrid) * up_full[t]
  6.  out = W_down[:, hot]  @ swiglu[hot]                    full-precision hot
          + T(W_down, τ_d)[:, cold] @ swiglu[cold]           ternary cold

where τ_d = α_d × mean(|W_down|).  α_d = 0 recovers the exp11 baseline
(sign cold down), α_d > 0 zeros out small W_down entries.

For comparison two extra variants are included at each (α_g, α_d) point:
  - "full_down":   use exact W_down for all channels (exp11 proxy-prior baseline)
  - "ternary_all": use T(W_down, τ_d) for ALL channels (no hot/cold split)

This lets us quantify: (a) how much quality ternary down costs vs full down,
and (b) whether the hot/cold split is worth maintaining for the down projection.

Sweeps
------
  - α_gate ∈ {0.75}  (sweet spot from exp10/11)
  - α_down ∈ {0.0 (sign), 0.25, 0.50, 0.75, 1.00, 1.50}
  - hot fractions ∈ {0.5%, 1%, 2%, 5%, 10%, 20%, 30%}
  - routing: proxy-prior (|gate_approx[t-1]|) only

Usage::

    python tools/profiler/exp12_ternary_down_cold.py \\
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
        description="Experiment 12: ternary W_down for cold SwiGLU channels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", default="ibm-granite/granite-4.2-3b")
    p.add_argument("--act-file", default="ffn_activations128_gate.npz")
    p.add_argument("--layers", nargs="*", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=2000)
    p.add_argument(
        "--tau-gate", type=float, default=0.75,
        metavar="AG",
        help="τ_gate = α_gate × mean(|W_gate|). Fixed at 0.75 from exp11.",
    )
    p.add_argument(
        "--tau-down-factors", nargs="+", type=float,
        default=[0.0, 0.25, 0.50, 0.75, 1.00, 1.50],
        metavar="AD",
        help="τ_down = α_down × mean(|W_down|).",
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
        f"α_gate={args.tau_gate}, "
        f"{len(args.tau_down_factors)} α_down values, "
        f"{len(args.hot_fractions)} hot-fractions.",
        file=sys.stderr,
    )

    from vllm import LLM
    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              kv_cache_memory_bytes=int(0.5 * 1024**3), max_model_len=512)
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model

    # results[ad][frac][scheme] = list of per-layer cos-sim values
    # schemes: "ternary_cold", "full_down" (baseline), "ternary_all"
    schemes = ("ternary_cold", "full_down", "ternary_all")
    results: dict[float, dict[float, dict[str, list[float]]]] = {
        ad: {f: {s: [] for s in schemes} for f in args.hot_fractions}
        for ad in args.tau_down_factors
    }
    # also track down zero-fraction per alpha_down
    down_zero_frac: dict[float, list[float]] = {ad: [] for ad in args.tau_down_factors}

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
        mean_abs_down = float(W_down.abs().mean())

        tau_gate = args.tau_gate * mean_abs_gate
        T_gate   = ternary(W_gate, tau_gate)            # (I, H)

        gate_full   = x @ W_gate.T                      # (T, I)
        up_full     = x @ W_up.T                        # (T, I) — always exact
        gate_approx = x @ T_gate.T                      # (T, I)

        # Full-precision reference
        swiglu_full = F.silu(gate_raw) * up_full        # (T, I)
        out_full    = swiglu_full @ W_down.T             # (T, H)

        # Adjacent-token pairs; prior = 0..T-2, current = 1..T-1
        T_pairs = x.shape[0] - 1
        gate_approx_prior = gate_approx[:T_pairs]       # proxy-prior routing signal
        gate_approx_cur   = gate_approx[1:]
        gate_full_cur     = gate_full[1:]
        up_full_cur       = up_full[1:]
        gate_raw_cur      = gate_raw[1:]
        out_full_cur      = out_full[1:]
        swiglu_full_cur   = F.silu(gate_raw_cur) * up_full_cur  # (T-1, I)

        for ad in args.tau_down_factors:
            tau_down = ad * mean_abs_down
            T_down   = ternary(W_down, tau_down)        # (H, I)  ternary W_down
            down_zero_frac[ad].append(float((T_down == 0).float().mean()))

            for frac in args.hot_fractions:
                k_hot = max(1, int(frac * I))
                hot   = top_k_mask(gate_approx_prior.abs(), k_hot)  # proxy-prior

                # Build hybrid SwiGLU (same as exp11 proxy-prior)
                gate_hybrid   = torch.where(hot, gate_full_cur, gate_approx_cur)
                swiglu_hybrid = F.silu(gate_hybrid) * up_full_cur   # (T-1, I)

                # "full_down": W_down for all channels (exp11 baseline)
                out_full_down = swiglu_hybrid @ W_down.T

                # "ternary_cold": exact W_down for hot, ternary for cold
                # out = W_down[:, hot] @ s[hot] + T_down[:, cold] @ s[cold]
                # = T_down @ s + (W_down - T_down)[:, hot] @ s[hot]
                residual_down = W_down - T_down                     # (H, I)
                out_ternary_cold = (
                    swiglu_hybrid @ T_down.T
                    + (swiglu_hybrid * hot.float()) @ residual_down.T
                )

                # "ternary_all": T_down for all channels
                out_ternary_all = swiglu_hybrid @ T_down.T

                results[ad][frac]["full_down"].append(
                    cosine_sim_mean(out_full_cur, out_full_down))
                results[ad][frac]["ternary_cold"].append(
                    cosine_sim_mean(out_full_cur, out_ternary_cold))
                results[ad][frac]["ternary_all"].append(
                    cosine_sim_mean(out_full_cur, out_ternary_all))

        print(
            f"  layer {layer_idx:3d}: "
            f"full_down@10%={results[0.0][0.10]['full_down'][-1]:.4f}  "
            f"tern_cold(α=0)@10%={results[0.0][0.10]['ternary_cold'][-1]:.4f}  "
            f"tern_cold(α=0.75)@10%={results[0.75][0.10]['ternary_cold'][-1]:.4f}",
            file=sys.stderr,
        )

        del x, gate_raw, W_gate, W_up, W_down, T_gate, T_down, residual_down
        del gate_full, up_full, gate_approx, swiglu_full, out_full
        if DEV.type == "mps":
            torch.mps.empty_cache()
        elif DEV.type == "cuda":
            torch.cuda.empty_cache()

    # --- Summary ---
    print("\n--- Experiment 12 Results ---\n")

    # W_down proxy quality table
    print("W_down proxy zero-fraction (mean across layers):")
    print(f"  {'α_down':>8}  {'zero%':>7}")
    for ad in args.tau_down_factors:
        zf = np.mean(down_zero_frac[ad]) * 100
        print(f"  {ad:>8.2f}  {zf:>6.1f}%")

    hdr = (f"  {'α_down':>8}  {'zero%':>6}"
           + "".join(f"  {f*100:>5.1f}%" for f in args.hot_fractions))

    for scheme_label, scheme_key in [
        ("Output cosine similarity — ternary_cold (exp12 scheme)", "ternary_cold"),
        ("Output cosine similarity — full_down (exp11 baseline)", "full_down"),
        ("Output cosine similarity — ternary_all (no hot/cold split)", "ternary_all"),
    ]:
        print(f"\n{scheme_label}:")
        print(hdr)
        for ad in args.tau_down_factors:
            zf  = np.mean(down_zero_frac[ad]) * 100
            row = f"  {ad:>8.2f}  {zf:>5.1f}%"
            for frac in args.hot_fractions:
                row += f"  {np.mean(results[ad][frac][scheme_key]):>6.4f}"
            print(row)

    # Delta: ternary_cold vs full_down
    print(f"\nΔ(ternary_cold − full_down) — quality cost of ternary cold down:")
    print(hdr)
    for ad in args.tau_down_factors:
        zf  = np.mean(down_zero_frac[ad]) * 100
        row = f"  {ad:>8.2f}  {zf:>5.1f}%"
        for frac in args.hot_fractions:
            tc = np.mean(results[ad][frac]["ternary_cold"])
            fd = np.mean(results[ad][frac]["full_down"])
            row += f"  {tc - fd:>+6.4f}"
        print(row)

    # Delta: hot/cold split vs ternary_all
    print(f"\nΔ(ternary_cold − ternary_all) — value of the hot/cold down split:")
    print(hdr)
    for ad in args.tau_down_factors:
        zf  = np.mean(down_zero_frac[ad]) * 100
        row = f"  {ad:>8.2f}  {zf:>5.1f}%"
        for frac in args.hot_fractions:
            tc = np.mean(results[ad][frac]["ternary_cold"])
            ta = np.mean(results[ad][frac]["ternary_all"])
            row += f"  {tc - ta:>+6.4f}"
        print(row)

    # Best config per hot fraction
    print(f"\nBest (α_down, scheme) per hot fraction:")
    print(f"  {'hot%':>6}  {'α_down':>8}  {'scheme':>14}  {'out_cos':>9}  "
          f"{'vs exp11':>10}")
    exp11_ref = {0.005: 0.675, 0.01: 0.665, 0.02: 0.654, 0.05: 0.630,
                 0.10: 0.600, 0.20: 0.552, 0.30: 0.511}
    for frac in args.hot_fractions:
        best_val = -1.0
        best_ad = None
        best_sk = None
        for ad in args.tau_down_factors:
            for sk in schemes:
                v = np.mean(results[ad][frac][sk])
                if v > best_val:
                    best_val, best_ad, best_sk = v, ad, sk
        e11 = exp11_ref.get(frac)
        d11 = f"{best_val - e11:+.4f}" if e11 is not None else "       n/a"
        print(f"  {frac*100:>6.1f}%  {best_ad:>8.2f}  {best_sk:>14}  "
              f"{best_val:>9.4f}  {d11:>10}")

    # Per-layer detail at α_down=0.75, hot=10%
    ad_ref = 0.75
    print(f"\nPer-layer detail: α_down={ad_ref}, hot=10%:")
    print(f"  {'layer':>5}  {'full_down':>10}  {'tern_cold':>10}  "
          f"{'tern_all':>10}  {'Δcold':>8}  {'Δall':>8}")
    for i, li in enumerate(layers):
        fd = results[ad_ref][0.10]["full_down"][i]
        tc = results[ad_ref][0.10]["ternary_cold"][i]
        ta = results[ad_ref][0.10]["ternary_all"][i]
        print(f"  {li:5d}  {fd:10.4f}  {tc:10.4f}  {ta:10.4f}  "
              f"{tc - fd:+8.4f}  {ta - fd:+8.4f}")


if __name__ == "__main__":
    main()
