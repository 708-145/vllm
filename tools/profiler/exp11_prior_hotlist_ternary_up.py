# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 11 – prior-token hotlist with ternary gate proxy + full-precision up.

Motivation
----------
Experiment 10 showed that pre-SiLU routing on |gate_approx| (ternary gate,
α=0.75) with full-precision up delivers 0.476 output cosine similarity at 10%
hot — beating exp5's prior-token hotlist (0.310) and exp4's sign-gate pre-SiLU
routing (0.245).

The question: can we push further by replacing the current-token routing signal
(|gate_approx[t]|) with the prior-token hotlist?  At inference time the prior
hotlist is free — the full-precision gate activations for token t-1 are already
computed on the critical path.  Alternatively, the ternary proxy of the prior
token (|gate_approx[t-1]|) can serve as a proxy prior, which is even cheaper.

This experiment evaluates three routing variants under the exp10 weight scheme:

  A. **exp10 baseline**: route on |gate_approx[t]| — current ternary gate signal
  B. **proxy prior**: route on |gate_approx[t-1]| — prior ternary gate (free if
     ternary pass was run last step; no full gate GEMM needed for routing)
  C. **oracle prior**: route on |gate_full[t-1]| — prior full-precision gate
     (free since the full gate is computed anyway for token t-1)

For all variants:
  - gate_hybrid[hot]  = W_gate[hot] @ x[t]   full-precision gate for hot
    gate_hybrid[cold] = T(W_gate, τ) @ x[t]  ternary gate for cold
  - up_full  = W_up @ x[t]                   full-precision up always
  - swiglu   = SiLU(gate_hybrid) * up_full
  - out      = W_down @ swiglu

Variant C is the ceiling: it tells us how much of exp5's advantage over exp10
was due to the routing oracle vs the hybrid weight scheme.

Sweeps:
  - τ_factor α ∈ {0.0 (sign), 0.75} for efficiency; α=0.75 is the exp10 sweet spot
  - hot fractions ∈ {0.5%, 1%, 2%, 5%, 10%, 20%, 30%}

Usage::

    python tools/profiler/exp11_prior_hotlist_ternary_up.py \\
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
        description="Experiment 11: prior-token hotlist + ternary gate + full-prec up.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", default="ibm-granite/granite-4.2-3b")
    p.add_argument("--act-file", default="ffn_activations128_gate.npz")
    p.add_argument("--layers", nargs="*", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=2000)
    p.add_argument(
        "--tau-factors", nargs="+", type=float,
        default=[0.0, 0.75],
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

    # routing variants: "current" (exp10), "proxy_prior", "oracle_prior"
    variants = ("current", "proxy_prior", "oracle_prior")
    # results: alpha -> variant -> hot_frac -> list of layer values
    results: dict[float, dict[str, dict[float, list[float]]]] = {
        a: {v: {f: [] for f in args.hot_fractions} for v in variants}
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

        # Full-precision projections — used for oracle prior and hot gate recompute
        gate_full = x @ W_gate.T   # (T, I)
        up_full   = x @ W_up.T     # (T, I) — always exact

        # Full-precision reference output (using recorded gate_raw to match exp9/10)
        swiglu_full = F.silu(gate_raw) * up_full   # (T, I)
        out_full    = swiglu_full @ W_down.T        # (T, H)

        for alpha in args.tau_factors:
            tau_gate    = alpha * mean_abs_gate
            T_gate      = ternary(W_gate, tau_gate)   # (I, H)
            gate_approx = x @ T_gate.T                # (T, I)

            # Slices for adjacent-token pairs (prior=0..T-2, current=1..T-1)
            T_pairs = x.shape[0] - 1

            gate_approx_prior = gate_approx[:T_pairs]   # (T-1, I)  proxy prior
            gate_full_prior   = gate_full  [:T_pairs]   # (T-1, I)  oracle prior
            gate_approx_cur   = gate_approx[1:]         # (T-1, I)  current approx
            gate_full_cur     = gate_full  [1:]         # (T-1, I)  current full (for hot recompute)
            up_full_cur       = up_full    [1:]         # (T-1, I)
            gate_raw_cur      = gate_raw   [1:]         # (T-1, I)
            out_full_cur      = out_full   [1:]         # (T-1, H)
            swiglu_full_cur   = swiglu_full[1:]         # (T-1, I)

            for frac in args.hot_fractions:
                k_hot = max(1, int(frac * I))

                routing_signals = {
                    "current":      gate_approx_cur.abs(),
                    "proxy_prior":  gate_approx_prior.abs(),
                    "oracle_prior": gate_full_prior.abs(),
                }

                for var, signal in routing_signals.items():
                    hot = top_k_mask(signal, k_hot)   # (T-1, I)

                    # Hot: full-precision gate; cold: ternary gate.  Up always exact.
                    gate_hybrid   = torch.where(hot, gate_full_cur, gate_approx_cur)
                    swiglu_hybrid = F.silu(gate_hybrid) * up_full_cur
                    out_hybrid    = swiglu_hybrid @ W_down.T

                    results[alpha][var][frac].append(
                        cosine_sim_mean(out_full_cur, out_hybrid))

        ref_a = 0.75 if 0.75 in args.tau_factors else args.tau_factors[-1]
        print(
            f"  layer {layer_idx:3d}: "
            f"out@10%(current)={results[ref_a]['current'][0.10][-1]:.4f}  "
            f"proxy_prior={results[ref_a]['proxy_prior'][0.10][-1]:.4f}  "
            f"oracle_prior={results[ref_a]['oracle_prior'][0.10][-1]:.4f}",
            file=sys.stderr,
        )

        del x, gate_raw, W_gate, W_up, W_down
        del gate_full, up_full, swiglu_full, out_full
        if DEV.type == "mps":
            torch.mps.empty_cache()
        elif DEV.type == "cuda":
            torch.cuda.empty_cache()

    # --- Summary ---
    print("\n--- Experiment 11 Results ---\n")

    for alpha in args.tau_factors:
        zfrac = (1 - alpha / (alpha + 1e-9)) if alpha > 0 else 0.0
        # Approximate zero fraction from exp10 mapping: 0.75→46%, 0.0→0%
        zf_approx = {0.0: 0, 0.75: 46}.get(alpha, "?")
        print(f"α={alpha:.2f}  (zero%≈{zf_approx}%)")
        hdr = f"  {'hot%':>6}" + "".join(f"  {v:>14}" for v in variants)
        print(hdr)
        for frac in args.hot_fractions:
            row = f"  {frac*100:>6.1f}%"
            for var in variants:
                row += f"  {np.mean(results[alpha][var][frac]):>14.4f}"
            print(row)
        print()

    # Best per hot fraction: compare all (alpha, variant) combos
    print("Best (α, variant) per hot fraction — output cosine similarity:")
    print(f"  {'hot%':>6}  {'α':>5}  {'variant':>14}  {'out_cos':>9}  "
          f"{'vs exp10':>10}  {'vs exp5':>9}")
    exp10_ref = {0.005: 0.633, 0.01: 0.612, 0.02: 0.586, 0.05: 0.536,
                 0.10: 0.476, 0.20: 0.386, 0.30: 0.314}
    exp5_ref  = {0.10: 0.310}
    for frac in args.hot_fractions:
        best_val  = -1.0
        best_a    = None
        best_var  = None
        for a in args.tau_factors:
            for var in variants:
                v = np.mean(results[a][var][frac])
                if v > best_val:
                    best_val, best_a, best_var = v, a, var
        e10 = exp10_ref.get(frac)
        e5  = exp5_ref.get(frac)
        d10 = f"{best_val - e10:+.4f}" if e10 is not None else "       n/a"
        d5  = f"{best_val - e5:+.4f}"  if e5  is not None else "       n/a"
        print(f"  {frac*100:>6.1f}%  {best_a:>5.2f}  {best_var:>14}  "
              f"{best_val:>9.4f}  {d10:>10}  {d5:>9}")

    # Per-layer detail at α=0.75, hot=10%
    ref_a = 0.75 if 0.75 in args.tau_factors else args.tau_factors[-1]
    print(f"\nPer-layer detail: α={ref_a}, hot=10%:")
    print(f"  {'layer':>5}  {'current':>9}  {'proxy_prior':>12}  "
          f"{'oracle_prior':>13}  {'Δ(oracle-current)':>18}")
    for i, li in enumerate(layers):
        oc = results[ref_a]["current"]     [0.10][i]
        op = results[ref_a]["proxy_prior"] [0.10][i]
        oo = results[ref_a]["oracle_prior"][0.10][i]
        print(f"  {li:5d}  {oc:9.4f}  {op:12.4f}  {oo:13.4f}  {oo - oc:+18.4f}")


if __name__ == "__main__":
    main()
