# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 7 – SwiGLU-routed hybrid: route on post-nonlinearity magnitude.

Scheme
------
Previous experiments routed on |gate_approx| (pre-SiLU logit magnitude).
Here we route on the magnitude of the full SwiGLU approximation output, i.e.
*after* the nonlinearity, which is the quantity that actually multiplies the
down-projection input.

Step by step:

  1. Low-precision gate and up pass:
       gate_approx = sign(W_gate) @ x        (T, I)
       up_approx   = sign(W_up)   @ x        (T, I)

  2. Cheap SwiGLU approximation:
       swiglu_approx = SiLU(gate_approx) * up_approx   (T, I)

  3. Route on |swiglu_approx|: select the top-F fraction (or abs > threshold)
     as hot neurons.  These are the neurons that actually contribute most to
     the down-projection output.

  4. Recompute hot neurons in full precision:
       gate_full_hot   = W_gate[hot] @ x         (sparse)
       up_full_hot     = W_up[hot]   @ x          (sparse)
       swiglu_full_hot = SiLU(gate_full_hot) * up_full_hot

  5. Build hybrid neuron vector:
       swiglu_hybrid = swiglu_approx
       swiglu_hybrid[hot] = swiglu_full_hot

  6. Down projection in full precision:
       out_hybrid = W_down @ swiglu_hybrid

Metrics (vs full precision reference):
  - Neuron cosine similarity:  cos(swiglu_full, swiglu_hybrid)
  - Output cosine similarity:  cos(W_down @ swiglu_full, W_down @ swiglu_hybrid)

Sweep over hot fractions [0.5%, 1%, 2%, 5%, 10%, 20%, 30%] and also an
absolute-threshold sweep to find the natural operating point.

Key question: does routing on |swiglu_approx| — the true signal of neuron
importance — give better output quality per hot-fraction than routing on
|gate_approx| (experiment 4) or the prior-token hotlist (experiment 5)?

Usage::

    python tools/profiler/exp7_swiglu_routed_hybrid.py \\
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


def get_weights(model, layer_idx: int):
    mlp = model.model.layers[layer_idx].mlp
    W = mlp.gate_up_proj.weight.detach().float().numpy()
    I = W.shape[0] // 2
    return W[:I], W[I:], mlp.down_proj.weight.detach().float().numpy()


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Experiment 7: SwiGLU-routed hybrid.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", default="ibm-granite/granite-4.2-3b")
    p.add_argument("--act-file", default="ffn_activations128_gate.npz")
    p.add_argument("--layers", nargs="*", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=2000)
    p.add_argument(
        "--hot-fractions", nargs="+", type=float,
        default=[0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30],
        metavar="F",
        help="Hot-neuron fractions to sweep (default: 0.5%% 1%% 2%% 5%% 10%% 20%% 30%%).",
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

    # frac -> {neuron_cos: [], out_cos: []}  accumulated over layers
    frac_results: dict[float, dict[str, list]] = {
        f: {"neuron_cos": [], "out_cos": []} for f in args.hot_fractions
    }
    # also track routing quality: how much of |swiglu_full| energy is captured
    frac_energy: dict[float, list[float]] = {f: [] for f in args.hot_fractions}

    for layer_idx in layers:
        pfx = f"layer{layer_idx}"
        x_np     = act_data[f"{pfx}/gate_up_input"][: args.max_tokens].astype(np.float32)
        graw_np  = act_data[f"{pfx}/gate_raw"][: args.max_tokens].astype(np.float32)

        W_gate_np, W_up_np, W_down_np = get_weights(model, layer_idx)

        x        = t(x_np)
        gate_raw = t(graw_np)
        W_gate   = t(W_gate_np)
        W_up     = t(W_up_np)
        W_down   = t(W_down_np)
        S_gate   = W_gate.sign()
        S_up     = W_up.sign()
        I        = W_gate.shape[0]

        # --- Full precision reference ---
        gate_full    = x @ W_gate.T                        # (T, I)
        up_full      = x @ W_up.T                         # (T, I)
        swiglu_full  = F.silu(gate_raw) * up_full         # (T, I)  reference neuron vec
        out_full     = swiglu_full @ W_down.T             # (T, H)  reference output

        # --- Low-precision SwiGLU approximation (routing signal) ---
        gate_approx  = x @ S_gate.T                       # (T, I)
        up_approx    = x @ S_up.T                         # (T, I)
        swiglu_approx = F.silu(gate_approx) * up_approx  # (T, I)

        abs_swiglu_approx = swiglu_approx.abs()           # routing scores

        for frac in args.hot_fractions:
            k_hot = max(1, int(frac * I))
            hot   = top_k_mask(abs_swiglu_approx, k_hot)  # (T, I)

            # Recompute hot neurons in full precision
            gate_hybrid  = torch.where(hot, gate_full,  gate_approx)
            up_hybrid    = torch.where(hot, up_full,    up_approx)
            swiglu_hybrid = F.silu(gate_hybrid) * up_hybrid   # (T, I)

            out_hybrid   = swiglu_hybrid @ W_down.T           # (T, H)

            neuron_cos = cosine_sim_mean(swiglu_full, swiglu_hybrid)
            out_cos    = cosine_sim_mean(out_full,    out_hybrid)

            # Energy fraction: how much of ||swiglu_full||^2 is captured by hot neurons
            energy_hot   = (swiglu_full * hot.float()).pow(2).sum(1)   # (T,)
            energy_total = swiglu_full.pow(2).sum(1).clamp(min=1e-9)
            energy_frac  = float((energy_hot / energy_total).mean())

            frac_results[frac]["neuron_cos"].append(neuron_cos)
            frac_results[frac]["out_cos"].append(out_cos)
            frac_energy[frac].append(energy_frac)

        print(
            f"  layer {layer_idx:3d}: "
            f"swiglu@10%={frac_results[0.10]['neuron_cos'][-1]:.4f}  "
            f"out@10%={frac_results[0.10]['out_cos'][-1]:.4f}  "
            f"energy@10%={frac_energy[0.10][-1]:.3f}",
            file=sys.stderr,
        )

        del (x, gate_raw, W_gate, W_up, W_down, S_gate, S_up,
             gate_full, up_full, swiglu_full, out_full,
             gate_approx, up_approx, swiglu_approx)
        if DEV.type == "mps":
            torch.mps.empty_cache()
        elif DEV.type == "cuda":
            torch.cuda.empty_cache()

    # --- Summary ---
    print("\n--- Experiment 7 Results ---")
    print(f"\nRouting on |SwiGLU_approx|: cosine similarity vs full precision")
    print(f"({'mean across all '+str(len(layers))+' layers'})\n")
    print(f"  {'hot%':>6}  {'neuron_cos':>11}  {'out_cos':>9}  "
          f"{'energy%':>8}  {'Δout vs exp4@same%':>20}")

    # exp4 out_cos reference values at matching hot fractions (from documented results)
    exp4_ref = {0.10: 0.2446, 0.20: 0.1966, 0.30: 0.1566}

    for frac in args.hot_fractions:
        nc   = np.mean(frac_results[frac]["neuron_cos"])
        oc   = np.mean(frac_results[frac]["out_cos"])
        enrg = np.mean(frac_energy[frac]) * 100
        ref  = exp4_ref.get(frac)
        delta_str = f"{oc - ref:+.4f}" if ref is not None else "  n/a"
        print(f"  {frac*100:>6.1f}%  {nc:>11.4f}  {oc:>9.4f}  "
              f"{enrg:>7.1f}%  {delta_str:>20}")

    print(f"\nPer-layer detail (neuron_cos / out_cos):")
    hdr = f"  {'layer':>5}" + "".join(
        f"  nc@{f*100:.0f}%  oc@{f*100:.0f}%" for f in args.hot_fractions)
    print(hdr)
    for i, li in enumerate(layers):
        row = f"  {li:5d}"
        for frac in args.hot_fractions:
            row += (f"  {frac_results[frac]['neuron_cos'][i]:8.4f}"
                    f"  {frac_results[frac]['out_cos'][i]:7.4f}")
        print(row)

    # Energy vs cosine similarity table to understand the routing quality
    print(f"\nEnergy fraction captured by hot neurons:")
    print(f"  {'hot%':>6}  {'energy%':>9}  {'neuron_cos':>11}  {'out_cos':>9}")
    for frac in args.hot_fractions:
        print(f"  {frac*100:>6.1f}%  "
              f"{np.mean(frac_energy[frac])*100:>8.1f}%  "
              f"{np.mean(frac_results[frac]['neuron_cos']):>11.4f}  "
              f"{np.mean(frac_results[frac]['out_cos']):>9.4f}")


if __name__ == "__main__":
    main()
