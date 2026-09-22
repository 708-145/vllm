# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 4 – is |gate_approx| a reliable routing signal?

The sign predictor gives gate_approx = sign(W_gate) @ x.  Experiment 2 used
|gate_approx| > thresh as the hot/cold routing criterion and found SwiGLU
cosine similarity collapsed immediately.  This script diagnoses *why* by
measuring:

  1. Spearman rank correlation between |gate_approx| and |gate_full| per layer
     — does the cheap magnitude predict the true magnitude well?
  2. For each channel-token pair, which quadrant does it fall into:
       A  correct-sign AND |gate_approx| large   (hot, correctly classified)
       B  wrong-sign   AND |gate_approx| large   (hot, mis-classified — BAD)
       C  correct-sign AND |gate_approx| small   (cold, correctly classified)
       D  wrong-sign   AND |gate_approx| small   (cold, mis-classified — but SiLU≈0 so safe)
  3. Given the quadrant breakdown, quantify how much SwiGLU error comes from
     each quadrant when channels in B are left as cold (wrong sign, large
     |gate_approx| — the dangerous ones).

Usage::

    python tools/profiler/exp4_gate_approx_predictor.py \\
        --model ibm-granite/granite-4.2-3b \\
        --act-file ffn_activations128_gate.npz
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr


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
    idx = torch.topk(scores, k, dim=1, largest=True, sorted=False).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Experiment 4: |gate_approx| as routing signal.",
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
        help="Hot-channel fractions to evaluate (top-F of |gate_approx|).",
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

    spearman_vals: list[float] = []
    quad_A, quad_B, quad_C, quad_D = [], [], [], []
    frac_results: dict[float, dict[str, list]] = {
        f: {"swiglu": [], "down": []} for f in args.hot_fractions
    }

    for layer_idx in layers:
        pfx = f"layer{layer_idx}"
        x_np        = act_data[f"{pfx}/gate_up_input"][: args.max_tokens].astype(np.float32)
        gate_raw_np = act_data[f"{pfx}/gate_raw"][: args.max_tokens].astype(np.float32)

        W_gate_np, W_up_np, W_down_np = get_weights(model, layer_idx)

        x        = t(x_np)
        gate_raw = t(gate_raw_np)
        W_gate   = t(W_gate_np)
        W_up     = t(W_up_np)
        W_down   = t(W_down_np)
        S_gate   = W_gate.sign()
        S_up     = W_up.sign()
        S_down   = W_down.sign()
        residual_W = W_down - S_down
        I = W_gate.shape[0]

        gate_approx = x @ S_gate.T        # (T, I)
        gate_full   = x @ W_gate.T        # (T, I)
        up_full     = x @ W_up.T          # (T, I)
        up_approx   = x @ S_up.T          # (T, I)
        swiglu_full = silu_t(gate_raw) * up_full
        down_full   = swiglu_full @ W_down.T

        # 1. Spearman rank correlation (pull to CPU once for scipy)
        abs_approx_cpu = gate_approx.abs().cpu().numpy().ravel()
        abs_full_cpu   = gate_full.abs().cpu().numpy().ravel()
        rho, _ = spearmanr(abs_approx_cpu, abs_full_cpu)
        spearman_vals.append(float(rho))

        # 2. Quadrant analysis (stays on accelerator)
        correct_sign  = gate_approx.sign() == gate_full.sign()
        median_approx = float(gate_approx.abs().median())
        large_approx  = gate_approx.abs() > median_approx
        A = float(( correct_sign &  large_approx).float().mean())
        B = float((~correct_sign &  large_approx).float().mean())
        C = float(( correct_sign & ~large_approx).float().mean())
        D = float((~correct_sign & ~large_approx).float().mean())
        quad_A.append(A); quad_B.append(B)
        quad_C.append(C); quad_D.append(D)

        # 3. Hybrid scheme using top-F |gate_approx| as hot mask
        for frac in args.hot_fractions:
            k_hot = max(1, int(frac * I))
            hot   = top_k_mask(gate_approx.abs(), k_hot)

            gate_hybrid   = torch.where(hot, gate_full,  gate_approx)
            up_hybrid     = torch.where(hot, up_full,    up_approx)
            swiglu_hybrid = silu_t(gate_hybrid) * up_hybrid
            down_hybrid   = (swiglu_hybrid @ S_down.T
                             + (swiglu_hybrid * hot.float()) @ residual_W.T)

            frac_results[frac]["swiglu"].append(
                cosine_sim_mean(swiglu_full, swiglu_hybrid))
            frac_results[frac]["down"].append(
                cosine_sim_mean(down_full, down_hybrid))

        print(
            f"  layer {layer_idx:3d}: spearman={spearman_vals[-1]:.3f}  "
            f"quadA={A:.3f} B={B:.3f} C={C:.3f} D={D:.3f}",
            file=sys.stderr,
        )

        del x, gate_raw, W_gate, W_up, W_down, S_gate, S_up, S_down, residual_W
        del gate_approx, gate_full, up_full, up_approx, swiglu_full, down_full
        if DEV.type == "mps":
            torch.mps.empty_cache()
        elif DEV.type == "cuda":
            torch.cuda.empty_cache()

    print("\n--- Experiment 4 Results ---")
    print(f"\nSpearman rank-corr( |gate_approx|, |gate_full| ):")
    print(f"  mean={np.mean(spearman_vals):.4f}  "
          f"min={np.min(spearman_vals):.4f}  max={np.max(spearman_vals):.4f}")

    print(f"\nQuadrant fractions (median of |gate_approx| as large/small split):")
    print(f"  A correct-sign & large-approx : {np.mean(quad_A):.3f}")
    print(f"  B wrong-sign   & large-approx : {np.mean(quad_B):.3f}  ← mis-routed hot (dangerous)")
    print(f"  C correct-sign & small-approx : {np.mean(quad_C):.3f}")
    print(f"  D wrong-sign   & small-approx : {np.mean(quad_D):.3f}  ← mis-routed cold (safe: SiLU≈0)")

    print(f"\nHybrid scheme (top-F |gate_approx| as hot) — mean cos-sim across layers:")
    print(f"  {'hot%':>5}  {'SwiGLU':>8}  {'down':>8}")
    for frac in args.hot_fractions:
        r = frac_results[frac]
        print(f"  {frac*100:5.0f}%  {np.mean(r['swiglu']):8.4f}  {np.mean(r['down']):8.4f}")

    print("\nPer-layer Spearman:")
    for i, li in enumerate(layers):
        print(f"  layer {li:3d}: {spearman_vals[i]:.4f}")


if __name__ == "__main__":
    main()
