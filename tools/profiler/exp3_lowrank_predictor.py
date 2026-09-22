# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 3 – low-rank SVD gate predictor vs sign predictor.

Represents the gate projection as a rank-r approximation W_gate ≈ U_r S_r V_r^T
(truncated SVD) and evaluates whether it gives a better gate predictor than
the sign matrix used in experiments 1–2.

Metrics per layer:
  1. Gate cosine similarity vs gate_raw — same metric as experiments 1–2.
  2. Sign agreement frac(sign(gate_approx) == sign(gate_full)) — routing
     accuracy that determines hybrid-scheme quality.

Also reports FLOP cost per token relative to the full GEMM.

Usage::

    python tools/profiler/exp3_lowrank_predictor.py \\
        --model ibm-granite/granite-4.2-3b \\
        --act-file ffn_activations128_gate.npz

    # Evaluate only specific ranks and layers:
    python tools/profiler/exp3_lowrank_predictor.py \\
        --ranks 16 64 256 \\
        --layers 0 10 20 30 39
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


def get_weights(model, layer_idx: int):
    """Return (W_gate, W_up, W_down) as float32 numpy arrays."""
    mlp = model.model.layers[layer_idx].mlp
    W = mlp.gate_up_proj.weight.detach().float().numpy()
    I = W.shape[0] // 2
    return W[:I], W[I:], mlp.down_proj.weight.detach().float().numpy()


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Experiment 3: low-rank SVD gate predictor.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", default="ibm-granite/granite-4.2-3b")
    p.add_argument("--act-file", default="ffn_activations128_gate.npz")
    p.add_argument("--layers", nargs="*", type=int, default=None,
                   help="Layer indices to evaluate (default: all).")
    p.add_argument("--max-tokens", type=int, default=2000)
    p.add_argument(
        "--ranks", nargs="+", type=int,
        default=[4, 16, 64, 256, 1024],
        metavar="R",
        help="SVD ranks to evaluate (default: 4 16 64 256 1024).",
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
    print(f"Evaluating {len(layers)} layer(s), ranks {args.ranks}.", file=sys.stderr)

    from vllm import LLM
    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              kv_cache_memory_bytes=int(0.5 * 1024**3), max_model_len=512)
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model

    # Results: rank -> {cos_sim: [], sign_agree: []}
    rank_results: dict[int, dict[str, list]] = {
        r: {"cos_sim": [], "sign_agree": []} for r in args.ranks
    }
    sign_results = {"cos_sim": [], "sign_agree": []}

    # Spectral energy fractions: rank -> list across layers
    energy_fracs: dict[int, list[float]] = {r: [] for r in args.ranks}

    for layer_idx in layers:
        pfx = f"layer{layer_idx}"
        x_np        = act_data[f"{pfx}/gate_up_input"][: args.max_tokens].astype(np.float32)
        gate_raw_np = act_data[f"{pfx}/gate_raw"][: args.max_tokens].astype(np.float32)

        W_gate_np, _, _ = get_weights(model, layer_idx)

        # SVD on CPU (numpy) — full SVD is expensive; we do it once per layer
        U, s, Vt = np.linalg.svd(W_gate_np, full_matrices=False)
        # s shape: (min(I, H),) = (2560,) for this model
        total_energy = float((s ** 2).sum())

        # Move activations and sign-weight matmul to accelerator
        x        = t(x_np)
        gate_raw = t(gate_raw_np)
        S_gate   = t(np.sign(W_gate_np))
        W_gate   = t(W_gate_np)

        gate_full   = x @ W_gate.T   # (T, I)  — reference full precision
        gate_approx_sign = x @ S_gate.T  # (T, I) — sign predictor

        # Sign predictor metrics
        sign_results["cos_sim"].append(cosine_sim_mean(gate_raw, gate_approx_sign))
        sign_results["sign_agree"].append(
            float((gate_approx_sign.sign() == gate_full.sign()).float().mean()))

        # Low-rank metrics for each rank
        for r in args.ranks:
            r_eff = min(r, len(s))
            # Reconstruct low-rank weight on CPU, then move to accelerator
            W_lr_np = (U[:, :r_eff] * s[:r_eff]) @ Vt[:r_eff]   # (I, H)
            W_lr = t(W_lr_np)
            gate_lr = x @ W_lr.T   # (T, I)

            rank_results[r]["cos_sim"].append(cosine_sim_mean(gate_raw, gate_lr))
            rank_results[r]["sign_agree"].append(
                float((gate_lr.sign() == gate_full.sign()).float().mean()))

            energy_frac = float((s[:r_eff] ** 2).sum()) / total_energy
            energy_fracs[r].append(energy_frac)

            del W_lr, gate_lr

        cos_r64 = rank_results[64]["cos_sim"][-1] if 64 in args.ranks else float("nan")
        print(
            f"  layer {layer_idx:3d}: "
            f"sign cos={sign_results['cos_sim'][-1]:.4f}  "
            f"sign_agree={sign_results['sign_agree'][-1]:.4f}  "
            f"r=64 cos={cos_r64:.4f}",
            file=sys.stderr,
        )

        del x, gate_raw, S_gate, W_gate, gate_full, gate_approx_sign
        if DEV.type == "mps":
            torch.mps.empty_cache()
        elif DEV.type == "cuda":
            torch.cuda.empty_cache()

    # --- Summary ---
    print("\n--- Experiment 3 Results ---")

    H, I_size = 2560, 8192
    full_flops = 2 * H * I_size

    print(f"\nGate cosine similarity vs gate_raw (mean ± over {len(layers)} layers):")
    print(f"  {'Method':>15}  {'Mean':>7}  {'Min':>7}  {'Max':>7}  "
          f"{'FLOPs':>10}  {'vs full':>8}  {'Energy%':>8}")
    print(f"  {'Sign':>15}  "
          f"{np.mean(sign_results['cos_sim']):7.4f}  "
          f"{np.min(sign_results['cos_sim']):7.4f}  "
          f"{np.max(sign_results['cos_sim']):7.4f}  "
          f"{'~'+str(full_flops//1_000_000)+'M adds':>10}  "
          f"{'≈0.1-0.2x':>8}  "
          f"{'100%':>8}")
    for r in args.ranks:
        flops = 2 * (H * r + r * I_size)
        ratio = flops / full_flops
        energy_pct = np.mean(energy_fracs[r]) * 100
        print(f"  {'Rank-'+str(r):>15}  "
              f"{np.mean(rank_results[r]['cos_sim']):7.4f}  "
              f"{np.min(rank_results[r]['cos_sim']):7.4f}  "
              f"{np.max(rank_results[r]['cos_sim']):7.4f}  "
              f"{flops:>10,}  "
              f"{ratio:>8.3f}x  "
              f"{energy_pct:>7.1f}%")
    print(f"  {'Full GEMM':>15}  {'1.0000':>7}  {'1.0000':>7}  {'1.0000':>7}  "
          f"{full_flops:>10,}  {'1.000x':>8}  {'100%':>8}")

    print(f"\nSign agreement frac(sign(approx)==sign(full)) — routing accuracy:")
    print(f"  {'Sign':>15}  {np.mean(sign_results['sign_agree']):.4f}")
    for r in args.ranks:
        print(f"  {'Rank-'+str(r):>15}  {np.mean(rank_results[r]['sign_agree']):.4f}")

    print(f"\nPer-layer detail:")
    hdr = f"  {'layer':>5}  {'sign_cos':>9}  {'sign_agr':>9}" + "".join(
        f"  r{r}_cos  r{r}_agr" for r in args.ranks)
    print(hdr)
    for i, li in enumerate(layers):
        row = (f"  {li:5d}  "
               f"{sign_results['cos_sim'][i]:9.4f}  "
               f"{sign_results['sign_agree'][i]:9.4f}")
        for r in args.ranks:
            row += (f"  {rank_results[r]['cos_sim'][i]:7.4f}"
                    f"  {rank_results[r]['sign_agree'][i]:7.4f}")
        print(row)


if __name__ == "__main__":
    main()
