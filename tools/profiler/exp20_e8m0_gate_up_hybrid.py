# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 20 – B=8 E8M0 sign encoding for both gate and up, hot-channel refinement.

Motivation
----------
Experiments 10–11 used a ternary gate proxy (α=0.75) with *full-precision* up.
The up projection (W_up @ x) was computed exactly for all channels on every
token, which is the dominant GEMM cost alongside W_gate.

Experiments 17–19 showed that B=8 E8M0 sign encoding achieves TARE=0.837 at
8.00× compression — the best quality-per-byte across all evaluated encodings.

This experiment applies B=8 E8M0 sign encoding to **both** gate and up
projections, then refines the hot channels of both to full precision using the
same proxy-prior routing from exp11.  The down projection always uses the full
SwiGLU vector (hot + cold contributions) at full precision.

Scheme
------
Pre-compute (once, stored):
  s_gate[b] = 2^round( tilt-weighted geometric mean of log2(|W_gate block b|) )
  s_up  [b] = 2^round( tilt-weighted geometric mean of log2(|W_up   block b|) )

Per token t:
  1. gate_approx[t] = sign(W_gate) * s_gate @ x[t]   B=8 E8M0 sign approx
  2. up_approx  [t] = sign(W_up)   * s_up   @ x[t]   B=8 E8M0 sign approx
  3. hot = top-k channels by |gate_approx[t-1]|       proxy-prior routing (free)
  4. gate_hybrid[hot]  = W_gate[hot,:] @ x[t]         full-precision gate, hot only
     gate_hybrid[cold] = gate_approx[t][cold]         encoded approx, cold
  5. up_hybrid[hot]    = W_up[hot,:]   @ x[t]         full-precision up, hot only
     up_hybrid[cold]   = up_approx[t][cold]           encoded approx, cold
  6. swiglu = SiLU(gate_hybrid) * up_hybrid           full hybrid SwiGLU
  7. out    = W_down @ swiglu                         full-precision down, all channels

Comparison baselines:
  exp10 current   α=0.75, full up, pre-SiLU routing    out cos-sim @10%: 0.476
  exp11 proxy     α=0.75, full up, proxy-prior routing  out cos-sim @10%: 0.600

Usage::

    python tools/profiler/exp20_e8m0_gate_up_hybrid.py \\
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
# Device
# ---------------------------------------------------------------------------

def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEV = _device()
BLOCK_SIZE = 8   # B=8 E8M0 sweet spot from exp19


# ---------------------------------------------------------------------------
# E8M0 sign encoding helpers
# ---------------------------------------------------------------------------

def _floor_eps(W: torch.Tensor, floor_percentile: float = 1.0) -> torch.Tensor:
    flat = W.abs().reshape(-1)
    k = max(1, int(len(flat) * floor_percentile / 100.0))
    return flat.kthvalue(k).values.clamp(min=1e-9)


def e8m0_optimal_scales(W: torch.Tensor, eps: torch.Tensor,
                         B: int = BLOCK_SIZE) -> torch.Tensor:
    """Per-block TARE-optimal E8M0 scales for W (O, I).

    Returns scales (O, ceil(I/B)) float32, values are powers of 2.
    """
    O, I = W.shape
    pad = (B - I % B) % B
    Wp = F.pad(W, (0, pad)) if pad else W
    W_b = Wp.reshape(-1, B)                           # (n_blocks_total, B)

    wa    = W_b.abs().clamp(min=eps)
    tilt  = torch.log1p(wa / eps)
    log2w = torch.log2(wa)
    log2_s = (tilt * log2w).sum(1) / tilt.sum(1).clamp(min=1e-9)
    e = log2_s.round().to(torch.int32)
    scales_f32 = (2.0 ** e.float()).clamp(min=1e-9)  # (n_blocks_total,)
    n_blocks_per_row = (I + B - 1) // B
    return scales_f32.reshape(O, n_blocks_per_row)    # (O, n_blocks_per_row)


def build_signed_weight(W: torch.Tensor,
                        scales: torch.Tensor,
                        B: int = BLOCK_SIZE) -> torch.Tensor:
    """Construct the encoded approximation W̃ = sign(W) * scale_per_block.

    Returns W_enc (O, I) float32 — the approximated weight matrix.
    Used to materialise gate_approx = W_enc @ x via a single GEMM.
    """
    O, I = W.shape
    pad = (B - I % B) % B
    n_blocks_per_row = scales.shape[1]
    # Expand scales: (O, n_blocks_per_row) → (O, n_blocks_per_row * B)
    s_exp = scales.unsqueeze(2).expand(O, n_blocks_per_row, B).reshape(O, n_blocks_per_row * B)
    Wp = F.pad(W, (0, pad)) if pad else W
    W_enc = Wp.sign() * s_exp                        # (O, I+pad)
    return W_enc[:, :I].contiguous()                 # (O, I)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_sim_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a, b, dim=-1).mean())


def top_k_mask(scores: torch.Tensor, k: int) -> torch.BoolTensor:
    idx = torch.topk(scores, k, dim=1, largest=True, sorted=False).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Experiment 20: B=8 E8M0 gate+up sign encoding with hot refinement.",
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
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    args = _parse_args(argv)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    print(f"Device: {DEV}  block size: {BLOCK_SIZE} (E8M0)", file=sys.stderr)

    act_data = np.load(args.act_file)
    layers = sorted(
        int(k.split("/")[0].replace("layer", ""))
        for k in act_data.files if k.endswith("/gate_raw")
    )
    if args.layers is not None:
        layers = [l for l in layers if l in args.layers]
    print(f"Evaluating {len(layers)} layers, "
          f"{len(args.hot_fractions)} hot-fractions.", file=sys.stderr)

    from vllm import LLM
    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              kv_cache_memory_bytes=int(0.5 * 1024**3), max_model_len=512)
    model = (llm.llm_engine.model_executor
             .driver_worker.worker.model_runner.model)

    # results[hot_frac] = list of per-layer cosine sims
    results: dict[float, list[float]] = {f: [] for f in args.hot_fractions}

    for layer_idx in layers:
        pfx = f"layer{layer_idx}"
        x_np    = act_data[f"{pfx}/gate_up_input"][:args.max_tokens].astype(np.float32)
        graw_np = act_data[f"{pfx}/gate_raw"]     [:args.max_tokens].astype(np.float32)

        mlp = model.model.layers[layer_idx].mlp
        W_fused = mlp.gate_up_proj.weight.detach().float()
        I = W_fused.shape[0] // 2
        W_gate = W_fused[:I].to(DEV)
        W_up   = W_fused[I:].to(DEV)
        W_down = mlp.down_proj.weight.detach().float().to(DEV)

        x        = torch.from_numpy(x_np).to(DEV)
        gate_raw = torch.from_numpy(graw_np).to(DEV)

        # ---- Build E8M0 sign-encoded weight matrices ----
        eps_gate = _floor_eps(W_gate)
        eps_up   = _floor_eps(W_up)
        scales_gate = e8m0_optimal_scales(W_gate, eps_gate)   # (I, ceil(H/B))
        scales_up   = e8m0_optimal_scales(W_up,   eps_up)
        W_gate_enc  = build_signed_weight(W_gate, scales_gate) # (I, H)
        W_up_enc    = build_signed_weight(W_up,   scales_up)   # (I, H)

        # ---- Full-precision references ----
        gate_full = x @ W_gate.T          # (T, I)
        up_full   = x @ W_up.T            # (T, I)
        swiglu_full = F.silu(gate_raw) * up_full
        out_full    = swiglu_full @ W_down.T   # (T, H)

        # ---- E8M0 approximations (full matrix, used for proxy prior + cold) ----
        gate_approx = x @ W_gate_enc.T    # (T, I)
        up_approx   = x @ W_up_enc.T      # (T, I)

        # Adjacent-token pairs: prior uses t-1, current is t
        T_pairs = x.shape[0] - 1
        gate_approx_prior = gate_approx[:T_pairs]   # routing signal (free)
        gate_approx_cold  = gate_approx[1:]         # cold gate values
        up_approx_cold    = up_approx  [1:]         # cold up values
        gate_full_hot_src = gate_full  [1:]         # for hot gate recompute
        up_full_hot_src   = up_full    [1:]         # for hot up recompute
        gate_raw_cur      = gate_raw   [1:]
        out_full_cur      = out_full   [1:]

        for frac in args.hot_fractions:
            k_hot = max(1, int(frac * I))

            # Route on prior-token |gate_approx| (proxy-prior, zero overhead)
            hot = top_k_mask(gate_approx_prior.abs(), k_hot)   # (T-1, I)

            # Hybrid gate: hot → full precision, cold → E8M0 approx
            gate_hybrid = torch.where(hot, gate_full_hot_src, gate_approx_cold)

            # Hybrid up: hot → full precision, cold → E8M0 approx
            up_hybrid = torch.where(hot, up_full_hot_src, up_approx_cold)

            # SwiGLU with hybrid gate and hybrid up
            # Note: gate_raw_cur is used as the full-precision gate reference;
            # for hot channels gate_hybrid == gate_full so this is consistent.
            swiglu_hybrid = F.silu(gate_hybrid) * up_hybrid

            # Full W_down on full SwiGLU vector (hot + cold contributions)
            out_hybrid = swiglu_hybrid @ W_down.T

            results[frac].append(cosine_sim_mean(out_full_cur, out_hybrid))

        print(
            f"  layer {layer_idx:3d}: "
            f"out@10%={results[0.10][-1]:.4f}  "
            f"out@5%={results[0.05][-1]:.4f}  "
            f"out@1%={results[0.01][-1]:.4f}",
            file=sys.stderr,
        )

        del W_gate, W_up, W_down, W_gate_enc, W_up_enc
        del gate_full, up_full, gate_approx, up_approx
        del swiglu_full, out_full
        if DEV.type == "mps":
            torch.mps.empty_cache()
        elif DEV.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n--- Experiment 20 Results ---\n")
    print("Scheme: B=8 E8M0 sign gate+up, proxy-prior routing, full W_down\n")

    # Reference values from exp11 proxy-prior, α=0.75
    exp11_ref = {0.005: 0.675, 0.01: 0.665, 0.02: 0.654,
                 0.05: 0.630, 0.10: 0.600, 0.20: 0.552, 0.30: 0.511}
    # Reference values from exp10 current, α=0.75
    exp10_ref = {0.005: 0.633, 0.01: 0.612, 0.02: 0.586,
                 0.05: 0.536, 0.10: 0.476, 0.20: 0.386, 0.30: 0.314}

    print(f"  {'hot%':>6}  {'exp20':>8}  {'exp11 proxy':>12}  "
          f"{'Δ vs exp11':>12}  {'exp10 curr':>12}  {'Δ vs exp10':>12}")
    for frac in args.hot_fractions:
        mean_v = float(np.mean(results[frac]))
        e11 = exp11_ref.get(frac)
        e10 = exp10_ref.get(frac)
        d11 = f"{mean_v - e11:+.4f}" if e11 else "         n/a"
        d10 = f"{mean_v - e10:+.4f}" if e10 else "         n/a"
        print(f"  {frac*100:>6.1f}%  {mean_v:>8.4f}  "
              f"{e11 if e11 else 'n/a':>12}  {d11:>12}  "
              f"{e10 if e10 else 'n/a':>12}  {d10:>12}")

    print("\nPer-layer detail at hot=10%:")
    print(f"  {'layer':>5}  {'exp20':>8}")
    for i, li in enumerate(layers):
        print(f"  {li:5d}  {results[0.10][i]:8.4f}")


if __name__ == "__main__":
    main()
