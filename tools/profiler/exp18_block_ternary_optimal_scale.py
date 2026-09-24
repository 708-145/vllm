# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 18 – block-ternary encoding with TARE-optimal per-block scales.

Motivation
----------
Experiment 17 used the block-maximum as the per-block scale and found TARE=1.376
for a pure sign (α=0) encoding.  The block-max over-estimates small weights
within each block, inflating the log-ratio error for all non-maximal elements.

For a sign encoding  w̃_i = sign(w_i) * s  the TARE loss w.r.t. s is:

    L(s) = Σ_i tilt_i * (log s - log|w_i|)²

which is a weighted least-squares problem in log-space.  The analytic minimiser
is the tilt-weighted geometric mean of the absolute weights per block:

    log s* = Σ_i tilt_i * log|w_i| / Σ_i tilt_i
    s*     = exp( tilt-weighted mean of log|w_i| )

where  tilt_i = log1p(|w_i| / eps)  and eps is the 1st-percentile of |W|
(the same tensor-level floor used in the TARE metric).

This experiment:
  1. Derives s* analytically for every block of B=16 in one vectorised pass.
  2. Stores s* as FP16 (same storage cost as exp17).
  3. Reports TARE for the optimal-scale sign encoding vs exp17's block-max baseline.
  4. Also sweeps α (ternary threshold = α * s*) to check whether sparsity helps
     once the scale is optimal.
  5. Compares three scale choices side-by-side:
       block-max     (exp17 baseline)
       block-rms     (simple closed-form alternative)
       TARE-optimal  (this experiment)

Storage is identical to exp17: 6 bytes per block of 16 (4 B codes + 2 B FP16 scale).

Usage::

    python tools/profiler/exp18_block_ternary_optimal_scale.py \\
        --model ibm-granite/granite-4.2-3b
"""

import argparse
import sys
from dataclasses import dataclass

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
BLOCK_SIZE = 16


# ---------------------------------------------------------------------------
# TARE metric  (same as exp17)
# ---------------------------------------------------------------------------

def _floor_eps(W: torch.Tensor, floor_percentile: float = 1.0) -> torch.Tensor:
    """Per-tensor floor eps: k-th value of |W| (avoids torch.quantile size limit)."""
    flat = W.abs().reshape(-1)
    k = max(1, int(len(flat) * floor_percentile / 100.0))
    return flat.kthvalue(k).values.clamp(min=1e-9)


def tare(w_true: torch.Tensor, w_approx: torch.Tensor,
         eps: torch.Tensor) -> float:
    """TARE given a pre-computed eps (avoids recomputing kthvalue per call)."""
    wt   = w_true.abs().clamp(min=eps)
    wa   = w_approx.abs().clamp(min=eps)
    lr   = torch.log(wa / wt)
    tilt = torch.log1p(w_true.abs() / eps)
    return float((lr.pow(2) * tilt).sum() / tilt.sum().clamp(min=1e-9)).real ** 0.5


# ---------------------------------------------------------------------------
# Per-block scale variants
# ---------------------------------------------------------------------------

def _blockwise(W: torch.Tensor) -> torch.Tensor:
    """Reshape W (O, I) → (n_blocks, B), padding if needed.

    Returns W_b (n_blocks_total, B) and the number of blocks per row.
    """
    O, I = W.shape
    B = BLOCK_SIZE
    pad = (B - I % B) % B
    if pad:
        W = F.pad(W, (0, pad))
    return W.view(-1, B), (I + B - 1) // B


def scale_block_max(W_b: torch.Tensor) -> torch.Tensor:
    """s = max(|w_b|) per block — exp17 baseline."""
    return W_b.abs().amax(dim=1).clamp(min=1e-9)


def scale_block_rms(W_b: torch.Tensor) -> torch.Tensor:
    """s = rms(|w_b|) per block — minimises MSE for sign encoding."""
    return W_b.pow(2).mean(dim=1).sqrt().clamp(min=1e-9)


def scale_block_tare_optimal(W_b: torch.Tensor,
                              eps: torch.Tensor) -> torch.Tensor:
    """s* = tilt-weighted geometric mean of |w_b| per block.

    Analytically minimises TARE for a pure sign encoding (α=0).

    Derivation:
        TARE ∝ Σ tilt_i * (log s - log|w_i|)²
        dL/d(log s) = 2 Σ tilt_i * (log s - log|w_i|) = 0
        log s* = Σ tilt_i * log|w_i| / Σ tilt_i
    """
    wa    = W_b.abs().clamp(min=eps)            # (n_blocks, B)
    tilt  = torch.log1p(wa / eps)               # (n_blocks, B)
    logw  = torch.log(wa)                       # (n_blocks, B)
    log_s = (tilt * logw).sum(dim=1) / tilt.sum(dim=1).clamp(min=1e-9)
    return log_s.exp().to(torch.float16).float().clamp(min=1e-9)


# ---------------------------------------------------------------------------
# Encode / decode with a given scale tensor
# ---------------------------------------------------------------------------

def encode_with_scale(
    W: torch.Tensor,       # (O, I) float32
    scales_f32: torch.Tensor,   # (n_blocks_total,) float32 — one per block
    alpha: float,
    I_orig: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode W using pre-computed per-block scales.

    Returns:
        codes:  (O, I_orig) int8  ∈ {-1, 0, +1}
        scales: (O, n_blocks_per_row) float16
    """
    O = W.shape[0]
    B = BLOCK_SIZE
    pad = (B - I_orig % B) % B
    if pad:
        W = F.pad(W, (0, pad))
    Ip = W.shape[1]
    n_blocks_per_row = Ip // B
    n_blocks_total   = O * n_blocks_per_row

    W_b  = W.view(n_blocks_total, B)
    tau  = (alpha * scales_f32).unsqueeze(1).expand_as(W_b)
    codes = W_b.sign() * (W_b.abs() >= tau).to(W_b.dtype)
    codes_i8 = codes.to(torch.int8).view(O, Ip)[:, :I_orig]

    # Store scales as FP16 (same cost as exp17)
    scales_f16 = scales_f32.to(torch.float16).view(O, n_blocks_per_row)
    return codes_i8, scales_f16


def decode_with_scale(
    codes: torch.Tensor,   # (O, I) int8
    scales: torch.Tensor,  # (O, n_blocks) float16
) -> torch.Tensor:
    O, I = codes.shape
    B = BLOCK_SIZE
    n_blocks = scales.shape[1]
    s = scales.float().unsqueeze(2).expand(O, n_blocks, B).reshape(O, n_blocks * B)
    return codes.float() * s[:, :I]


# ---------------------------------------------------------------------------
# Per-matrix analysis
# ---------------------------------------------------------------------------

@dataclass
class MatrixStats:
    name: str
    scale_type: str
    alpha: float
    zero_frac: float
    tare_score: float
    bytes_bf16: int
    bytes_encoded: int


def analyse_matrix(
    name: str,
    W: torch.Tensor,     # (O, I) float32 on DEV
    alphas: list[float],
    eps: torch.Tensor,   # pre-computed per-tensor floor
) -> list[MatrixStats]:
    O, I = W.shape
    B = BLOCK_SIZE
    n_blocks_per_row = (I + B - 1) // B
    n_blocks_total = O * n_blocks_per_row

    bytes_bf16    = O * I * 2
    bytes_encoded = n_blocks_total * (B // 4 + 2)  # 4 B codes + 2 B FP16 scale

    W_b, _ = _blockwise(W)   # (n_blocks_total, B)

    # Compute the three scale variants once
    s_max  = scale_block_max(W_b)
    s_rms  = scale_block_rms(W_b)
    s_tare = scale_block_tare_optimal(W_b, eps)

    results = []
    for scale_type, scales_f32 in [("block_max",  s_max),
                                    ("block_rms",  s_rms),
                                    ("tare_opt",   s_tare)]:
        for alpha in alphas:
            codes, scales_f16 = encode_with_scale(W, scales_f32, alpha, I)
            W_approx = decode_with_scale(codes, scales_f16)
            zero_frac = float((codes == 0).float().mean())
            score = tare(W, W_approx, eps)
            results.append(MatrixStats(
                name=name,
                scale_type=scale_type,
                alpha=alpha,
                zero_frac=zero_frac,
                tare_score=score,
                bytes_bf16=bytes_bf16,
                bytes_encoded=bytes_encoded,
            ))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Experiment 18: block-ternary with TARE-optimal per-block scales.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", default="ibm-granite/granite-4.2-3b")
    p.add_argument(
        "--alphas", nargs="+", type=float,
        default=[0.0, 0.25, 0.50, 0.75, 1.00],
        metavar="A",
    )
    p.add_argument(
        "--layers", nargs="+", type=int, default=None,
        metavar="L",
        help="Layer indices to analyse (default: all layers).",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    print(f"Device: {DEV}", file=sys.stderr)
    print(f"Block size: {BLOCK_SIZE}  alphas: {args.alphas}", file=sys.stderr)

    from transformers import AutoModelForCausalLM
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16)
    layers = hf_model.model.layers

    layer_indices = args.layers if args.layers is not None else list(range(len(layers)))
    print(f"Analysing {len(layer_indices)} layer(s)", file=sys.stderr)

    from collections import defaultdict
    # acc[proj_name][scale_type][alpha] = list of TARE scores across layers
    tare_acc: dict[str, dict[str, dict[float, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    zero_acc: dict[str, dict[str, dict[float, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    storage: dict[str, tuple[int, int]] = {}  # proj_name → (bf16, enc)

    proj_names = ["gate", "up", "down"]

    for li in layer_indices:
        layer = layers[li]
        mlp = layer.mlp
        weights = {
            "gate": mlp.gate_proj.weight.detach().float().to(DEV),
            "up":   mlp.up_proj.weight.detach().float().to(DEV),
            "down": mlp.down_proj.weight.detach().float().to(DEV),
        }
        for pname, W in weights.items():
            eps = _floor_eps(W)
            stats_list = analyse_matrix(pname, W, args.alphas, eps)
            if pname not in storage:
                storage[pname] = (stats_list[0].bytes_bf16,
                                  stats_list[0].bytes_encoded)
            for s in stats_list:
                tare_acc[pname][s.scale_type][s.alpha].append(s.tare_score)
                zero_acc[pname][s.scale_type][s.alpha].append(s.zero_frac)
        del weights

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    scale_types = ["block_max", "block_rms", "tare_opt"]
    scale_labels = {
        "block_max": "block-max (exp17)",
        "block_rms": "block-RMS         ",
        "tare_opt":  "TARE-optimal      ",
    }

    print("\n--- Experiment 18: Block-Ternary with TARE-Optimal Scale ---\n")
    print(f"  B={BLOCK_SIZE}, storage=6 B/block (4 B codes + 2 B FP16 scale)\n")

    for pname in proj_names:
        bf16_b, enc_b = storage[pname]
        print(f"  {pname}_proj  "
              f"BF16={bf16_b/1024**2:.1f} MiB  encoded={enc_b/1024**2:.1f} MiB  "
              f"ratio={bf16_b/enc_b:.2f}×")

        # Header
        alpha_hdr = "".join(f"  α={a:.2f}" for a in args.alphas)
        print(f"    {'scale':22}{alpha_hdr}")

        for st in scale_types:
            row = f"    {scale_labels[st]:<22}"
            best_alpha = min(args.alphas,
                             key=lambda a: sum(tare_acc[pname][st][a]))
            for alpha in args.alphas:
                mean_t = sum(tare_acc[pname][st][alpha]) / len(tare_acc[pname][st][alpha])
                marker = "*" if alpha == best_alpha else " "
                row += f"  {mean_t:.4f}{marker}"
            print(row)
        print()

    # Cross-scale comparison at α=0 (sign encoding — most relevant)
    print("  Sign encoding (α=0) TARE comparison across scale types:\n")
    print(f"  {'proj':<6}  {'block-max':>10}  {'block-RMS':>10}  "
          f"{'TARE-opt':>10}  {'Δ opt-vs-max':>13}  {'Δ opt-vs-rms':>13}")
    for pname in proj_names:
        t_max = sum(tare_acc[pname]["block_max"][0.0]) / len(tare_acc[pname]["block_max"][0.0])
        t_rms = sum(tare_acc[pname]["block_rms"][0.0]) / len(tare_acc[pname]["block_rms"][0.0])
        t_opt = sum(tare_acc[pname]["tare_opt"][0.0])  / len(tare_acc[pname]["tare_opt"][0.0])
        print(f"  {pname:<6}  {t_max:>10.4f}  {t_rms:>10.4f}  "
              f"{t_opt:>10.4f}  {t_opt-t_max:>+13.4f}  {t_opt-t_rms:>+13.4f}")


if __name__ == "__main__":
    main()
