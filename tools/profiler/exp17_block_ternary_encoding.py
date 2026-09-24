# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 17 – block-ternary weight encoding under the TARE metric.

Motivation
----------
Previous experiments used a global ternary threshold τ = α × mean(|W|) for
the gate predictor.  This experiment evaluates a structured block-ternary
encoding suitable for actual deployment: the weight matrix is divided into
contiguous blocks of 16 elements (along the input/column dimension), each
block gets its own FP16 scale derived from the block maximum, and the ternary
values are determined by a per-block threshold optimised to minimise the
scale-tilted log relative error (TARE).

This is a pure encoding experiment — no inference, no monkey-patching.
For each MLP weight matrix (gate, up, down) we report:

  • TARE score  (lower is better; 0 = exact)
  • Memory as encoded  (bytes)
  • Memory as BF16     (bytes, reference)
  • Compression ratio

Encoding
--------
For a weight matrix W of shape (O, I):

  1. Reshape to blocks of B=16: W_blocks shape (O * I // B, B)
  2. Per-block FP16 scale:  s = max(|w_b|).to(float16)  — one value per block
  3. Per-block threshold:   τ_b = α × s  (α swept; also α=0 = pure sign)
  4. Ternary codes:         t_b = sign(w_b) * (|w_b| >= τ_b)  ∈ {-1, 0, +1}
  5. Packed storage:        2 bits per element → B//4 = 4 bytes per block
     + 2 bytes FP16 scale = 6 bytes per block of 16 weights

  Approximate weight:  w̃_b = t_b * s_b
  (s_b is the block max, so the approximation over-estimates small weights —
  a per-block mean-abs or RMS scale would be more accurate and will be
  explored in a follow-up experiment.)

TARE (scale-Tilted Anchored Relative Error)
-------------------------------------------
  eps   = quantile(|W|, 1%)               anchored floor
  tilt  = log1p(|w| / eps)               per-element tilt weight
  TARE  = sqrt( Σ tilt * log²(|w̃|/|w|) / Σ tilt )

TARE treats sign-errors on large weights as much more costly than errors
on near-zero weights, and is scale-invariant (uses log ratio rather than
absolute difference).

Usage::

    python tools/profiler/exp17_block_ternary_encoding.py \\
        --model ibm-granite/granite-4.2-3b
"""

import argparse
import sys
from dataclasses import dataclass

import torch


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


# ---------------------------------------------------------------------------
# TARE metric
# ---------------------------------------------------------------------------

def tare(w_true: torch.Tensor, w_approx: torch.Tensor,
         floor_percentile: float = 1.0) -> float:
    """Scale-Tilted Anchored Relative Error (TARE).

    Args:
        w_true: Reference weight tensor (any shape, float32).
        w_approx: Approximated weight tensor, same shape.
        floor_percentile: Percentile of |w_true| used as the floor eps.

    Returns:
        Scalar TARE score (lower = better; 0 = exact reconstruction).
    """
    # torch.quantile fails for large tensors (>16M elements on MPS, CPU alike).
    # Approximate the floor percentile via kthvalue on a flat sorted tensor.
    flat = w_true.abs().reshape(-1)
    k = max(1, int(len(flat) * floor_percentile / 100.0))
    eps = flat.kthvalue(k).values.clamp(min=1e-9)
    wt    = w_true.abs().clamp(min=eps)
    wa    = w_approx.abs().clamp(min=eps)
    lr    = torch.log(wa / wt)
    tilt  = torch.log1p(w_true.abs() / eps)   # 0.69 at floor, ~7.3 at max
    mse_w = (lr.pow(2) * tilt).sum() / tilt.sum()
    return float(mse_w.sqrt())


# ---------------------------------------------------------------------------
# Block-ternary encoding
# ---------------------------------------------------------------------------

BLOCK_SIZE = 16   # elements per block (will be tuned later)


def encode_block_ternary(
    W: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode W (O, I) as block-ternary with FP16 per-block scales.

    Args:
        W: Weight matrix shape (O, I), float32.  I must be divisible by
           BLOCK_SIZE; if not, the last partial block is zero-padded.
        alpha: Threshold factor — ternary threshold per block = alpha * scale.
               alpha=0 collapses to pure sign (no zeros).

    Returns:
        codes:  int8 tensor shape (O, I) with values in {-1, 0, +1}.
                (Bit-packing to 2 bits/element is omitted here so the
                approximation can be reconstructed directly for quality
                measurement; production storage uses 4 bytes per block.)
        scales: float16 tensor shape (O, I // BLOCK_SIZE) — one scale per block.
    """
    O, I = W.shape
    B = BLOCK_SIZE
    # Pad I to a multiple of B if needed
    pad = (B - I % B) % B
    if pad:
        W = torch.nn.functional.pad(W, (0, pad))
    Ip = W.shape[1]
    n_blocks = Ip // B

    W_b = W.view(O * n_blocks, B)              # (n_blocks_total, B)

    # Per-block scale: max absolute value, stored as FP16
    block_max = W_b.abs().amax(dim=1)          # (n_blocks_total,)
    scales_f32 = block_max.clamp(min=1e-9)
    scales_f16 = scales_f32.to(torch.float16)  # FP16 storage
    scales_f32_rt = scales_f16.float()         # round-tripped through FP16

    # Per-block threshold: α × scale
    tau = alpha * scales_f32_rt                # (n_blocks_total,)

    # Ternary codes: sign(w) * (|w| >= tau)
    tau_b = tau.unsqueeze(1).expand_as(W_b)    # broadcast to (n_blocks_total, B)
    codes = W_b.sign() * (W_b.abs() >= tau_b).to(W_b.dtype)  # {-1, 0, +1}
    codes_i8 = codes.to(torch.int8).view(O, Ip)[:, :I]       # trim padding

    # scales back to (O, n_blocks_per_row) — trim padded columns if needed
    n_blocks_per_row = (I + B - 1) // B
    scales_f16_out = scales_f16.view(O, n_blocks)[:, :n_blocks_per_row]

    return codes_i8, scales_f16_out


def decode_block_ternary(
    codes: torch.Tensor,    # (O, I) int8
    scales: torch.Tensor,   # (O, I // BLOCK_SIZE) float16
) -> torch.Tensor:
    """Reconstruct float32 approximation from block-ternary codes + scales.

    w̃_b = codes_b * scale_b   (block max scale — upper bound on |w̃|)
    """
    O, I = codes.shape
    B = BLOCK_SIZE
    n_blocks = scales.shape[1]                       # blocks per row
    # Expand scales from (O, n_blocks) → (O, n_blocks * B) then trim
    s = scales.float().unsqueeze(2).expand(O, n_blocks, B).reshape(O, n_blocks * B)
    s = s[:, :I]                                     # trim to actual I
    return codes.float() * s                         # (O, I)


# ---------------------------------------------------------------------------
# Per-matrix stats
# ---------------------------------------------------------------------------

@dataclass
class MatrixStats:
    name: str
    shape: tuple[int, int]
    alpha: float
    zero_frac: float    # fraction of ternary zeros
    tare_score: float
    bytes_fp32: int
    bytes_bf16: int
    bytes_encoded: int  # 2 bits/elem + FP16 scale per block

    @property
    def ratio_vs_bf16(self) -> float:
        return self.bytes_bf16 / self.bytes_encoded

    def row(self) -> str:
        O, I = self.shape
        return (
            f"  {self.name:<28}  α={self.alpha:.2f}"
            f"  zero={self.zero_frac*100:5.1f}%"
            f"  TARE={self.tare_score:.4f}"
            f"  {self.bytes_bf16/1024**2:6.1f} MiB → {self.bytes_encoded/1024**2:5.1f} MiB"
            f"  ({self.ratio_vs_bf16:.2f}× vs BF16)"
        )


def analyse_matrix(
    name: str,
    W: torch.Tensor,     # (O, I) on DEV, float32
    alphas: list[float],
) -> list[MatrixStats]:
    O, I = W.shape
    B = BLOCK_SIZE
    n_blocks_per_row = (I + B - 1) // B

    # Storage sizes
    bytes_fp32    = O * I * 4
    bytes_bf16    = O * I * 2
    # 2 bits per weight element (packed) + 2 bytes FP16 scale per block
    bytes_encoded = O * n_blocks_per_row * (B // 4 + 2)

    results = []
    for alpha in alphas:
        codes, scales = encode_block_ternary(W, alpha)
        W_approx = decode_block_ternary(codes, scales)
        zero_frac = float((codes == 0).float().mean())
        score = tare(W, W_approx)
        results.append(MatrixStats(
            name=name,
            shape=(O, I),
            alpha=alpha,
            zero_frac=zero_frac,
            tare_score=score,
            bytes_fp32=bytes_fp32,
            bytes_bf16=bytes_bf16,
            bytes_encoded=bytes_encoded,
        ))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Experiment 17: block-ternary encoding quality under TARE.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", default="ibm-granite/granite-4.2-3b")
    p.add_argument(
        "--alphas", nargs="+", type=float,
        default=[0.0, 0.25, 0.50, 0.75, 1.00, 1.50],
        metavar="A",
        help="Threshold factors to sweep (default: 0.0 0.25 0.50 0.75 1.00 1.50).",
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
    print(f"Analysing {len(layer_indices)} layer(s): {layer_indices}", file=sys.stderr)

    # Collect stats across layers; average TARE and zero-fraction per (name, alpha)
    from collections import defaultdict
    tare_acc:  dict[tuple, list[float]] = defaultdict(list)
    zero_acc:  dict[tuple, list[float]] = defaultdict(list)
    # Record storage once (same shape every layer)
    storage: dict[str, tuple[int, int, int]] = {}   # name → (fp32, bf16, enc) bytes

    for li in layer_indices:
        layer = layers[li]
        mlp = layer.mlp
        # HuggingFace Granite uses separate gate_proj / up_proj / down_proj
        W_gate = mlp.gate_proj.weight.detach().float().to(DEV)
        W_up   = mlp.up_proj.weight.detach().float().to(DEV)
        W_down = mlp.down_proj.weight.detach().float().to(DEV)

        for proj_name, W in [("gate_up_proj/gate", W_gate),
                              ("gate_up_proj/up",   W_up),
                              ("down_proj",          W_down)]:
            stats_list = analyse_matrix(proj_name, W, args.alphas)
            if proj_name not in storage:
                s = stats_list[0]
                storage[proj_name] = (s.bytes_fp32, s.bytes_bf16, s.bytes_encoded)
            for s in stats_list:
                key = (proj_name, s.alpha)
                tare_acc[key].append(s.tare_score)
                zero_acc[key].append(s.zero_frac)

        del W_gate, W_up, W_down

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n--- Experiment 17: Block-Ternary Encoding Quality ---\n")
    print(f"  Block size B={BLOCK_SIZE}, scale dtype FP16")
    print(f"  Storage per block: {BLOCK_SIZE//4} B (codes, 2 bit/elem) + 2 B (FP16 scale)"
          f" = {BLOCK_SIZE//4 + 2} B  vs  {BLOCK_SIZE*2} B (BF16)  "
          f"or  {BLOCK_SIZE*4} B (FP32)")
    print()

    for proj_name in ["gate_up_proj/gate", "gate_up_proj/up", "down_proj"]:
        fp32_b, bf16_b, enc_b = storage[proj_name]
        ratio = bf16_b / enc_b
        print(f"  {proj_name:<28}  "
              f"BF16={bf16_b/1024**2:.1f} MiB  encoded={enc_b/1024**2:.1f} MiB  "
              f"ratio={ratio:.2f}×")
        print(f"  {'':28}  {'alpha':>6}  {'zero%':>6}  {'TARE':>8}")
        for alpha in args.alphas:
            key = (proj_name, alpha)
            mean_tare = float(torch.tensor(tare_acc[key]).mean())
            mean_zero = float(torch.tensor(zero_acc[key]).mean())
            marker = "  ← best" if alpha == args.alphas[
                min(range(len(args.alphas)),
                    key=lambda i: sum(tare_acc[(proj_name, args.alphas[i])]) )
            ] else ""
            print(f"  {'':28}  {alpha:>6.2f}  {mean_zero*100:>5.1f}%  "
                  f"{mean_tare:>8.4f}{marker}")
        print()

    # Best alpha per projection (minimum mean TARE)
    print("  Best α per projection (min mean TARE across layers):")
    for proj_name in ["gate_up_proj/gate", "gate_up_proj/up", "down_proj"]:
        best_alpha = min(args.alphas,
                         key=lambda a: sum(tare_acc[(proj_name, a)]))
        best_tare  = float(torch.tensor(tare_acc[(proj_name, best_alpha)]).mean())
        best_zero  = float(torch.tensor(zero_acc[(proj_name, best_alpha)]).mean())
        print(f"    {proj_name:<28}  α={best_alpha:.2f}  "
              f"zero={best_zero*100:.1f}%  TARE={best_tare:.4f}")


if __name__ == "__main__":
    main()
