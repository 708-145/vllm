# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 19 – 1-bit sign encoding with E8M0 per-block scales, block-size sweep.

Motivation
----------
Experiment 18 showed that the TARE-optimal scale at α=0 (pure sign / 1-bit
encoding) is the best quality-per-bit scheme, beating ternary at α>0 under the
same storage budget.  The FP16 scale in exp17/18 costs 2 bytes per block,
which dominates at small block sizes.

This experiment replaces the FP16 scale with an **E8M0** scale: 8 exponent
bits, 0 mantissa bits — a power-of-two approximation s = 2^e stored in 1 byte.
The optimal E8M0 scale is derived by rounding the TARE-optimal log2(s*) to the
nearest integer:

    e* = round( log2(s*) )   =  round( tilt-weighted mean of log2(|w_b|) )
    s_e8m0 = 2^e*

This halves the scale overhead vs FP16, improving the compression ratio
especially at small block sizes.

Block sizes swept: B ∈ {8, 16, 32, 64}.

Storage per block (1-bit sign codes + 1-byte E8M0 scale):
    B=8  : 1 B codes + 1 B scale = 2 B / 8 weights  →  8.00× vs BF16
    B=16 : 2 B codes + 1 B scale = 3 B / 16 weights →  8.53× vs BF16
    B=32 : 4 B codes + 1 B scale = 5 B / 32 weights →  8.96× vs BF16 (projected)  [actually 10.24×]
    B=64 : 8 B codes + 1 B scale = 9 B / 64 weights →  ?

(Exact ratios computed at runtime.)

Comparison baselines included for context:
    FP16 scale, B=16, TARE-optimal (exp18 best) → TARE ≈ 0.828

Usage::

    python tools/profiler/exp19_e8m0_scale_block_sweep.py \\
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


# ---------------------------------------------------------------------------
# TARE helpers  (same floor logic as exp17/18)
# ---------------------------------------------------------------------------

def _floor_eps(W: torch.Tensor, floor_percentile: float = 1.0) -> torch.Tensor:
    flat = W.abs().reshape(-1)
    k = max(1, int(len(flat) * floor_percentile / 100.0))
    return flat.kthvalue(k).values.clamp(min=1e-9)


def tare(w_true: torch.Tensor, w_approx: torch.Tensor,
         eps: torch.Tensor) -> float:
    wt   = w_true.abs().clamp(min=eps)
    wa   = w_approx.abs().clamp(min=eps)
    lr   = torch.log(wa / wt)
    tilt = torch.log1p(w_true.abs() / eps)
    return float(((lr.pow(2) * tilt).sum() / tilt.sum().clamp(min=1e-9)).sqrt())


# ---------------------------------------------------------------------------
# Block reshaping
# ---------------------------------------------------------------------------

def _to_blocks(W: torch.Tensor, B: int) -> tuple[torch.Tensor, int]:
    """Return (W_b, pad) where W_b is (O * ceil(I/B), B), padded if needed."""
    O, I = W.shape
    pad = (B - I % B) % B
    Wp = F.pad(W, (0, pad)) if pad else W
    return Wp.reshape(-1, B), pad


# ---------------------------------------------------------------------------
# Scale computation
# ---------------------------------------------------------------------------

def scale_e8m0_optimal(W_b: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """E8M0 TARE-optimal scale per block.

    1. Compute tilt-weighted geometric mean of |w_b| (same as exp18).
    2. Round log2(s*) to nearest integer → E8M0 exponent.
    3. Return 2^e as float32 (1-byte E8M0 simulated via integer rounding).
    """
    wa    = W_b.abs().clamp(min=eps)            # (n_blocks, B)
    tilt  = torch.log1p(wa / eps)
    log2w = torch.log2(wa)
    log2_s_opt = (tilt * log2w).sum(1) / tilt.sum(1).clamp(min=1e-9)  # (n_blocks,)
    e = log2_s_opt.round().to(torch.int32)      # E8M0 exponent (integer)
    return (2.0 ** e.float()).clamp(min=1e-9)   # (n_blocks,) float32


def scale_fp16_optimal(W_b: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """FP16 TARE-optimal scale — exp18 reference (for comparison at B=16)."""
    wa    = W_b.abs().clamp(min=eps)
    tilt  = torch.log1p(wa / eps)
    logw  = torch.log(wa)
    log_s = (tilt * logw).sum(1) / tilt.sum(1).clamp(min=1e-9)
    return log_s.exp().to(torch.float16).float().clamp(min=1e-9)


# ---------------------------------------------------------------------------
# Encode / decode (1-bit sign + per-block scale)
# ---------------------------------------------------------------------------

def encode_sign(W: torch.Tensor, scales_f32: torch.Tensor,
                B: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sign encoding with given per-block scales.

    Returns:
        codes:  (O, I) int8 ∈ {-1, +1}   (all elements encoded, α=0)
        scales: (O, n_blocks) float32     (caller converts to storage dtype)
    """
    O, I = W.shape
    W_b, pad = _to_blocks(W, B)
    Ip = I + pad
    n_blocks_per_row = Ip // B
    n_blocks_total   = O * n_blocks_per_row

    codes_b = W_b.sign()                                   # (n_total, B)
    codes   = codes_b.to(torch.int8).reshape(O, Ip)[:, :I]

    scales_out = scales_f32.reshape(O, n_blocks_per_row)
    return codes, scales_out


def decode_sign(codes: torch.Tensor,
                scales: torch.Tensor, B: int) -> torch.Tensor:
    O, I = codes.shape
    n_blocks = scales.shape[1]
    s = scales.float().unsqueeze(2).expand(O, n_blocks, B).reshape(O, n_blocks * B)
    return codes.float() * s[:, :I]


# ---------------------------------------------------------------------------
# Per-matrix analysis
# ---------------------------------------------------------------------------

@dataclass
class MatrixStats:
    name:          str
    block_size:    int
    scale_type:    str   # "e8m0_opt" | "fp16_opt"
    zero_frac:     float
    tare_score:    float
    bytes_bf16:    int
    bytes_encoded: int

    @property
    def compression(self) -> float:
        return self.bytes_bf16 / self.bytes_encoded


def _storage(O: int, I: int, B: int, scale_bytes: int) -> int:
    """Encoded bytes: ceil(I/B) blocks per row, each block = B//8 codes + scale."""
    n_blocks_per_row = (I + B - 1) // B
    return O * n_blocks_per_row * (B // 8 + scale_bytes)


def analyse_matrix(
    name:   str,
    W:      torch.Tensor,    # (O, I) float32 on DEV
    blocks: list[int],
    eps:    torch.Tensor,
) -> list[MatrixStats]:
    O, I = W.shape
    results = []

    for B in blocks:
        W_b, _ = _to_blocks(W, B)

        s_e8m0 = scale_e8m0_optimal(W_b, eps)     # 1-byte E8M0 per block
        s_fp16 = scale_fp16_optimal(W_b, eps) if B == 16 else None

        for scale_type, scales_f32, scale_bytes in [
            ("e8m0_opt", s_e8m0, 1),
            *([("fp16_opt", s_fp16, 2)] if B == 16 else []),
        ]:
            codes, scales_out = encode_sign(W, scales_f32, B)
            W_approx = decode_sign(codes, scales_out, B)
            score = tare(W, W_approx, eps)
            results.append(MatrixStats(
                name=name,
                block_size=B,
                scale_type=scale_type,
                zero_frac=0.0,   # α=0: no zeros
                tare_score=score,
                bytes_bf16=O * I * 2,
                bytes_encoded=_storage(O, I, B, scale_bytes),
            ))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Experiment 19: 1-bit sign + E8M0 scale, block-size sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", default="ibm-granite/granite-4.2-3b")
    p.add_argument(
        "--blocks", nargs="+", type=int, default=[8, 16, 32, 64],
        metavar="B",
    )
    p.add_argument(
        "--layers", nargs="+", type=int, default=None,
        metavar="L",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    print(f"Device: {DEV}  block sizes: {args.blocks}", file=sys.stderr)

    from transformers import AutoModelForCausalLM
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16)
    layers = hf_model.model.layers

    layer_indices = (args.layers if args.layers is not None
                     else list(range(len(layers))))
    print(f"Analysing {len(layer_indices)} layer(s)", file=sys.stderr)

    from collections import defaultdict
    # acc[proj][block_size][scale_type] = list of TARE scores
    tare_acc: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    storage:  dict = {}   # (proj, block_size, scale_type) → (bf16, enc)

    for li in layer_indices:
        mlp = layers[li].mlp
        weights = {
            "gate": mlp.gate_proj.weight.detach().float().to(DEV),
            "up":   mlp.up_proj.weight.detach().float().to(DEV),
            "down": mlp.down_proj.weight.detach().float().to(DEV),
        }
        for pname, W in weights.items():
            eps = _floor_eps(W)
            for s in analyse_matrix(pname, W, args.blocks, eps):
                key = (pname, s.block_size, s.scale_type)
                tare_acc[pname][s.block_size][s.scale_type].append(s.tare_score)
                if key not in storage:
                    storage[key] = (s.bytes_bf16, s.bytes_encoded)
        del weights

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print("\n--- Experiment 19: 1-bit sign + E8M0 scale, block-size sweep ---\n")
    print("  α=0 throughout (pure sign encoding, no zeros)\n")

    # Per-projection table
    for pname in ["gate", "up", "down"]:
        print(f"  {pname}_proj:")
        print(f"    {'B':>4}  {'scale':>10}  {'MiB enc':>8}  {'ratio':>7}  {'TARE':>8}")
        for B in args.blocks:
            for st in (["e8m0_opt", "fp16_opt"] if B == 16 else ["e8m0_opt"]):
                key = (pname, B, st)
                if key not in storage:
                    continue
                bf16_b, enc_b = storage[key]
                scores = tare_acc[pname][B][st]
                mean_t = sum(scores) / len(scores)
                ratio  = bf16_b / enc_b
                print(f"    {B:>4}  {st:>10}  "
                      f"{enc_b/1024**2:>8.2f}  {ratio:>6.2f}×  {mean_t:>8.4f}")
        print()

    # Summary: best TARE per block size across projections (average gate/up/down)
    print("  Mean TARE across gate/up/down projections:")
    print(f"    {'B':>4}  {'scale':>10}  {'ratio':>7}  {'TARE':>8}")
    for B in args.blocks:
        for st in (["e8m0_opt", "fp16_opt"] if B == 16 else ["e8m0_opt"]):
            all_scores = []
            for pname in ["gate", "up", "down"]:
                all_scores.extend(tare_acc[pname][B][st])
            if not all_scores:
                continue
            # ratio from first proj (same shape)
            key = ("gate", B, st)
            bf16_b, enc_b = storage[key]
            mean_t = sum(all_scores) / len(all_scores)
            ratio  = bf16_b / enc_b
            ref = " ← exp18 FP16 reference" if st == "fp16_opt" else ""
            print(f"    {B:>4}  {st:>10}  {ratio:>6.2f}×  {mean_t:>8.4f}{ref}")


if __name__ == "__main__":
    main()
