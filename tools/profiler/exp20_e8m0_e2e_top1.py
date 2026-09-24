# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 20 – E2E top-1 perturbation: B=8 E8M0 sign gate+up, hot refinement.

Motivation
----------
Exp14 measured end-to-end top-1 perturbation for the ternary gate + full-precision
up scheme (exp10/11) and found 48% of token predictions change at 10% hot channels.

This experiment replaces both gate AND up projections with B=8 E8M0 sign encoding
(the sweet spot from exp19: 8.00× compression, TARE=0.837), then refines hot
channels of both to full precision.  Down projection always runs at full precision
on the complete SwiGLU vector (hot + cold contributions).

Scheme (per token t, all 40 layers simultaneously)
------
Pre-computed (stored per layer, ~7.5 MiB each for gate and up):
  W_gate_enc = sign(W_gate) * s_gate   E8M0 sign encoding, B=8
  W_up_enc   = sign(W_up)   * s_up

Per token:
  1. gate_approx = W_gate_enc @ x        cheap sign-scaled GEMM
  2. up_approx   = W_up_enc   @ x        cheap sign-scaled GEMM
  3. hot = top-k by |gate_approx[t-1]|   proxy-prior routing (zero overhead)
  4. gate_hybrid[hot]  = W_gate[hot] @ x full-precision gate, hot rows only
     gate_hybrid[cold] = gate_approx[cold]
  5. up_hybrid[hot]    = W_up[hot]   @ x full-precision up, hot rows only
     up_hybrid[cold]   = up_approx[cold]
  6. swiglu = SiLU(gate_hybrid) * up_hybrid
  7. out    = W_down @ swiglu             full-precision, full vector

Comparison targets:
  exp14 at 10% hot (ternary gate α=0.75 + full up):   48% perturbation / 52% match
  exp14 at 20% hot:                                    41% perturbation / 59% match
  exp14 at 30% hot:                                    37% perturbation / 63% match

Usage::

    python tools/profiler/exp20_e8m0_e2e_top1.py \\
        --model ibm-granite/granite-4.2-3b \\
        --calibration-set bartowski-imatrix-v5-semantic.txt
"""

import argparse
import os
import sys
from pathlib import Path

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
BLOCK_SIZE = 8


# ---------------------------------------------------------------------------
# E8M0 sign encoding
# ---------------------------------------------------------------------------

def _floor_eps(W: torch.Tensor, pct: float = 1.0) -> torch.Tensor:
    flat = W.abs().reshape(-1)
    k = max(1, int(len(flat) * pct / 100.0))
    return flat.kthvalue(k).values.clamp(min=1e-9)


def build_e8m0_encoded(W: torch.Tensor, B: int = BLOCK_SIZE) -> torch.Tensor:
    """Return W_enc (O, I) float32: sign(W) * E8M0_optimal_scale_per_block.

    Scale per block = 2^round(tilt-weighted geometric mean of log2(|w_b|)).
    Stored in the same dtype as W for fast GEMM; actual storage would be
    1-bit codes + 1-byte exponent (8× compression vs BF16).
    """
    O, I = W.shape
    eps = _floor_eps(W)
    pad = (B - I % B) % B
    Wp  = F.pad(W, (0, pad)) if pad else W
    W_b = Wp.reshape(-1, B)                              # (n_blocks, B)

    wa     = W_b.abs().clamp(min=eps)
    tilt   = torch.log1p(wa / eps)
    log2w  = torch.log2(wa)
    log2_s = (tilt * log2w).sum(1) / tilt.sum(1).clamp(min=1e-9)
    e      = log2_s.round().to(torch.int32)
    scales = (2.0 ** e.float()).clamp(min=1e-9)          # (n_blocks,)

    n_blocks_per_row = Wp.shape[1] // B
    s_exp = scales.reshape(O, n_blocks_per_row).unsqueeze(2) \
                  .expand(O, n_blocks_per_row, B) \
                  .reshape(O, Wp.shape[1])                # (O, I+pad)
    W_enc = Wp.sign() * s_exp
    return W_enc[:, :I].contiguous()


# ---------------------------------------------------------------------------
# top-k hot mask
# ---------------------------------------------------------------------------

def top_k_mask(scores: torch.Tensor, k: int) -> torch.BoolTensor:
    idx  = torch.topk(scores, k, dim=-1, largest=True, sorted=False).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    return mask


# ---------------------------------------------------------------------------
# HybridMLP  (replaces GraniteMLP.forward)
# ---------------------------------------------------------------------------

class HybridMLP:
    """E8M0 sign encoding for gate and up; full-precision hot-channel refinement.

    Stores W_gate_enc and W_up_enc (float32, same shape as original weights)
    so the cheap pass is a single matmul per projection.  Full-precision weights
    are read from the model at call time to avoid doubling memory.

    Proxy-prior routing: the prior token's gate_approx (already computed last
    step) selects hot channels with zero extra GEMMs on the critical path.
    """

    def __init__(self, mlp, k_hot: int):
        self._mlp   = mlp
        self._k_hot = k_hot
        self._prior: torch.Tensor | None = None   # gate_approx[t-1]

        W_fused = mlp.gate_up_proj.weight.detach().float()
        I = W_fused.shape[0] // 2
        self._I = I
        # Pre-compute encoded matrices (materialised as float32 for GEMM)
        self._W_gate_enc = build_e8m0_encoded(W_fused[:I]).to(W_fused.device)
        self._W_up_enc   = build_e8m0_encoded(W_fused[I:]).to(W_fused.device)

    def reset_prior(self) -> None:
        self._prior = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        xf = x.float()
        I  = self._I

        # Cheap encoded pass — two sign-scaled GEMMs
        gate_approx = xf @ self._W_gate_enc.T    # (T, I)
        up_approx   = xf @ self._W_up_enc.T      # (T, I)

        # Routing: proxy-prior if available, else current (first token of seq)
        signal = self._prior.abs() if self._prior is not None else gate_approx.abs()
        hot    = top_k_mask(signal, self._k_hot)  # (T, I)

        # Hot-channel full-precision recompute — read weights from model (no copy)
        W_fused = self._mlp.gate_up_proj.weight.detach().float()  # (2I, H)
        W_gate  = W_fused[:I]
        W_up    = W_fused[I:]
        gate_full = xf @ W_gate.T   # (T, I)
        up_full   = xf @ W_up.T     # (T, I)

        # Hybrid: hot → full precision, cold → encoded approx
        gate_hybrid = torch.where(hot, gate_full,  gate_approx)
        up_hybrid   = torch.where(hot, up_full,    up_approx)

        swiglu = F.silu(gate_hybrid) * up_hybrid  # (T, I)

        W_down = self._mlp.down_proj.weight.detach().float()  # (H, I)
        out    = (swiglu @ W_down.T).to(orig_dtype)

        # Save gate_approx as proxy prior for the next token
        self._prior = gate_approx.detach()
        return out


# ---------------------------------------------------------------------------
# Norm-hook capture  (identical to exp14/15)
# ---------------------------------------------------------------------------

class NormCapture:
    def __init__(self, W_U: torch.Tensor):
        self._W_U = W_U.float()
        self.ids: list[int] = []
        self._handle = None

    def attach(self, norm_module) -> None:
        self._handle = norm_module.register_forward_hook(self._hook)

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _hook(self, module, args, output) -> None:
        self.ids.extend(
            (output.float() @ self._W_U.T).argmax(dim=-1).cpu().tolist())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_prompts(path: str, n: int) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip()][:n]


def _get_norm(llm):
    """Return model.model.norm regardless of single-process vs worker path."""
    e = llm.llm_engine
    # V1 single-process path (VLLM_ENABLE_V1_MULTIPROCESSING=0)
    try:
        return e.model_executor.driver_worker.worker.model_runner.model.model.norm
    except AttributeError:
        pass
    # Older / alternate path
    return e.model_executor.driver_worker.model_runner.model.model.norm


def _get_layers(llm):
    e = llm.llm_engine
    try:
        return e.model_executor.driver_worker.worker.model_runner.model.model.layers
    except AttributeError:
        return e.model_executor.driver_worker.model_runner.model.model.layers


def _get_W_U(llm) -> torch.Tensor:
    e = llm.llm_engine
    try:
        return e.model_executor.driver_worker.worker.model_runner.model.lm_head.weight.detach().float()
    except AttributeError:
        return e.model_executor.driver_worker.model_runner.model.lm_head.weight.detach().float()


def _run(llm, prompts: list[str], W_U: torch.Tensor) -> np.ndarray:
    from vllm import SamplingParams
    cap = NormCapture(W_U)
    cap.attach(_get_norm(llm))
    llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0.0),
                 use_tqdm=False)
    cap.detach()
    ids = cap.ids
    n_decode = len(prompts)
    return np.array(ids[:-n_decode] if n_decode < len(ids) else ids,
                    dtype=np.int32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Experiment 20: E2E top-1 perturbation, B=8 E8M0 gate+up hybrid.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", default="ibm-granite/granite-4.2-3b")
    p.add_argument("--calibration-set", default="bartowski-imatrix-v5-semantic.txt")
    p.add_argument("--num-prompts", type=int, default=8)
    p.add_argument(
        "--hot-fractions", nargs="+", type=float,
        default=[0.05, 0.10, 0.20, 0.30],
        metavar="F",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    args = _parse_args(argv)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    print(f"Device: {DEV}  B={BLOCK_SIZE} E8M0  "
          f"hot fractions: {args.hot_fractions}", file=sys.stderr)

    prompts = _load_prompts(args.calibration_set, args.num_prompts)
    print(f"Loaded {len(prompts)} prompts.", file=sys.stderr)

    from vllm import LLM
    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              kv_cache_memory_bytes=int(0.5 * 1024**3), max_model_len=512,
              enable_prefix_caching=False)

    W_U    = _get_W_U(llm)
    layers = _get_layers(llm)
    I      = layers[0].mlp.gate_up_proj.weight.shape[0] // 2

    orig_forwards = [layer.mlp.forward for layer in layers]

    # Baseline pass
    print("Baseline pass...", file=sys.stderr)
    baseline = _run(llm, prompts, W_U)
    n_tok = len(baseline)
    print(f"  {n_tok} prefill-token predictions captured.", file=sys.stderr)

    results: dict[float, float] = {}
    n_passes = len(args.hot_fractions)

    for pass_idx, frac in enumerate(args.hot_fractions, 1):
        k_hot = max(1, int(frac * I))

        print(f"  [{pass_idx}/{n_passes}] hot={frac*100:.0f}%  "
              f"encoding E8M0 weights...", file=sys.stderr, flush=True)

        hybrids: list[HybridMLP] = []
        for layer in layers:
            h = HybridMLP(mlp=layer.mlp, k_hot=k_hot)
            hybrids.append(h)
            layer.mlp.forward = h

        for h in hybrids:
            h.reset_prior()

        print(f"  [{pass_idx}/{n_passes}] hot={frac*100:.0f}%  running...",
              end="  ", file=sys.stderr, flush=True)
        hybrid_ids = _run(llm, prompts, W_U)[:n_tok]
        match   = float((hybrid_ids == baseline).mean())
        perturb = 1.0 - match
        results[frac] = match
        print(f"match={match:.4f}  perturb={perturb:.4f}", file=sys.stderr)

        for layer, fwd in zip(layers, orig_forwards):
            layer.mlp.forward = fwd
        del hybrids

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n--- Experiment 20 Results ---\n")
    print("Scheme: B=8 E8M0 sign gate+up, proxy-prior routing, full W_down\n")

    # exp14 references (ternary gate α=0.75 + full up)
    exp14_match = {0.05: 0.456, 0.10: 0.516, 0.20: 0.588, 0.30: 0.632}

    print(f"  {'hot%':>6}  {'match':>8}  {'perturb':>9}  "
          f"{'exp14 match':>12}  {'Δ vs exp14':>12}")
    for frac in args.hot_fractions:
        match   = results[frac]
        perturb = 1.0 - match
        e14     = exp14_match.get(frac)
        delta   = f"{match - e14:+.4f}" if e14 is not None else "         n/a"
        e14_str = f"{e14:.4f}" if e14 is not None else "        n/a"
        print(f"  {frac*100:>6.1f}%  {match:>8.4f}  {perturb:>9.4f}  "
              f"{e14_str:>12}  {delta:>12}")

    print("\nEnd-to-end top-1 match rate  (↑ better, 1.0 = identical to baseline):")
    hdr = "  " + "".join(f"  {f*100:>5.0f}%" for f in args.hot_fractions)
    print(hdr)
    row = "  exp20" + "".join(f"  {results[f]:>6.4f}" for f in args.hot_fractions)
    print(row)
    row = "  exp14" + "".join(
        f"  {exp14_match[f]:>6.4f}" if f in exp14_match else "     n/a"
        for f in args.hot_fractions)
    print(row)


if __name__ == "__main__":
    main()
