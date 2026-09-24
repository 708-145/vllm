# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 21 – E2E top-1: B=8 E8M0 gate only, full-precision up + down.

Motivation
----------
Exp20 showed that encoding W_up cold channels alongside the gate is too costly
(match 0.194 vs 0.516 at 10% hot).  The correct split is:

  W_gate  → B=8 E8M0 sign encoding  (8× compression, TARE=0.837)
  W_up    → full precision
  W_down  → full precision

This is the same weight split as exp14 (ternary gate α=0.75 + full up), but
with the B=8 E8M0-optimal gate encoding replacing the global-α ternary proxy.
The E8M0 encoding uses a per-block TARE-optimal power-of-two scale, which
should produce a better gate approximation than the global α=0.75 ternary
(lower TARE: 0.837 vs 0.856 from exp14's ternary scheme).

Both routing variants from exp11 are tested:
  current    : route on |gate_approx[t]|     (exp10-style)
  proxy_prior: route on |gate_approx[t-1]|   (exp11-style, zero overhead)

Usage::

    python tools/profiler/exp21_e8m0_gate_only_e2e.py \\
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
# B=8 E8M0-optimal sign encoding
# ---------------------------------------------------------------------------

def _floor_eps(W: torch.Tensor, pct: float = 1.0) -> torch.Tensor:
    flat = W.abs().reshape(-1)
    k = max(1, int(len(flat) * pct / 100.0))
    return flat.kthvalue(k).values.clamp(min=1e-9)


def build_e8m0_encoded(W: torch.Tensor, B: int = BLOCK_SIZE) -> torch.Tensor:
    """W_enc (O, I) float32 = sign(W) * E8M0_optimal_scale_per_block.

    Logically stored as 1-bit codes + 1-byte exponent per block (8× vs BF16);
    materialised here as float32 for direct GEMM use.
    """
    O, I = W.shape
    eps = _floor_eps(W)
    pad = (B - I % B) % B
    Wp  = F.pad(W, (0, pad)) if pad else W
    W_b = Wp.reshape(-1, B)

    wa     = W_b.abs().clamp(min=eps)
    tilt   = torch.log1p(wa / eps)
    log2_s = (tilt * torch.log2(wa)).sum(1) / tilt.sum(1).clamp(min=1e-9)
    scales = (2.0 ** log2_s.round().to(torch.int32).float()).clamp(min=1e-9)

    n_blk  = Wp.shape[1] // B
    s_exp  = scales.reshape(O, n_blk).unsqueeze(2).expand(O, n_blk, B).reshape(O, Wp.shape[1])
    return (Wp.sign() * s_exp)[:, :I].contiguous()


# ---------------------------------------------------------------------------
# top-k hot mask
# ---------------------------------------------------------------------------

def top_k_mask(scores: torch.Tensor, k: int) -> torch.BoolTensor:
    idx  = torch.topk(scores, k, dim=-1, largest=True, sorted=False).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    return mask


# ---------------------------------------------------------------------------
# HybridMLP
# ---------------------------------------------------------------------------

class HybridMLP:
    """B=8 E8M0 gate encoding, full-precision up+down, hot gate refinement.

    Stores only W_gate_enc (float32).  Full-precision weights are read from
    the model at call time to avoid doubling memory.
    """

    def __init__(self, mlp, k_hot: int, routing: str):
        self._mlp    = mlp
        self._k_hot  = k_hot
        self._routing = routing          # "current" | "proxy_prior"
        self._prior: torch.Tensor | None = None

        W_fused = mlp.gate_up_proj.weight.detach().float()
        I = W_fused.shape[0] // 2
        self._I          = I
        self._W_gate_enc = build_e8m0_encoded(W_fused[:I])

    def reset_prior(self) -> None:
        self._prior = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        xf = x.float()
        I  = self._I

        # Cheap gate pass (E8M0 encoded)
        gate_approx = xf @ self._W_gate_enc.T        # (T, I)

        # Routing signal
        if self._routing == "proxy_prior" and self._prior is not None:
            signal = self._prior.abs()
        else:
            signal = gate_approx.abs()

        hot = top_k_mask(signal, self._k_hot)         # (T, I)

        # Full-precision weights from model (no copy stored)
        W_fused = self._mlp.gate_up_proj.weight.detach().float()
        W_gate  = W_fused[:I]
        W_up    = W_fused[I:]
        W_down  = self._mlp.down_proj.weight.detach().float()

        gate_full = xf @ W_gate.T                     # (T, I)
        up_full   = xf @ W_up.T                       # (T, I) — always exact

        # Hybrid gate: hot → full precision, cold → E8M0 approx
        gate_hybrid   = torch.where(hot, gate_full, gate_approx)
        swiglu_hybrid = F.silu(gate_hybrid) * up_full  # full up always
        out = (swiglu_hybrid @ W_down.T).to(orig_dtype)

        self._prior = gate_approx.detach()
        return out


# ---------------------------------------------------------------------------
# NormCapture / helpers  (identical to exp14/15/20)
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


def _load_prompts(path: str, n: int) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip()][:n]


def _get_internals(llm):
    e = llm.llm_engine
    try:
        mr = e.model_executor.driver_worker.worker.model_runner
    except AttributeError:
        mr = e.model_executor.driver_worker.model_runner
    return mr.model.model.norm, mr.model.model.layers, mr.model.lm_head.weight


def _run(llm, prompts, W_U, norm) -> np.ndarray:
    from vllm import SamplingParams
    cap = NormCapture(W_U.detach().float())
    cap.attach(norm)
    llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0.0),
                 use_tqdm=False)
    cap.detach()
    ids = cap.ids
    n_dec = len(prompts)
    return np.array(ids[:-n_dec] if n_dec < len(ids) else ids, dtype=np.int32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Exp21: E2E top-1, B=8 E8M0 gate only, full up+down.",
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
    print(f"Device: {DEV}  B={BLOCK_SIZE} E8M0 gate only  "
          f"hot: {args.hot_fractions}", file=sys.stderr)

    prompts = _load_prompts(args.calibration_set, args.num_prompts)
    print(f"Loaded {len(prompts)} prompts.", file=sys.stderr)

    from vllm import LLM
    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              kv_cache_memory_bytes=int(0.5 * 1024**3), max_model_len=512,
              enable_prefix_caching=False)

    norm, layers, W_U = _get_internals(llm)
    I = layers[0].mlp.gate_up_proj.weight.shape[0] // 2
    orig_forwards = [l.mlp.forward for l in layers]

    # Baseline
    print("Baseline pass...", file=sys.stderr)
    baseline = _run(llm, prompts, W_U, norm)
    n_tok = len(baseline)
    print(f"  {n_tok} prefill-token predictions captured.", file=sys.stderr)

    configs   = ["current", "proxy_prior"]
    results:  dict[str, dict[float, float]] = {c: {} for c in configs}
    n_passes  = len(configs) * len(args.hot_fractions)
    pass_idx  = 0

    for cfg in configs:
        for frac in args.hot_fractions:
            pass_idx += 1
            k_hot = max(1, int(frac * I))

            hybrids = []
            for layer in layers:
                h = HybridMLP(mlp=layer.mlp, k_hot=k_hot, routing=cfg)
                hybrids.append(h)
                layer.mlp.forward = h
            for h in hybrids:
                h.reset_prior()

            print(f"  [{pass_idx}/{n_passes}] {cfg}  hot={frac*100:.0f}%...",
                  end="  ", file=sys.stderr, flush=True)
            hybrid_ids = _run(llm, prompts, W_U, norm)[:n_tok]
            match = float((hybrid_ids == baseline).mean())
            results[cfg][frac] = match
            print(f"match={match:.4f}  perturb={1-match:.4f}", file=sys.stderr)

            for layer, fwd in zip(layers, orig_forwards):
                layer.mlp.forward = fwd
            del hybrids

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n--- Experiment 21 Results ---\n")
    print("Scheme: B=8 E8M0 gate, full up+down\n")

    # Reference values from exp14 (ternary gate α=0.75 + full up)
    exp14 = {0.05: 0.456, 0.10: 0.516, 0.20: 0.588, 0.30: 0.632}

    hdr = (f"  {'config':<14}"
           + "".join(f"  {f*100:>5.0f}%" for f in args.hot_fractions))
    print("End-to-end top-1 match rate  (↑ better):")
    print(hdr)
    for cfg in configs:
        row = f"  {cfg:<14}" + "".join(
            f"  {results[cfg][f]:>6.4f}" for f in args.hot_fractions)
        print(row)
    row = f"  {'exp14 ref':<14}" + "".join(
        f"  {exp14.get(f, float('nan')):>6.4f}" for f in args.hot_fractions)
    print(row)

    print("\nΔ match vs exp14 reference  (+ = exp21 better):")
    print(hdr)
    for cfg in configs:
        row = f"  {cfg:<14}" + "".join(
            f"  {results[cfg][f] - exp14.get(f, float('nan')):>+6.4f}"
            for f in args.hot_fractions)
        print(row)


if __name__ == "__main__":
    main()
