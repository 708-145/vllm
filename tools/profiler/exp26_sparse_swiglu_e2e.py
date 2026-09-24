# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 26 – sparse SwiGLU: hot channels full precision, cold channels zero.

Motivation
----------
Exp23/24 used gate_approx as the cold-channel gate value fed into SiLU.
This experiment tests a cleaner alternative: cold channels are set to zero
entirely in the SwiGLU intermediate vector.  No approximation is used for
values — only for routing.

Scheme
------
1. gate_approx = x @ W_gate_enc.T          (E5M3 cheap pass, routing only)
2. hot = |gate_approx| > T * mean(|gate_approx|)
3. gate_full = x @ W_gate.T                (full precision, all channels)
4. up_full   = x @ W_up.T                  (full precision, all channels)
5. swiglu = silu(gate_full) * up_full * hot (cold channels zeroed)
6. out = swiglu @ W_down.T                 (full precision down projection)

Steps 3+4 still compute all channels — a future sparse GEMM kernel would
compute only the hot rows — but this measures the output quality of the scheme
as if that sparse kernel existed.

Compared to exp23/24, this removes the gate_approx cold-channel contribution
entirely.  From exp25b we know cold-channel gate_approx has SNR ~0–2 dB
(nearly noise), but SiLU suppresses it anyway.  Zeroing should be equivalent
or slightly better since it eliminates any residual cold-channel leakage.

Thresholds swept: same range as exp24 (T=0.20 to T=0.80).

Usage::

    python tools/profiler/exp26_sparse_swiglu_e2e.py \\
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
EPS = 1e-9


# ---------------------------------------------------------------------------
# E5M3 encoding (identical to exp22/23)
# ---------------------------------------------------------------------------

_M3_FRAC_LOG2 = torch.log2(1.0 + torch.arange(8).float() / 8.0)


def _floor_eps(W: torch.Tensor, pct: float = 1.0) -> torch.Tensor:
    flat = W.abs().reshape(-1)
    k = max(1, int(len(flat) * pct / 100.0))
    return flat.kthvalue(k).values.clamp(min=EPS)


def build_e5m3_encoded(W: torch.Tensor, B: int = BLOCK_SIZE) -> torch.Tensor:
    O, I = W.shape
    eps = _floor_eps(W)
    pad = (B - I % B) % B
    Wp  = F.pad(W, (0, pad)) if pad else W
    W_b = Wp.reshape(-1, B)

    wa     = W_b.abs().clamp(min=eps)
    tilt   = torch.log1p(wa / eps)
    log2_s = (tilt * torch.log2(wa)).sum(1) / tilt.sum(1).clamp(min=EPS)

    e        = log2_s.floor().to(torch.int32)
    frac     = log2_s - e.float()
    m_lut    = _M3_FRAC_LOG2.to(frac.device)
    m_best   = (frac.unsqueeze(1) - m_lut).abs().argmin(1)
    log2_s_q = e.float() + m_lut[m_best]
    scales   = (2.0 ** log2_s_q).clamp(min=EPS)

    n_blk  = Wp.shape[1] // B
    s_exp  = scales.reshape(O, n_blk).unsqueeze(2).expand(O, n_blk, B).reshape(O, Wp.shape[1])
    return (Wp.sign() * s_exp)[:, :I].contiguous()


# ---------------------------------------------------------------------------
# Sparse SwiGLU MLP
# ---------------------------------------------------------------------------

class SparseSwigluMLP:
    """Exp26: E5M3 routing, full-precision hot gate+up, cold channels zeroed.

    swiglu[cold] = 0  (not gate_approx — completely discarded)
    swiglu[hot]  = silu(gate_full[hot]) * up_full[hot]
    out          = swiglu @ W_down.T   (full precision)
    """

    def __init__(self, mlp, threshold_factor: float):
        self._mlp              = mlp
        self._threshold_factor = threshold_factor

        W_fused = mlp.gate_up_proj.weight.detach().float()
        I = W_fused.shape[0] // 2
        self._I          = I
        self._W_gate_enc = build_e5m3_encoded(W_fused[:I])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        xf = x.float()
        I  = self._I

        gate_approx = xf @ self._W_gate_enc.T              # (T, I) routing only
        thresh      = self._threshold_factor * gate_approx.abs().mean(dim=-1, keepdim=True)
        hot         = gate_approx.abs() > thresh            # (T, I)

        W_fused = self._mlp.gate_up_proj.weight.detach().float()
        W_gate  = W_fused[:I]
        W_up    = W_fused[I:]
        W_down  = self._mlp.down_proj.weight.detach().float()

        gate_full = xf @ W_gate.T                          # (T, I)
        up_full   = xf @ W_up.T                            # (T, I)
        swiglu    = F.silu(gate_full) * up_full * hot      # cold = 0
        out       = (swiglu @ W_down.T).to(orig_dtype)
        return out


# ---------------------------------------------------------------------------
# NormCapture / infra (shared with exp14/22/23/24)
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
        description="Exp26: sparse SwiGLU — hot full precision, cold zero.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", default="ibm-granite/granite-4.2-3b")
    p.add_argument("--calibration-set", default="bartowski-imatrix-v5-semantic.txt")
    p.add_argument("--num-prompts", type=int, default=8)
    p.add_argument(
        "--thresholds", nargs="+", type=float,
        default=[0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
        metavar="T",
        help="Threshold factors (hot = |gate_approx| > T * mean|gate_approx|).",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    args = _parse_args(argv)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    print(f"Device: {DEV}  B={BLOCK_SIZE} E5M3", file=sys.stderr)

    prompts = _load_prompts(args.calibration_set, args.num_prompts)
    print(f"Loaded {len(prompts)} prompts.", file=sys.stderr)

    from vllm import LLM
    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              kv_cache_memory_bytes=int(0.5 * 1024**3), max_model_len=512,
              enable_prefix_caching=False)

    norm, layers, W_U = _get_internals(llm)
    orig_forwards = [l.mlp.forward for l in layers]

    print("Baseline pass...", file=sys.stderr)
    baseline = _run(llm, prompts, W_U, norm)
    n_tok = len(baseline)
    print(f"  {n_tok} token predictions captured.", file=sys.stderr)

    # exp23/24 reference points for comparison
    exp23_ref = {0.20: 0.818, 0.30: 0.798, 0.40: 0.779, 0.50: 0.736,
                 0.60: 0.695, 0.70: 0.659, 0.80: 0.632}

    results: dict[float, float] = {}
    hot_fracs: dict[float, float] = {}

    print("\n=== Exp26: sparse SwiGLU (cold=zero) ===", file=sys.stderr)
    n = len(args.thresholds)
    for i, T in enumerate(args.thresholds):
        mlps = [SparseSwigluMLP(l.mlp, T) for l in layers]
        for l, m in zip(layers, mlps):
            l.mlp.forward = m

        print(f"  [{i+1}/{n}] T={T:.2f}...", end="  ", file=sys.stderr, flush=True)
        ids = _run(llm, prompts, W_U, norm)[:n_tok]
        match = float((ids == baseline).mean())
        results[T] = match

        # Estimate hot fraction from a single forward via the hook's gate_approx
        # We don't instrument hot% here; approximate from exp25b data.
        # Instead compute it analytically from a quick weight-only sample:
        hot_fracs[T] = 1.0 - T  # rough placeholder — will be overridden below

        print(f"match={match:.4f}  perturb={1-match:.4f}", file=sys.stderr)
        for l, fwd in zip(layers, orig_forwards):
            l.mlp.forward = fwd
        del mlps

    # ----------------------------------------------------------------
    # Compute actual hot% via a measurement pass with a hook
    # ----------------------------------------------------------------
    # Reuse exp25b-style measurement to get real hot fracs
    class HotFracHook:
        def __init__(self, mlp, T):
            W_fused = mlp.gate_up_proj.weight.detach().float()
            I = W_fused.shape[0] // 2
            self._I          = I
            self._W_gate_enc = build_e5m3_encoded(W_fused[:I])
            self._T          = T
            self.fracs: list[float] = []
            self._mlp = mlp

        def __call__(self, x):
            xf  = x.float()
            g   = xf @ self._W_gate_enc.T
            hot = g.abs() > self._T * g.abs().mean(dim=-1, keepdim=True)
            self.fracs.append(hot.float().mean().item())
            W_fused = self._mlp.gate_up_proj.weight.detach().float()
            I = self._I
            W_up   = W_fused[I:]
            W_down = self._mlp.down_proj.weight.detach().float()
            gate_f = xf @ W_fused[:I].T
            up_f   = xf @ W_up.T
            swiglu = F.silu(gate_f) * up_f
            return (swiglu @ W_down.T).to(x.dtype)

    print("\nMeasuring hot fractions...", file=sys.stderr)
    from vllm import SamplingParams
    for T in args.thresholds:
        hooks = [HotFracHook(l.mlp, T) for l in layers]
        for l, h in zip(layers, hooks):
            l.mlp.forward = h
        llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0.0),
                     use_tqdm=False)
        for l, fwd in zip(layers, orig_forwards):
            l.mlp.forward = fwd
        hot_fracs[T] = float(np.mean([f for h in hooks for f in h.fracs]))
        del hooks

    # ----------------------------------------------------------------
    # Summary table
    # ----------------------------------------------------------------
    print("\n" + "="*75)
    print("Exp26: sparse SwiGLU — cold channels zeroed")
    print("="*75)
    print(f"{'T':>5}  {'hot%':>6}  {'match':>7}  {'perturb':>8}  "
          f"{'exp24 match':>11}  {'Δ vs exp24':>10}")
    print("-"*75)
    for T in args.thresholds:
        m    = results[T]
        hf   = hot_fracs[T] * 100
        ref  = exp23_ref.get(T, float("nan"))
        delta = m - ref
        print(f"{T:>5.2f}  {hf:>6.1f}%  {m:>7.4f}  {1-m:>8.4f}  "
              f"{ref:>11.4f}  {delta:>+10.4f}")
    print("="*75)
    print()
    print("exp24 ref: E5M3 threshold, cold channels use gate_approx value")
    print("exp26:     E5M3 threshold, cold channels forced to zero")
    print("Δ > 0 means exp26 (zero cold) is better than exp24 (approx cold)")


if __name__ == "__main__":
    main()
