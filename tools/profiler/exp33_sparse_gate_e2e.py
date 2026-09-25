# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 33 – end-to-end top-1 with sparse W_gate routing (unstructured, kr=0.5+ft).

Motivation
----------
Exp32 showed that unstructured magnitude pruning (keep=0.5) + 300-step Adam
fine-tune gives F1=0.902 @20% hot vs E5M3's F1=0.784 — a 11.8 pp improvement
in routing quality.  This experiment measures whether that routing improvement
translates to better end-to-end top-1 match rate.

Scheme
------
  W_sparse[l] = magnitude_prune_unstructured(W_gate[l], keep=0.5)
             +  300 Adam steps (MSE vs x @ W_gate_full.T, mask fixed)

  routing:  hot = top-k(|x @ W_sparse.T|)
  hot:      gate_full  = x @ W_gate_full.T   (full precision)
            up_full    = x @ W_up_full.T     (full precision; always, as in exp24)
  cold:     gate_cold  = x @ W_sparse.T      (sparse approx, reusing routing signal)
  merge:    gate       = where(hot, gate_full, gate_cold)
  swiglu:   out        = silu(gate) * up_full
  down:     out        = swiglu @ W_down.T    (full precision)

Note: cold up is always full-precision, matching exp24's best configuration.
The sparse W_gate doubles as both routing signal and cold approximation value,
eliminating the E5M3 encoding step entirely.

Hot fractions swept: 0.20, 0.30, 0.50
Also sweeps keep_rates: 0.5 (with ft), 0.3 (with ft), 0.5 (no ft), 0.3 (no ft)
  to measure whether fine-tuning matters for e2e and whether 0.3 is competitive.

Reference points:
  exp24: E5M3 threshold T=0.20 → match=0.818 @88% hot
  exp24: hot=20%  → match not directly available (threshold; hot% varies)
  exp27: rank=1024 union  50% hot → 0.834
  exp14: ternary @30% hot → 0.632

Fine-tuning uses activations from ffn_activations128.npz (pre-recorded).

Usage::

    python tools/profiler/exp33_sparse_gate_e2e.py \\
        --model ibm-granite/granite-4.2-3b \\
        --calibration-set bartowski-imatrix-v5-semantic.txt \\
        --activations ffn_activations128.npz
"""

import argparse
import os
import sys
import time
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
I_DIM = 8192
EPS = 1e-9


# ---------------------------------------------------------------------------
# Sparsification
# ---------------------------------------------------------------------------

def magnitude_prune_unstructured(W: torch.Tensor, keep_rate: float) -> torch.Tensor:
    """Zero the (1-keep_rate) weights with smallest |w| element-wise."""
    flat   = W.abs().reshape(-1)
    k      = max(1, int(keep_rate * flat.numel()))
    thresh = flat.kthvalue(flat.numel() - k + 1).values
    return W * (W.abs() >= thresh)


def finetune_sparse(W_init: torch.Tensor,
                    mask: torch.Tensor,
                    X_tr: torch.Tensor,
                    Y_tr: torch.Tensor,
                    n_steps: int = 300,
                    lr: float = 1e-3,
                    batch_size: int = 512) -> torch.Tensor:
    """300 Adam steps on MSE(x @ W.T, x @ W_gate_full.T) with mask fixed."""
    W   = W_init.clone().requires_grad_(True)
    opt = torch.optim.Adam([W], lr=lr)
    N   = X_tr.shape[0]
    for _ in range(n_steps):
        idx  = torch.randint(0, N, (batch_size,), device=X_tr.device)
        pred = X_tr[idx] @ (W * mask).T
        loss = F.mse_loss(pred, Y_tr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            W.mul_(mask)
    return (W * mask).detach()


def build_sparse_gate(W_gate: torch.Tensor,
                      X: torch.Tensor,
                      keep_rate: float,
                      finetune_steps: int) -> torch.Tensor:
    """Return sparse W_gate (I, H) on CPU (float32).

    Fine-tuning is done on DEV (MPS/CUDA) for speed, then moved to CPU for
    use in the vLLM forward pass which runs on CPU on this machine.
    """
    W_u  = magnitude_prune_unstructured(W_gate, keep_rate)
    if finetune_steps <= 0:
        return W_u.cpu()
    # 80/20 train split on the recorded activations
    n_tr = int(X.shape[0] * 0.8)
    X_tr = X[:n_tr]
    with torch.no_grad():
        Y_tr = X_tr @ W_gate.T   # oracle targets
    mask = (W_u != 0).float()
    return finetune_sparse(W_u, mask, X_tr, Y_tr, n_steps=finetune_steps).cpu()


# ---------------------------------------------------------------------------
# Hybrid MLP wrapper
# ---------------------------------------------------------------------------

class SparseGateHybridMLP:
    """Top-k routing on |x @ W_sparse.T|; cold gate from sparse, up always full.

    Args:
        mlp:        the layer's MLP module
        W_sparse:   (I, H) pre-computed sparse W_gate
        k_hot:      number of hot channels per token
    """

    def __init__(self, mlp, W_sparse: torch.Tensor, k_hot: int):
        self._mlp     = mlp
        self._W_sp    = W_sparse    # (I, H) float32
        self._k_hot   = k_hot
        W_fused       = mlp.gate_up_proj.weight.detach().float()
        self._I       = W_fused.shape[0] // 2

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        xf = x.float()                        # (T, H)
        I  = self._I

        # Routing + cold approximation from sparse W_gate
        gate_sparse = xf @ self._W_sp.T       # (T, I)
        k = min(self._k_hot, I)
        idx = torch.topk(gate_sparse.abs(), k, dim=-1, sorted=False).indices
        hot = torch.zeros(xf.shape[0], I, dtype=torch.bool, device=xf.device)
        hot.scatter_(-1, idx, True)

        # Full-precision weights
        W_fused = self._mlp.gate_up_proj.weight.detach().float()
        W_gate  = W_fused[:I]
        W_up    = W_fused[I:]
        W_down  = self._mlp.down_proj.weight.detach().float()

        gate_full = xf @ W_gate.T             # (T, I)
        up_full   = xf @ W_up.T              # (T, I) — always full

        gate = torch.where(hot, gate_full, gate_sparse)
        swiglu = F.silu(gate) * up_full
        return (swiglu @ W_down.T).to(orig_dtype)


# ---------------------------------------------------------------------------
# NormCapture / e2e infra (identical across all e2e experiments)
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
        description="Exp33: sparse W_gate e2e top-1 assessment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model",           default="ibm-granite/granite-4.2-3b")
    p.add_argument("--calibration-set", default="bartowski-imatrix-v5-semantic.txt")
    p.add_argument("--activations",     default="ffn_activations128.npz")
    p.add_argument("--num-prompts",     type=int, default=8)
    p.add_argument(
        "--keep-rates", nargs="+", type=float, default=[0.5, 0.3],
        help="Keep rates to evaluate (default: 0.5 0.3).")
    p.add_argument(
        "--hot-fractions", nargs="+", type=float, default=[0.20, 0.30, 0.50],
        metavar="F",
        help="Hot channel fractions.")
    p.add_argument(
        "--finetune-steps", type=int, default=300,
        help="Adam fine-tune steps (0 to disable).")
    p.add_argument(
        "--no-finetune", action="store_true",
        help="Also run without fine-tuning for comparison.")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    args = _parse_args(argv)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    print(f"Device: {DEV}", file=sys.stderr)
    print(f"Keep rates: {args.keep_rates}  hot fractions: {args.hot_fractions}  "
          f"finetune_steps: {args.finetune_steps}", file=sys.stderr)

    data = np.load(args.activations)

    prompts = _load_prompts(args.calibration_set, args.num_prompts)
    print(f"Loaded {len(prompts)} prompts.", file=sys.stderr)

    from vllm import LLM
    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              kv_cache_memory_bytes=int(0.5 * 1024**3), max_model_len=512,
              enable_prefix_caching=False)

    norm, layers, W_U = _get_internals(llm)
    orig_forwards = [l.mlp.forward for l in layers]
    I = layers[0].mlp.gate_up_proj.weight.shape[0] // 2

    # ------------------------------------------------------------------
    # Pre-compute sparse W_gate for all (keep_rate, ft) combinations
    # ------------------------------------------------------------------
    # variants: list of (label, keep_rate, finetune_steps)
    variants = []
    for kr in args.keep_rates:
        variants.append((f"kr={kr:.1f}+ft", kr, args.finetune_steps))
        if args.no_finetune:
            variants.append((f"kr={kr:.1f}   ", kr, 0))

    # sparse_weights[label] = list of (I,H) tensors, one per layer
    sparse_weights: dict[str, list[torch.Tensor]] = {}

    for label, kr, ft_steps in variants:
        print(f"\nBuilding sparse W_gate  label={label} ...", file=sys.stderr)
        t0 = time.time()
        ws = []
        for li, layer in enumerate(layers):
            W_fused = layer.mlp.gate_up_proj.weight.detach().float()
            W_gate  = W_fused[:I].to(DEV)
            X       = torch.from_numpy(
                          data[f"layer{li}/gate_up_input"]).float().to(DEV)
            W_sp = build_sparse_gate(W_gate, X, kr, ft_steps)
            ws.append(W_sp)
            print(f"  layer {li:2d}", end="\r", file=sys.stderr, flush=True)
        sparse_weights[label] = ws
        print(f"  done  ({time.time()-t0:.0f}s)", file=sys.stderr)

    # ------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------
    print("\nBaseline pass...", file=sys.stderr)
    baseline = _run(llm, prompts, W_U, norm)
    n_tok = len(baseline)
    print(f"  {n_tok} prefill-token predictions captured.", file=sys.stderr)

    # References
    exp24_ref = {0.20: None, 0.30: 0.798, 0.50: 0.736}   # T=0.20 gives 0.818 @88% hot (adaptive)
    exp27_ref = {0.20: 0.612, 0.30: 0.702, 0.50: 0.834}
    exp14_ref = {0.20: 0.588, 0.30: 0.632, 0.50: None}

    # Results[label][frac] = match
    results: dict[str, dict[float, float]] = {lab: {} for lab, _, _ in variants}

    n_total = len(variants) * len(args.hot_fractions)
    idx = 0
    for label, kr, ft_steps in variants:
        for frac in args.hot_fractions:
            idx += 1
            k_hot   = max(1, int(frac * I))
            hybrids = [
                SparseGateHybridMLP(layers[li].mlp, sparse_weights[label][li], k_hot)
                for li in range(len(layers))
            ]
            for l, h in zip(layers, hybrids):
                l.mlp.forward = h

            print(f"  [{idx:2d}/{n_total}] {label}  hot={frac*100:.0f}% ...",
                  end="  ", file=sys.stderr, flush=True)
            ids   = _run(llm, prompts, W_U, norm)[:n_tok]
            match = float((ids == baseline).mean())
            results[label][frac] = match
            print(f"match={match:.4f}  perturb={1-match:.4f}", file=sys.stderr)

            for l, fwd in zip(layers, orig_forwards):
                l.mlp.forward = fwd
            del hybrids

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    fracs = args.hot_fractions
    hdr   = f"  {'scheme':<18}" + "".join(f"  {f*100:>5.0f}%" for f in fracs)

    print("\n" + "="*72)
    print("Exp33: sparse W_gate routing, full-precision up (matching exp24 scheme)")
    print("routing: top-k(|x @ W_sparse.T|)  cold gate: x @ W_sparse.T")
    print("="*72)
    print("\nTop-1 match rate (↑ better):")
    print(hdr)
    for label, _, _ in variants:
        row = f"  {label:<18}" + "".join(
            f"  {results[label][f]:>6.4f}" for f in fracs)
        print(row)
    # Reference rows
    print(f"  {'exp24 (thresh)':18}" +
          "".join(f"  {exp24_ref[f]:>6.4f}" if exp24_ref[f] is not None
                  else "     —    " for f in fracs))
    print(f"  {'exp27 r1024 union':18}" +
          "".join(f"  {exp27_ref.get(f, float('nan')):>6.4f}" for f in fracs))
    print(f"  {'exp14 ternary':18}" +
          "".join(f"  {exp14_ref[f]:>6.4f}" if exp14_ref.get(f) is not None
                  else "     —    " for f in fracs))

    print("\nΔ vs exp27 r1024 union (+ = exp33 better):")
    print(hdr)
    for label, _, _ in variants:
        row = f"  {label:<18}" + "".join(
            f"  {results[label][f] - exp27_ref.get(f, float('nan')):>+6.4f}"
            for f in fracs)
        print(row)
    print("="*72)

    print("\nNotes:")
    print("  exp24: threshold routing on E5M3 gate_approx; up always full precision")
    print("  exp27: low-rank SVD union routing; cold E5M3 gate+up")
    print("  exp33: sparse W_gate routing (top-k); cold gate from sparse, up full")
    for label, kr, ft_steps in variants:
        nnz    = int(kr * I_DIM * 2560)
        saving = (1.0 - kr) * 100
        print(f"  {label}: {nnz:>8d} non-zeros in W_gate  "
              f"({saving:.0f}% zeroed, {ft_steps} Adam ft steps)")


if __name__ == "__main__":
    main()
