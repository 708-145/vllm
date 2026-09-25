# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 34 – thermal match rate for exp33 best scheme and key references.

Motivation
----------
All prior e2e experiments report strict top-1 match (temperature=0 greedy).
This experiment introduces the thermal match rate: a perturbation is forgiven
when the full-precision top-1 token is still "thermally accessible" in the
hybrid logits — i.e. the hybrid's preference gap over the correct token is
below T·ln(2), meaning the correct token would still win >33% of pairwise
draws at temperature T.

  gap       = lh[hybrid_top1] − lh[baseline_top1]   (≥ 0 when perturbed)
  forgiven  = gap < T · ln(2)
  thermal_match(T) = mean(exact_match OR forgiven)

This experiment re-runs the top schemes from the log and reports:
  - strict_match      (= current metric, T=0)
  - thermal_match_07  (T=0.7, τ=ln2  — typical chat)
  - thermal_match_10  (T=1.0, τ=ln2  — canonical sampling)
  - mean_gap          (mean logit gap among perturbed tokens — distribution insight)

Schemes evaluated:
  A. exp33 best: sparse W_gate kr=0.5+ft, hot=20%/30%/50%  (new best)
  B. exp24 ref:  E5M3 gate, threshold T=0.20  (best prior single-pass scheme)
  C. exp27 ref:  SVD rank=1024 union, hot=50%  (best prior at 50% hot)
  D. Baseline:   full precision (sanity: should give strict_match=1.0)

Usage::

    python tools/profiler/exp34_thermal_match.py \\
        --model ibm-granite/granite-4.2-3b \\
        --calibration-set bartowski-imatrix-v5-semantic.txt \\
        --activations ffn_activations128.npz
"""

import argparse
import math
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
BLOCK_SIZE = 8
I_DIM = 8192
EPS = 1e-9
_M3_FRAC_LOG2 = torch.log2(1.0 + torch.arange(8).float() / 8.0)
LN2 = math.log(2)


# ---------------------------------------------------------------------------
# Thermal match metric
# ---------------------------------------------------------------------------

def thermal_match(
    lf: torch.Tensor,
    lh: torch.Tensor,
    temperature: float = 0.7,
    tau: float = LN2,
) -> tuple[float, float, float]:
    """Return (strict_match, thermal_match, mean_gap_among_perturbed).

    Args:
        lf:          (T_seq, vocab) full-precision logits
        lh:          (T_seq, vocab) hybrid logits
        temperature: inference temperature (default 0.7)
        tau:         noise-floor multiplier (default ln2)

    Returns:
        strict_match:   fraction where argmax(lf) == argmax(lh)
        thermal_match:  fraction where exact OR gap < temperature * tau
        mean_gap:       mean logit gap among perturbed (non-exact) tokens
    """
    baseline_top1 = lf.argmax(-1)                                   # (T_seq,)
    hybrid_top1   = lh.argmax(-1)                                   # (T_seq,)
    exact_match   = (baseline_top1 == hybrid_top1)                  # (T_seq,) bool

    hybrid_rank1_logit = lh.gather(
        -1, hybrid_top1.unsqueeze(-1)).squeeze(-1)                  # (T_seq,)
    baseline_in_hybrid = lh.gather(
        -1, baseline_top1.unsqueeze(-1)).squeeze(-1)                # (T_seq,)

    gap = hybrid_rank1_logit - baseline_in_hybrid                   # (T_seq,) ≥ 0

    forgiven      = (~exact_match) & (gap < temperature * tau)
    strict        = float(exact_match.float().mean())
    thermal       = float((exact_match | forgiven).float().mean())

    perturbed_gaps = gap[~exact_match]
    mean_gap = float(perturbed_gaps.mean()) if perturbed_gaps.numel() > 0 else 0.0

    return strict, thermal, mean_gap


# ---------------------------------------------------------------------------
# E5M3 encoding (identical to exp22–27)
# ---------------------------------------------------------------------------

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
    e      = log2_s.floor().to(torch.int32)
    frac   = log2_s - e.float()
    m_lut  = _M3_FRAC_LOG2.to(frac.device)
    m_best = (frac.unsqueeze(1) - m_lut).abs().argmin(1)
    scales = (2.0 ** (e.float() + m_lut[m_best])).clamp(min=EPS)
    n_blk  = Wp.shape[1] // B
    s_exp  = scales.reshape(O, n_blk).unsqueeze(2).expand(
        O, n_blk, B).reshape(O, Wp.shape[1])
    return (Wp.sign() * s_exp)[:, :I].contiguous()


# ---------------------------------------------------------------------------
# Sparse W_gate helpers (identical to exp33)
# ---------------------------------------------------------------------------

def magnitude_prune_unstructured(W: torch.Tensor, keep_rate: float) -> torch.Tensor:
    flat   = W.abs().reshape(-1)
    k      = max(1, int(keep_rate * flat.numel()))
    thresh = flat.kthvalue(flat.numel() - k + 1).values
    return W * (W.abs() >= thresh)


def finetune_sparse(W_init, mask, X_tr, Y_tr,
                    n_steps=300, lr=1e-3, batch_size=512):
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


def build_sparse_gate(W_gate, X, keep_rate, finetune_steps):
    W_u  = magnitude_prune_unstructured(W_gate, keep_rate)
    if finetune_steps <= 0:
        return W_u.cpu()
    n_tr = int(X.shape[0] * 0.8)
    X_tr = X[:n_tr]
    with torch.no_grad():
        Y_tr = X_tr @ W_gate.T
    mask = (W_u != 0).float()
    return finetune_sparse(W_u, mask, X_tr, Y_tr, n_steps=finetune_steps).cpu()


# ---------------------------------------------------------------------------
# MLP wrappers
# ---------------------------------------------------------------------------

class SparseGateHybridMLP:
    """Exp33 scheme: sparse W_gate routing + cold, up always full."""

    def __init__(self, mlp, W_sparse: torch.Tensor, k_hot: int):
        self._mlp  = mlp
        self._W_sp = W_sparse      # (I, H) float32 CPU
        self._k_hot = k_hot
        self._I = mlp.gate_up_proj.weight.shape[0] // 2

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        xf = x.float()
        I  = self._I
        gate_sparse = xf @ self._W_sp.T
        k   = min(self._k_hot, I)
        idx = torch.topk(gate_sparse.abs(), k, dim=-1, sorted=False).indices
        hot = torch.zeros(xf.shape[0], I, dtype=torch.bool, device=xf.device)
        hot.scatter_(-1, idx, True)
        W_fused = self._mlp.gate_up_proj.weight.detach().float()
        W_gate  = W_fused[:I];  W_up = W_fused[I:]
        W_down  = self._mlp.down_proj.weight.detach().float()
        gate_full = xf @ W_gate.T
        up_full   = xf @ W_up.T
        gate   = torch.where(hot, gate_full, gate_sparse)
        swiglu = F.silu(gate) * up_full
        return (swiglu @ W_down.T).to(orig_dtype)


class E5M3ThresholdMLP:
    """Exp24 scheme: E5M3 gate, threshold routing, up always full."""

    def __init__(self, mlp, threshold: float):
        self._mlp       = mlp
        self._threshold = threshold
        W_fused = mlp.gate_up_proj.weight.detach().float()
        I = W_fused.shape[0] // 2
        self._I          = I
        self._W_gate_enc = build_e5m3_encoded(W_fused[:I])   # CPU

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        xf = x.float()
        I  = self._I
        gate_approx = xf @ self._W_gate_enc.T
        thresh = self._threshold * gate_approx.abs().mean(dim=-1, keepdim=True)
        hot    = gate_approx.abs() > thresh
        W_fused   = self._mlp.gate_up_proj.weight.detach().float()
        W_gate    = W_fused[:I];  W_up = W_fused[I:]
        W_down    = self._mlp.down_proj.weight.detach().float()
        gate_full = xf @ W_gate.T
        up_full   = xf @ W_up.T
        gate_hybrid = torch.where(hot, gate_full, gate_approx)
        swiglu      = F.silu(gate_hybrid) * up_full
        return (swiglu @ W_down.T).to(orig_dtype)


# ---------------------------------------------------------------------------
# LogitCapture — stores full logit tensors (not just argmax)
# ---------------------------------------------------------------------------

class LogitCapture:
    """Hook on model.model.norm; computes full logits via lm_head GEMM."""

    def __init__(self, W_U: torch.Tensor):
        self._W_U   = W_U.float()   # (vocab, H)
        self.logits: list[torch.Tensor] = []   # list of (T_i, vocab) cpu tensors
        self._handle = None

    def attach(self, norm_module) -> None:
        self._handle = norm_module.register_forward_hook(self._hook)

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _hook(self, module, args, output) -> None:
        lgt = (output.float() @ self._W_U.T).cpu()   # (T_i, vocab)
        self.logits.append(lgt)

    def all_logits(self, n_dec: int) -> torch.Tensor:
        """Concatenate all captured logits, dropping the decode token(s)."""
        cat = torch.cat(self.logits, dim=0)           # (total_tokens, vocab)
        if n_dec < cat.shape[0]:
            cat = cat[:-n_dec]
        return cat


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


def _run_logits(llm, prompts, W_U, norm) -> torch.Tensor:
    """Run inference and return (N_prefill, vocab) full logit tensor."""
    from vllm import SamplingParams
    cap = LogitCapture(W_U.detach().float())
    cap.attach(norm)
    llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0.0),
                 use_tqdm=False)
    cap.detach()
    return cap.all_logits(n_dec=len(prompts))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Exp34: thermal match rate for key schemes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model",           default="ibm-granite/granite-4.2-3b")
    p.add_argument("--calibration-set", default="bartowski-imatrix-v5-semantic.txt")
    p.add_argument("--activations",     default="ffn_activations128.npz")
    p.add_argument("--num-prompts",     type=int, default=8)
    p.add_argument("--temperatures",    nargs="+", type=float, default=[0.7, 1.0],
                   help="Temperatures for thermal match (default: 0.7 1.0).")
    p.add_argument("--tau",             type=float, default=LN2,
                   help=f"Noise-floor multiplier tau (default: ln2={LN2:.4f}).")
    p.add_argument(
        "--hot-fractions", nargs="+", type=float, default=[0.20, 0.30, 0.50])
    p.add_argument("--finetune-steps",  type=int, default=300)
    p.add_argument("--keep-rate",       type=float, default=0.5)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    args = _parse_args(argv)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    print(f"Device: {DEV}  tau={args.tau:.4f}  temps={args.temperatures}",
          file=sys.stderr)

    data    = np.load(args.activations)
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
    # Build sparse weights (exp33 scheme, kr=0.5+ft)
    # ------------------------------------------------------------------
    kr = args.keep_rate
    ft = args.finetune_steps
    print(f"\nBuilding sparse W_gate kr={kr} ft={ft} for all layers ...",
          file=sys.stderr)
    t0 = time.time()
    sparse_weights = []
    for li, layer in enumerate(layers):
        W_fused = layer.mlp.gate_up_proj.weight.detach().float()
        W_gate  = W_fused[:I].to(DEV)
        X       = torch.from_numpy(
                      data[f"layer{li}/gate_up_input"]).float().to(DEV)
        sparse_weights.append(build_sparse_gate(W_gate, X, kr, ft))
        print(f"  layer {li:2d}", end="\r", file=sys.stderr, flush=True)
    print(f"  done ({time.time()-t0:.0f}s)", file=sys.stderr)

    # ------------------------------------------------------------------
    # Build E5M3 encoded weights for exp24 reference (CPU)
    # ------------------------------------------------------------------
    print("\nBuilding E5M3 encoded weights for exp24 reference ...",
          file=sys.stderr)
    e5m3_weights = []
    for layer in layers:
        W_fused = layer.mlp.gate_up_proj.weight.detach().float()
        W_enc   = build_e5m3_encoded(W_fused[:I])   # CPU
        e5m3_weights.append(W_enc)

    # ------------------------------------------------------------------
    # Baseline (full precision)
    # ------------------------------------------------------------------
    print("\nBaseline pass (full precision) ...", file=sys.stderr)
    lf_base = _run_logits(llm, prompts, W_U, norm)
    n_tok   = lf_base.shape[0]
    print(f"  {n_tok} prefill tokens captured, vocab={lf_base.shape[1]}",
          file=sys.stderr)

    # Sanity: baseline vs itself
    s, tm07, mg = thermal_match(lf_base, lf_base, 0.7, args.tau)
    print(f"  Sanity check — strict={s:.4f} thermal_07={tm07:.4f} "
          f"(should both be 1.0)", file=sys.stderr)

    # ------------------------------------------------------------------
    # Define schemes to evaluate
    # ------------------------------------------------------------------
    # Each entry: (label, constructor_fn) where constructor_fn() installs hooks
    # and returns a list of MLP wrappers (one per layer)

    schemes = []

    # exp33 sparse kr=0.5+ft at various hot fractions
    for frac in args.hot_fractions:
        k_hot = max(1, int(frac * I))
        label = f"sparse kr={kr}+ft  hot={frac*100:.0f}%"
        ws    = sparse_weights   # captured by closure — same list, ok
        schemes.append((label, lambda _ws=ws, _k=k_hot: [
            SparseGateHybridMLP(layers[li].mlp, _ws[li], _k)
            for li in range(len(layers))
        ]))

    # exp24 E5M3 threshold T=0.20
    schemes.append(("exp24 E5M3 T=0.20", lambda: [
        E5M3ThresholdMLP(layers[li].mlp, 0.20)
        for li in range(len(layers))
    ]))

    # exp24 E5M3 threshold T=0.50 (reference at ~50% effective hot)
    schemes.append(("exp24 E5M3 T=0.50", lambda: [
        E5M3ThresholdMLP(layers[li].mlp, 0.50)
        for li in range(len(layers))
    ]))

    # ------------------------------------------------------------------
    # Evaluate each scheme
    # ------------------------------------------------------------------
    # results[label] = dict with strict, thermal_T for each T, mean_gap
    results: dict[str, dict] = {}

    for idx, (label, build_fn) in enumerate(schemes):
        hybrids = build_fn()
        for l, h in zip(layers, hybrids):
            l.mlp.forward = h

        print(f"\n[{idx+1}/{len(schemes)}] {label} ...",
              end="  ", file=sys.stderr, flush=True)
        lh = _run_logits(llm, prompts, W_U, norm)[:n_tok]

        rec: dict = {}
        # Strict match (T=0)
        s, _, _ = thermal_match(lf_base, lh, temperature=1.0, tau=0.0)
        rec["strict"] = s

        # Thermal match at each requested temperature
        for T in args.temperatures:
            _, tm, mg = thermal_match(lf_base, lh, temperature=T, tau=args.tau)
            rec[f"thermal_{T}"] = tm
            rec[f"gap_{T}"] = mg   # mean gap among perturbed tokens

        results[label] = rec
        print(
            "strict={:.4f}  ".format(rec["strict"]) +
            "  ".join(
                "thermal_{T}={v:.4f}".format(T=T, v=rec[f"thermal_{T}"])
                for T in args.temperatures
            ),
            file=sys.stderr,
        )

        for l, fwd in zip(layers, orig_forwards):
            l.mlp.forward = fwd
        del hybrids

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    temps = args.temperatures
    hdr_parts = ["strict"] + [f"therm T={T}" for T in temps] + \
                [f"gap T={T}" for T in temps]
    col_w = 12

    print("\n" + "=" * 80)
    print("Exp34: thermal match rate  (τ=ln2={:.4f},  forgive if gap < T·τ)".format(
        args.tau))
    print("gap = lh[hybrid_top1] − lh[baseline_top1]  (logit units)")
    print("=" * 80)

    hdr = f"  {'scheme':<35}" + "".join(f"  {h:>{col_w}}" for h in hdr_parts)
    print(hdr)
    print("  " + "-" * (35 + (col_w + 2) * len(hdr_parts)))

    for label, rec in results.items():
        row = f"  {label:<35}"
        row += f"  {rec['strict']:>{col_w}.4f}"
        for T in temps:
            row += f"  {rec[f'thermal_{T}']:>{col_w}.4f}"
        for T in temps:
            row += f"  {rec[f'gap_{T}']:>{col_w}.4f}"
        print(row)

    print()
    print("  'gap' columns = mean logit gap among perturbed tokens (lower = milder errors)")
    print("  strict perturb = 1 - strict_match")
    print("  thermal perturb(T) = 1 - thermal_match(T)")
    print()
    print("  Reference: FP8 equivalent ~3-5% strict perturbation")
    print("             NVFP4 equivalent ~10-20% strict perturbation")
    print("=" * 80)

    # Gap distribution detail
    print("\nStrict vs thermal perturbation (%):")
    hdr2 = (f"  {'scheme':<35}  {'strict%':>8}  " +
            "  ".join(f"{'therm%@'+str(T):>10}" for T in temps) +
            "  " + "  ".join(f"{'Δ@'+str(T):>8}" for T in temps))
    print(hdr2)
    for label, rec in results.items():
        strict_p = (1 - rec["strict"]) * 100
        row = f"  {label:<35}  {strict_p:>7.1f}%"
        for T in temps:
            tp = (1 - rec[f"thermal_{T}"]) * 100
            row += f"  {tp:>9.1f}%"
        for T in temps:
            tp   = (1 - rec[f"thermal_{T}"]) * 100
            delta = strict_p - tp
            row += f"  {delta:>7.1f}pp"
        print(row)


if __name__ == "__main__":
    main()
