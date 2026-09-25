# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 36 – E5M3 B=8 cold up with sparse gate routing.

Motivation
----------
Exp35 showed that:
  - sparse_gate_only (cold up = full precision):   strict 12.0%, thermal@1.0 4.8% @30% hot
  - sparse_gate+up   (cold up = sparse approx):    strict 18.4%, thermal@1.0 7.0% @30% hot
  - zero_cold        (cold up = 0):                strict 63.0% — catastrophically hard

The question: can E5M3 B=8 encoding for cold W_up (established as the sweet spot
in exp19/22) bridge the gap between sparse_gate_only and sparse_gate+up?

E5M3 B=8 achieved TARE=0.837 (exp19), close to the TARE-optimal sign encoding.
In exp27, E5M3 cold up was shown to be tolerable when routing quality is high.
With sparse W_gate routing (F1=0.902 @20%, exp32) — higher quality than SVD
rank=1024 (F1=0.745) used in exp27 — cold up errors should matter even less.

Scheme
------
  routing:   hot = top-k(|x @ W_gate_sparse.T|)    [kr=0.5+ft from exp35 cache]
  hot gate:  x @ W_gate_full.T                      [full precision]
  cold gate: x @ W_gate_sparse.T                    [sparse approx, free]
  hot up:    x @ W_up_full.T                        [full precision]
  cold up:   x @ W_up_e5m3.T                        [E5M3 B=8 encoded]
  down:      swiglu @ W_down.T                      [always full precision]

Compared against:
  - sparse_gate_only (full up cold, exp35 reprise) — upper bound
  - exp24 E5M3 threshold T=0.20 — prior best single-pass scheme

Hot fractions: 0.20, 0.25, 0.30, 0.50
Both strict and thermal match (T=0.7 and T=1.0, τ=ln2) reported.

Usage::

    python tools/profiler/exp36_e5m3_cold_up.py \\
        --model ibm-granite/granite-4.2-3b \\
        --calibration-set bartowski-imatrix-v5-semantic.txt \\
        --activations ffn_activations128.npz \\
        --weight-cache exp35_sparse_weights.pt
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


DEV   = _device()
I_DIM = 8192
LN2   = math.log(2)
EPS   = 1e-9
BLOCK = 8

_M3_FRAC_LOG2 = torch.log2(1.0 + torch.arange(8).float() / 8.0)


# ---------------------------------------------------------------------------
# E5M3 B=8 encoding  (identical to exp22–27)
# ---------------------------------------------------------------------------

def _floor_eps(W: torch.Tensor, pct: float = 1.0) -> torch.Tensor:
    flat = W.abs().reshape(-1)
    k = max(1, int(len(flat) * pct / 100.0))
    return flat.kthvalue(k).values.clamp(min=EPS)


def build_e5m3_encoded(W: torch.Tensor, B: int = BLOCK) -> torch.Tensor:
    """Return W_enc (O, I) float32 = sign(W) * E5M3_optimal_scale_per_block."""
    O, I = W.shape
    eps  = _floor_eps(W)
    pad  = (B - I % B) % B
    Wp   = F.pad(W, (0, pad)) if pad else W
    W_b  = Wp.reshape(-1, B)

    wa     = W_b.abs().clamp(min=eps)
    tilt   = torch.log1p(wa / eps)
    log2_s = (tilt * torch.log2(wa)).sum(1) / tilt.sum(1).clamp(min=EPS)

    e      = log2_s.floor().to(torch.int32)
    frac   = log2_s - e.float()
    m_lut  = _M3_FRAC_LOG2.to(frac.device)
    m_best = (frac.unsqueeze(1) - m_lut).abs().argmin(1)
    scales = (2.0 ** (e.float() + m_lut[m_best])).clamp(min=EPS)

    n_blk = Wp.shape[1] // B
    s_exp = (scales.reshape(O, n_blk)
                   .unsqueeze(2)
                   .expand(O, n_blk, B)
                   .reshape(O, Wp.shape[1]))
    return (Wp.sign() * s_exp)[:, :I].contiguous()


# ---------------------------------------------------------------------------
# Thermal match metric  (identical to exp34/35)
# ---------------------------------------------------------------------------

def thermal_match(
    lf: torch.Tensor,
    lh: torch.Tensor,
    temperature: float = 0.7,
    tau: float = LN2,
) -> tuple[float, float, float]:
    """(strict_match, thermal_match, mean_gap_among_perturbed)."""
    baseline_top1 = lf.argmax(-1)
    hybrid_top1   = lh.argmax(-1)
    exact_match   = (baseline_top1 == hybrid_top1)

    hybrid_rank1_logit = lh.gather(-1, hybrid_top1.unsqueeze(-1)).squeeze(-1)
    baseline_in_hybrid = lh.gather(-1, baseline_top1.unsqueeze(-1)).squeeze(-1)
    gap = hybrid_rank1_logit - baseline_in_hybrid

    forgiven = (~exact_match) & (gap < temperature * tau)
    strict   = float(exact_match.float().mean())
    thermal  = float((exact_match | forgiven).float().mean())
    gaps     = gap[~exact_match]
    mean_gap = float(gaps.mean()) if gaps.numel() > 0 else 0.0
    return strict, thermal, mean_gap


# ---------------------------------------------------------------------------
# Sparse W_gate helper  (magnitude pruning only — weights loaded from cache)
# ---------------------------------------------------------------------------

def magnitude_prune_unstructured(W: torch.Tensor, keep_rate: float) -> torch.Tensor:
    flat   = W.abs().reshape(-1)
    k      = max(1, int(keep_rate * flat.numel()))
    thresh = flat.kthvalue(flat.numel() - k + 1).values
    return W * (W.abs() >= thresh)


# ---------------------------------------------------------------------------
# MLP wrappers
# ---------------------------------------------------------------------------

class SparseGateOnlyMLP:
    """Baseline: sparse gate routing + cold gate sparse, up always full (exp35)."""

    def __init__(self, mlp, W_gate_sp: torch.Tensor, k_hot: int):
        self._mlp    = mlp
        self._W_g_sp = W_gate_sp   # (I, H) CPU float32
        self._k_hot  = k_hot
        self._I      = mlp.gate_up_proj.weight.shape[0] // 2

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        xf = x.float()
        I  = self._I
        gate_sp = xf @ self._W_g_sp.T
        k   = min(self._k_hot, I)
        idx = torch.topk(gate_sp.abs(), k, dim=-1, sorted=False).indices
        hot = torch.zeros(xf.shape[0], I, dtype=torch.bool, device=xf.device)
        hot.scatter_(-1, idx, True)
        W         = self._mlp.gate_up_proj.weight.detach().float()
        gate_full = xf @ W[:I].T
        up_full   = xf @ W[I:].T
        W_down    = self._mlp.down_proj.weight.detach().float()
        gate   = torch.where(hot, gate_full, gate_sp)
        swiglu = F.silu(gate) * up_full
        return (swiglu @ W_down.T).to(orig_dtype)


class SparseGateE5M3UpMLP:
    """Exp36 scheme: sparse gate routing, cold up = E5M3 B=8 encoded."""

    def __init__(self, mlp, W_gate_sp: torch.Tensor,
                 W_up_enc: torch.Tensor, k_hot: int):
        self._mlp    = mlp
        self._W_g_sp = W_gate_sp   # (I, H) CPU float32  — sparse gate
        self._W_u_enc = W_up_enc   # (I, H) CPU float32  — E5M3 encoded up
        self._k_hot  = k_hot
        self._I      = mlp.gate_up_proj.weight.shape[0] // 2

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        xf = x.float()
        I  = self._I
        gate_sp = xf @ self._W_g_sp.T
        k   = min(self._k_hot, I)
        idx = torch.topk(gate_sp.abs(), k, dim=-1, sorted=False).indices
        hot = torch.zeros(xf.shape[0], I, dtype=torch.bool, device=xf.device)
        hot.scatter_(-1, idx, True)
        W         = self._mlp.gate_up_proj.weight.detach().float()
        gate_full = xf @ W[:I].T
        up_full   = xf @ W[I:].T
        up_enc    = xf @ self._W_u_enc.T   # cold up via E5M3
        W_down    = self._mlp.down_proj.weight.detach().float()
        gate   = torch.where(hot, gate_full, gate_sp)
        up     = torch.where(hot, up_full, up_enc)
        swiglu = F.silu(gate) * up
        return (swiglu @ W_down.T).to(orig_dtype)


# ---------------------------------------------------------------------------
# E5M3 threshold MLP  (exp24 reference — identical to exp34/35)
# ---------------------------------------------------------------------------

class E5M3ThresholdMLP:
    def __init__(self, mlp, threshold: float):
        self._mlp       = mlp
        self._threshold = threshold
        W = mlp.gate_up_proj.weight.detach().float()
        I = W.shape[0] // 2
        self._I          = I
        self._W_gate_enc = build_e5m3_encoded(W[:I])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        xf = x.float()
        I  = self._I
        gate_approx = xf @ self._W_gate_enc.T
        thresh = self._threshold * gate_approx.abs().mean(dim=-1, keepdim=True)
        hot    = gate_approx.abs() > thresh
        W      = self._mlp.gate_up_proj.weight.detach().float()
        gate_full = xf @ W[:I].T
        up_full   = xf @ W[I:].T
        W_down    = self._mlp.down_proj.weight.detach().float()
        gate   = torch.where(hot, gate_full, gate_approx)
        swiglu = F.silu(gate) * up_full
        return (swiglu @ W_down.T).to(orig_dtype)


# ---------------------------------------------------------------------------
# LogitCapture  (identical to exp34/35)
# ---------------------------------------------------------------------------

class LogitCapture:
    def __init__(self, W_U: torch.Tensor):
        self._W_U   = W_U.float()
        self.logits: list[torch.Tensor] = []
        self._handle = None

    def attach(self, norm_module) -> None:
        self._handle = norm_module.register_forward_hook(self._hook)

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _hook(self, module, args, output) -> None:
        self.logits.append((output.float() @ self._W_U.T).cpu())

    def all_logits(self, n_dec: int) -> torch.Tensor:
        cat = torch.cat(self.logits, dim=0)
        return cat[:-n_dec] if n_dec < cat.shape[0] else cat


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
        description="Exp36: E5M3 B=8 cold up with sparse gate routing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model",           default="ibm-granite/granite-4.2-3b")
    p.add_argument("--calibration-set", default="bartowski-imatrix-v5-semantic.txt")
    p.add_argument("--activations",     default="ffn_activations128.npz",
                   help="Only needed if weight-cache is missing.")
    p.add_argument("--num-prompts",     type=int, default=8)
    p.add_argument("--keep-rate",       type=float, default=0.5)
    p.add_argument("--finetune-steps",  type=int, default=300)
    p.add_argument(
        "--hot-fractions", nargs="+", type=float, default=[0.20, 0.25, 0.30, 0.50])
    p.add_argument(
        "--temperatures", nargs="+", type=float, default=[0.7, 1.0])
    p.add_argument("--tau",             type=float, default=LN2)
    p.add_argument("--weight-cache",    default="exp35_sparse_weights.pt",
                   help="Cache of sparse gate weights from exp35.")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    args = _parse_args(argv)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    print(f"Device: {DEV}  tau={args.tau:.4f}  temps={args.temperatures}",
          file=sys.stderr)

    prompts = _load_prompts(args.calibration_set, args.num_prompts)
    print(f"Loaded {len(prompts)} prompts.", file=sys.stderr)

    from vllm import LLM
    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              kv_cache_memory_bytes=int(0.5 * 1024**3), max_model_len=512,
              enable_prefix_caching=False)

    norm, layers, W_U = _get_internals(llm)
    orig_forwards = [l.mlp.forward for l in layers]
    I  = layers[0].mlp.gate_up_proj.weight.shape[0] // 2
    kr = args.keep_rate
    ft = args.finetune_steps

    # ------------------------------------------------------------------
    # Load sparse W_gate from exp35 cache
    # ------------------------------------------------------------------
    cache_path = Path(args.weight_cache)
    cache_key  = f"kr{kr}_ft{ft}"
    sp_gate    = None

    if cache_path.exists():
        print(f"\nLoading sparse W_gate from {cache_path} (key={cache_key}) ...",
              file=sys.stderr)
        saved = torch.load(cache_path, weights_only=True)
        if cache_key in saved:
            sp_gate = saved[cache_key]["gate"]
            print(f"  loaded {len(sp_gate)} layers.", file=sys.stderr)

    if sp_gate is None:
        raise RuntimeError(
            f"Sparse gate weights not found in {cache_path} for key {cache_key}. "
            f"Run exp35 first to build the cache.")

    # ------------------------------------------------------------------
    # Build E5M3 encoded W_up for all layers  (CPU, no activations needed)
    # ------------------------------------------------------------------
    print("\nBuilding E5M3 B=8 encoded W_up for all layers ...", file=sys.stderr)
    t0 = time.time()
    e5m3_up = []
    for li, layer in enumerate(layers):
        W_fused = layer.mlp.gate_up_proj.weight.detach().float()
        W_up    = W_fused[I:]                         # (I, H)
        e5m3_up.append(build_e5m3_encoded(W_up))      # CPU float32
        print(f"  layer {li:2d}", end="\r", file=sys.stderr, flush=True)
    print(f"  done ({time.time()-t0:.0f}s)", file=sys.stderr)

    # ------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------
    print("\nBaseline pass ...", file=sys.stderr)
    lf_base = _run_logits(llm, prompts, W_U, norm)
    n_tok   = lf_base.shape[0]
    print(f"  {n_tok} prefill tokens, vocab={lf_base.shape[1]}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Define runs
    # ------------------------------------------------------------------
    temps = args.temperatures
    runs: list[tuple[str, object]] = []

    for frac in args.hot_fractions:
        k = max(1, int(frac * I))
        pct = f"{frac*100:.0f}%"

        runs.append((f"sparse_gate+e5m3_up  hot={pct}", lambda _k=k: [
            SparseGateE5M3UpMLP(layers[li].mlp, sp_gate[li], e5m3_up[li], _k)
            for li in range(len(layers))
        ]))
        runs.append((f"sparse_gate_only     hot={pct}", lambda _k=k: [
            SparseGateOnlyMLP(layers[li].mlp, sp_gate[li], _k)
            for li in range(len(layers))
        ]))

    # exp24 reference: E5M3 threshold T=0.20
    runs.append(("exp24 E5M3 T=0.20 (~88% hot)", lambda: [
        E5M3ThresholdMLP(layers[li].mlp, 0.20) for li in range(len(layers))
    ]))

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    results: dict[str, dict] = {}

    for idx, (label, build_fn) in enumerate(runs):
        hybrids = build_fn()
        for l, h in zip(layers, hybrids):
            l.mlp.forward = h

        print(f"\n[{idx+1:2d}/{len(runs)}] {label} ...",
              end="  ", file=sys.stderr, flush=True)
        lh = _run_logits(llm, prompts, W_U, norm)[:n_tok]

        rec: dict = {}
        s, _, _ = thermal_match(lf_base, lh, temperature=1.0, tau=0.0)
        rec["strict"] = s
        for T in temps:
            _, tm, mg = thermal_match(lf_base, lh, T, args.tau)
            rec[f"thermal_{T}"] = tm
            rec[f"gap_{T}"]     = mg
        results[label] = rec

        print(
            "strict={:.4f}  ".format(rec["strict"]) +
            "  ".join("th@{T}={v:.4f}".format(T=T, v=rec[f"thermal_{T}"])
                      for T in temps),
            file=sys.stderr,
        )
        for l, fwd in zip(layers, orig_forwards):
            l.mlp.forward = fwd
        del hybrids

    # ------------------------------------------------------------------
    # Summary tables
    # ------------------------------------------------------------------
    col_w = 11

    def _hdr():
        cols = ["strict"] + [f"th@T={T}" for T in temps] + \
               [f"gap@{T}" for T in temps]
        return f"  {'scheme':<36}" + "".join(f"  {c:>{col_w}}" for c in cols)

    def _row(label, rec):
        r = f"  {label:<36}  {rec['strict']:>{col_w}.4f}"
        for T in temps:
            r += f"  {rec[f'thermal_{T}']:>{col_w}.4f}"
        for T in temps:
            r += f"  {rec[f'gap_{T}']:>{col_w}.4f}"
        return r

    print("\n" + "=" * 88)
    print(f"Exp36: E5M3 B=8 cold up  (sparse gate routing kr={kr}+ft, τ=ln2={args.tau:.4f})")
    print("cold gate = x @ W_gate_sparse.T  |  cold up = x @ W_up_e5m3.T  |  down = full")
    print("=" * 88)
    print(_hdr())
    print("  " + "-" * 84)
    for label, rec in results.items():
        print(_row(label, rec))

    print()
    print("  gap = mean logit gap among perturbed tokens (logit units)")
    print("  FP8-equivalent: strict ~3-5%  thermal ~2-4%")
    print("  NVFP4-equivalent: strict ~10-20%  thermal ~8-16%")
    print("=" * 88)

    # Perturbation % breakdown
    print("\nPerturbation rates (%):")
    hdr2 = (f"  {'scheme':<36}  {'strict%':>8}" +
            "".join(f"  {'th%@'+str(T):>9}" for T in temps) +
            "".join(f"  {'Δ@'+str(T):>8}" for T in temps))
    print(hdr2)
    for label, rec in results.items():
        sp  = (1 - rec["strict"]) * 100
        row = f"  {label:<36}  {sp:>7.1f}%"
        for T in temps:
            tp = (1 - rec[f"thermal_{T}"]) * 100
            row += f"  {tp:>8.1f}%"
        for T in temps:
            tp    = (1 - rec[f"thermal_{T}"]) * 100
            delta = sp - tp
            row  += f"  {delta:>7.1f}pp"
        print(row)

    # Delta between E5M3 up and full up, per fraction
    print("\nΔ (sparse_gate+e5m3_up  −  sparse_gate_only)  [+ = e5m3 up worse]:")
    hdr3 = f"  {'hot%':<8}  {'Δ strict':>10}  {'Δ th@0.7':>10}  {'Δ th@1.0':>10}  {'Δ gap@0.7':>11}"
    print(hdr3)
    for frac in args.hot_fractions:
        pct   = f"{frac*100:.0f}%"
        k_e5m = f"sparse_gate+e5m3_up  hot={pct}"
        k_full = f"sparse_gate_only     hot={pct}"
        if k_e5m not in results or k_full not in results:
            continue
        re, rf = results[k_e5m], results[k_full]
        ds  = (1 - re["strict"])  - (1 - rf["strict"])
        d07 = (1 - re["thermal_0.7"]) - (1 - rf["thermal_0.7"])
        d10 = (1 - re["thermal_1.0"]) - (1 - rf["thermal_1.0"])
        dg  = re["gap_0.7"] - rf["gap_0.7"]
        print(f"  {pct:<8}  {ds*100:>+9.1f}pp  {d07*100:>+9.1f}pp  "
              f"{d10*100:>+9.1f}pp  {dg:>+10.3f}L")


if __name__ == "__main__":
    main()
