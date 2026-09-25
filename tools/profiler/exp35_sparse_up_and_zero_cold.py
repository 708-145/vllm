# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 35 – sparse up projection and zero-cold comparison.

Motivation
----------
Exp33 kept W_up at full precision for cold channels, matching exp24's design.
This experiment tests two extensions, both targeting the 20–30% hot regime:

  A. sparse_gate_up  : routing from |x @ W_gate_sparse.T|, hot channels full
                       precision for both gate and up; cold channels use
                       x @ W_gate_sparse.T  and  x @ W_up_sparse.T
                       W_up_sparse built with the same magnitude-prune+ft scheme
                       (kr=0.5, 300 Adam steps targeting W_up_full responses).
                       Down projection always full precision.

  B. zero_cold       : same routing as sparse_gate_up, hot channels full
                       precision, cold channels zeroed (feed 0 to down proj).
                       Answers: are cold-channel errors hard or soft?

  C. sparse_gate_only (exp33 reprise, for direct comparison at 20/30% hot)

Hot fractions: 0.20, 0.30  (target regime; 0.50 included for context).

Both strict and thermal match (T=0.7 and T=1.0, τ=ln2) are reported.

Usage::

    python tools/profiler/exp35_sparse_up_and_zero_cold.py \\
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


DEV    = _device()
I_DIM  = 8192
LN2    = math.log(2)
EPS    = 1e-9


# ---------------------------------------------------------------------------
# Thermal match metric (identical to exp34)
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
# Sparse weight helpers
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


def build_sparse(W: torch.Tensor, X: torch.Tensor,
                 keep_rate: float, finetune_steps: int) -> torch.Tensor:
    """Return sparse W (I, H) on CPU.  X is the input activations on DEV."""
    W_u  = magnitude_prune_unstructured(W, keep_rate)
    if finetune_steps <= 0:
        return W_u.cpu()
    n_tr = int(X.shape[0] * 0.8)
    X_tr = X[:n_tr]
    with torch.no_grad():
        Y_tr = X_tr @ W.T          # oracle targets for this projection
    mask = (W_u != 0).float()
    return finetune_sparse(W_u, mask, X_tr, Y_tr, n_steps=finetune_steps).cpu()


# ---------------------------------------------------------------------------
# MLP wrappers
# ---------------------------------------------------------------------------

class SparseGateOnlyMLP:
    """Exp33 scheme: gate routing + sparse cold gate, up always full."""

    def __init__(self, mlp, W_gate_sp: torch.Tensor, k_hot: int):
        self._mlp     = mlp
        self._W_g_sp  = W_gate_sp    # (I, H) CPU float32
        self._k_hot   = k_hot
        self._I       = mlp.gate_up_proj.weight.shape[0] // 2

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        xf = x.float()
        I  = self._I
        gate_sp = xf @ self._W_g_sp.T
        k   = min(self._k_hot, I)
        idx = torch.topk(gate_sp.abs(), k, dim=-1, sorted=False).indices
        hot = torch.zeros(xf.shape[0], I, dtype=torch.bool, device=xf.device)
        hot.scatter_(-1, idx, True)
        W   = self._mlp.gate_up_proj.weight.detach().float()
        gate_full = xf @ W[:I].T
        up_full   = xf @ W[I:].T
        W_down    = self._mlp.down_proj.weight.detach().float()
        gate   = torch.where(hot, gate_full, gate_sp)
        swiglu = F.silu(gate) * up_full
        return (swiglu @ W_down.T).to(orig_dtype)


class SparseGateUpMLP:
    """Scheme A: sparse routing + sparse cold for both gate and up, up full for hot."""

    def __init__(self, mlp, W_gate_sp: torch.Tensor,
                 W_up_sp: torch.Tensor, k_hot: int):
        self._mlp    = mlp
        self._W_g_sp = W_gate_sp    # (I, H) CPU float32
        self._W_u_sp = W_up_sp      # (I, H) CPU float32
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
        up_sp     = xf @ self._W_u_sp.T
        W_down    = self._mlp.down_proj.weight.detach().float()
        gate   = torch.where(hot, gate_full, gate_sp)
        up     = torch.where(hot, up_full,   up_sp)
        swiglu = F.silu(gate) * up
        return (swiglu @ W_down.T).to(orig_dtype)


class ZeroColdMLP:
    """Scheme B: same routing as sparse_gate_up, but cold channels → 0."""

    def __init__(self, mlp, W_gate_sp: torch.Tensor, k_hot: int):
        self._mlp    = mlp
        self._W_g_sp = W_gate_sp    # (I, H) CPU float32
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
        # Cold channels zeroed — contributes 0 to down projection
        gate   = torch.where(hot, gate_full, torch.zeros_like(gate_full))
        up     = torch.where(hot, up_full,   torch.zeros_like(up_full))
        swiglu = F.silu(gate) * up
        return (swiglu @ W_down.T).to(orig_dtype)


# ---------------------------------------------------------------------------
# LogitCapture (full logits, identical to exp34)
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
        description="Exp35: sparse up + zero-cold comparison with thermal metric.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model",           default="ibm-granite/granite-4.2-3b")
    p.add_argument("--calibration-set", default="bartowski-imatrix-v5-semantic.txt")
    p.add_argument("--activations",     default="ffn_activations128.npz")
    p.add_argument("--num-prompts",     type=int, default=8)
    p.add_argument("--keep-rate",       type=float, default=0.5)
    p.add_argument("--finetune-steps",  type=int, default=300)
    p.add_argument(
        "--hot-fractions", nargs="+", type=float, default=[0.20, 0.30, 0.50])
    p.add_argument(
        "--temperatures", nargs="+", type=float, default=[0.7, 1.0])
    p.add_argument("--tau",             type=float, default=LN2)
    p.add_argument("--weight-cache",    default="exp35_sparse_weights.pt",
                   help="Cache file for sparse weights (avoids recomputing ft).")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    args = _parse_args(argv)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    print(f"Device: {DEV}  kr={args.keep_rate}  ft={args.finetune_steps}  "
          f"tau={args.tau:.4f}  temps={args.temperatures}", file=sys.stderr)

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
    kr, ft = args.keep_rate, args.finetune_steps

    # ------------------------------------------------------------------
    # Build (or load cached) sparse W_gate and W_up
    # ------------------------------------------------------------------
    cache_path = Path(args.weight_cache)
    cache_key  = f"kr{kr}_ft{ft}"
    sp_gate = sp_up = None

    if cache_path.exists():
        print(f"\nLoading sparse weights from {cache_path} ...", file=sys.stderr)
        saved = torch.load(cache_path, weights_only=True)
        if cache_key in saved:
            sp_gate = saved[cache_key]["gate"]
            sp_up   = saved[cache_key]["up"]
            print(f"  loaded key={cache_key}.", file=sys.stderr)

    if sp_gate is None:
        print(f"\nBuilding sparse W_gate  kr={kr}  ft={ft} ...", file=sys.stderr)
        t0 = time.time()
        sp_gate = []
        for li, layer in enumerate(layers):
            W  = layer.mlp.gate_up_proj.weight.detach().float()
            Wg = W[:I].to(DEV)
            X  = torch.from_numpy(
                     data[f"layer{li}/gate_up_input"]).float().to(DEV)
            sp_gate.append(build_sparse(Wg, X, kr, ft))
            print(f"  gate layer {li:2d}", end="\r", file=sys.stderr, flush=True)
        print(f"  done ({time.time()-t0:.0f}s)", file=sys.stderr)

        print(f"\nBuilding sparse W_up    kr={kr}  ft={ft} ...", file=sys.stderr)
        t0 = time.time()
        sp_up = []
        for li, layer in enumerate(layers):
            W  = layer.mlp.gate_up_proj.weight.detach().float()
            Wu = W[I:].to(DEV)
            X  = torch.from_numpy(
                     data[f"layer{li}/gate_up_input"]).float().to(DEV)
            sp_up.append(build_sparse(Wu, X, kr, ft))
            print(f"  up   layer {li:2d}", end="\r", file=sys.stderr, flush=True)
        print(f"  done ({time.time()-t0:.0f}s)", file=sys.stderr)

        # Save cache
        cache_data = {}
        if cache_path.exists():
            cache_data = torch.load(cache_path, weights_only=True)
        cache_data[cache_key] = {"gate": sp_gate, "up": sp_up}
        torch.save(cache_data, cache_path)
        print(f"  saved to {cache_path}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------
    print("\nBaseline pass ...", file=sys.stderr)
    lf_base = _run_logits(llm, prompts, W_U, norm)
    n_tok   = lf_base.shape[0]
    print(f"  {n_tok} prefill tokens, vocab={lf_base.shape[1]}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Define schemes × hot fractions
    # ------------------------------------------------------------------
    # Each entry: (label, wrapper_builder)
    runs: list[tuple[str, object]] = []
    for frac in args.hot_fractions:
        k = max(1, int(frac * I))
        label_g  = f"sparse_gate_only   hot={frac*100:.0f}%"
        label_gu = f"sparse_gate+up     hot={frac*100:.0f}%"
        label_z  = f"zero_cold          hot={frac*100:.0f}%"

        runs.append((label_g, lambda _k=k: [
            SparseGateOnlyMLP(layers[li].mlp, sp_gate[li], _k)
            for li in range(len(layers))
        ]))
        runs.append((label_gu, lambda _k=k: [
            SparseGateUpMLP(layers[li].mlp, sp_gate[li], sp_up[li], _k)
            for li in range(len(layers))
        ]))
        runs.append((label_z, lambda _k=k: [
            ZeroColdMLP(layers[li].mlp, sp_gate[li], _k)
            for li in range(len(layers))
        ]))

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    temps   = args.temperatures
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
        return f"  {'scheme':<32}" + "".join(f"  {c:>{col_w}}" for c in cols)

    def _row(label, rec):
        r = f"  {label:<32}  {rec['strict']:>{col_w}.4f}"
        for T in temps:
            r += f"  {rec[f'thermal_{T}']:>{col_w}.4f}"
        for T in temps:
            r += f"  {rec[f'gap_{T}']:>{col_w}.4f}"
        return r

    print("\n" + "=" * 84)
    print(f"Exp35: sparse up + zero-cold  (kr={kr}, ft={ft}, τ=ln2={args.tau:.4f})")
    print("routing = top-k(|x @ W_gate_sparse.T|)  down = always full precision")
    print("=" * 84)
    print(_hdr())
    print("  " + "-" * 80)

    for frac in args.hot_fractions:
        for suffix in ["sparse_gate_only", "sparse_gate+up", "zero_cold"]:
            key = [k for k in results if f"hot={frac*100:.0f}%" in k
                   and suffix in k][0]
            print(_row(key, results[key]))
        print()

    print("  gap = mean logit gap among perturbed tokens (logit units)")
    print("  strict perturb = 1 - strict;  thermal perturb = 1 - thermal")
    print("=" * 84)

    # Perturbation % table
    print("\nPerturbation rates (%):")
    hdr2 = (f"  {'scheme':<32}  {'strict%':>8}" +
            "".join(f"  {'th%@'+str(T):>9}" for T in temps) +
            "".join(f"  {'Δ@'+str(T):>8}" for T in temps))
    print(hdr2)
    for frac in args.hot_fractions:
        for suffix in ["sparse_gate_only", "sparse_gate+up", "zero_cold"]:
            key = [k for k in results if f"hot={frac*100:.0f}%" in k
                   and suffix in k][0]
            rec  = results[key]
            sp   = (1 - rec["strict"]) * 100
            row  = f"  {key:<32}  {sp:>7.1f}%"
            for T in temps:
                tp = (1 - rec[f"thermal_{T}"]) * 100
                row += f"  {tp:>8.1f}%"
            for T in temps:
                tp    = (1 - rec[f"thermal_{T}"]) * 100
                delta = sp - tp
                row  += f"  {delta:>7.1f}pp"
            print(row)
        print()

    print("FP8-equivalent: strict ~3-5%, thermal ~2-4%")
    print("NVFP4-equivalent: strict ~10-20%, thermal ~8-16%")


if __name__ == "__main__":
    main()
