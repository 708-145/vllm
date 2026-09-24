# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routing quality of low-rank SVD predictor vs oracle (gate_full / up_full).

For each rank and hot fraction, measures precision/recall/F1/IoU of the
union-top-k hot set produced by (gate_lr, up_lr) vs the oracle hot set
produced by the same threshold applied to (gate_full, up_full).

Also reports the fraction of oracle-hot channels captured by each signal
separately (gate-only recall, up-only recall) to diagnose where the loss
comes from.

Usage::

    python tools/profiler/exp_svd_routing_quality.py \\
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


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEV = _device()
EPS = 1e-9


# ---------------------------------------------------------------------------
# SVD low-rank approximation (float32, from cache)
# ---------------------------------------------------------------------------

def _lowrank(xf: torch.Tensor,
             U: torch.Tensor, s: torch.Tensor, Vt: torch.Tensor,
             rank: int) -> torch.Tensor:
    """Compute x @ W_lr.T where W_lr = U[:,:r] diag(s[:r]) Vt[:r,:]."""
    r = min(rank, s.shape[0])
    h = (xf @ Vt[:r].T) * s[:r]   # (T, r)
    return h @ U[:, :r].T          # (T, I)


# ---------------------------------------------------------------------------
# Hook
# ---------------------------------------------------------------------------

class SVDRoutingQualityHook:
    """Measures hot-set agreement between SVD predictor and oracle."""

    def __init__(self, mlp, svd_gate, svd_up, ranks: list[int], k_hot: int):
        """
        Args:
            svd_gate: (U_g, s_g, Vt_g) float32 cpu
            svd_up:   (U_u, s_u, Vt_u) float32 cpu
            ranks:    list of SVD ranks to evaluate
            k_hot:    top-k per signal (same for gate and up)
        """
        self._mlp   = mlp
        self._ranks = ranks
        self._k_hot = k_hot

        self._U_g, self._s_g, self._Vt_g = svd_gate
        self._U_u, self._s_u, self._Vt_u = svd_up

        W_fused = mlp.gate_up_proj.weight.detach().float()
        I = W_fused.shape[0] // 2
        self._I      = I
        self._W_gate = W_fused[:I]
        self._W_up   = W_fused[I:]

        # Accumulators: tp/fp/fn/tn per rank
        self._tp = {r: 0.0 for r in ranks}
        self._fp = {r: 0.0 for r in ranks}
        self._fn = {r: 0.0 for r in ranks}
        self._tn = {r: 0.0 for r in ranks}
        self._n  = {r: 0.0 for r in ranks}

        # Per-signal recall accumulators (gate-only, up-only)
        self._tp_g = {r: 0.0 for r in ranks}   # oracle-hot caught by gate_lr alone
        self._tp_u = {r: 0.0 for r in ranks}   # oracle-hot caught by up_lr alone
        self._n_oracle = {r: 0.0 for r in ranks}

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        I  = self._I
        k  = min(self._k_hot, I)

        gate_full = xf @ self._W_gate.T   # (T, I) oracle
        up_full   = xf @ self._W_up.T

        # Oracle hot set: union top-k of |gate_full| and |up_full|
        idx_gf = torch.topk(gate_full.abs(), k, dim=-1, sorted=False).indices
        idx_uf = torch.topk(up_full.abs(),   k, dim=-1, sorted=False).indices
        hot_oracle = torch.zeros(xf.shape[0], I, dtype=torch.bool)
        hot_oracle.scatter_(-1, idx_gf, True)
        hot_oracle.scatter_(-1, idx_uf, True)

        for rank in self._ranks:
            gate_lr = _lowrank(xf, self._U_g, self._s_g, self._Vt_g, rank)
            up_lr   = _lowrank(xf, self._U_u, self._s_u, self._Vt_u, rank)

            idx_ga = torch.topk(gate_lr.abs(), k, dim=-1, sorted=False).indices
            idx_ua = torch.topk(up_lr.abs(),   k, dim=-1, sorted=False).indices
            hot_gate_lr = torch.zeros_like(hot_oracle)
            hot_up_lr   = torch.zeros_like(hot_oracle)
            hot_gate_lr.scatter_(-1, idx_ga, True)
            hot_up_lr.scatter_(-1,   idx_ua, True)
            hot_approx = hot_gate_lr | hot_up_lr

            tp = ( hot_approx &  hot_oracle).float().sum().item()
            fp = ( hot_approx & ~hot_oracle).float().sum().item()
            fn = (~hot_approx &  hot_oracle).float().sum().item()
            tn = (~hot_approx & ~hot_oracle).float().sum().item()
            self._tp[rank] += tp
            self._fp[rank] += fp
            self._fn[rank] += fn
            self._tn[rank] += tn
            self._n [rank] += hot_approx.numel()

            # Per-signal recall
            n_oracle = hot_oracle.float().sum().item()
            self._tp_g[rank]     += (hot_gate_lr & hot_oracle).float().sum().item()
            self._tp_u[rank]     += (hot_up_lr   & hot_oracle).float().sum().item()
            self._n_oracle[rank] += n_oracle

        # Unmodified forward
        W_down = self._mlp.down_proj.weight.detach().float()
        swiglu = F.silu(gate_full) * up_full
        return (swiglu @ W_down.T).to(x.dtype)

    def summary(self) -> list[dict]:
        results = []
        for rank in self._ranks:
            tp, fp, fn, tn, n = (self._tp[rank], self._fp[rank],
                                  self._fn[rank], self._tn[rank], self._n[rank])
            prec   = tp / (tp + fp + EPS)
            recall = tp / (tp + fn + EPS)
            f1     = 2 * prec * recall / (prec + recall + EPS)
            iou    = tp / (tp + fp + fn + EPS)
            acc    = (tp + tn) / (n + EPS)
            hot_frac_approx = (tp + fp) / (n + EPS)
            hot_frac_oracle = (tp + fn) / (n + EPS)
            recall_g = self._tp_g[rank] / (self._n_oracle[rank] + EPS)
            recall_u = self._tp_u[rank] / (self._n_oracle[rank] + EPS)
            results.append(dict(
                rank=rank,
                hot_frac_approx=hot_frac_approx,
                hot_frac_oracle=hot_frac_oracle,
                precision=prec, recall=recall, f1=f1, iou=iou, acc=acc,
                recall_gate_only=recall_g,
                recall_up_only=recall_u,
            ))
        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_prompts(path: str, n: int) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip()][:n]


def _get_internals(llm):
    e = llm.llm_engine
    try:
        mr = e.model_executor.driver_worker.worker.model_runner
    except AttributeError:
        mr = e.model_executor.driver_worker.model_runner
    return mr.model.model.layers


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="SVD routing quality vs oracle.")
    p.add_argument("--model", default="ibm-granite/granite-4.2-3b")
    p.add_argument("--calibration-set", default="bartowski-imatrix-v5-semantic.txt")
    p.add_argument("--num-prompts", type=int, default=8)
    p.add_argument(
        "--ranks", nargs="+", type=int,
        default=[16, 32, 64, 128, 256, 512, 1024],
        metavar="R",
    )
    p.add_argument(
        "--hot-fractions", nargs="+", type=float,
        default=[0.20, 0.50],
        metavar="F",
    )
    p.add_argument("--svd-cache", default="svd_factors_r1024.pt")
    args = p.parse_args(argv)

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    prompts = _load_prompts(args.calibration_set, args.num_prompts)
    print(f"Loaded {len(prompts)} prompts.", file=sys.stderr)

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              kv_cache_memory_bytes=int(0.5 * 1024**3), max_model_len=512,
              enable_prefix_caching=False)

    layers = _get_internals(llm)
    orig_forwards = [l.mlp.forward for l in layers]
    I = layers[0].mlp.gate_up_proj.weight.shape[0] // 2

    print(f"Loading SVD from {args.svd_cache} ...", file=sys.stderr)
    raw = torch.load(args.svd_cache, weights_only=True)
    # cache layout: (U_g, s_g, Vt_g, U_u, s_u, Vt_u) per layer
    svd = [(f[0].float(), f[1].float(), f[2].float(),
            f[3].float(), f[4].float(), f[5].float()) for f in raw]

    for frac in args.hot_fractions:
        k_hot = max(1, int(frac * I))
        print(f"\n--- hot fraction = {frac*100:.0f}%  (k={k_hot}) ---",
              file=sys.stderr)

        hooks = [
            SVDRoutingQualityHook(
                layers[li].mlp,
                svd_gate=(svd[li][0], svd[li][1], svd[li][2]),
                svd_up  =(svd[li][3], svd[li][4], svd[li][5]),
                ranks=args.ranks,
                k_hot=k_hot,
            )
            for li in range(len(layers))
        ]
        for l, h in zip(layers, hooks):
            l.mlp.forward = h

        print("Running ...", file=sys.stderr)
        llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0.0),
                     use_tqdm=False)

        for l, fwd in zip(layers, orig_forwards):
            l.mlp.forward = fwd

        # Macro-average across layers
        all_sums = [h.summary() for h in hooks]
        # all_sums[layer][rank_idx]
        n_ranks = len(args.ranks)
        by_rank = [[layer_sums[ri] for layer_sums in all_sums]
                   for ri in range(n_ranks)]

        def _m(tier, key):
            return float(np.mean([s[key] for s in tier]))

        print(f"\n{'SVD routing quality vs oracle — hot={:.0f}%  (macro-avg 40 layers)'.format(frac*100)}")
        print("oracle = union top-k(|gate_full|, k) ∪ top-k(|up_full|, k)")
        print("approx = union top-k(|gate_lr|,   k) ∪ top-k(|up_lr|,   k)")
        print()
        print(f"{'rank':>6}  {'hot%(A)':>8}  {'hot%(O)':>8}  "
              f"{'prec':>7}  {'recall':>7}  {'F1':>7}  {'IoU':>7}  "
              f"{'rec_gate':>9}  {'rec_up':>7}")
        print("-" * 80)
        for ri, rank in enumerate(args.ranks):
            s = by_rank[ri]
            print(f"{rank:>6}  "
                  f"{_m(s,'hot_frac_approx')*100:>7.1f}%  "
                  f"{_m(s,'hot_frac_oracle')*100:>7.1f}%  "
                  f"{_m(s,'precision'):>7.4f}  "
                  f"{_m(s,'recall'):>7.4f}  "
                  f"{_m(s,'f1'):>7.4f}  "
                  f"{_m(s,'iou'):>7.4f}  "
                  f"{_m(s,'recall_gate_only'):>9.4f}  "
                  f"{_m(s,'recall_up_only'):>7.4f}")
        print()
        # Also show E5M3 reference from exp25c for comparison
        print("  E5M3 ref (exp25c): prec=0.779, recall=0.790, F1=0.784, IoU=0.645  @hot=53%")
        print("  E5M3 ref (exp25c): prec=0.901, recall=0.905, F1=0.903, IoU=0.823  @hot=88%")
        del hooks


if __name__ == "__main__":
    main()
