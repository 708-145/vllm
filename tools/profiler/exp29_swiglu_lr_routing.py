# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 29 – SwiGLU low-rank routing: top-k on |SiLU(gate_lr) * up_lr|.

Motivation
----------
Exp27 routed on the union of top-k(|gate_lr|) and top-k(|up_lr|) separately.
This selects channels where *either* projection is large, which over-selects:
a channel with large |gate_lr| but tiny |up_lr| contributes ≈ 0 to the MLP
output after SwiGLU, yet still counts as hot.

This experiment routes directly on the SwiGLU-combined signal:

  signal[i] = |SiLU(gate_lr[i]) * up_lr[i]|

This is the low-rank estimate of the actual contribution of channel i to
the MLP output.  Top-k on this signal selects the channels that matter most,
without inflating the hot set with single-projection outliers.

Part 1: routing quality (precision/recall/F1/IoU vs oracle)
  Oracle: top-k(|SiLU(gate_full) * up_full|) = top-k(|down_input|)
  Compare:
    A. SwiGLU-LR:   top-k(|SiLU(gate_lr) * up_lr|)         [this exp]
    B. Union-LR:    top-k(|gate_lr|) ∪ top-k(|up_lr|)       [exp27]
    C. Gate-only:   top-k(|gate_lr|)                         [exp22/23 style]

Part 2: e2e top-1 perturbation rate (same hot/cold hybrid as exp27)
  hot channels:  gate_full, up_full (full precision)
  cold channels: gate_enc, up_enc   (E5M3 B=8)
  routing:       SwiGLU-LR signal

Ranks swept: 64, 256, 1024.
Hot fractions: 0.20, 0.30, 0.50.

Usage::

    python tools/profiler/exp29_swiglu_lr_routing.py \\
        --model ibm-granite/granite-4.2-3b \\
        --calibration-set bartowski-imatrix-v5-semantic.txt
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
BLOCK_SIZE = 8
EPS = 1e-9


# ---------------------------------------------------------------------------
# E5M3 encoding (identical to exp22+)
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
# Low-rank helper
# ---------------------------------------------------------------------------

def _lowrank(xf: torch.Tensor,
             U: torch.Tensor, s: torch.Tensor, Vt: torch.Tensor,
             rank: int) -> torch.Tensor:
    """x @ W_lr.T  where W_lr = U[:,:r] diag(s[:r]) Vt[:r,:]."""
    r = min(rank, s.shape[0])
    return ((xf @ Vt[:r].T) * s[:r]) @ U[:, :r].T   # (T, I)


# ---------------------------------------------------------------------------
# Part 1 — routing quality hook (offline, no vLLM needed)
# ---------------------------------------------------------------------------

def routing_quality_from_npz(npz_path: str, svd_cache: str,
                              ranks: list[int], hot_fracs: list[float],
                              sample_layers: list[int]) -> None:
    """Compare SwiGLU-LR, union-LR, gate-only routing vs oracle."""
    data = np.load(npz_path)
    raw  = torch.load(svd_cache, weights_only=True)
    svd  = [(f[0].float(), f[1].float(), f[2].float(),
             f[3].float(), f[4].float(), f[5].float()) for f in raw]

    EPS2 = 1e-9

    def _metrics(hot_pred, hot_oracle):
        tp = ( hot_pred &  hot_oracle).float().sum().item()
        fp = ( hot_pred & ~hot_oracle).float().sum().item()
        fn = (~hot_pred &  hot_oracle).float().sum().item()
        prec   = tp / (tp + fp + EPS2)
        recall = tp / (tp + fn + EPS2)
        f1     = 2 * prec * recall / (prec + recall + EPS2)
        iou    = tp / (tp + fp + fn + EPS2)
        hot_a  = (tp + fp) / (hot_pred.numel() + EPS2)
        return dict(prec=prec, recall=recall, f1=f1, iou=iou, hot_frac=hot_a)

    def _topk(scores, k):
        k = min(k, scores.shape[-1])
        idx = torch.topk(scores.abs(), k, dim=-1, sorted=False).indices
        m = torch.zeros_like(scores, dtype=torch.bool)
        m.scatter_(-1, idx, True)
        return m

    # accumulators: [rank_idx][frac_idx] → list of metric dicts
    n_r, n_f = len(ranks), len(hot_fracs)
    acc_swiglu = [[[] for _ in range(n_f)] for _ in range(n_r)]
    acc_union  = [[[] for _ in range(n_f)] for _ in range(n_r)]
    acc_gate   = [[[] for _ in range(n_f)] for _ in range(n_r)]

    for li in sample_layers:
        X = torch.from_numpy(data[f"layer{li}/gate_up_input"]).float()  # (N, H)
        Y = torch.from_numpy(data[f"layer{li}/down_input"]).float()     # (N, I)
        U_g, s_g, Vt_g = svd[li][0], svd[li][1], svd[li][2]
        U_u, s_u, Vt_u = svd[li][3], svd[li][4], svd[li][5]

        for fi, frac in enumerate(hot_fracs):
            k = max(1, int(frac * Y.shape[1]))
            # Oracle: top-k by |SwiGLU_full| = |down_input|
            hot_oracle = _topk(Y, k)

            for ri, rank in enumerate(ranks):
                gate_lr = _lowrank(X, U_g, s_g, Vt_g, rank)   # (N, I)
                up_lr   = _lowrank(X, U_u, s_u, Vt_u, rank)   # (N, I)

                # A: SwiGLU-LR — combined signal
                swiglu_lr = F.silu(gate_lr) * up_lr            # (N, I)
                hot_swiglu = _topk(swiglu_lr, k)

                # B: Union-LR (exp27 style)
                hot_g = _topk(gate_lr, k)
                hot_u = _topk(up_lr, k)
                hot_union = hot_g | hot_u

                # C: Gate-only (exp22/23 style)
                hot_gate_only = _topk(gate_lr, k)

                acc_swiglu[ri][fi].append(_metrics(hot_swiglu, hot_oracle))
                acc_union [ri][fi].append(_metrics(hot_union,  hot_oracle))
                acc_gate  [ri][fi].append(_metrics(hot_gate_only, hot_oracle))

    def _avg(acc, ri, fi, key):
        return float(np.mean([d[key] for d in acc[ri][fi]]))

    for fi, frac in enumerate(hot_fracs):
        print(f"\n{'='*85}")
        print(f"Routing quality  hot={frac*100:.0f}%  k={max(1,int(frac*8192))}"
              f"  (avg layers {sample_layers})")
        print(f"oracle = top-k(|SiLU(gate_full)*up_full|)")
        print(f"{'='*85}")
        print(f"  {'signal':<22}  {'rank':>5}  {'hot%(A)':>8}  "
              f"{'prec':>7}  {'recall':>7}  {'F1':>7}  {'IoU':>7}")
        print(f"  {'-'*70}")
        for ri, rank in enumerate(ranks):
            for label, acc in [("SwiGLU-LR (this exp)", acc_swiglu),
                                ("Union-LR  (exp27)",   acc_union),
                                ("Gate-only (exp22/23)",acc_gate)]:
                print(f"  {label:<22}  {rank:>5}  "
                      f"{_avg(acc,ri,fi,'hot_frac')*100:>7.1f}%  "
                      f"{_avg(acc,ri,fi,'prec'):>7.4f}  "
                      f"{_avg(acc,ri,fi,'recall'):>7.4f}  "
                      f"{_avg(acc,ri,fi,'f1'):>7.4f}  "
                      f"{_avg(acc,ri,fi,'iou'):>7.4f}")
            print()


# ---------------------------------------------------------------------------
# Part 2 — e2e top-1 MLP hook
# ---------------------------------------------------------------------------

class SwiGLULRRoutingMLP:
    """Exp29: route on |SiLU(gate_lr)*up_lr|; hot=full-prec, cold=E5M3."""

    def __init__(self, mlp,
                 U_g, s_g, Vt_g,
                 U_u, s_u, Vt_u,
                 rank: int, k_hot: int):
        self._mlp   = mlp
        self._k_hot = k_hot

        r = min(rank, s_g.shape[0])
        self._U_g  = U_g[:, :r].float().contiguous()   # (I, r)
        self._s_g  = s_g[  :r].float()
        self._Vt_g = Vt_g[:r].float().contiguous()     # (r, H)
        self._U_u  = U_u[:, :r].float().contiguous()
        self._s_u  = s_u[  :r].float()
        self._Vt_u = Vt_u[:r].float().contiguous()

        W_fused = mlp.gate_up_proj.weight.detach().float()
        I = W_fused.shape[0] // 2
        self._I          = I
        self._W_gate_enc = build_e5m3_encoded(W_fused[:I])
        self._W_up_enc   = build_e5m3_encoded(W_fused[I:])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        xf = x.float()
        I  = self._I

        # Low-rank routing signal: |SiLU(gate_lr) * up_lr|
        gate_lr   = ((xf @ self._Vt_g.T) * self._s_g) @ self._U_g.T   # (T, I)
        up_lr     = ((xf @ self._Vt_u.T) * self._s_u) @ self._U_u.T
        signal    = (F.silu(gate_lr) * up_lr).abs()                    # (T, I)

        k = min(self._k_hot, I)
        idx = torch.topk(signal, k, dim=-1, sorted=False).indices
        hot = torch.zeros(xf.shape[0], I, dtype=torch.bool, device=xf.device)
        hot.scatter_(-1, idx, True)

        W_fused  = self._mlp.gate_up_proj.weight.detach().float()
        W_gate   = W_fused[:I]
        W_up     = W_fused[I:]
        W_down   = self._mlp.down_proj.weight.detach().float()

        gate_full = xf @ W_gate.T
        up_full   = xf @ W_up.T
        gate_cold = xf @ self._W_gate_enc.T
        up_cold   = xf @ self._W_up_enc.T

        gate   = torch.where(hot, gate_full, gate_cold)
        up     = torch.where(hot, up_full,   up_cold)
        return (F.silu(gate) * up @ W_down.T).to(orig_dtype)


# ---------------------------------------------------------------------------
# e2e infra (shared pattern)
# ---------------------------------------------------------------------------

class NormCapture:
    def __init__(self, W_U):
        self._W_U = W_U.float()
        self.ids: list[int] = []
        self._handle = None

    def attach(self, norm):
        self._handle = norm.register_forward_hook(self._hook)

    def detach(self):
        if self._handle:
            self._handle.remove()
            self._handle = None

    def _hook(self, module, args, output):
        self.ids.extend(
            (output.float() @ self._W_U.T).argmax(dim=-1).cpu().tolist())


def _load_prompts(path, n):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip()][:n]


def _get_internals(llm):
    e = llm.llm_engine
    try:
        mr = e.model_executor.driver_worker.worker.model_runner
    except AttributeError:
        mr = e.model_executor.driver_worker.model_runner
    return mr.model.model.norm, mr.model.model.layers, mr.model.lm_head.weight


def _run(llm, prompts, W_U, norm):
    from vllm import SamplingParams
    cap = NormCapture(W_U.detach().float())
    cap.attach(norm)
    llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0.0),
                 use_tqdm=False)
    cap.detach()
    ids = cap.ids
    return np.array(ids[:-len(prompts)] if len(prompts) < len(ids) else ids,
                    dtype=np.int32)


def _load_svd(cache_path, max_rank):
    raw = torch.load(cache_path, weights_only=True)
    return [(f[0].float(), f[1].float(), f[2].float(),
             f[3].float(), f[4].float(), f[5].float()) for f in raw]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Exp29: SwiGLU-LR routing + E5M3 cold hybrid.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", default="ibm-granite/granite-4.2-3b")
    p.add_argument("--calibration-set", default="bartowski-imatrix-v5-semantic.txt")
    p.add_argument("--num-prompts", type=int, default=8)
    p.add_argument("--ranks", nargs="+", type=int, default=[64, 256, 1024])
    p.add_argument("--hot-fractions", nargs="+", type=float,
                   default=[0.20, 0.30, 0.50])
    p.add_argument("--svd-cache", default="svd_factors_r1024.pt")
    p.add_argument("--activations", default="ffn_activations128.npz",
                   help="NPZ file for Part 1 routing quality analysis.")
    p.add_argument("--quality-layers", nargs="+", type=int,
                   default=list(range(0, 40, 4)),
                   help="Layers to use for Part 1 quality analysis.")
    p.add_argument("--skip-quality", action="store_true",
                   help="Skip Part 1 and go straight to e2e.")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = _parse_args(argv)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    # ----------------------------------------------------------------
    # Part 1 — routing quality (no model load needed)
    # ----------------------------------------------------------------
    if not args.skip_quality:
        print("=" * 60, file=sys.stderr)
        print("Part 1: routing quality analysis", file=sys.stderr)
        routing_quality_from_npz(
            npz_path=args.activations,
            svd_cache=args.svd_cache,
            ranks=args.ranks,
            hot_fracs=args.hot_fractions,
            sample_layers=args.quality_layers,
        )

    # ----------------------------------------------------------------
    # Part 2 — e2e top-1
    # ----------------------------------------------------------------
    print("\n" + "=" * 60, file=sys.stderr)
    print("Part 2: e2e top-1 perturbation", file=sys.stderr)

    prompts = _load_prompts(args.calibration_set, args.num_prompts)
    print(f"Loaded {len(prompts)} prompts.", file=sys.stderr)

    from vllm import LLM
    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              kv_cache_memory_bytes=int(0.5 * 1024**3), max_model_len=512,
              enable_prefix_caching=False)

    norm, layers, W_U = _get_internals(llm)
    I = layers[0].mlp.gate_up_proj.weight.shape[0] // 2
    orig_forwards = [l.mlp.forward for l in layers]

    svd = _load_svd(args.svd_cache, max(args.ranks))

    print("Baseline pass ...", file=sys.stderr)
    baseline = _run(llm, prompts, W_U, norm)
    n_tok = len(baseline)
    print(f"  {n_tok} tokens.", file=sys.stderr)

    # Reference points
    exp24_ref = {0.20: 0.818, 0.30: 0.798, 0.50: 0.736}
    exp27_ref = {(64,0.20):0.392,(64,0.30):0.472,(64,0.50):0.678,
                 (256,0.20):0.520,(256,0.30):0.606,(256,0.50):0.764,
                 (1024,0.20):0.612,(1024,0.30):0.702,(1024,0.50):0.834}

    results: dict[tuple, float] = {}
    n_total = len(args.ranks) * len(args.hot_fractions)
    idx = 0

    for rank in args.ranks:
        for frac in args.hot_fractions:
            idx += 1
            k_hot = max(1, int(frac * I))
            mlps = [
                SwiGLULRRoutingMLP(
                    layers[li].mlp,
                    U_g=svd[li][0], s_g=svd[li][1], Vt_g=svd[li][2],
                    U_u=svd[li][3], s_u=svd[li][4], Vt_u=svd[li][5],
                    rank=rank, k_hot=k_hot,
                )
                for li in range(len(layers))
            ]
            for l, m in zip(layers, mlps):
                l.mlp.forward = m

            print(f"  [{idx}/{n_total}] rank={rank:4d}  hot={frac*100:.0f}% ...",
                  end="  ", file=sys.stderr, flush=True)
            ids = _run(llm, prompts, W_U, norm)[:n_tok]
            match = float((ids == baseline).mean())
            results[(rank, frac)] = match
            print(f"match={match:.4f}  perturb={1-match:.4f}", file=sys.stderr)

            for l, fwd in zip(layers, orig_forwards):
                l.mlp.forward = fwd
            del mlps

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    fracs = args.hot_fractions
    hdr   = f"  {'rank':>6}" + "".join(f"  {f*100:>5.0f}%" for f in fracs)

    print("\n" + "="*75)
    print("Exp29: SwiGLU-LR routing — top-k(|SiLU(gate_lr)*up_lr|)")
    print("cold: E5M3 B=8 for gate and up")
    print("="*75)
    print("\nTop-1 match (↑ better):")
    print(hdr)
    for rank in args.ranks:
        row = f"  {rank:>6}" + "".join(f"  {results[(rank,f)]:>6.4f}" for f in fracs)
        print(row)
    print(f"  {'exp24':>6}" + "".join(f"  {exp24_ref.get(f,float('nan')):>6.4f}" for f in fracs))
    print(f"  {'exp27':>6} (union-LR):")
    for rank in args.ranks:
        row = f"  {rank:>6}" + "".join(
            f"  {exp27_ref.get((rank,f),float('nan')):>6.4f}" for f in fracs)
        print(row)

    print("\nΔ vs exp27 same rank (+ = exp29 better):")
    print(hdr)
    for rank in args.ranks:
        row = f"  {rank:>6}" + "".join(
            f"  {results[(rank,f)] - exp27_ref.get((rank,f), float('nan')):>+6.4f}"
            for f in fracs)
        print(row)
    print("="*75)


if __name__ == "__main__":
    main()
