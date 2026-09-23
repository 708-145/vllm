# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment 13 – logit-space metrics for best configurations from exp10–12.

Motivation
----------
Experiments 10–12 used output cosine similarity in hidden space as the primary
metric.  This is magnitude-blind and not directly tied to top-1 token
prediction quality.  This experiment re-evaluates the best configurations from
each prior experiment under four logit-space metrics computed via a single
W_U GEMM:

  - top1_match  : fraction of tokens where argmax(logit_hybrid) == argmax(logit_full)
  - kl_div      : KL(softmax(logit_full) || softmax(logit_hybrid))
  - logit_cos   : cosine similarity in logit space
  - hidden_cos  : cosine similarity in hidden space (for reference / continuity)

Configurations evaluated (all use α_gate=0.75, full-precision up, full W_down):

  A. exp10-best  : pre-SiLU routing on |gate_approx[t]|  (current token)
  B. exp11-proxy : pre-SiLU routing on |gate_approx[t-1]| (proxy-prior token)
  C. exp11-oracle: pre-SiLU routing on |gate_full[t-1]|   (oracle-prior token)

Hot fractions: {0.5%, 1%, 2%, 5%, 10%, 20%, 30%}

W_U GEMM is expensive at vocab=100352; we subsample to 512 tokens per layer.

Usage::

    python tools/profiler/exp13_logit_metrics.py \\
        --model ibm-granite/granite-4.2-3b \\
        --act-file ffn_activations128_gate.npz
"""

import argparse
import os
import sys

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


def t(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(a).to(DEV, dtype=torch.float32)


def top_k_mask(scores: torch.Tensor, k: int) -> torch.BoolTensor:
    idx = torch.topk(scores, k, dim=1, largest=True, sorted=False).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask


def ternary(W: torch.Tensor, tau: float) -> torch.Tensor:
    return W.sign() * (W.abs() >= tau).float()


def logit_metrics(
    out_full:   torch.Tensor,   # (T, H)
    out_hybrid: torch.Tensor,   # (T, H)
    W_U:        torch.Tensor,   # (vocab, H)
) -> dict[str, float]:
    """KL div, top-1 match, logit cos-sim, hidden cos-sim."""
    lf = out_full.float()   @ W_U.T.float()   # (T, vocab)
    lh = out_hybrid.float() @ W_U.T.float()
    # Stabilise softmax: subtract per-row max before exp to prevent overflow
    pf = torch.softmax(lf - lf.max(dim=-1, keepdim=True).values, dim=-1)
    ph = torch.softmax(lh - lh.max(dim=-1, keepdim=True).values, dim=-1)
    kl         = float((pf * (pf / (ph + 1e-9)).log()).sum(-1).mean())
    top1_match = float((lf.argmax(-1) == lh.argmax(-1)).float().mean())
    logit_cos  = float(F.cosine_similarity(lf, lh, dim=-1).mean())
    hidden_cos = float(F.cosine_similarity(out_full, out_hybrid, dim=-1).mean())
    return dict(kl=kl, top1=top1_match, lcos=logit_cos, hcos=hidden_cos)


def get_weights(model, layer_idx: int):
    mlp = model.model.layers[layer_idx].mlp
    W = mlp.gate_up_proj.weight.detach().float().numpy()
    I = W.shape[0] // 2
    return W[:I], W[I:], mlp.down_proj.weight.detach().float().numpy()


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Experiment 13: logit-space metrics for exp10–12 best configs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", default="ibm-granite/granite-4.2-3b")
    p.add_argument("--act-file", default="ffn_activations128_gate.npz")
    p.add_argument("--layers", nargs="*", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=512,
                   help="Tokens per layer (W_U GEMM is expensive; default 512).")
    p.add_argument(
        "--hot-fractions", nargs="+", type=float,
        default=[0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30],
        metavar="F",
    )
    p.add_argument("--tau-gate", type=float, default=0.75)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    print(f"Using device: {DEV}", file=sys.stderr)

    act_data = np.load(args.act_file)
    layers = sorted(
        int(k.split("/")[0].replace("layer", ""))
        for k in act_data.files if k.endswith("/gate_raw")
    )
    if args.layers is not None:
        layers = [l for l in layers if l in args.layers]
    print(
        f"Evaluating {len(layers)} layers, "
        f"{len(args.hot_fractions)} hot-fractions, "
        f"{args.max_tokens} tokens/layer.",
        file=sys.stderr,
    )

    from vllm import LLM
    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              kv_cache_memory_bytes=int(0.5 * 1024**3), max_model_len=512)
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model

    W_U = model.lm_head.weight.detach().float().to(DEV)  # (vocab, H)

    configs = ("exp10_current", "exp11_proxy", "exp11_oracle")
    metric_keys = ("kl", "top1", "lcos", "hcos")

    # results[config][frac][metric] = list of per-layer values
    results: dict[str, dict[float, dict[str, list[float]]]] = {
        c: {f: {m: [] for m in metric_keys} for f in args.hot_fractions}
        for c in configs
    }

    for layer_idx in layers:
        pfx = f"layer{layer_idx}"
        x_np    = act_data[f"{pfx}/gate_up_input"][: args.max_tokens].astype(np.float32)
        graw_np = act_data[f"{pfx}/gate_raw"][: args.max_tokens].astype(np.float32)

        W_gate_np, W_up_np, W_down_np = get_weights(model, layer_idx)

        x        = t(x_np)
        gate_raw = t(graw_np)
        W_gate   = t(W_gate_np)
        W_up     = t(W_up_np)
        W_down   = t(W_down_np)
        I        = W_gate.shape[0]

        tau_gate    = args.tau_gate * float(W_gate.abs().mean())
        T_gate      = ternary(W_gate, tau_gate)

        gate_full   = x @ W_gate.T     # (T, I)
        up_full     = x @ W_up.T       # (T, I)
        gate_approx = x @ T_gate.T     # (T, I)

        swiglu_full = F.silu(gate_raw) * up_full   # (T, I)
        out_full    = swiglu_full @ W_down.T        # (T, H)

        # Adjacent-token slices (prior=0..T-2, current=1..T-1)
        T_pairs           = x.shape[0] - 1
        gate_approx_prior = gate_approx[:T_pairs]
        gate_full_prior   = gate_full[:T_pairs]
        gate_approx_cur   = gate_approx[1:]
        gate_full_cur     = gate_full[1:]
        up_full_cur       = up_full[1:]
        gate_raw_cur      = gate_raw[1:]
        out_full_cur      = out_full[1:]

        routing_signals = {
            "exp10_current": gate_approx_cur.abs(),
            "exp11_proxy":   gate_approx_prior.abs(),
            "exp11_oracle":  gate_full_prior.abs(),
        }

        for frac in args.hot_fractions:
            k_hot = max(1, int(frac * I))

            for cfg, signal in routing_signals.items():
                hot           = top_k_mask(signal, k_hot)
                gate_hybrid   = torch.where(hot, gate_full_cur, gate_approx_cur)
                swiglu_hybrid = F.silu(gate_hybrid) * up_full_cur
                out_hybrid    = swiglu_hybrid @ W_down.T

                m = logit_metrics(out_full_cur, out_hybrid, W_U)
                for mk in metric_keys:
                    results[cfg][frac][mk].append(m[mk])

        print(
            f"  layer {layer_idx:3d}: "
            f"top1@10%  exp10={results['exp10_current'][0.10]['top1'][-1]:.4f}  "
            f"proxy={results['exp11_proxy'][0.10]['top1'][-1]:.4f}  "
            f"oracle={results['exp11_oracle'][0.10]['top1'][-1]:.4f}",
            file=sys.stderr,
        )

        del x, gate_raw, W_gate, W_up, W_down, T_gate
        del gate_full, up_full, gate_approx, swiglu_full, out_full
        if DEV.type == "mps":
            torch.mps.empty_cache()
        elif DEV.type == "cuda":
            torch.cuda.empty_cache()

    # --- Summary ---
    print("\n--- Experiment 13 Results ---\n")

    cfg_labels = {
        "exp10_current": "exp10 current |gate_approx[t]|",
        "exp11_proxy":   "exp11 proxy   |gate_approx[t-1]|",
        "exp11_oracle":  "exp11 oracle  |gate_full[t-1]|",
    }

    for mk, mk_label in [
        ("top1",  "Top-1 preservation rate (higher = better)"),
        ("kl",    "KL divergence full||hybrid (lower = better; NaN = diverged layer excluded)"),
        ("lcos",  "Logit cosine similarity (higher = better)"),
        ("hcos",  "Hidden cosine similarity (higher = better, ref)"),
    ]:
        print(f"\n{mk_label}:")
        hdr = f"  {'config':<38}" + "".join(
            f"  {f*100:>5.1f}%" for f in args.hot_fractions)
        print(hdr)
        for cfg in configs:
            row = f"  {cfg_labels[cfg]:<38}"
            for frac in args.hot_fractions:
                vals = np.array(results[cfg][frac][mk], dtype=np.float64)
                row += f"  {np.nanmean(vals):>6.4f}"
            print(row)

    # Best config per hot fraction (by top-1)
    print(f"\nBest config per hot fraction — Top-1 preservation rate:")
    print(f"  {'hot%':>6}  {'best config':>22}  {'top1':>8}  {'kl (nanmean)':>14}  "
          f"{'lcos':>8}  {'hcos':>8}")
    for frac in args.hot_fractions:
        best_cfg = max(configs, key=lambda c: np.mean(results[c][frac]["top1"]))
        top1 = np.mean(results[best_cfg][frac]["top1"])
        kl   = np.nanmean(np.array(results[best_cfg][frac]["kl"], dtype=np.float64))
        lcos = np.mean(results[best_cfg][frac]["lcos"])
        hcos = np.mean(results[best_cfg][frac]["hcos"])
        print(f"  {frac*100:>6.1f}%  {best_cfg:>22}  {top1:>8.4f}  {kl:>14.4f}  "
              f"{lcos:>8.4f}  {hcos:>8.4f}")

    # Per-layer detail at hot=10%, proxy config
    print(f"\nPer-layer detail: hot=10%, α_gate={args.tau_gate}:")
    print(f"  {'layer':>5}  {'exp10 top1':>11}  {'proxy top1':>11}  "
          f"{'oracle top1':>12}  {'exp10 kl':>10}  {'proxy kl':>10}")
    for i, li in enumerate(layers):
        t10  = results["exp10_current"][0.10]["top1"][i]
        tp   = results["exp11_proxy"]  [0.10]["top1"][i]
        tor  = results["exp11_oracle"] [0.10]["top1"][i]
        k10  = results["exp10_current"][0.10]["kl"][i]
        kp   = results["exp11_proxy"]  [0.10]["kl"][i]
        print(f"  {li:5d}  {t10:11.4f}  {tp:11.4f}  {tor:12.4f}  "
              f"{k10:10.6f}  {kp:10.6f}")


if __name__ == "__main__":
    main()
