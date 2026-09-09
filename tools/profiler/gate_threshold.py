# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gate-threshold sparsity: skip up/down projection for neurons whose
gate_raw < threshold.

Two modes:

  threshold mode (default)
    Sweep one or more explicit gate_raw thresholds.  For each (layer, threshold)
    pair report: fraction inactive, compute saved, MSE, cosim_mean, cosim_p5.

  target mode (--target-cosim)
    For each layer binary-search the highest threshold T such that the chosen
    cosim metric (mean or p5, controlled by --target-metric) stays >= the
    requested floor.  Reports the derived threshold, inactive fraction, and
    compute saved alongside the quality metrics.

Per-layer output columns:
  threshold     -- gate_raw cutoff used
  inactive      -- mean fraction of neurons skipped per token
  compute_saved -- MLP compute saving = 2 * inactive_frac / 3
                   (gate always fully computed; only up+down are partial)
  MSE           -- mean squared error of sparse vs full down_proj output
  cosim         -- cosine similarity between sparse and full output (scale-free,
                   layer-comparable; equivalent to quality metric in gate_predict.py)

Usage::

    # Threshold mode: single value, all layers
    python tools/profiler/gate_threshold.py \\
        --model ibm-granite/granite-4.2-3b \\
        --npz   ffn_activations128_gate.npz \\
        --threshold 0.0

    # Threshold mode: sweep
    python tools/profiler/gate_threshold.py \\
        --model ibm-granite/granite-4.2-3b \\
        --npz   ffn_activations128_gate.npz \\
        --threshold -1.0 -0.5 0.0 0.5 1.0

    # Target mode: find threshold that keeps cosim_mean >= 0.95
    python tools/profiler/gate_threshold.py \\
        --model ibm-granite/granite-4.2-3b \\
        --npz   ffn_activations128_gate.npz \\
        --target-cosim 0.95

    # Target mode: use cosim_p5 as the quality metric, floor 0.90
    python tools/profiler/gate_threshold.py \\
        --model ibm-granite/granite-4.2-3b \\
        --npz   ffn_activations128_gate.npz \\
        --target-cosim 0.90 --target-metric p5

    # Target mode: restrict to specific layers
    python tools/profiler/gate_threshold.py \\
        --model ibm-granite/granite-4.2-3b \\
        --npz   ffn_activations128_gate.npz \\
        --target-cosim 0.95 \\
        --layers 0 15 33 39
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def _find_model_dir(model_name_or_path: str) -> Path:
    p = Path(model_name_or_path)
    if p.exists():
        return p
    slug = model_name_or_path.replace("/", "--")
    candidates = sorted(
        glob.glob(str(Path.home() / ".cache/huggingface/hub"
                      / f"models--{slug}/snapshots/*/"))
    )
    if not candidates:
        raise FileNotFoundError(
            f"Could not find model '{model_name_or_path}' locally."
        )
    return Path(candidates[-1])


def _load_mlp_weights(
    model_dir: Path, layer_idx: int, device: torch.device, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    pfx = f"model.layers.{layer_idx}.mlp."
    needed = {pfx + k for k in ("gate_proj.weight", "up_proj.weight", "down_proj.weight")}
    found: dict[str, torch.Tensor] = {}
    for shard in sorted(model_dir.glob("model*.safetensors")):
        if not needed:
            break
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in list(needed):
                if key in f.keys():
                    found[key] = f.get_tensor(key).to(device=device, dtype=dtype)
                    needed.discard(key)
    if needed:
        raise KeyError(f"Missing weights for layer {layer_idx}: {needed}")
    return {
        "gate": found[pfx + "gate_proj.weight"],
        "up":   found[pfx + "up_proj.weight"],
        "down": found[pfx + "down_proj.weight"],
    }


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_threshold(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    threshold: float,
) -> dict[str, float]:
    """Run one (layer, threshold) evaluation.

    Args:
        x: [T, H] gate_up_input hidden states.
        weights: dict with keys "gate", "up", "down".
        threshold: neurons with gate_raw < threshold are skipped.

    Returns:
        Dict with inactive_frac, compute_saved, mse_mean, mse_std, cosim_mean, cosim_p5.
    """
    gate_w, up_w, down_w = weights["gate"], weights["up"], weights["down"]
    T, H = x.shape
    I = gate_w.shape[0]

    # Full gate projection (always computed)
    gate_raw = x @ gate_w.T          # [T, I]

    # Active mask: gate_raw >= threshold
    active = gate_raw >= threshold    # [T, I] bool

    # Full reference output
    full_out = (F.silu(gate_raw) * (x @ up_w.T)) @ down_w.T  # [T, H]

    # Sparse output: zero out inactive neurons before up/down
    active_f = active.float()
    sparse_gate = F.silu(gate_raw) * active_f        # [T, I]
    sparse_up   = (x @ up_w.T) * active_f            # [T, I]
    sparse_out  = (sparse_gate * sparse_up) @ down_w.T  # [T, H]

    # Metrics
    inactive_frac = (~active).float().mean().item()
    compute_saved = 2.0 * inactive_frac / 3.0

    diff = sparse_out - full_out                     # [T, H]
    mse_per_token = diff.pow(2).mean(dim=1)          # [T]
    mse_mean = mse_per_token.mean().item()
    mse_std  = mse_per_token.std().item()

    cs = F.cosine_similarity(sparse_out, full_out, dim=1)  # [T]

    return {
        "inactive_frac": inactive_frac,
        "compute_saved": compute_saved,
        "mse_mean":      mse_mean,
        "mse_std":       mse_std,
        "cosim_mean":    cs.mean().item(),
        "cosim_p5":      cs.quantile(0.05).item(),
    }


# ---------------------------------------------------------------------------
# Target-cosim mode: binary-search threshold
# ---------------------------------------------------------------------------

def find_threshold_for_cosim(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    target: float,
    metric: str = "mean",
    tol: float = 1e-3,
    max_iter: int = 40,
) -> dict[str, float]:
    """Binary-search the highest gate_raw threshold that keeps cosim >= target.

    Searches over the range [gate_raw.min(), gate_raw.max()].  A higher
    threshold means more neurons are pruned; the search finds the maximum T
    where quality is still acceptable.

    Args:
        x: [T, H] gate_up_input hidden states.
        weights: dict with keys "gate", "up", "down".
        target: minimum acceptable cosim value.
        metric: "mean" or "p5" — which cosim statistic to constrain.
        tol: convergence tolerance on threshold (default 1e-3).
        max_iter: maximum bisection iterations (default 40).

    Returns:
        Same dict as evaluate_threshold, plus "threshold".
    """
    gate_w = weights["gate"]

    # Pre-compute gate_raw and full outputs once — reused across all iterations
    gate_raw = x @ gate_w.T                                          # [T, I]
    up_out   = x @ weights["up"].T                                   # [T, I]
    full_out = (F.silu(gate_raw) * up_out) @ weights["down"].T       # [T, H]
    silu_raw = F.silu(gate_raw)                                      # [T, I]

    lo = gate_raw.min().item()
    hi = gate_raw.max().item()

    def _cosim_at(thr: float) -> float:
        active_f = (gate_raw >= thr).float()
        sparse_out = (silu_raw * active_f * up_out) @ weights["down"].T
        cs = F.cosine_similarity(sparse_out, full_out, dim=1)
        return cs.mean().item() if metric == "mean" else cs.quantile(0.05).item()

    # Quick sanity: if even T=lo doesn't meet the floor, return T=lo
    if _cosim_at(lo) < target:
        best_thr = lo
    else:
        # Bisect: find highest T where cosim >= target
        best_thr = lo
        for _ in range(max_iter):
            mid = (lo + hi) / 2.0
            if _cosim_at(mid) >= target:
                best_thr = mid
                lo = mid
            else:
                hi = mid
            if (hi - lo) < tol:
                break

    return {**evaluate_threshold(x, weights, best_thr), "threshold": best_thr}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gate-threshold MLP sparsity: skip up/down for gate_raw < T.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", required=True,
                   help="HuggingFace model ID or local path.")
    p.add_argument("--npz", required=True,
                   help=".npz file from record_ffn_activations.py.")

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--threshold", nargs="+", type=float, default=None,
                      metavar="T",
                      help="Threshold mode: gate_raw threshold(s). Neurons with "
                           "gate_raw < T are skipped. Multiple values produce a "
                           "sweep. (default when no mode flag given: 0.0)")
    mode.add_argument("--target-cosim", type=float, default=None,
                      metavar="C",
                      help="Target mode: binary-search the highest threshold that "
                           "keeps cosim >= C per layer.")

    p.add_argument("--target-metric", default="mean", choices=["mean", "p5"],
                   help="Which cosim statistic to constrain in target mode "
                        "(default: mean).")
    p.add_argument("--layers", nargs="*", type=int, default=None, metavar="N",
                   help="Layer indices to evaluate. Default: all layers in npz.")
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "bfloat16"],
                   help="Compute dtype (default: float32).")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda", "mps"],
                   help="Torch device (default: auto).")
    return p.parse_args(argv)


def _pick_device(choice: str) -> torch.device:
    if choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _print_row(li_label: str, thr: float, r: dict) -> None:
    thr_s = f"{thr:>+10.4f}" if thr == thr else f"{'nan':>10}"  # nan check
    print(
        f"{li_label:>5}  {thr_s}  "
        f"{r['inactive_frac']*100:>8.1f}%  "
        f"{r['compute_saved']*100:>8.1f}%  "
        f"{r['mse_mean']:>12.6f}  "
        f"{r['mse_std']:>12.6f}  "
        f"{r['cosim_mean']:>8.5f}  "
        f"{r['cosim_p5']:>9.5f}"
    )


def main(argv=None) -> None:
    args = _parse_args(argv)

    device = _pick_device(args.device)
    dtype  = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    print(f"Device: {device}  dtype: {dtype}", file=sys.stderr)

    try:
        model_dir = _find_model_dir(args.model)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Weights: {model_dir}", file=sys.stderr)

    data = np.load(args.npz)
    available = sorted(
        int(k.split("/")[0][5:]) for k in data.files if k.endswith("/gate_up_input")
    )
    layers = args.layers if args.layers is not None else available
    layers = [l for l in layers if l in available]
    if not layers:
        print("ERROR: no matching layers found in npz.", file=sys.stderr)
        sys.exit(1)

    hdr = (f"{'layer':>5}  {'threshold':>10}  {'inactive':>9}  {'saved':>9}  "
           f"{'MSE':>12}  {'MSE_std':>12}  {'cosim':>8}  {'cosim_p5':>9}")
    sep = "─" * len(hdr)

    # ------------------------------------------------------------------ #
    # Target mode                                                          #
    # ------------------------------------------------------------------ #
    if args.target_cosim is not None:
        target = args.target_cosim
        metric = args.target_metric
        print(f"\ntarget cosim_{metric} >= {target:.4f}")
        print(hdr)
        print(sep)

        totals: dict[str, list] = {
            "inactive": [], "saved": [], "mse": [], "cosim": [], "cosim_p5": [],
        }
        for li in layers:
            x = torch.from_numpy(
                data[f"layer{li}/gate_up_input"].astype(np.float32)
            ).to(device=device, dtype=dtype)
            try:
                w = _load_mlp_weights(model_dir, li, device, dtype)
            except Exception as e:
                print(f"{li:>5}  {'WEIGHT LOAD ERROR':>52}  {e}", file=sys.stderr)
                continue

            r = find_threshold_for_cosim(x, w, target, metric=metric)

            totals["inactive"].append(r["inactive_frac"])
            totals["saved"].append(r["compute_saved"])
            totals["mse"].append(r["mse_mean"])
            totals["cosim"].append(r["cosim_mean"])
            totals["cosim_p5"].append(r["cosim_p5"])

            _print_row(str(li), r["threshold"], r)

        n = len(totals["inactive"])
        if n:
            print(sep)
            _print_row(
                "mean",
                float("nan"),
                {
                    "inactive_frac": sum(totals["inactive"]) / n,
                    "compute_saved": sum(totals["saved"]) / n,
                    "mse_mean":      sum(totals["mse"]) / n,
                    "mse_std":       float("nan"),
                    "cosim_mean":    sum(totals["cosim"]) / n,
                    "cosim_p5":      sum(totals["cosim_p5"]) / n,
                },
            )
        return

    # ------------------------------------------------------------------ #
    # Threshold mode                                                       #
    # ------------------------------------------------------------------ #
    thresholds = sorted(args.threshold if args.threshold is not None else [0.0])

    thr_totals: dict[float, dict[str, list]] = {
        t: {"inactive": [], "saved": [], "mse": [], "cosim": [], "cosim_p5": []}
        for t in thresholds
    }

    for thr in thresholds:
        print(f"\nthreshold = {thr:+.4f}")
        print(hdr)
        print(sep)

        for li in layers:
            x = torch.from_numpy(
                data[f"layer{li}/gate_up_input"].astype(np.float32)
            ).to(device=device, dtype=dtype)
            try:
                w = _load_mlp_weights(model_dir, li, device, dtype)
            except Exception as e:
                print(f"{li:>5}  {thr:>+10.4f}  {'WEIGHT LOAD ERROR':>40}  {e}",
                      file=sys.stderr)
                continue

            r = evaluate_threshold(x, w, thr)

            thr_totals[thr]["inactive"].append(r["inactive_frac"])
            thr_totals[thr]["saved"].append(r["compute_saved"])
            thr_totals[thr]["mse"].append(r["mse_mean"])
            thr_totals[thr]["cosim"].append(r["cosim_mean"])
            thr_totals[thr]["cosim_p5"].append(r["cosim_p5"])

            _print_row(str(li), thr, r)

        vals = thr_totals[thr]
        n = len(vals["inactive"])
        if n:
            print(sep)
            _print_row(
                "mean",
                thr,
                {
                    "inactive_frac": sum(vals["inactive"]) / n,
                    "compute_saved": sum(vals["saved"]) / n,
                    "mse_mean":      sum(vals["mse"]) / n,
                    "mse_std":       float("nan"),
                    "cosim_mean":    sum(vals["cosim"]) / n,
                    "cosim_p5":      sum(vals["cosim_p5"]) / n,
                },
            )


if __name__ == "__main__":
    main()
