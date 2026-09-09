# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gate-threshold sparsity: skip up/down projection for neurons whose
gate_raw < threshold.

Computes the full gate projection, applies a scalar threshold on the
pre-SiLU gate logits to decide which neurons are "inactive", and runs
up_proj + down_proj only on the active neurons.  Compares the sparse
output to the full-compute reference and reports MSE.

Per-layer output columns:
  threshold   -- the gate_raw cutoff used
  inactive    -- mean fraction of neurons skipped per token
  compute_saved -- MLP compute saving = 2 * inactive_frac / 3
  MSE         -- mean squared error of sparse vs full down_proj output

Usage::

    # Single threshold, all layers
    python tools/profiler/gate_threshold.py \\
        --model ibm-granite/granite-4.2-3b \\
        --npz   ffn_activations128_gate.npz \\
        --threshold 0.0

    # Sweep multiple thresholds
    python tools/profiler/gate_threshold.py \\
        --model ibm-granite/granite-4.2-3b \\
        --npz   ffn_activations128_gate.npz \\
        --threshold -1.0 -0.5 0.0 0.5 1.0

    # Restrict layers
    python tools/profiler/gate_threshold.py \\
        --model ibm-granite/granite-4.2-3b \\
        --npz   ffn_activations128_gate.npz \\
        --threshold 0.0 \\
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
        Dict with inactive_frac, compute_saved, mse_mean, mse_std.
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

    return {
        "inactive_frac": inactive_frac,
        "compute_saved": compute_saved,
        "mse_mean":      mse_mean,
        "mse_std":       mse_std,
    }


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
    p.add_argument("--threshold", nargs="+", type=float, default=[0.0],
                   metavar="T",
                   help="gate_raw threshold(s). Neurons with gate_raw < T are "
                        "skipped. Multiple values produce a sweep (default: 0.0).")
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

    thresholds = sorted(args.threshold)

    hdr  = f"{'layer':>5}  {'threshold':>10}  {'inactive':>9}  {'saved':>9}  {'MSE':>12}  {'MSE_std':>12}"
    sep  = "─" * len(hdr)

    # Accumulate per-threshold totals for the summary line
    totals: dict[float, dict[str, list]] = {
        t: {"inactive": [], "saved": [], "mse": []} for t in thresholds
    }

    prev_thr: float | None = None
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

            totals[thr]["inactive"].append(r["inactive_frac"])
            totals[thr]["saved"].append(r["compute_saved"])
            totals[thr]["mse"].append(r["mse_mean"])

            print(
                f"{li:>5}  {thr:>+10.4f}  "
                f"{r['inactive_frac']*100:>8.1f}%  "
                f"{r['compute_saved']*100:>8.1f}%  "
                f"{r['mse_mean']:>12.6f}  "
                f"{r['mse_std']:>12.6f}"
            )

        vals = totals[thr]
        if vals["inactive"]:
            n = len(vals["inactive"])
            print(sep)
            print(
                f"{'mean':>5}  {thr:>+10.4f}  "
                f"{sum(vals['inactive'])/n*100:>8.1f}%  "
                f"{sum(vals['saved'])/n*100:>8.1f}%  "
                f"{sum(vals['mse'])/n:>12.6f}"
            )


if __name__ == "__main__":
    main()
