# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline validation of KV-retrieval-based FFN activation prediction.

Given a .npz file produced by record_ffn_activations.py, this script
evaluates how well an OR-of-top-K nearest-neighbour lookup can predict
which intermediate neurons will be active, without running the actual
MLP computation.

For each query token the predictor:
  1. Finds the K most similar stored input vectors (cosine similarity).
  2. ORs their binarised activation masks together.
  3. Compares the predicted mask against the ground-truth mask.

Reported per-layer metrics:
  recall    – fraction of truly active neurons that were predicted active
              (false negatives corrupt output; this is the safety metric)
  precision – fraction of predicted-active neurons that are truly active
              (false positives waste compute)
  density   – fraction of neurons predicted active (= sparsity cost)

Usage::

    # Basic: evaluate layer 0 with default settings
    python tools/profiler/O1_predict.py --input ffn_activations.npz

    # Sweep K and threshold across layers 0 and 15
    python tools/profiler/O1_predict.py \\
        --input ffn_activations.npz \\
        --layers 0 15 \\
        --top-k 1 3 5 \\
        --threshold-pct 70 80 90

    # Save per-token detail to CSV
    python tools/profiler/O1_predict.py \\
        --input ffn_activations.npz \\
        --output-csv results.csv
"""

import argparse
import contextlib
import csv
import sys
from itertools import product

import numpy as np

# ---------------------------------------------------------------------------
# Core prediction logic
# ---------------------------------------------------------------------------


def cosine_similarity_matrix(a: np.ndarray) -> np.ndarray:
    """Return [T, T] pairwise cosine similarity for row vectors in a."""
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    a_norm = a / norms
    return a_norm @ a_norm.T


def binarise(acts: np.ndarray, threshold_pct: float) -> np.ndarray:
    """Return boolean mask: True where |act| exceeds the per-token percentile.

    Args:
        acts: shape [T, I] float32 activation matrix.
        threshold_pct: percentile (0–100) used as the activity threshold.
            E.g. 70 keeps the top-30 % of neurons per token.

    Returns:
        Boolean array of shape [T, I].
    """
    thresholds = np.percentile(np.abs(acts), threshold_pct, axis=1, keepdims=True)
    return np.abs(acts) > thresholds


def predict_masks(
    inputs: np.ndarray,
    masks: np.ndarray,
    top_k: int,
    sim: np.ndarray | None = None,
) -> np.ndarray:
    """Predict activation masks via OR of top-K nearest neighbours.

    Args:
        inputs: shape [T, H] input hidden states.
        masks: shape [T, I] ground-truth binary masks.
        top_k: number of nearest neighbours to retrieve and OR.
        sim: precomputed [T, T] cosine similarity matrix; computed if None.

    Returns:
        predicted: boolean array of shape [T, I].
    """
    if sim is None:
        sim = cosine_similarity_matrix(inputs)

    T = inputs.shape[0]
    predicted = np.zeros_like(masks)

    for i in range(T):
        row = sim[i].copy()
        row[i] = -2.0  # exclude self
        neighbours = np.argpartition(row, -top_k)[-top_k:]
        predicted[i] = masks[neighbours].any(axis=0)

    return predicted


def evaluate(
    masks_true: np.ndarray,
    masks_pred: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute per-token recall, precision, and predicted density.

    Args:
        masks_true: shape [T, I] ground-truth boolean masks.
        masks_pred: shape [T, I] predicted boolean masks.

    Returns:
        Dict with keys ``recall``, ``precision``, ``density``, each a
        float32 array of shape [T].
    """
    tp = (masks_true & masks_pred).sum(axis=1).astype(np.float32)
    actual_pos = masks_true.sum(axis=1).astype(np.float32)
    pred_pos = masks_pred.sum(axis=1).astype(np.float32)

    recall = np.where(actual_pos > 0, tp / actual_pos, 1.0)
    precision = np.where(pred_pos > 0, tp / pred_pos, 0.0)
    density = pred_pos / masks_true.shape[1]

    return {"recall": recall, "precision": precision, "density": density}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline KV-retrieval activation prediction validator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--input",
        required=True,
        help=".npz file produced by record_ffn_activations.py.",
    )
    p.add_argument(
        "--layers",
        nargs="*",
        type=int,
        default=None,
        metavar="N",
        help="Layer indices to evaluate. Omit to evaluate all available layers.",
    )
    p.add_argument(
        "--top-k",
        nargs="+",
        type=int,
        default=[3],
        metavar="K",
        help="Number of neighbours to retrieve and OR (default: 3). "
        "Multiple values produce a sweep.",
    )
    p.add_argument(
        "--threshold-pct",
        nargs="+",
        type=float,
        default=[70.0],
        metavar="P",
        help="Percentile threshold for binarising activations (default: 70). "
        "Neurons above this percentile of |activation| are considered active. "
        "Multiple values produce a sweep.",
    )
    p.add_argument(
        "--output-csv",
        default=None,
        metavar="PATH",
        help="If given, write per-token results to this CSV file.",
    )
    return p.parse_args(argv)


def _available_layers(data: np.lib.npyio.NpzFile) -> list[int]:
    layers = set()
    for key in data.files:
        part = key.split("/")[0]
        if part.startswith("layer"):
            with contextlib.suppress(ValueError):
                layers.add(int(part[len("layer") :]))
    return sorted(layers)


def main(argv=None) -> None:
    args = _parse_args(argv)

    data = np.load(args.input)
    all_layers = _available_layers(data)

    if not all_layers:
        print("ERROR: no layer data found in the .npz file.", file=sys.stderr)
        sys.exit(1)

    layers = args.layers if args.layers is not None else all_layers
    missing = [idx for idx in layers if idx not in all_layers]
    if missing:
        print(
            f"WARNING: layers {missing} not found in file; available: {all_layers}",
            file=sys.stderr,
        )
        layers = [idx for idx in layers if idx in all_layers]

    csv_rows: list[dict] = []

    for layer_idx in layers:
        prefix = f"layer{layer_idx}"
        inputs_key = f"{prefix}/gate_up_input"
        acts_key = f"{prefix}/down_input"

        if inputs_key not in data.files or acts_key not in data.files:
            print(
                f"  layer {layer_idx}: missing gate_up_input or down_input, skipping.",
                file=sys.stderr,
            )
            continue

        inputs = data[inputs_key].astype(np.float32)  # [T, H]
        acts = data[acts_key].astype(np.float32)  # [T, I]
        T = inputs.shape[0]

        print(
            f"\nLayer {layer_idx}:  {T} tokens,  "
            f"H={inputs.shape[1]},  I={acts.shape[1]}"
        )
        print(
            f"  {'K':>4}  {'thresh%':>8}  "
            f"{'recall':>8}  {'precision':>10}  {'density':>8}"
        )
        print("  " + "-" * 46)

        # Precompute similarity once per layer
        sim = cosine_similarity_matrix(inputs)

        for threshold_pct, top_k in product(args.threshold_pct, args.top_k):
            masks_true = binarise(acts, threshold_pct)
            masks_pred = predict_masks(inputs, masks_true, top_k, sim=sim)
            metrics = evaluate(masks_true, masks_pred)

            mean_recall = metrics["recall"].mean()
            mean_precision = metrics["precision"].mean()
            mean_density = metrics["density"].mean()

            print(
                f"  {top_k:>4}  {threshold_pct:>8.1f}  "
                f"{mean_recall:>8.3f}  {mean_precision:>10.3f}  "
                f"{mean_density:>8.3f}"
            )

            if args.output_csv is not None:
                for token_idx in range(T):
                    csv_rows.append(
                        {
                            "layer": layer_idx,
                            "token": token_idx,
                            "top_k": top_k,
                            "threshold_pct": threshold_pct,
                            "recall": float(metrics["recall"][token_idx]),
                            "precision": float(metrics["precision"][token_idx]),
                            "density": float(metrics["density"][token_idx]),
                        }
                    )

    if args.output_csv is not None and csv_rows:
        fieldnames = [
            "layer",
            "token",
            "top_k",
            "threshold_pct",
            "recall",
            "precision",
            "density",
        ]
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(
            f"\nPer-token results written to {args.output_csv}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
