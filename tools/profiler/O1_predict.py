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

    # Use 90 % of tokens as the lookup library, evaluate on the rest
    python tools/profiler/O1_predict.py \\
        --input ffn_activations.npz \\
        --train-split 0.9

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


def binarise_pct(acts: np.ndarray, threshold_pct: float) -> np.ndarray:
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


def binarise_mult(
    acts: np.ndarray,
    multiplier: float,
    global_median: float,
) -> np.ndarray:
    """Return boolean mask: True where |act| exceeds multiplier × global_median.

    Args:
        acts: shape [T, I] float32 activation matrix.
        multiplier: threshold = multiplier × global_median.
        global_median: median of |training activations|, computed once from
            the library split and reused for both library and eval.

    Returns:
        Boolean array of shape [T, I].
    """
    return np.abs(acts) > multiplier * global_median


def predict_masks(
    query_inputs: np.ndarray,
    query_masks: np.ndarray,
    library_masks: np.ndarray,
    top_k: int,
    sim: np.ndarray | None = None,
) -> np.ndarray:
    """Predict activation masks via OR of top-K nearest neighbours.

    Args:
        query_inputs: shape [Q, H] input hidden states for eval tokens.
        query_masks: shape [Q, I] ground-truth masks for eval tokens (unused
            in prediction; kept for API symmetry and future use).
        library_masks: shape [L, I] ground-truth masks for library tokens.
        top_k: number of nearest neighbours to retrieve and OR.
        sim: precomputed [Q, L] cosine similarity matrix; computed if None.

    Returns:
        predicted: boolean array of shape [Q, I].
    """
    if sim is None:
        sim = cosine_similarity_matrix(
            np.vstack([query_inputs,
                       np.zeros((library_masks.shape[0], query_inputs.shape[1]),
                                dtype=query_inputs.dtype)])
        )[:query_inputs.shape[0], query_inputs.shape[0]:]

    Q = query_inputs.shape[0]
    predicted = np.zeros_like(query_masks)

    for i in range(Q):
        neighbours = np.argpartition(sim[i], -top_k)[-top_k:]
        predicted[i] = library_masks[neighbours].any(axis=0)

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
        "--train-split",
        type=float,
        default=0.8,
        metavar="F",
        help="Fraction of tokens used as the lookup library (default: 0.8). "
        "The remaining tokens are used for evaluation. "
        "Tokens are split in order (first F*T for library, rest for eval).",
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
        default=None,
        metavar="P",
        help="Per-token percentile threshold (0–100). Neurons whose |activation| "
        "exceeds this percentile of their own token are considered active. "
        "E.g. 70 → top-30%% of neurons per token. Multiple values produce a sweep. "
        "Mutually exclusive with --threshold-mult.",
    )
    p.add_argument(
        "--threshold-mult",
        nargs="+",
        type=float,
        default=None,
        metavar="M",
        help="Global-median multiplier threshold. Neurons whose |activation| "
        "exceeds M × median(|train_activations|) are considered active. "
        "E.g. 3 5 10 sweeps three thresholds. "
        "Mutually exclusive with --threshold-pct. "
        "If neither flag is given, defaults to --threshold-mult 3 5 10.",
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

    if args.threshold_pct is not None and args.threshold_mult is not None:
        print(
            "ERROR: --threshold-pct and --threshold-mult are mutually exclusive.",
            file=sys.stderr,
        )
        sys.exit(1)

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

        n_train = max(1, int(T * args.train_split))
        n_eval = T - n_train
        if n_eval == 0:
            print(
                f"  layer {layer_idx}: not enough tokens for eval split "
                f"(T={T}, train_split={args.train_split}), skipping.",
                file=sys.stderr,
            )
            continue

        train_inputs = inputs[:n_train]
        eval_inputs = inputs[n_train:]
        train_acts = acts[:n_train]
        eval_acts = acts[n_train:]

        # Resolve which threshold mode and values to sweep
        use_mult = args.threshold_mult is not None or args.threshold_pct is None
        if use_mult:
            thresh_values = args.threshold_mult if args.threshold_mult is not None else [3.0, 5.0, 10.0]
            global_median: float = float(np.median(np.abs(train_acts)))
            thresh_label = "mult"
            header_thresh = "     mult"
        else:
            thresh_values = args.threshold_pct  # type: ignore[assignment]
            global_median = 0.0  # unused in pct mode
            thresh_label = "pct"
            header_thresh = "  thresh%"

        print(
            f"\nLayer {layer_idx}:  {T} tokens  "
            f"(library={n_train}, eval={n_eval}),  "
            f"H={inputs.shape[1]},  I={acts.shape[1]}"
            + (f"  median|act|={global_median:.4f}" if use_mult else "")
        )
        print(
            f"  {'K':>4}  {header_thresh}  "
            f"{'recall':>8}  {'precision':>10}  {'density':>8}"
        )
        print("  " + "-" * 46)

        # Precompute query→library similarity once per layer
        train_norms = np.linalg.norm(train_inputs, axis=1, keepdims=True)
        train_norms = np.where(train_norms == 0, 1e-9, train_norms)
        eval_norms = np.linalg.norm(eval_inputs, axis=1, keepdims=True)
        eval_norms = np.where(eval_norms == 0, 1e-9, eval_norms)
        sim = (eval_inputs / eval_norms) @ (train_inputs / train_norms).T  # [Q, L]

        for thresh, top_k in product(thresh_values, args.top_k):
            top_k = int(top_k)
            if use_mult:
                train_masks = binarise_mult(train_acts, thresh, global_median)
                eval_masks  = binarise_mult(eval_acts,  thresh, global_median)
                thresh_col  = f"{thresh:>8.1f}x"
            else:
                train_masks = binarise_pct(train_acts, thresh)
                eval_masks  = binarise_pct(eval_acts,  thresh)
                thresh_col  = f"{thresh:>8.1f}"

            masks_pred = predict_masks(
                eval_inputs, eval_masks, train_masks, top_k, sim=sim
            )
            metrics = evaluate(eval_masks, masks_pred)

            mean_recall    = metrics["recall"].mean()
            mean_precision = metrics["precision"].mean()
            mean_density   = metrics["density"].mean()

            print(
                f"  {top_k:>4}  {thresh_col}  "
                f"{mean_recall:>8.3f}  {mean_precision:>10.3f}  "
                f"{mean_density:>8.3f}"
            )

            if args.output_csv is not None:
                for token_idx in range(n_eval):
                    csv_rows.append(
                        {
                            "layer": layer_idx,
                            "token": n_train + token_idx,
                            "top_k": top_k,
                            thresh_label: thresh,
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
