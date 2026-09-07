# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compute the cosine similarity of the FFN intermediate channels.

Groups intermediate FFN channels into highly similar groups of size 64.
Reads ``gate_raw`` (gate logits before SiLU) from files produced by
``record_ffn_activations.py``; falls back to ``down_input`` for files
recorded before the gate_raw switch.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch


def load_activation(
    data: Dict[str, np.ndarray],
    layer_idx: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Retrieve the FFN intermediate activations for a layer.

    Prefers ``gate_raw`` (gate logits before SiLU, recorded by the current
    version of ``record_ffn_activations.py``).  Falls back to ``down_input``
    for NPZ files recorded before the gate_raw switch.
    """
    for key in (f"layer{layer_idx}/gate_raw", f"layer{layer_idx}/down_input"):
        if key in data:
            label = key.split("/")[1]
            print(f"  Loading {label} for layer {layer_idx}...")
            return torch.from_numpy(data[key]).to(device).float()
    raise KeyError(
        f"Expected key 'layer{layer_idx}/gate_raw' not found in NPZ data. "
        f"Available keys: {[k for k in data.keys() if k.startswith(f'layer{layer_idx}')]}"
    )


# Backwards-compatible alias used by existing tests and callers.
load_down_input = load_activation


def binarise_pct(acts: torch.Tensor, threshold_pct: float) -> torch.Tensor:
    """Return float mask: 1.0 where |act| exceeds the per-token percentile, else 0.0.

    Matches the thresholding behavior of O1_predict.py.
    """
    q = threshold_pct / 100.0
    thresholds = torch.quantile(torch.abs(acts), q, dim=1, keepdim=True)
    return (torch.abs(acts) > thresholds).float()


def normalize_channels(activations: torch.Tensor) -> torch.Tensor:
    """Normalize the columns of activations to unit L2 norm."""
    norms = torch.norm(activations, dim=0, keepdim=True)
    # Avoid division by zero for inactive/dead channels
    norms = torch.where(norms == 0.0, torch.ones_like(norms), norms)
    return activations / norms


def compute_cosine_similarity(activations: torch.Tensor) -> torch.Tensor:
    """Compute the C x C pairwise cosine similarity matrix of the channels."""
    # activations shape: (T, C) where C is intermediate_size
    normalized = normalize_channels(activations)
    similarity_matrix = torch.mm(normalized.t(), normalized)
    return similarity_matrix


def group_channels_greedy(similarity_matrix: torch.Tensor,
                          group_size: int = 64) -> List[List[int]]:
    """Group channels into equal-sized groups using a similarity-seeding approach.

    1. Measures overall density (mean of top-K similarity) for each channel.
    2. Sorts candidates so dense neighborhoods are seeded first.
    3. Greedily grabs the best unassigned seed and its nearest neighbors.
    """
    # Force CPU execution to avoid GPU/MPS synchronization latency during sequential loops
    similarity_matrix = similarity_matrix.cpu().clone().float()
    C = similarity_matrix.shape[0]
    if C % group_size != 0:
        raise ValueError(
            f"Number of channels {C} is not divisible by group_size {group_size}"
        )

    unassigned = torch.ones(C, dtype=torch.bool, device=similarity_matrix.device)
    groups = []

    # Calculate candidate seed density scores using top_k similarities
    top_k_sims, _ = torch.topk(similarity_matrix, k=group_size, dim=1)
    seed_scores = top_k_sims.mean(dim=1)
    seed_order = torch.argsort(seed_scores, descending=True)

    for seed in seed_order:
        seed_idx = seed.item()
        if not unassigned[seed_idx]:
            continue

        # Get similarities for the current seed, masking already assigned channels
        sims = similarity_matrix[seed_idx].clone()
        sims[~unassigned] = -1e9

        # Grab top nearest unassigned neighbors (including seed itself)
        _, top_indices = torch.topk(sims, k=group_size)
        group_indices = top_indices.tolist()

        groups.append(group_indices)
        unassigned[group_indices] = False

    return groups


def group_channels_kmeans(
    normalized_channels: torch.Tensor,
    group_size: int = 64,
    num_iters: int = 10,
    seed: int = 42,
) -> List[List[int]]:
    """Group channels into equal-sized groups using Balanced K-Means.

    Iteratively updates cluster centroids and assigns channels to centroids
    respecting the size capacity constraint (exactly group_size channels per cluster).
    """
    import random
    # Force CPU execution to avoid GPU/MPS synchronization latency during sequential loops
    normalized_channels = normalized_channels.cpu()
    C = normalized_channels.shape[1]
    if C % group_size != 0:
        raise ValueError(
            f"Number of channels {C} is not divisible by group_size {group_size}"
        )

    K = C // group_size  # Number of clusters
    device = torch.device("cpu")

    # Set random seeds for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)

    # Randomly initialize centroids by selecting K channels
    shuffled_indices = torch.randperm(C, device=device)
    centroids = normalized_channels[:, shuffled_indices[:K]].clone()  # (T, K)

    channel_to_cluster = torch.full((C,), -1, dtype=torch.long, device=device)

    for _ in range(num_iters):
        # Compute similarity between all channels and all centroids
        sims = torch.mm(normalized_channels.t(), centroids)

        # Pre-sort centroid choices for each channel in parallel
        sorted_centroid_indices = torch.argsort(sims, dim=1, descending=True)
        # Convert to nested list to avoid tensor indexing and .item() overhead in Python loop
        sorted_centroid_indices_list = sorted_centroid_indices.tolist()

        channel_to_cluster.fill_(-1)
        cluster_counts = torch.zeros(K, dtype=torch.long, device=device)

        # Sort channels by their maximum similarity to any centroid (Strongest Preference First)
        max_sims, _ = sims.max(dim=1)
        channel_order = torch.argsort(max_sims, descending=True).tolist()

        # Iterate channel-by-channel in Strongest-Preference-First order
        for ch in channel_order:
            for cl in sorted_centroid_indices_list[ch]:
                if cluster_counts[cl] < group_size:
                    channel_to_cluster[ch] = cl
                    cluster_counts[cl] += 1
                    break

        # Update centroids to be the normalized mean of each cluster
        for cl in range(K):
            ch_indices = (channel_to_cluster == cl).nonzero(as_tuple=True)[0]
            if len(ch_indices) > 0:
                cl_channels = normalized_channels[:, ch_indices]
                mean_channel = cl_channels.mean(dim=1)
                mean_norm = torch.norm(mean_channel)
                if mean_norm > 0:
                    centroids[:, cl] = mean_channel / mean_norm
                else:
                    centroids[:, cl] = cl_channels[:, 0]

    # Reconstruct groups
    groups = [[] for _ in range(K)]
    for ch, cl in enumerate(channel_to_cluster.tolist()):
        groups[cl].append(ch)

    return groups


def group_channels_random(C: int, group_size: int = 64) -> List[List[int]]:
    """Partition channels randomly as a baseline."""
    perm = torch.randperm(C).tolist()
    return [perm[i : i + group_size] for i in range(0, C, group_size)]


def evaluate_grouping(similarity_matrix: torch.Tensor,
                      groups: List[List[int]]) -> Dict[str, float]:
    """Evaluate pairwise similarity statistics across all groups."""
    # Move to CPU for fast, robust evaluation
    similarity_matrix = similarity_matrix.cpu()
    groups_tensor = torch.tensor(groups, device=similarity_matrix.device)
    group_size = groups_tensor.shape[1]

    # Extract unique pairs using upper triangle indices
    triu_indices = torch.triu_indices(group_size, group_size, offset=1)

    # Gather rows and columns for parallel indexing
    rows = groups_tensor[:, triu_indices[0]].flatten()
    cols = groups_tensor[:, triu_indices[1]].flatten()

    flat_sims = similarity_matrix[rows, cols]
    group_pairwise_sims = flat_sims.reshape(groups_tensor.shape[0], -1)

    # Compute mean pairwise similarity for each of the K groups
    mean_pair_sims = group_pairwise_sims.mean(dim=1)

    return {
        "mean": float(mean_pair_sims.mean().item()),
        "min": float(mean_pair_sims.min().item()),
        "max": float(mean_pair_sims.max().item()),
        "std": float(mean_pair_sims.std().item()),
    }


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "input",
        help="Path to the .npz activations file (e.g. ffn_activations128.npz)",
    )
    p.add_argument(
        "--output",
        default="channel_groups.json",
        help="Output file path (ends with .json or .npz) to save results (default: channel_groups.json)",
    )
    p.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="Size of each channel group (default: 64)",
    )
    p.add_argument(
        "--layers",
        nargs="*",
        type=int,
        default=None,
        help="Specific layers to process (e.g., 0 2). If omitted, processes all layers in the NPZ.",
    )
    p.add_argument(
        "--threshold-pct",
        type=float,
        default=None,
        help="If set, binarize activations using this percentile threshold (0-100) per token before computing similarity.",
    )
    p.add_argument(
        "--method",
        choices=["greedy", "random", "kmeans", "all"],
        default="all",
        help="Grouping method to evaluate (default: all)",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Device to run torch computations on (cpu, mps, cuda). Auto-detected if not specified.",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    # Determine device
    if args.device:
        device = args.device
    else:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    print(f"Using device: {device}", file=sys.stderr)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading activations from {input_path}...", file=sys.stderr)
    data = np.load(input_path)

    # Detect available layers in NPZ
    layers_in_npz = sorted(
        list({
            int(k.split("/")[0].replace("layer", ""))
            for k in data.keys()
            if k.startswith("layer")
        })
    )

    if not layers_in_npz:
        print(f"Error: No layers found in NPZ file '{input_path}'.", file=sys.stderr)
        sys.exit(1)

    target_layers = args.layers if args.layers is not None else layers_in_npz
    for lyr in target_layers:
        if lyr not in layers_in_npz:
            print(f"Error: Layer {lyr} not present in the NPZ file.", file=sys.stderr)
            sys.exit(1)

    print(f"Processing layers: {target_layers}", file=sys.stderr)

    results: Dict[str, Any] = {}
    saved_groupings: Dict[str, Union[List[List[int]], np.ndarray]] = {}

    for lyr in target_layers:
        print(f"\n--- Layer {lyr} ---", file=sys.stderr)
        try:
            activations = load_activation(data, lyr, device=device)
        except Exception as e:
            print(f"Failed to load activations for layer {lyr}: {e}", file=sys.stderr)
            continue

        T, C = activations.shape
        print(f"  Shape: {T} tokens, {C} channels", file=sys.stderr)

        if args.threshold_pct is not None:
            print(f"  Binarizing activations with threshold-pct={args.threshold_pct}...", file=sys.stderr)
            activations = binarise_pct(activations, args.threshold_pct)

        print("  Computing cosine similarity matrix...", file=sys.stderr)
        sim_matrix = compute_cosine_similarity(activations)

        layer_res = {}

        if args.method in ("kmeans", "all"):
            print("  Grouping channels (balanced k-means method)...", file=sys.stderr)
            norm_channels = normalize_channels(activations)
            kmeans_groups = group_channels_kmeans(norm_channels, group_size=args.group_size)
            kmeans_stats = evaluate_grouping(sim_matrix, kmeans_groups)
            layer_res["kmeans"] = kmeans_stats
            saved_groupings[f"layer{lyr}"] = kmeans_groups
            print(
                f"  [K-Means] Intra-group similarity: "
                f"mean={kmeans_stats['mean']:.5f}, "
                f"min={kmeans_stats['min']:.5f}, "
                f"max={kmeans_stats['max']:.5f}, "
                f"std={kmeans_stats['std']:.5f}",
                file=sys.stderr,
            )

        if args.method in ("greedy", "all"):
            print("  Grouping channels (greedy method)...", file=sys.stderr)
            greedy_groups = group_channels_greedy(sim_matrix, group_size=args.group_size)
            greedy_stats = evaluate_grouping(sim_matrix, greedy_groups)
            layer_res["greedy"] = greedy_stats
            if args.method == "greedy" or args.method == "all":
                # Let K-Means take priority if "all" is used, otherwise set it
                if f"layer{lyr}" not in saved_groupings:
                    saved_groupings[f"layer{lyr}"] = greedy_groups
            print(
                f"  [Greedy] Intra-group similarity: "
                f"mean={greedy_stats['mean']:.5f}, "
                f"min={greedy_stats['min']:.5f}, "
                f"max={greedy_stats['max']:.5f}, "
                f"std={greedy_stats['std']:.5f}",
                file=sys.stderr,
            )

        if args.method in ("random", "all"):
            print("  Grouping channels (random method)...", file=sys.stderr)
            random_groups = group_channels_random(C, group_size=args.group_size)
            random_stats = evaluate_grouping(sim_matrix, random_groups)
            layer_res["random"] = random_stats
            if args.method == "random":
                saved_groupings[f"layer{lyr}"] = random_groups
            print(
                f"  [Random] Intra-group similarity: "
                f"mean={random_stats['mean']:.5f}, "
                f"min={random_stats['min']:.5f}, "
                f"max={random_stats['max']:.5f}, "
                f"std={random_stats['std']:.5f}",
                file=sys.stderr,
            )

        results[f"layer{lyr}"] = layer_res

    # Save results if requested
    output_path = Path(args.output)
    print(f"\nSaving channel groupings to {output_path}...", file=sys.stderr)
    if output_path.suffix == ".json":
        # Save as readable JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(saved_groupings, f, indent=2)
    elif output_path.suffix == ".npz":
        # Save as npz archive
        npz_dict = {k: np.array(v) for k, v in saved_groupings.items()}
        np.savez(output_path, **npz_dict)
    else:
        # Fallback to JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(saved_groupings, f, indent=2)

    print("Done!", file=sys.stderr)


if __name__ == "__main__":
    main()
