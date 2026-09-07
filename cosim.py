# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compute the cosine similarity of the down projection input channels.

Groups intermediate FFN channels into highly similar groups of size 64.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch


def load_down_input(
    data: Dict[str, np.ndarray],
    layer_idx: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Retrieve the down projection input activations for a layer."""
    down_input_key = f"layer{layer_idx}/down_input"
    if down_input_key not in data:
        raise KeyError(
            f"Expected key '{down_input_key}' not found in NPZ data.")

    print(f"  Loading down_input for layer {layer_idx}...")
    down_input = torch.from_numpy(data[down_input_key]).to(device).float()
    return down_input


def compute_cosine_similarity(activations: torch.Tensor) -> torch.Tensor:
    """Compute the C x C pairwise cosine similarity matrix of the channels."""
    # activations shape: (T, C) where C is intermediate_size
    norms = torch.norm(activations, dim=0, keepdim=True)
    # Avoid division by zero for inactive/dead channels
    norms = torch.where(norms == 0.0, torch.ones_like(norms), norms)
    normalized = activations / norms
    similarity_matrix = torch.mm(normalized.t(), normalized)
    return similarity_matrix


def group_channels_greedy(similarity_matrix: torch.Tensor,
                          group_size: int = 64) -> List[List[int]]:
    """Group channels into equal-sized groups using a similarity-seeding approach.

    1. Measures overall density (mean of top-K similarity) for each channel.
    2. Sorts candidates so dense neighborhoods are seeded first.
    3. Greedily grabs the best unassigned seed and its nearest neighbors.
    """
    C = similarity_matrix.shape[0]
    if C % group_size != 0:
        raise ValueError(
            f"Number of channels {C} is not divisible by group_size {group_size}"
        )

    similarity_matrix = similarity_matrix.clone().float()
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


def group_channels_random(C: int, group_size: int = 64) -> List[List[int]]:
    """Partition channels randomly as a baseline."""
    perm = torch.randperm(C).tolist()
    return [perm[i : i + group_size] for i in range(0, C, group_size)]


def evaluate_grouping(similarity_matrix: torch.Tensor,
                      groups: List[List[int]]) -> Dict[str, float]:
    """Evaluate pairwise similarity statistics across all groups."""
    group_size = len(groups[0])
    triu_indices = torch.triu_indices(group_size, group_size, offset=1)

    all_sims = []
    for g in groups:
        sims_g = similarity_matrix[g][:, g]
        mean_pair_sim = sims_g[triu_indices[0], triu_indices[1]].mean().item()
        all_sims.append(mean_pair_sim)

    all_sims = np.array(all_sims)
    return {
        "mean": float(np.mean(all_sims)),
        "min": float(np.min(all_sims)),
        "max": float(np.max(all_sims)),
        "std": float(np.std(all_sims)),
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
        "--method",
        choices=["greedy", "random", "both"],
        default="both",
        help="Grouping method to evaluate (default: both)",
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
            down_input = load_down_input(data, lyr, device=device)
        except Exception as e:
            print(f"Failed to load down_input for layer {lyr}: {e}", file=sys.stderr)
            continue

        T, C = down_input.shape
        print(f"  Shape: {T} tokens, {C} channels", file=sys.stderr)

        print("  Computing cosine similarity matrix...", file=sys.stderr)
        sim_matrix = compute_cosine_similarity(down_input)

        layer_res = {}

        if args.method in ("greedy", "both"):
            print("  Grouping channels (greedy method)...", file=sys.stderr)
            greedy_groups = group_channels_greedy(sim_matrix, group_size=args.group_size)
            greedy_stats = evaluate_grouping(sim_matrix, greedy_groups)
            layer_res["greedy"] = greedy_stats
            saved_groupings[f"layer{lyr}"] = greedy_groups
            print(
                f"  [Greedy] Intra-group similarity: "
                f"mean={greedy_stats['mean']:.5f}, "
                f"min={greedy_stats['min']:.5f}, "
                f"max={greedy_stats['max']:.5f}, "
                f"std={greedy_stats['std']:.5f}",
                file=sys.stderr,
            )

        if args.method in ("random", "both"):
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
