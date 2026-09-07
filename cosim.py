# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compute the cosine similarity of the outputs of the gate projection.

Groups intermediate FFN channels into highly similar groups of size 64.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def get_gate_weight(model: torch.nn.Module, layer_idx: int) -> torch.Tensor:
    """Retrieve the gate projection weight for the given layer.

    Handles both split (gate_proj) and fused (gate_up_proj) layouts.
    """
    # Try common attributes for different model structures
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layer = model.model.layers[layer_idx]
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layer = model.transformer.h[layer_idx]
    else:
        # Generic fallback search
        for name, module in model.named_modules():
            if f"layers.{layer_idx}.mlp" in name or f"h.{layer_idx}.mlp" in name:
                if hasattr(module, "gate_proj"):
                    return module.gate_proj.weight.data
                elif hasattr(module, "gate_up_proj"):
                    weight = module.gate_up_proj.weight.data
                    intermediate_size = weight.shape[0] // 2
                    return weight[:intermediate_size]
        raise RuntimeError(f"Could not find layer {layer_idx} MLP in model")

    if hasattr(layer, "mlp"):
        mlp = layer.mlp
    elif hasattr(layer, "ffn"):
        mlp = layer.ffn
    elif hasattr(layer, "feed_forward"):
        mlp = layer.feed_forward
    else:
        raise RuntimeError(f"Could not find MLP module in layer {layer_idx}")

    for attr in ("gate_proj", "gate_up_proj", "c_fc", "fc1", "dense_h_to_4h"):
        proj = getattr(mlp, attr, None)
        if proj is not None:
            weight = proj.weight.data
            if attr == "gate_up_proj":
                # Fused gate_up_proj, gate is the first half
                # shape is (2 * intermediate_size, hidden_size)
                intermediate_size = weight.shape[0] // 2
                return weight[:intermediate_size]
            return weight

    raise RuntimeError(
        f"Could not find gate projection weight in layer {layer_idx} MLP")


def load_gate_activations(
    data: Dict[str, np.ndarray],
    layer_idx: int,
    model: torch.nn.Module | None = None,
    device: str = "cpu",
) -> torch.Tensor:
    """Retrieve or compute the raw gate projection outputs for a layer."""
    gate_raw_key = f"layer{layer_idx}/gate_raw"
    if gate_raw_key in data:
        print(f"  Found pre-computed gate_raw for layer {layer_idx}.")
        return torch.from_numpy(data[gate_raw_key]).to(device).float()

    input_key = f"layer{layer_idx}/gate_up_input"
    if input_key not in data:
        raise KeyError(
            f"Neither '{gate_raw_key}' nor '{input_key}' found in NPZ data.")

    if model is None:
        raise ValueError(
            f"Layer {layer_idx} requires model weights to compute gate_raw from "
            f"'{input_key}', but no model was provided.")

    print(f"  Computing gate_raw for layer {layer_idx} via model weights...")
    gate_up_input = torch.from_numpy(data[input_key]).to(device).float()
    gate_weight = get_gate_weight(model, layer_idx).to(device).float()

    # Project inputs: (num_tokens, hidden_size) @ (intermediate_size, hidden_size).T
    # -> (num_tokens, intermediate_size)
    with torch.no_grad():
        gate_raw = torch.mm(gate_up_input, gate_weight.t())

    return gate_raw


def compute_cosine_similarity(gate_outputs: torch.Tensor) -> torch.Tensor:
    """Compute the C x C pairwise cosine similarity matrix of the channels."""
    # gate_outputs shape: (T, C) where C is intermediate_size
    norms = torch.norm(gate_outputs, dim=0, keepdim=True)
    # Avoid division by zero for inactive/dead channels
    norms = torch.where(norms == 0.0, torch.ones_like(norms), norms)
    normalized = gate_outputs / norms
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
        "--input",
        required=True,
        help="Path to the .npz activations file (e.g. ffn_activations128.npz)",
    )
    p.add_argument(
        "--model",
        default="ibm-granite/granite-4.2-3b",
        help="HuggingFace model ID/path to load weights from (default: ibm-granite/granite-4.2-3b)",
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

    # Check if we need to load the model
    needs_model = False
    for lyr in target_layers:
        if f"layer{lyr}/gate_raw" not in data:
            needs_model = True
            break

    model = None
    if needs_model:
        print(f"Loading weights from model '{args.model}' to compute gate_raw...", file=sys.stderr)
        # Load the model config first, then weights
        model = AutoModelForCausalLM.from_pretrained(args.model, low_cpu_mem_usage=True)
        model.eval()

    results: Dict[str, Any] = {}
    saved_groupings: Dict[str, Union[List[List[int]], np.ndarray]] = {}

    for lyr in target_layers:
        print(f"\n--- Layer {lyr} ---", file=sys.stderr)
        try:
            gate_raw = load_gate_activations(data, lyr, model=model, device=device)
        except Exception as e:
            print(f"Failed to load/compute gate activations for layer {lyr}: {e}", file=sys.stderr)
            continue

        T, C = gate_raw.shape
        print(f"  Shape: {T} tokens, {C} channels", file=sys.stderr)

        print("  Computing cosine similarity matrix...", file=sys.stderr)
        sim_matrix = compute_cosine_similarity(gate_raw)

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
