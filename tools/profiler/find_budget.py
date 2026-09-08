# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""For a given output-quality floor, find the lowest neuron compute budget
and best grouping configuration per layer.

Searches over:
  - uniform kmeans groups at group sizes 64, 32, 16, 8, 4
    (pre-built JSON files supplied via --groups)
  - variable-size groups (64/16/4/1 tiers, built on-the-fly from gate_raw)

For each (layer, group-config) pair, binary-searches the top-pct parameter
to find the minimum neuron budget that satisfies all quality constraints.
Prints the Pareto-optimal choice per layer and a summary config table.

Quality constraints (all must be met simultaneously):
  --min-cosim-mean   minimum mean cosine similarity  (default: 0.90)
  --min-cosim-p5     minimum 5th-percentile cosim    (default: 0.85)
  --min-neuron-rec   minimum neuron recall            (default: 0.80)

Usage::

    python tools/profiler/find_budget.py \\
        --model ibm-granite/granite-4.2-3b \\
        --npz ffn_activations128_gate.npz \\
        --groups channel_groups.json channel_groups_gs32.json \\
                 channel_groups_gs16.json channel_groups_gs8.json \\
                 channel_groups_gs4.json

    # Stricter quality floor
    python tools/profiler/find_budget.py \\
        --model ibm-granite/granite-4.2-3b \\
        --npz ffn_activations128_gate.npz \\
        --groups channel_groups*.json \\
        --min-cosim-mean 0.95 --min-cosim-p5 0.90 --min-neuron-rec 0.85
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def _find_model_dir(model_name_or_path: str) -> Path:
    import glob
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


def load_mlp_weights(model_dir: Path, layer_idx: int) -> dict[str, torch.Tensor]:
    from safetensors import safe_open
    shards = sorted(model_dir.glob("model*.safetensors"))
    needed = {
        f"model.layers.{layer_idx}.mlp.gate_proj.weight",
        f"model.layers.{layer_idx}.mlp.up_proj.weight",
        f"model.layers.{layer_idx}.mlp.down_proj.weight",
    }
    found: dict[str, torch.Tensor] = {}
    for shard in shards:
        if not needed:
            break
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in list(needed):
                if key in f.keys():
                    found[key] = f.get_tensor(key).float()
                    needed.discard(key)
    if needed:
        raise KeyError(f"Missing weights for layer {layer_idx}: {needed}")
    pfx = f"model.layers.{layer_idx}.mlp."
    return {
        "gate": found[pfx + "gate_proj.weight"],
        "up":   found[pfx + "up_proj.weight"],
        "down": found[pfx + "down_proj.weight"],
    }


# ---------------------------------------------------------------------------
# Variable-size grouper
# ---------------------------------------------------------------------------

def build_variable_groups(
    gate_raw: torch.Tensor,
    tiers: list[tuple[float, int]],
) -> list[list[int]]:
    """Greedy similarity-based grouper with tiered size caps."""
    T, I = gate_raw.shape
    col_norms = gate_raw.norm(dim=0, keepdim=True).clamp(min=1e-9)
    gate_n = (gate_raw / col_norms).T.float()   # [I, T]
    BLOCK = 512
    sim = torch.zeros(I, I)
    for i in range(0, I, BLOCK):
        a = gate_n[i:i+BLOCK]
        for j in range(0, I, BLOCK):
            sim[i:i+BLOCK, j:j+BLOCK] = a @ gate_n[j:j+BLOCK].T
    sim.fill_diagonal_(0.0)
    unassigned = torch.ones(I, dtype=torch.bool)
    groups: list[list[int]] = []
    top_sum, _ = sim.topk(min(64, I - 1), dim=1)
    order = top_sum.sum(dim=1).argsort(descending=True).tolist()
    for seed in order:
        if not unassigned[seed]:
            continue
        row = sim[seed].clone()
        row[~unassigned] = -1.0
        row[seed] = -1.0
        for min_sim, max_size in tiers:
            if max_size == 1:
                groups.append([seed])
                unassigned[seed] = False
                break
            candidates = (row >= min_sim).nonzero(as_tuple=True)[0]
            if len(candidates) == 0:
                continue
            k = min(max_size - 1, len(candidates))
            members = candidates[row[candidates].topk(k).indices].tolist() + [seed]
            for m in members:
                unassigned[m] = False
            groups.append(members)
            break
        else:
            groups.append([seed])
            unassigned[seed] = False
    return groups


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def build_predictor(gate_w: torch.Tensor, groups: list[list[int]]) -> torch.Tensor:
    G, H = len(groups), gate_w.shape[1]
    pred = torch.zeros(G, H)
    for g, idxs in enumerate(groups):
        pred[g] = gate_w[idxs].sum(0)
    return pred


def build_group_index(groups: list[list[int]], I: int) -> torch.Tensor:
    group_of = torch.zeros(I, dtype=torch.long)
    for g, idxs in enumerate(groups):
        group_of[idxs] = g
    return group_of


def evaluate_at_budget(
    x: torch.Tensor,
    gate_raw: torch.Tensor,
    weights: dict[str, torch.Tensor],
    groups: list[list[int]],
    target_neuron_frac: float,
) -> dict[str, float]:
    """Evaluate quality at the lowest top_pct that reaches target_neuron_frac."""
    gate_w, up_w, down_w = weights["gate"], weights["up"], weights["down"]
    I = gate_w.shape[0]
    G = len(groups)
    group_sizes = torch.tensor([len(g) for g in groups], dtype=torch.float32)

    pred     = build_predictor(gate_w, groups)
    group_of = build_group_index(groups, I)
    scores   = x @ pred.T   # [T, G]

    # Binary-search for K (number of groups to activate) achieving target budget
    lo, hi = 1, G
    for _ in range(20):
        K = (lo + hi) // 2
        kth = scores.kthvalue(G - K + 1, dim=1, keepdim=True).values
        ag  = scores >= kth
        avg_neurons = (ag.float() @ group_sizes).mean().item() / I
        if avg_neurons < target_neuron_frac:
            lo = K + 1
        else:
            hi = K
    K = hi
    kth = scores.kthvalue(G - K + 1, dim=1, keepdim=True).values
    ag  = scores >= kth
    nm  = ag[:, group_of]

    full = (F.silu(x @ gate_w.T) * (x @ up_w.T)) @ down_w.T
    sp   = (F.silu(x @ gate_w.T) * nm * (x @ up_w.T) * nm) @ down_w.T
    cs   = F.cosine_similarity(sp, full, dim=1)
    rec  = ((gate_raw > 0) & nm).sum().item() / max((gate_raw > 0).sum().item(), 1)

    return {
        "cosim_mean":  cs.mean().item(),
        "cosim_p5":    cs.quantile(0.05).item(),
        "neuron_rec":  rec,
        "neuron_frac": (ag.float() @ group_sizes).mean().item() / I,
        "K": K,
        "G": G,
    }


def find_min_budget(
    x: torch.Tensor,
    gate_raw: torch.Tensor,
    weights: dict[str, torch.Tensor],
    groups: list[list[int]],
    min_cosim_mean: float,
    min_cosim_p5: float,
    min_neuron_rec: float,
    budget_step: float = 0.02,
) -> dict | None:
    """Binary-search the lowest neuron budget satisfying all quality constraints."""
    lo, hi = 0.0, 1.0
    best = None
    for _ in range(20):
        mid = (lo + hi) / 2
        m = evaluate_at_budget(x, gate_raw, weights, groups, mid)
        ok = (m["cosim_mean"] >= min_cosim_mean
              and m["cosim_p5"] >= min_cosim_p5
              and m["neuron_rec"] >= min_neuron_rec)
        if ok:
            best = m
            hi = mid
        else:
            lo = mid
        if hi - lo < budget_step / 2:
            break
    return best


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Find lowest compute budget per layer satisfying quality constraints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", required=True,
                   help="HuggingFace model ID or local path.")
    p.add_argument("--npz", required=True,
                   help=".npz file from record_ffn_activations.py.")
    p.add_argument("--groups", nargs="+", required=True, metavar="FILE",
                   help="One or more channel_groups*.json files to search over. "
                        "Variable-size grouping is always added automatically.")
    p.add_argument("--layers", nargs="*", type=int, default=None, metavar="N",
                   help="Layer indices to evaluate. Default: all layers in npz.")
    p.add_argument("--min-cosim-mean", type=float, default=0.90, metavar="F",
                   help="Minimum mean cosine similarity (default: 0.90).")
    p.add_argument("--min-cosim-p5", type=float, default=0.85, metavar="F",
                   help="Minimum p5 cosine similarity (default: 0.85).")
    p.add_argument("--min-neuron-rec", type=float, default=0.80, metavar="F",
                   help="Minimum neuron recall (default: 0.80).")
    p.add_argument("--variable-tiers", nargs="+", type=str,
                   default=["0.90:64", "0.70:16", "0.50:4", "0.0:1"],
                   metavar="SIM:SIZE",
                   help="Variable-group tiers as sim:max_size pairs "
                        "(default: 0.90:64 0.70:16 0.50:4 0.0:1).")
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "bfloat16"],
                   help="Compute dtype (default: float32).")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    tiers = []
    for t in args.variable_tiers:
        sim_s, size_s = t.split(":")
        tiers.append((float(sim_s), int(size_s)))
    tiers.sort(key=lambda x: -x[0])

    try:
        model_dir = _find_model_dir(args.model)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr); sys.exit(1)
    print(f"Reading weights from {model_dir}", file=sys.stderr)

    data = np.load(args.npz)
    available_layers = sorted(
        int(k.split("/")[0][5:]) for k in data.files if k.endswith("/gate_up_input")
    )
    layers = args.layers if args.layers is not None else available_layers
    layers = [l for l in layers if l in available_layers]

    # Load all uniform group files
    uniform_configs: list[tuple[str, dict]] = []
    for path in args.groups:
        gj = json.load(open(path))
        label = Path(path).stem   # e.g. "channel_groups_gs32"
        uniform_configs.append((label, gj))
        print(f"Loaded {path}: {len(gj)} layers", file=sys.stderr)

    dt = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    qfloor = (f"cosim_mean≥{args.min_cosim_mean}  "
              f"cosim_p5≥{args.min_cosim_p5}  "
              f"neuron_rec≥{args.min_neuron_rec}")
    print(f"\nQuality floor: {qfloor}", file=sys.stderr)

    # Summary table
    summary: list[dict] = []

    for layer_idx in layers:
        x        = torch.from_numpy(data[f"layer{layer_idx}/gate_up_input"]).to(dt)
        gate_raw = torch.from_numpy(data[f"layer{layer_idx}/gate_raw"]).to(dt)

        try:
            weights = {k: v.to(dt) for k, v in load_mlp_weights(model_dir, layer_idx).items()}
        except Exception as e:
            print(f"  layer {layer_idx}: weight load failed: {e}", file=sys.stderr)
            continue

        print(f"\n  Layer {layer_idx}", file=sys.stderr)

        configs: list[tuple[str, list[list[int]]]] = []

        # Uniform configs available for this layer
        for label, gj in uniform_configs:
            key = f"layer{layer_idx}"
            if key in gj:
                configs.append((label, gj[key]))

        # Variable groups (built on-the-fly)
        print(f"    building variable groups...", file=sys.stderr, end="", flush=True)
        vgroups = build_variable_groups(gate_raw.float(), tiers)
        cnt = Counter(len(g) for g in vgroups)
        print(f" {len(vgroups)} groups {dict(sorted(cnt.items()))}", file=sys.stderr)
        configs.append(("variable", vgroups))

        best_budget = 1.01
        best_result = None
        best_config = None

        for cfg_label, groups in configs:
            result = find_min_budget(
                x, gate_raw, weights, groups,
                args.min_cosim_mean, args.min_cosim_p5, args.min_neuron_rec,
            )
            if result is None:
                print(f"    {cfg_label:30s}  INFEASIBLE", file=sys.stderr)
                continue

            flag = " ◀ best" if result["neuron_frac"] < best_budget else ""
            print(
                f"    {cfg_label:30s}  "
                f"budget={result['neuron_frac']*100:5.1f}%  "
                f"cosim_mean={result['cosim_mean']:.4f}  "
                f"cosim_p5={result['cosim_p5']:.4f}  "
                f"rec={result['neuron_rec']:.4f}  "
                f"K={result['K']}/{result['G']}{flag}",
                file=sys.stderr,
            )
            if result["neuron_frac"] < best_budget:
                best_budget = result["neuron_frac"]
                best_result = result
                best_config = cfg_label

        summary.append({
            "layer": layer_idx,
            "config": best_config,
            "neuron_budget": best_budget,
            "cosim_mean": best_result["cosim_mean"] if best_result else None,
            "cosim_p5":   best_result["cosim_p5"]   if best_result else None,
            "neuron_rec": best_result["neuron_rec"]  if best_result else None,
            "K": best_result["K"] if best_result else None,
            "G": best_result["G"] if best_result else None,
        })

    # Print summary to stdout
    print(f"\n{'='*72}")
    print(f"Quality floor: {qfloor}")
    print(f"{'='*72}")
    print(f"{'layer':>6}  {'best_config':>30}  {'budget':>7}  "
          f"{'cosim_mean':>11}  {'cosim_p5':>9}  {'rec':>7}  {'K/G':>12}")
    print("  " + "-"*78)
    total_saving = 0.0
    n_layers = 0
    for row in summary:
        if row["config"] is None:
            print(f"{row['layer']:>6}  {'INFEASIBLE':>30}")
            continue
        saving = (1.0 - row["neuron_budget"]) * 100
        total_saving += saving
        n_layers += 1
        print(
            f"{row['layer']:>6}  {row['config']:>30}  "
            f"{row['neuron_budget']*100:>6.1f}%  "
            f"{row['cosim_mean']:>11.5f}  "
            f"{row['cosim_p5']:>9.5f}  "
            f"{row['neuron_rec']:>7.4f}  "
            f"{row['K']:>5}/{row['G']}"
        )
    if n_layers:
        print(f"\n  Mean compute saving across {n_layers} layers: "
              f"{total_saving/n_layers:.1f}%")


if __name__ == "__main__":
    main()
