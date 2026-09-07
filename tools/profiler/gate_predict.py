# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Group-level FFN gate predictor — quality evaluation.

For each transformer layer this script:

1. Loads the channel groups produced by ``cosim.py`` (a JSON file mapping
   layer → list-of-groups, each group being a list of intermediate-neuron
   indices).

2. Builds one **group predictor vector** per group by summing the gate-projection
   weight rows that belong to that group::

       pred[g] = sum(gate_proj.weight[i]  for i in group[g])   # shape [H]

3. For every eval token, scores all groups::

       score[g, t] = hidden[t] · pred[g]                       # dot product

   and activates the top ``--top-pct`` percent of groups by score.  Using a
   percentage rather than an absolute threshold keeps the compute budget
   independent of the scale of the scores, which varies across layers.

4. For each token, runs the MLP only for predicted-active groups:
   - gate and up projections restricted to those neuron columns,
   - SiLU applied,
   - down projection restricted to those neuron rows.
   Skipped-group neurons contribute zero to the down-proj output.

5. Compares the resulting ``down_proj`` output against the full-compute
   baseline (all groups active) using per-token cosine similarity.

Reported metrics per layer:
  - grp_active   – mean fraction of groups predicted active (= top-pct / 100)
  - cosim_mean/std/p5  – cosine similarity of predicted vs full output
  - neuron_rec   – fraction of truly-active (gate_raw > 0) neurons that fall
                   inside a predicted-active group

Usage::

    python tools/profiler/gate_predict.py \\
        --model ibm-granite/granite-4.2-3b \\
        --npz ffn_activations128_gate.npz \\
        --groups channel_groups.json

    # Sweep top-pct values to find the cosim/savings tradeoff
    python tools/profiler/gate_predict.py \\
        --model ibm-granite/granite-4.2-3b \\
        --npz ffn_activations128_gate.npz \\
        --groups channel_groups.json \\
        --top-pct 100 90 75 50 25 10

    # Evaluate only layers 0 and 15
    python tools/profiler/gate_predict.py \\
        --model ibm-granite/granite-4.2-3b \\
        --npz ffn_activations128_gate.npz \\
        --groups channel_groups.json \\
        --layers 0 15

The ``--model`` argument accepts a HuggingFace model ID or a local directory.
Weights are read directly from the safetensors checkpoint — vLLM is not
required and the model is never loaded into an LLM engine.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Weight extraction — read directly from safetensors, no vLLM engine needed
# ---------------------------------------------------------------------------

def _find_model_dir(model_name_or_path: str) -> Path:
    """Resolve a HF model ID or local path to the directory containing weights."""
    p = Path(model_name_or_path)
    if p.exists():
        return p
    # HuggingFace cache layout
    slug = model_name_or_path.replace("/", "--")
    candidates = sorted(
        glob.glob(str(Path.home() / ".cache/huggingface/hub"
                      / f"models--{slug}/snapshots/*/"))
    )
    if not candidates:
        raise FileNotFoundError(
            f"Could not find model '{model_name_or_path}' locally. "
            "Pass a local directory path or download the model first."
        )
    return Path(candidates[-1])   # latest snapshot


def load_mlp_weights(
    model_dir: Path,
    layer_idx: int,
) -> dict[str, torch.Tensor]:
    """Load gate_proj, up_proj, down_proj for one layer from safetensors.

    Returns dict with keys ``gate``, ``up``, ``down``, each float32 CPU tensor:
      gate: [intermediate, hidden]
      up:   [intermediate, hidden]
      down: [hidden, intermediate]
    """
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError("pip install safetensors")

    shards = sorted(model_dir.glob("model*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No model*.safetensors found in {model_dir}")

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

    prefix = f"model.layers.{layer_idx}.mlp."
    return {
        "gate": found[prefix + "gate_proj.weight"],   # [I, H]
        "up":   found[prefix + "up_proj.weight"],     # [I, H]
        "down": found[prefix + "down_proj.weight"],   # [H, I]
    }


# ---------------------------------------------------------------------------
# Predictor construction
# ---------------------------------------------------------------------------

def build_group_predictors(
    gate_w: torch.Tensor,           # [I, H]
    groups: list[list[int]],
) -> torch.Tensor:
    """Sum gate-weight rows within each group → predictor matrix [G, H].

    Each row is the aggregated "signature" of that group in input space.
    A dot product with a hidden state gives a scalar activation score.
    """
    G = len(groups)
    H = gate_w.shape[1]
    pred = torch.zeros(G, H, dtype=torch.float32)
    for g, idxs in enumerate(groups):
        pred[g] = gate_w[idxs].sum(dim=0)
    return pred   # [G, H]


# ---------------------------------------------------------------------------
# Sparse MLP forward
# ---------------------------------------------------------------------------

def mlp_full(
    x: torch.Tensor,        # [T, H]
    gate_w: torch.Tensor,   # [I, H]
    up_w: torch.Tensor,     # [I, H]
    down_w: torch.Tensor,   # [H, I]
) -> torch.Tensor:
    """Full MLP: SiLU(gate(x)) * up(x) → down.  Returns [T, H]."""
    gate_out = F.silu(x @ gate_w.T)   # [T, I]
    up_out   = x @ up_w.T             # [T, I]
    return (gate_out * up_out) @ down_w.T   # [T, H]


def _build_group_index(groups: list[list[int]], I: int) -> torch.Tensor:
    """Return [I] int tensor mapping neuron → group index."""
    group_of = torch.zeros(I, dtype=torch.long)
    for g, idxs in enumerate(groups):
        group_of[idxs] = g
    return group_of


def evaluate_layer(
    x: torch.Tensor,                # [T, H]  gate_up_input from npz
    gate_raw: torch.Tensor,         # [T, I]  recorded gate logits (ground truth)
    weights: dict[str, torch.Tensor],
    groups: list[list[int]],
    top_pct: float,                 # 0–100: activate top N% of groups per token
) -> dict[str, float]:
    gate_w = weights["gate"]    # [I, H]
    up_w   = weights["up"]      # [I, H]
    down_w = weights["down"]    # [H, I]
    I = gate_w.shape[0]
    G = len(groups)

    pred     = build_group_predictors(gate_w, groups)   # [G, H]
    group_of = _build_group_index(groups, I)            # [I]

    # Scores: [T, G]
    scores = x @ pred.T

    # Activate the top-K groups per token where K = ceil(top_pct/100 * G).
    # Using a per-token score quantile as threshold keeps budget fixed regardless
    # of the absolute scale of the scores (which varies widely across layers).
    K = max(1, round(top_pct / 100.0 * G))
    if K >= G:
        active_groups = torch.ones(x.shape[0], G, dtype=torch.bool)
    else:
        # threshold per token = the K-th largest score value
        kth_vals = scores.kthvalue(G - K + 1, dim=1, keepdim=True).values
        active_groups = scores >= kth_vals          # [T, G] bool

    # Expand group decisions to per-neuron mask: neuron i is active when its
    # group is active.  group_of[i] gives the group index for neuron i.
    neuron_mask = active_groups[:, group_of]      # [T, I] bool

    # Full-compute baseline
    full_out = mlp_full(x, gate_w, up_w, down_w)   # [T, H]

    # Sparse forward — vectorised over the token batch.
    # Apply the neuron mask by zeroing out inactive neurons after the full
    # gate/up projections; this avoids a Python loop over tokens while still
    # correctly zeroing skipped neurons before the down projection.
    gate_out = F.silu(x @ gate_w.T) * neuron_mask   # [T, I]
    up_out   = (x @ up_w.T)         * neuron_mask   # [T, I]
    sparse_out = (gate_out * up_out) @ down_w.T      # [T, H]

    # Per-token cosine similarity
    cosim = F.cosine_similarity(sparse_out, full_out, dim=1)   # [T]

    # Neuron recall: of neurons where gate_raw > 0, how many are in active groups?
    truly_active = gate_raw > 0
    recalled     = (truly_active & neuron_mask).sum().item()
    total_active = truly_active.sum().item()
    recall       = recalled / total_active if total_active > 0 else 1.0

    return {
        "cosim_mean": cosim.mean().item(),
        "cosim_std":  cosim.std().item(),
        "cosim_p5":   cosim.quantile(0.05).item(),
        "groups_active_frac": active_groups.float().mean().item(),
        "neuron_recall": recall,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Group-level FFN gate predictor quality evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", required=True,
                   help="HuggingFace model ID or local path. "
                        "Weights are read directly from safetensors shards — "
                        "vLLM is not required.")
    p.add_argument("--npz", required=True,
                   help=".npz file from record_ffn_activations.py.")
    p.add_argument("--groups", required=True,
                   help="channel_groups.json from cosim.py.")
    p.add_argument("--layers", nargs="*", type=int, default=None,
                   metavar="N",
                   help="Layer indices to evaluate. Default: all layers in npz.")
    p.add_argument("--top-pct", nargs="+", type=float, default=[100, 90, 75, 50, 25],
                   metavar="P",
                   help="Percentage(s) of groups to activate per token, ranked by "
                        "dot-product score (default: 100 90 75 50 25). "
                        "100 activates all groups (full compute baseline). "
                        "Multiple values sweep the compute/quality tradeoff.")
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "bfloat16"],
                   help="Compute dtype (default: float32).")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    # ---- Load groups ----
    groups_path = Path(args.groups)
    if not groups_path.exists():
        print(f"ERROR: groups file not found: {groups_path}", file=sys.stderr)
        sys.exit(1)
    with open(groups_path) as f:
        all_groups: dict[str, list[list[int]]] = json.load(f)
    print(f"Loaded groups from {groups_path}  "
          f"({len(all_groups)} layers, "
          f"{len(next(iter(all_groups.values())))} groups each)",
          file=sys.stderr)

    # ---- Load npz ----
    npz_path = Path(args.npz)
    if not npz_path.exists():
        print(f"ERROR: npz file not found: {npz_path}", file=sys.stderr)
        sys.exit(1)
    data = np.load(npz_path)
    available_layers = sorted(
        int(k.split("/")[0][5:]) for k in data.files if k.endswith("/gate_up_input")
    )
    layers = args.layers if args.layers is not None else available_layers
    missing = [l for l in layers if l not in available_layers]
    if missing:
        print(f"WARNING: layers {missing} not in npz; skipping.", file=sys.stderr)
        layers = [l for l in layers if l in available_layers]

    # ---- Resolve model directory ----
    try:
        model_dir = _find_model_dir(args.model)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Reading weights from {model_dir}", file=sys.stderr)

    dt = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    # ---- Header ----
    print(
        f"\n{'layer':>6}  {'top_pct':>9}  "
        f"{'cosim_mean':>11}  {'cosim_std':>10}  {'cosim_p5':>9}  "
        f"{'grp_active':>11}  {'neuron_rec':>11}"
    )
    print("  " + "-" * 78)

    for layer_idx in layers:
        group_key = f"layer{layer_idx}"
        if group_key not in all_groups:
            print(f"  layer {layer_idx}: no groups in JSON, skipping.", file=sys.stderr)
            continue

        groups = all_groups[group_key]

        x_np       = data[f"layer{layer_idx}/gate_up_input"]  # [T, H]
        gate_np    = data[f"layer{layer_idx}/gate_raw"]        # [T, I]

        x        = torch.from_numpy(x_np).to(dt)
        gate_raw = torch.from_numpy(gate_np).to(dt)

        try:
            weights = load_mlp_weights(model_dir, layer_idx)
        except Exception as e:
            print(f"  layer {layer_idx}: could not load weights: {e}", file=sys.stderr)
            continue
        weights = {k: v.to(dt) for k, v in weights.items()}

        for top_pct in args.top_pct:
            metrics = evaluate_layer(x, gate_raw, weights, groups, top_pct)
            print(
                f"{layer_idx:>6}  {top_pct:>8.1f}%  "
                f"{metrics['cosim_mean']:>11.5f}  "
                f"{metrics['cosim_std']:>10.5f}  "
                f"{metrics['cosim_p5']:>9.5f}  "
                f"{metrics['groups_active_frac']:>11.4f}  "
                f"{metrics['neuron_recall']:>11.4f}"
            )

if __name__ == "__main__":
    main()
