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

3. For every eval token, computes a scalar score per group::

       score[g, t] = hidden[t] · pred[g]                       # dot product

   and predicts a group "active" when ``score[g, t] > threshold``.

4. For each token, runs the MLP only for predicted-active groups:
   - gate and up projections restricted to those neuron columns,
   - SiLU applied,
   - down projection restricted to those neuron rows.
   Skipped-group neurons contribute zero to the down-proj output.

5. Compares the resulting ``down_proj`` output against the full-compute
   baseline (all groups active) using per-token cosine similarity.

Reported metrics per layer:
  - groups_predicted  – mean fraction of groups predicted active
  - cosim_mean/std    – mean/std cosine similarity of predicted vs full output
  - recall_neurons    – mean fraction of truly-active neurons (gate_raw > 0)
                        that fall inside a predicted-active group

Usage::

    python tools/profiler/gate_predict.py \\
        --model ibm-granite/granite-4.2-3b \\
        --npz ffn_activations128_gate.npz \\
        --groups channel_groups.json

    # Sweep thresholds
    python tools/profiler/gate_predict.py \\
        --model ibm-granite/granite-4.2-3b \\
        --npz ffn_activations128_gate.npz \\
        --groups channel_groups.json \\
        --threshold 0.0 1.0 2.0 4.0

    # Evaluate only layers 0 and 15
    python tools/profiler/gate_predict.py \\
        --model ibm-granite/granite-4.2-3b \\
        --npz ffn_activations128_gate.npz \\
        --groups channel_groups.json \\
        --layers 0 15
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Weight extraction
# ---------------------------------------------------------------------------

def _get_layers(model: torch.nn.Module) -> list:
    for attr in ("model",):
        inner = getattr(model, attr, None)
        if inner is not None:
            layers = getattr(inner, "layers", None)
            if layers is not None:
                return list(layers)
    transformer = getattr(model, "transformer", None)
    if transformer is not None:
        return list(getattr(transformer, "h", []))
    raise RuntimeError("Could not locate transformer layers.")


def _get_mlp(layer: torch.nn.Module) -> torch.nn.Module:
    for attr in ("mlp", "ffn", "feed_forward"):
        m = getattr(layer, attr, None)
        if m is not None:
            return m
    raise RuntimeError(f"Could not find MLP in {type(layer).__name__}.")


def _get_proj(mlp: torch.nn.Module, *candidates: str) -> torch.nn.Module:
    for name in candidates:
        m = getattr(mlp, name, None)
        if m is not None:
            return m
    raise RuntimeError(f"Could not find projection {candidates}.")


def _extract_weight(proj: torch.nn.Module) -> torch.Tensor:
    """Return the weight tensor as float32 on CPU regardless of vLLM wrapper."""
    w = getattr(proj, "weight", None)
    if w is None:
        raise RuntimeError(f"No weight attribute on {type(proj).__name__}.")
    return w.detach().float().cpu()


def extract_mlp_weights(
    model: torch.nn.Module,
    layer_idx: int,
) -> dict[str, torch.Tensor]:
    """Return gate_proj, up_proj, down_proj weights for one layer.

    For models where gate and up are fused into ``gate_up_proj``, split the
    weight at the midpoint.

    Returns dict with keys ``gate``, ``up``, ``down``, each float32 CPU tensor:
      gate: [intermediate, hidden]
      up:   [intermediate, hidden]
      down: [hidden, intermediate]
    """
    layers = _get_layers(model)
    mlp = _get_mlp(layers[layer_idx])

    # Prefer separate gate_proj / up_proj; fall back to fused gate_up_proj
    gate_mod = getattr(mlp, "gate_proj", None)
    up_mod   = getattr(mlp, "up_proj",   None)
    if gate_mod is not None and up_mod is not None:
        gate_w = _extract_weight(gate_mod)   # [I, H]
        up_w   = _extract_weight(up_mod)     # [I, H]
    else:
        fused = _get_proj(mlp, "gate_up_proj", "c_fc", "dense_h_to_4h", "fc1")
        fused_w = _extract_weight(fused)     # [2*I, H]
        I = fused_w.shape[0] // 2
        gate_w, up_w = fused_w[:I], fused_w[I:]

    down_mod = _get_proj(mlp, "down_proj", "c_proj", "dense_4h_to_h", "fc2")
    down_w   = _extract_weight(down_mod)     # [H, I]

    return {"gate": gate_w, "up": up_w, "down": down_w}


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


def mlp_sparse(
    x: torch.Tensor,                # [T, H]
    gate_w: torch.Tensor,           # [I, H]
    up_w: torch.Tensor,             # [I, H]
    down_w: torch.Tensor,           # [H, I]
    active_neurons: torch.Tensor,   # [I] bool — which neurons to compute
) -> torch.Tensor:
    """Sparse MLP: only compute active neurons, zero the rest.  Returns [T, H]."""
    idx = active_neurons.nonzero(as_tuple=True)[0]   # [K]
    if idx.numel() == 0:
        return torch.zeros(x.shape[0], down_w.shape[0], dtype=x.dtype)
    gate_out = F.silu(x @ gate_w[idx].T)    # [T, K]
    up_out   = x @ up_w[idx].T              # [T, K]
    # down_w: [H, I] → select columns idx → [H, K] → transpose to [K, H]
    return (gate_out * up_out) @ down_w[:, idx].T     # [T, H]


def predict_active_neurons(
    x: torch.Tensor,            # [T, H]
    pred: torch.Tensor,         # [G, H]
    groups: list[list[int]],
    I: int,
    threshold: float,
) -> torch.Tensor:
    """Return [T, I] bool mask: True for neurons in predicted-active groups."""
    scores = x @ pred.T                          # [T, G]
    active_groups = scores > threshold           # [T, G] bool
    # expand groups to neuron mask
    mask = torch.zeros(x.shape[0], I, dtype=torch.bool)
    for g, idxs in enumerate(groups):
        col = torch.tensor(idxs, dtype=torch.long)
        mask[:, col] |= active_groups[:, g : g + 1]
    return mask   # [T, I]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_layer(
    x: torch.Tensor,                # [T, H]  gate_up_input from npz
    gate_raw: torch.Tensor,         # [T, I]  recorded gate logits (ground truth)
    weights: dict[str, torch.Tensor],
    groups: list[list[int]],
    threshold: float,
) -> dict[str, float]:
    gate_w = weights["gate"]    # [I, H]
    up_w   = weights["up"]      # [I, H]
    down_w = weights["down"]    # [H, I]
    I = gate_w.shape[0]

    pred = build_group_predictors(gate_w, groups)   # [G, H]

    # Full-compute baseline
    full_out = mlp_full(x, gate_w, up_w, down_w)   # [T, H]

    # Per-token sparse forward
    neuron_mask = predict_active_neurons(x, pred, groups, I, threshold)  # [T, I]
    sparse_out = torch.zeros_like(full_out)
    for t in range(x.shape[0]):
        sparse_out[t] = mlp_sparse(x[t : t + 1], gate_w, up_w, down_w,
                                    neuron_mask[t])

    # Cosine similarity between sparse and full outputs, per token
    cosim = F.cosine_similarity(sparse_out, full_out, dim=1)   # [T]

    # Group density: mean fraction of groups predicted active
    scores = x @ pred.T                       # [T, G]
    active_groups = (scores > threshold)      # [T, G]
    group_density = active_groups.float().mean().item()

    # Neuron recall: of neurons where gate_raw > 0, how many are in active groups?
    truly_active = gate_raw > 0               # [T, I]
    recalled = (truly_active & neuron_mask).sum().item()
    total_active = truly_active.sum().item()
    recall = recalled / total_active if total_active > 0 else 1.0

    return {
        "cosim_mean": cosim.mean().item(),
        "cosim_std":  cosim.std().item(),
        "cosim_p5":   cosim.quantile(0.05).item(),
        "groups_active_frac": group_density,
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
                   help="Model name or path (same as used for recording).")
    p.add_argument("--npz", required=True,
                   help=".npz file from record_ffn_activations.py.")
    p.add_argument("--groups", required=True,
                   help="channel_groups.json from cosim.py.")
    p.add_argument("--layers", nargs="*", type=int, default=None,
                   metavar="N",
                   help="Layer indices to evaluate. Default: all layers in npz.")
    p.add_argument("--threshold", nargs="+", type=float, default=[0.0],
                   metavar="T",
                   help="Group activation threshold(s) on the dot-product score "
                        "(default: 0.0). Multiple values sweep. "
                        "A group is predicted active when "
                        "hidden · sum(gate_rows) > T.")
    p.add_argument("--train-split", type=float, default=0.8, metavar="F",
                   help="Fraction of tokens used as training split (unused here "
                        "— all eval tokens used). Kept for consistency. "
                        "(default: 0.8)")
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "bfloat16"],
                   help="Compute dtype (default: float32).")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    # Suppress vLLM startup noise
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

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

    # ---- Load model ----
    print(f"Loading model: {args.model}", file=sys.stderr)
    from vllm import LLM
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        enforce_eager=True,
        kv_cache_memory_bytes=int(1e9),
        max_model_len=512,
    )
    nn_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    print("Model loaded.", file=sys.stderr)

    dt = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    # ---- Header ----
    print(
        f"\n{'layer':>6}  {'thresh':>8}  "
        f"{'cosim_mean':>11}  {'cosim_std':>10}  {'cosim_p5':>9}  "
        f"{'grp_active':>11}  {'neuron_rec':>11}"
    )
    print("  " + "-" * 76)

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
            weights = extract_mlp_weights(nn_model, layer_idx)
        except Exception as e:
            print(f"  layer {layer_idx}: could not extract weights: {e}", file=sys.stderr)
            continue
        # cast weights to compute dtype
        weights = {k: v.to(dt) for k, v in weights.items()}

        for thresh in args.threshold:
            metrics = evaluate_layer(x, gate_raw, weights, groups, thresh)
            print(
                f"{layer_idx:>6}  {thresh:>8.2f}  "
                f"{metrics['cosim_mean']:>11.5f}  "
                f"{metrics['cosim_std']:>10.5f}  "
                f"{metrics['cosim_p5']:>9.5f}  "
                f"{metrics['groups_active_frac']:>11.4f}  "
                f"{metrics['neuron_recall']:>11.4f}"
            )

    # Free model before exit
    del nn_model, llm


if __name__ == "__main__":
    main()
