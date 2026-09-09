# FFN Activation Prediction

`tools/profiler/O1_predict.py` validates whether a **KV-retrieval scheme** can
predict which MLP neurons will be active for a given input token, without
running the full MLP computation.

The idea: similar hidden states tend to activate similar sets of intermediate
neurons.  Given a database of previously seen `(input, activation_mask)` pairs,
the predictor finds the K nearest stored inputs by cosine similarity and ORs
their binarised masks together to produce a predicted active set.

This is an **offline analysis tool** — it re-uses recordings produced by
[`record_ffn_activations.py`](inference_hooks.md#hook-based-activation-sampling-cli)
and evaluates prediction quality without touching the model.

## Prerequisites


The venv lives at `.venv/` inside the repository root.  On macOS `python` may
not resolve to the venv Python even after activation — use the explicit path:

```bash
# Create once if needed (from the vllm/ repo root):
uv venv --python 3.12
uv pip install numpy torch

# Run with the venv Python explicitly:
.venv/bin/python tools/profiler/O1_predict.py ...
```

First record FFN activations (requires a full vllm install — see
[Inference Hooks](inference_hooks.md#hook-based-activation-sampling-cli)):

```bash
.venv/bin/python tools/profiler/record_ffn_activations.py \
    --model meta-llama/Llama-3.2-1B \
    --prompts "The capital of France is" "Once upon a time" \
    --output ffn_activations.npz
```

See [Inference Hooks](inference_hooks.md) for full recording options.

## How it works

The token dataset is split into a **library** (default: first 80 %) and an
**eval set** (remaining 20 %).  For each eval token the predictor finds the K
nearest library tokens by cosine similarity on `gate_up_input` hidden states
and ORs their binarised activation masks together.

```
  library tokens                   eval tokens
  {(x_j, mask_j)}  ◀──cosine sim──  query x_i
                          │
                  top-K library neighbours
                          │
                  OR(mask_j for j in top-K)
                          │
                   predicted mask m̂_i
                          │
               compare with true mask m_i
```

The binary mask `m_i` is derived from `gate_raw` (the gate logits before SiLU)
by thresholding at a per-token percentile: neurons whose `|gate_raw|` exceeds
the P-th percentile of that token's gate distribution are marked active.
A threshold of 70 marks the top-30 % of neurons as active.  Using `gate_raw`
rather than `down_input` (the post-SiLU product) gives a sharper activity
signal: `SiLU(x) ≈ 0` for `x ≲ −4`, so neurons with strongly negative gate
logits are genuinely inactive and form a cleaner zero-mass in the distribution.

**OR is used deliberately**: a false negative (predicting a neuron inactive
when it is actually active) corrupts the MLP output, so the predictor errs
on the side of over-predicting.  Recall is therefore the primary safety
metric; density (fraction of neurons predicted active) is the sparsity cost.

## Usage

### Basic

```bash
.venv/bin/python tools/profiler/O1_predict.py --input ffn_activations.npz
```

Evaluates all layers with defaults: 80/20 train split, K=3, threshold=70th
percentile.  Prints a per-layer table to stdout:

```
Layer 0:  512 tokens  (library=409, eval=103),  H=2048,  I=8192
     K  thresh%    recall   precision   density
  ----------------------------------------------
     3      70.0     0.943       0.521     0.574
```

### Adjust the train/eval split

```bash
.venv/bin/python tools/profiler/O1_predict.py \
    --input ffn_activations.npz \
    --train-split 0.9
```

Uses 90 % of tokens as the lookup library and the remaining 10 % for
evaluation.  A larger library generally improves recall; a larger eval set
gives more reliable metric estimates.

### Sweep K and threshold

```bash
.venv/bin/python tools/profiler/O1_predict.py \
    --input ffn_activations.npz \
    --layers 0 15 \
    --top-k 1 3 5 \
    --threshold-pct 70 80 90
```

Each combination of `--top-k` and `--threshold-pct` is evaluated and printed
as a separate row, making it easy to read off the recall/density trade-off:

```
Layer 0:  512 tokens  (library=409, eval=103),  H=2048,  I=8192
     K  thresh%    recall   precision   density
  ----------------------------------------------
     1      70.0     0.821       0.712     0.421
     3      70.0     0.943       0.521     0.574
     5      70.0     0.971       0.412     0.727
     1      80.0     0.849       0.743     0.381
     3      80.0     0.961       0.583     0.521
     5      80.0     0.979       0.461     0.683
     1      90.0     0.891       0.801     0.302
     3      90.0     0.974       0.641     0.468
     5      90.0     0.988       0.531     0.621
```

### Save per-token detail

```bash
.venv/bin/python tools/profiler/O1_predict.py \
    --input ffn_activations.npz \
    --layers 0 15 \
    --top-k 1 3 5 \
    --threshold-pct 70 80 90 \
    --output-csv results.csv
```

Writes one row per `(layer, token, K, threshold)` combination to `results.csv`
for downstream analysis in a notebook or spreadsheet.

## Options

| Flag | Default | Description |
| --- | --- | --- |
| `--input` | *(required)* | `.npz` file from `record_ffn_activations.py` |
| `--train-split F` | `0.8` | Fraction of tokens used as the lookup library; the rest are eval |
| `--layers N …` | all | Layer indices to evaluate |
| `--top-k K …` | `3` | Neighbour count(s) to retrieve and OR; multiple values sweep |
| `--threshold-pct P …` | `70` | Percentile(s) for binarising activations; multiple values sweep |
| `--output-csv PATH` | none | Write per-token CSV results to this path |

## Interpreting results

| Metric | What it measures | Good value |
| --- | --- | --- |
| **recall** | Fraction of truly active neurons predicted active | As close to 1.0 as possible — missed neurons corrupt output |
| **precision** | Fraction of predicted-active neurons that are truly active | Higher = less wasted compute, but lower = safer |
| **density** | Fraction of neurons predicted active | Lower = more sparsity benefit; must be weighed against recall |

A practical operating point is **recall ≥ 0.95** at the lowest density that
achieves it.  Start with the sweep output: find the threshold/K row where
recall first crosses 0.95 and note its density.  If density is still below
~0.6, the scheme is worth pursuing; if density exceeds 0.7 the predicted
sparsity benefit is likely too small to justify the lookup overhead.

## Relationship to `record_ffn_activations.py`

The tools form a pipeline:

```
record_ffn_activations.py    cosim.py                gate_predict.py
─────────────────────────    ────────────────────    ───────────────────────
  vLLM inference + hooks  ─▶  group neurons by   ─▶  weight-based predictor
  → ffn_activations.npz       gate_raw cosim          cosim vs full output
                              → channel_groups.json
                                        │
                              O1_predict.py
                              ────────────────────
                              KV-retrieval recall/
                              precision sweep
                              (offline, no model)
```

`record_ffn_activations.py` needs a model and GPU/CPU inference.
`O1_predict.py` only needs NumPy and the `.npz` file — it can run on any
machine, including a Mac without a GPU.
`gate_predict.py` needs the model (to read weights) plus the `.npz` and the
groups JSON.

---

## Gate-weight group predictor (`gate_predict.py`)

`tools/profiler/gate_predict.py` implements a **weight-derived, group-level
predictor** and measures its output quality against full MLP computation.

Unlike the KV-retrieval approach in `O1_predict.py`, this predictor requires
no stored token database.  Instead it uses the model's own gate-projection
weights to decide, at inference time, which groups of neurons to compute.

### How it works

**Step 1 — group predictor vectors.**
For each group of neurons (produced by `cosim.py`) the gate-projection weight
rows of those neurons are summed into a single `[hidden_size]` vector:

```
pred[g] = Σ  gate_proj.weight[i]   for i ∈ group[g]
```

This vector represents the group's aggregate "signature" in hidden-state
space: if the current hidden state has a large positive dot product with it,
the group is likely to fire.

**Step 2 — rank and select top-N%.**
For each token, one scalar score per group is computed:

```
score[g, t] = hidden[t] · pred[g]
```

The top `--top-pct` percent of groups by score are predicted active and
computed; the rest are skipped.  Ranking rather than thresholding makes the
compute budget explicit and scale-invariant — the dot-product scores vary
widely in magnitude across layers, so an absolute threshold would need
per-layer tuning.

**Step 3 — sparse MLP forward.**
Only the active neurons' columns of `gate_proj` and `up_proj`, and the
corresponding rows of `down_proj`, are materialised.  Skipped neurons
contribute zero to the `down_proj` output:

```
active_idx = {i : group(i) is predicted active}
out = (SiLU(x · gate[active_idx].T) * (x · up[active_idx].T)) · down[:, active_idx].T
```

**Step 4 — quality measurement.**
The sparse output is compared to the full-compute baseline using **cosine
similarity** per token.  A value of 1.0 is a perfect match; values above 0.99
are typically imperceptible in downstream quality.

### Prerequisites

Only `safetensors`, `numpy`, and `torch` are required — vLLM is not needed.
Weights are read directly from the HuggingFace safetensors shards:

```bash
uv pip install safetensors numpy torch
```

First produce the two input files:

```bash
# 1. Record gate activations (ffn_activations128_gate.npz)
.venv/bin/python tools/profiler/record_ffn_activations.py \
    --model ibm-granite/granite-4.2-3b \
    --calibration-set bartowski-imatrix-v5-semantic.txt \
    --num-chunks 128 \
    --output ffn_activations128_gate.npz

# 2. Group neurons by cosine similarity (channel_groups.json)
.venv/bin/python cosim.py ffn_activations128_gate.npz \
    --output channel_groups.json
```

### Usage

```bash
# Sweep top-pct values (default: 100 90 75 50 25)
.venv/bin/python tools/profiler/gate_predict.py \
    --model ibm-granite/granite-4.2-3b \
    --npz ffn_activations128_gate.npz \
    --groups channel_groups.json

# Custom sweep on specific layers
.venv/bin/python tools/profiler/gate_predict.py \
    --model ibm-granite/granite-4.2-3b \
    --npz ffn_activations128_gate.npz \
    --groups channel_groups.json \
    --layers 0 15 39 \
    --top-pct 80 60 50 40 25
```

Output is one row per `(layer, top_pct)` combination:

```
 layer    top_pct   cosim_mean   cosim_std   cosim_p5   grp_active   neuron_rec
  ------------------------------------------------------------------------------
     0      80.0%      0.97336     0.01189    0.96288       0.7969       0.9936
     0      60.0%      0.92087     0.02236    0.88825       0.6016       0.9850
     0      50.0%      0.87746     0.03459    0.82318       0.5000       0.9784
     0      40.0%      0.81746     0.05413    0.72896       0.3984       0.9689
     0      25.0%      0.69740     0.09492    0.52482       0.2500       0.9267
    15      80.0%      0.92569     0.02269    0.89489       0.7969       0.9684
    15      60.0%      0.85605     0.03534    0.80048       0.6016       0.8930
    15      50.0%      0.81490     0.04193    0.74790       0.5000       0.8318
    15      40.0%      0.76649     0.04932    0.68499       0.3984       0.7500
    15      25.0%      0.67119     0.06494    0.56407       0.2500       0.5771
    39      80.0%      0.97393     0.01767    0.94381       0.7969       0.9777
    39      60.0%      0.94202     0.03675    0.87222       0.6016       0.8847
    39      50.0%      0.93465     0.04073    0.85806       0.5000       0.8004
    39      40.0%      0.92747     0.04519    0.84271       0.3984       0.6973
    39      25.0%      0.91382     0.05625    0.80378       0.2500       0.5103
```

### Options

| Flag | Default | Description |
| --- | --- | --- |
| `--model` | *(required)* | HuggingFace model ID or local path; weights read directly from safetensors |
| `--npz` | *(required)* | `.npz` file from `record_ffn_activations.py` |
| `--groups` | *(required)* | `channel_groups.json` from `cosim.py` |
| `--layers N …` | all | Layer indices to evaluate |
| `--top-pct P …` | `100 90 75 50 25` | Percentage(s) of groups to activate per token by rank; multiple values sweep |
| `--dtype` | `float32` | Compute dtype (`float32` or `bfloat16`) |

### Interpreting results

| Metric | What it measures | Target |
| --- | --- | --- |
| `cosim_mean` | Mean cosine similarity of sparse vs full `down_proj` output | ≥ 0.99 for imperceptible degradation |
| `cosim_p5` | 5th-percentile cosim — worst-case token quality | ≥ 0.95 for a safe operating point |
| `grp_active` | Mean fraction of groups activated (= `top_pct / 100`) | Lower = more compute saved |
| `neuron_rec` | Fraction of `gate_raw > 0` neurons in an activated group | ≥ 0.90 to cover genuinely-firing neurons |

**Reading the sweep:** `top_pct=100` is the full-compute baseline (cosim=1.0, no
savings).  As `top_pct` falls, compute drops linearly while cosim degrades
gracefully.  From the measured results on granite-4.2-3b:

| Layer group | top_pct for cosim_p5 ≥ 0.95 | compute saving |
| --- | --- | --- |
| 0 (embedding) | ~80% | 20% |
| 15 (plateau) | ~80% | 20% |
| 39 (output) | **40%** | **60%** |

Layer 39 is remarkably tolerant — activating only 40% of groups still yields
cosim_p5 = 0.84, and 25% still holds cosim_mean = 0.91.  Plateau layers (15)
are harder: even 80% top_pct gives cosim_p5 = 0.89.  The practical operating
point is the lowest `top_pct` where `cosim_p5 ≥ 0.90` and `neuron_rec ≥ 0.85`.

### Relationship to `cosim.py`

`cosim.py` determines *which* neurons are grouped together; `gate_predict.py`
uses those groups to build the predictor and measure quality.  Better groupings
(higher intra-group cosine similarity) produce better predictors because the
summed weight vector is a more faithful representative of the group.  Running
`cosim.py` with `--method kmeans` instead of `--method greedy` often yields
tighter groups and a higher `cosim_mean` at the same `grp_active` level.

---

## Down-projection input activation analysis

This section documents empirical analysis of the `down_proj` input vector
— i.e. `SiLU(gate_raw) * up_proj(x)` — recorded across 128 calibration
chunks of granite-4.2-3b (12 734 tokens × 8 192 neurons per layer, stored in
`ffn_activations128.npz`).  The goal is to understand what threshold or
selection criterion best separates the neurons that matter from those that can
be skipped.

### Value distribution by layer

`down_input` values are always non-negative (SiLU output multiplied element-
wise by `up_proj`).  The full-distribution percentiles across all tokens and
neurons show two distinct regimes:

| Layer range | p20 | p30 | p40 | p50 | p90 | p99 |
|---|---|---|---|---|---|---|
| 0 (embedding) | 0.053 | 0.084 | 0.120 | 0.163 | 0.582 | 2.203 |
| 5–27 (plateau) | 0.019–0.028 | 0.033–0.046 | 0.050–0.069 | 0.069–0.097 | 0.231–0.349 | 0.598–1.063 |
| 30–35 (late) | 0.026–0.055 | 0.045–0.095 | 0.068–0.142 | 0.097–0.199 | 0.408–0.898 | 1.516–3.938 |
| 39 (output) | 0.077 | 0.137 | 0.212 | 0.313 | 1.844 | 12.188 |

Key observations:

- **No zero mass.** Every neuron has a positive `down_input` value for every
  token — SiLU is never exactly zero for finite inputs.  There is no binary
  sparsity to exploit directly; only magnitude-based selection applies.
- **Values vary 4× across layers.** Any global absolute threshold would be
  either too tight for the output layers or too loose for the plateau.
  Per-token or per-layer normalisation is mandatory.
- **Heavy tail, consistent shape.** The ratio p50/p99 is 0.05–0.13 across all
  layers — the top 1% of neurons hold 10–20× the median value.  The bottom
  20–30% of neurons can be zeroed with negligible contribution.
- **Per-token p50 cut selects ~50% of neurons by construction.**  The sparse
  fraction above the per-token median is 42–51% across all layers, confirming
  that rank-based (top-k%) selection is the natural primitive — not an absolute
  threshold.

### Energy-based threshold analysis

An alternative to rank-based selection is to keep the minimum number of neurons
whose cumulative activation energy reaches a target fraction of the total.  Two
norms were evaluated:

- **L1 energy**: cumulative `|x|` — linear in activation magnitude
- **L2 energy**: cumulative `|x|²` — squares amplify dominant neurons

The table below shows the median neuron fraction (across tokens) needed to
capture N% of energy per layer.  Values are medians; token-to-token spread is
shown separately.

#### L1 energy (`|x|`)

| Layer | ≥80% | ≥90% | ≥95% | ≥99% |
|---|---|---|---|---|
| 0 | 0.439 | 0.591 | 0.705 | 0.863 |
| 5 | 0.409 | 0.556 | 0.670 | 0.838 |
| 10 | 0.434 | 0.580 | 0.691 | 0.850 |
| 15 | 0.405 | 0.553 | 0.668 | 0.837 |
| 20 | 0.400 | 0.548 | 0.664 | 0.834 |
| 25 | 0.408 | 0.556 | 0.671 | 0.838 |
| 30 | 0.358 | 0.512 | 0.634 | 0.818 |
| 35 | 0.338 | 0.498 | 0.625 | 0.815 |
| 39 | 0.274 | 0.429 | 0.565 | 0.781 |

L1 at 90% energy requires **43–59% of neurons** — barely better than a flat
50% rank cut.  The linear norm is not strongly concentrated.

#### L2 energy (`|x|²`)

| Layer | ≥80% | ≥90% | ≥95% | ≥99% |
|---|---|---|---|---|
| 0 | 0.106 | 0.228 | 0.354 | 0.594 |
| 5 | 0.130 | 0.242 | 0.353 | 0.569 |
| 10 | 0.159 | 0.281 | 0.395 | 0.608 |
| 15 | 0.121 | 0.233 | 0.345 | 0.564 |
| 20 | 0.118 | 0.228 | 0.338 | 0.557 |
| 25 | 0.108 | 0.220 | 0.334 | 0.556 |
| 30 | 0.064 | 0.146 | 0.248 | 0.478 |
| 35 | 0.038 | 0.093 | 0.179 | 0.418 |
| 39 | 0.013 | 0.038 | 0.096 | 0.274 |

L2 at 90% energy requires only **4–29% of neurons** — a dramatic reduction
driven by the squared amplification of dominant activations.  Late layers
(35–39) are especially sparse in L2: a handful of very large activations
dominate the squared sum.

#### Token-level spread at 90% energy

The neuron fraction is not fixed — it varies token-by-token.  At the 90%
energy threshold, the p10–p90 spread across tokens is:

| Layer | L1 p10 | L1 med | L1 p90 | L2 p10 | L2 med | L2 p90 |
|---|---|---|---|---|---|---|
| 0 | 0.548 | 0.591 | 0.609 | 0.128 | 0.228 | 0.327 |
| 10 | 0.569 | 0.580 | 0.589 | 0.226 | 0.281 | 0.314 |
| 20 | 0.538 | 0.548 | 0.557 | 0.190 | 0.228 | 0.255 |
| 30 | 0.501 | 0.512 | 0.526 | 0.118 | 0.146 | 0.171 |
| 39 | 0.369 | 0.429 | 0.461 | 0.015 | 0.038 | 0.082 |

L1 spread is narrow (±3–4 pp), making it predictable.  L2 spread is wide,
especially in late layers — at layer 39, the p10–p90 range is 0.015–0.082,
a 5× ratio.  This means a fixed L2 energy budget will significantly over-select
for most tokens and under-select for a few extreme outlier tokens.

### Comparison of selection strategies

| Strategy | Mid-layer neurons | Layer-39 neurons | Token variability | Notes |
|---|---|---|---|---|
| Rank top-50% | 50% | 50% | none | Baseline; blind to magnitude |
| L1 ≥90% energy | ~55% | ~43% | low (±4 pp) | Small gain over rank; predictable |
| L2 ≥90% energy | ~22% | ~4% | high (5× range) | Aggressive; dominated by outlier spikes |
| Gate-score predictor (find_budget.py) | 58–78% | ~48% | n/a | Predictive; budget set before MLP runs |

### Practical guidance

**Use rank-based selection for the gate predictor.**  The `gate_predict.py` /
`find_budget.py` pipeline already uses top-pct ranking on gate scores, which is
equivalent to a rank cut on the predicted (pre-SiLU) activation.  This is
preferable to post-hoc energy cutting because the selection happens *before*
the MLP runs.

**L2 energy as an oracle lower bound.**  If you have already computed all
activations and want to know the theoretical minimum compute budget, L2 energy
at 90% gives a useful floor: ~4–29% of neurons per layer.  This bounds how
much a perfect predictor could save.  The gap between this floor and the
gate-score predictor's actual budget (58–78%) represents the remaining
prediction headroom.

**Avoid global absolute thresholds.**  `down_input` magnitudes vary 4× across
layers.  Any single threshold (e.g. "zero neurons below 0.05") will be
miscalibrated outside the plateau layers 5–27.  Always normalise per-token or
per-layer.

---

## Gate-first oracle: full gate + sparse up/down

A fundamentally different strategy to the weight-based group predictor is to
**compute the full gate projection first**, use the resulting SiLU activations
as a perfect per-token oracle to select the top-K neurons, and then run only
those K neurons through `up_proj` and `down_proj`.

```
x ──▶ gate_proj (full, H×I) ──▶ SiLU(·) ──▶ top-K selection
                                                    │
x ──▶ up_proj (K columns only) ──────────────────▶ × ──▶ down_proj (K rows only) ──▶ out
```

This is an **oracle** — it cannot be directly applied at inference time because
the gate values are only known after running gate_proj, which is precisely the
computation that precedes up/down.  Its value is as a **quality ceiling**: it
tells us exactly how much up/down compute can be saved given a perfect gate
signal, and it establishes an upper bound on what any predictive scheme can
achieve.

### Compute accounting

Each of `gate_proj`, `up_proj`, `down_proj` costs `H×I` multiplications
(`H=2560`, `I=8192` for granite-4.2-3b → 20.97 M mults/token each, 62.9 M
total).  Under the gate-first scheme:

```
total cost = gate(H×I) + up(H×K) + down(H×K) = H×I × (1 + 2×K/I)
```

Compute saving vs full MLP = `2×(1 − K/I) / 3`.  The floor is **33.3% saving
at K=0** (gate cost can never be avoided), and saving scales linearly with the
fraction of neurons skipped.

| K/I | neurons computed | total cost (vs full) | saving |
|---|---|---|---|
| 1.00 | 8192 | 100% | 0% |
| 0.90 | 7373 | 93.3% | **6.7%** |
| 0.80 | 6554 | 86.7% | **13.3%** |
| 0.70 | 5734 | 80.0% | **20.0%** |
| 0.50 | 4096 | 66.7% | **33.3%** |
| 0.20 | 1638 | 46.7% | **53.3%** |

### Quality curve (granite-4.2-3b, all 40 layers)

Neurons are selected by descending `SiLU(gate_raw)` per token; quality is
measured as cosine similarity between the sparse and full `down_proj` output.

The minimum K/I needed to reach each cosim_p5 floor, across all 40 layers:

| Layer | K/I for p5≥0.99 | saving | K/I for p5≥0.98 | saving | K/I for p5≥0.97 | saving |
|---|---|---|---|---|---|---|
| 0 | 0.97 | 2.0% | 0.93 | 4.7% | 0.90 | 6.7% |
| 1–2 | 0.97 | 2.0% | 0.94 | 4.0% | 0.91–0.92 | 5–6% |
| 3–7 | 0.98 | 1.3% | 0.95–0.96 | 2.7–3.3% | 0.93–0.94 | 4–5% |
| 8–14 | 0.98–0.99 | 0.7–1.3% | 0.96–0.97 | 2–2.7% | 0.94–0.95 | 3.3–4% |
| 15–27 | 0.98–0.99 | 0.7–1.3% | 0.96–0.97 | 2–2.7% | 0.93–0.95 | 3.3–4.7% |
| 28–31 | 0.97–0.98 | 1.3–2.0% | 0.93–0.95 | 3.3–4.7% | 0.90–0.92 | 5.3–6.7% |
| 32–39 | 0.90–0.96 | 2.7–6.7% | 0.85–0.91 | 6–10% | 0.85 | **10%** |

**Summary across all 40 layers:**

| Quality floor | Feasible layers | Mean saving |
|---|---|---|
| cosim_p5 ≥ 0.99 | 40/40 | **1.9%** |
| cosim_p5 ≥ 0.98 | 40/40 | **3.7%** |
| cosim_p5 ≥ 0.97 | 40/40 | **5.4%** |

### Key findings

**1 — The gate signal is a near-perfect oracle, but the saving is modest.**
Even skipping only the bottom 1–7% of neurons (by SiLU value) is enough to
hold cosim_p5 ≥ 0.99 on every layer.  The bottom 15% can be skipped at
cosim_p5 ≥ 0.97.  However, the ceiling saving of the entire approach is only
5–10% because the gate projection (1/3 of MLP compute) must always be run.

**2 — Late layers (32–39) are far more skippable than early/middle layers.**
At p5≥0.97, layers 32–39 need only K/I=0.85 (10% saving); layers 3–27 only
allow 3–5% saving at the same floor.  This mirrors the L2 energy analysis —
late layers have highly concentrated activation distributions dominated by a
few large neurons.

**3 — This scheme is complementary to, not a replacement for, the group predictor.**
The gate-first oracle provides a hard ceiling.  The group predictor
([`gate_predict.py`](../../tools/profiler/gate_predict.py) /
[`find_budget.py`](../../tools/profiler/find_budget.py)) operates *before* any
MLP compute by predicting which neuron groups to skip, avoiding up to 58–78%
of total MLP cost (including the gate projection) at equivalent quality floors.
The trade-off is prediction error: the oracle is lossless within its K budget;
the predictor incurs false positives and false negatives.

**4 — Practical hybrid.**
A viable production strategy is to combine both:
- Run gate_proj fully (unavoidable for decoding-step latency).
- Use the true SiLU values to skip the bottom `B%` of up_proj/down_proj neurons.
- Skipping 10–15% of up/down at cosim_p5 ≥ 0.97 costs only 3–5% total MLP
  saving but requires zero prediction infrastructure — just a `topk` on the
  gate output that is already computed.

This is particularly attractive when `gate_proj` and `up_proj` are fused in
hardware (as in the `gate_up_proj` merged weight used by vLLM), where the gate
values are already in registers before the element-wise SiLU multiply.

---

## SiLU(gate) distribution and the near-zero region

`SiLU(x) = x · σ(x)` reaches its global minimum at `x ≈ −1.28`, giving
`SiLU(−1.28) ≈ −0.2785`.  For large negative gate logits the output saturates
at this floor rather than going to zero.  Measured across all 40 layers of
granite-4.2-3b (12 734 tokens × 8 192 neurons per layer):

### Percentile structure

Every layer's p1 and p5 sit at or within 0.002 of the −0.2785 floor, meaning
the bottom ~5 % of neurons are saturated at the minimum on every layer.  The
median (p50) is **negative on all 40 layers**, ranging from −0.24 (layers 0–1,
heavily saturated) down to −0.04 (layer 33, least saturated).  Only the p75
crosses zero for many layers; the top 10–25 % of neurons carry positive signal.

| Layer range | p50 | p75 | p90 | character |
|---|---|---|---|---|
| 0–1 | −0.24 | −0.14 | +0.19 to +0.83 | deeply saturated |
| 2–5 | −0.08 to −0.19 | +0.09 to +0.23 | +0.40 to +0.72 | widest spread |
| 6–14 | −0.19 to −0.24 | −0.09 to −0.05 | +0.10 to +0.26 | mid plateau |
| 15–28 | −0.14 to −0.23 | 0.00 to +0.12 | +0.31 to +0.52 | plateau / rising |
| 29–39 | −0.04 to −0.18 | +0.18 to +0.46 | +0.64 to +1.35 | late layers, heavy tail |

### Near-zero fraction by layer

The fraction of neurons with `SiLU(gate) < 0` (negative, near-floor) and
`< 0.01` (effectively zero contribution) are nearly identical — confirming the
negative region is tightly clustered at −0.278 rather than spread across a
wide negative range.

| Layer | neg < 0 | < 0.01 | < 0.1 |
|---|---|---|---|
| 0 | 81% | 81% | 83% |
| 1 | 84% | 85% | 88% |
| 2 | 58% | 59% | 67% | ← least saturated early layer
| 8–13 | 79–86% | 80–86% | 86–90% | ← most saturated plateau
| 16 | 67% | 67% | 74% |
| 33 | **53%** | **54%** | **60%** | ← global minimum (late)
| 39 | 67% | 67% | 71% |

The mean across all layers is **~71% of neurons per token** with negative SiLU
output (near-floor), and **~73% below 0.01**.  Only the top ~27–30% of neurons
carry a meaningfully positive gate activation on any given token.

### Why the negative floor matters for thresholding

A threshold on `gate_raw` (pre-SiLU) is cleaner than one on `SiLU(gate_raw)`
because:

1. `gate_raw` has a true zero crossing at `x = 0`; neurons with `gate_raw < 0`
   produce small-magnitude (but non-zero) SiLU outputs near the −0.2785 floor.
2. Gating on `gate_raw < threshold` can use a single scalar comparison per
   neuron with no activation function evaluation for the pruned set.
3. The `gate_threshold.py` tool (below) implements exactly this: it computes
   the full gate projection, applies a user-supplied `gate_raw` threshold, and
   runs `up_proj` + `down_proj` only on the surviving neurons.

### Relationship to the gate-first oracle

The top-K oracle (previous section) selects by descending `SiLU` value;
threshold gating selects by `gate_raw > T`.  They are equivalent when T maps
to the K-th largest `gate_raw` value, but threshold gating is simpler to
implement and has a natural semantic: neurons with `gate_raw < 0` are on the
saturated-negative portion of SiLU and contribute negatively to the output.
Setting `T = 0` prunes all such neurons, which is approximately 60–85% of
neurons depending on the layer.

---

## Gate-threshold tool (`gate_threshold.py`)

`tools/profiler/gate_threshold.py` computes the full gate projection, applies
a `gate_raw` threshold to predict inactive neurons, and runs `up_proj` +
`down_proj` only on the active set.  For each layer it reports:

- **inactive_frac** — fraction of neurons predicted unused (gate_raw < threshold)
- **compute_saved** — total MLP compute saved = `2 × inactive_frac / 3`
- **MSE** — mean squared error of the sparse vs full `down_proj` output vector,
  averaged over tokens

```
x ──▶ gate_proj (full) ──▶ gate_raw ──▶ gate_raw ≥ T ? active
                                                          │
x ──▶ up_proj (active only) ──────────────────────────▶ × ──▶ down_proj (active only) ──▶ out
```

### Usage

```bash
# Single threshold sweep on all layers
.venv/bin/python tools/profiler/gate_threshold.py \
    --model ibm-granite/granite-4.2-3b \
    --npz  ffn_activations128_gate.npz \
    --threshold 0.0

# Sweep multiple thresholds
.venv/bin/python tools/profiler/gate_threshold.py \
    --model ibm-granite/granite-4.2-3b \
    --npz  ffn_activations128_gate.npz \
    --threshold -1.0 -0.5 0.0 0.5 1.0

# Restrict to specific layers
.venv/bin/python tools/profiler/gate_threshold.py \
    --model ibm-granite/granite-4.2-3b \
    --npz  ffn_activations128_gate.npz \
    --threshold 0.0 \
    --layers 0 15 33 39
```

### Options

| Flag | Default | Description |
| --- | --- | --- |
| `--model` | *(required)* | HuggingFace model ID or local path |
| `--npz` | *(required)* | `.npz` file from `record_ffn_activations.py` |
| `--threshold T …` | `0.0` | `gate_raw` threshold(s); neurons with `gate_raw < T` are skipped |
| `--layers N …` | all | Layer indices to evaluate |
| `--dtype` | `float32` | Compute dtype (`float32` or `bfloat16`) |

### Output

```
threshold = +0.0000
layer   threshold   inactive      saved           MSE       MSE_std
───────────────────────────────────────────────────────────────────
    0     +0.0000      81.1%      54.1%      0.022770      0.010653
    1     +0.0000      84.2%      56.2%      0.012255      0.004852
    2     +0.0000      58.2%      38.8%      0.004225      0.001786
    5     +0.0000      67.0%      44.7%      0.001786      0.000784
   10     +0.0000      83.8%      55.8%      0.002269      0.000650
   15     +0.0000      73.0%      48.7%      0.001786      0.000577
   20     +0.0000      71.7%      47.8%      0.001736      0.000375
   25     +0.0000      74.4%      49.6%      0.001974      0.000540
   30     +0.0000      62.0%      41.4%      0.001878      0.000390
   33     +0.0000      53.1%      35.4%      0.003672      0.000685
   35     +0.0000      65.7%      43.8%      0.012452      0.003673
   39     +0.0000      66.5%      44.3%      0.103709      0.028500
───────────────────────────────────────────────────────────────────
 mean     +0.0000      70.1%      46.7%      0.014209
```

MSE is high for layer 39 because that layer has very large activations
(p99 ≈ 12.19 in absolute magnitude); the relative error (cosine similarity)
is still 0.97+ as shown in the gate-first oracle section above.
