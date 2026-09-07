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

**Step 2 — score and threshold.**
For each token, one scalar score per group is computed:

```
score[g, t] = hidden[t] · pred[g]
```

Groups whose score exceeds a threshold `T` are predicted active and will be
computed; the rest are skipped and contribute zero to the output.

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

Same as `O1_predict.py` plus `vllm` installed (the model is loaded to read
its weights):

```bash
uv pip install -r requirements/test/cuda.in   # or requirements/cpu.txt on CPU
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
# Evaluate all layers, default threshold 0.0
.venv/bin/python tools/profiler/gate_predict.py \
    --model ibm-granite/granite-4.2-3b \
    --npz ffn_activations128_gate.npz \
    --groups channel_groups.json

# Sweep thresholds to find the cosim/savings tradeoff
.venv/bin/python tools/profiler/gate_predict.py \
    --model ibm-granite/granite-4.2-3b \
    --npz ffn_activations128_gate.npz \
    --groups channel_groups.json \
    --threshold 0.0 1.0 2.0 4.0 8.0

# Evaluate only layers 0, 15, and 39
.venv/bin/python tools/profiler/gate_predict.py \
    --model ibm-granite/granite-4.2-3b \
    --npz ffn_activations128_gate.npz \
    --groups channel_groups.json \
    --layers 0 15 39 \
    --threshold 0.0 2.0 4.0 8.0
```

Output is one row per `(layer, threshold)` combination:

```
 layer   thresh   cosim_mean   cosim_std  cosim_p5  grp_active  neuron_rec
  -----------------------------------------------------------------------
     0     0.00      1.00000     0.00000   1.00000      1.0000      1.0000
     0     2.00      0.99712     0.00431   0.98801      0.7234      0.9531
     0     4.00      0.98103     0.01820   0.94210      0.4891      0.8762
```

### Options

| Flag | Default | Description |
| --- | --- | --- |
| `--model` | *(required)* | Model name or path — used only to read weights |
| `--npz` | *(required)* | `.npz` file from `record_ffn_activations.py` |
| `--groups` | *(required)* | `channel_groups.json` from `cosim.py` |
| `--layers N …` | all | Layer indices to evaluate |
| `--threshold T …` | `0.0` | Dot-product threshold(s) for group activation; multiple values sweep |
| `--dtype` | `float32` | Compute dtype (`float32` or `bfloat16`) |

### Interpreting results

| Metric | What it measures | Target |
| --- | --- | --- |
| `cosim_mean` | Mean cosine similarity of sparse vs full `down_proj` output | ≥ 0.99 for imperceptible degradation |
| `cosim_p5` | 5th-percentile cosim — worst-case token quality | ≥ 0.95 for a safe operating point |
| `grp_active` | Mean fraction of groups predicted active | Lower = more compute saved; `1 − grp_active` is the skip rate |
| `neuron_recall` | Fraction of `gate_raw > 0` neurons inside a predicted-active group | ≥ 0.90 to avoid corrupting strongly-firing neurons |

**Reading the threshold sweep:** at threshold = 0.0 every group scores above
zero so `grp_active = 1.0` and `cosim = 1.0` (baseline, no savings).  As the
threshold rises, groups are shed and `grp_active` drops.  The practical
operating point is the highest threshold where `cosim_p5 ≥ 0.95` and
`neuron_recall ≥ 0.90`.

### Relationship to `cosim.py`

`cosim.py` determines *which* neurons are grouped together; `gate_predict.py`
uses those groups to build the predictor and measure quality.  Better groupings
(higher intra-group cosine similarity) produce better predictors because the
summed weight vector is a more faithful representative of the group.  Running
`cosim.py` with `--method kmeans` instead of `--method greedy` often yields
tighter groups and a higher `cosim_mean` at the same `grp_active` level.
