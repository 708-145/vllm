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

For each token position in the recorded data the predictor performs a
**leave-one-out** evaluation: it treats each token as the query, finds its
K nearest neighbours among all other tokens in the same layer using cosine
similarity on the `gate_up_input` hidden states, and ORs their activation
masks together.

```
query x_i  ──cosine sim──▶  top-K neighbours {x_j}
                                    │
                            OR(mask_j for j in top-K)
                                    │
                             predicted mask m̂_i
                                    │
                         compare with true mask m_i
```

The binary mask `m_i` is derived from `down_input` (the post-SiluAndMul
activations) by thresholding at a per-token percentile: neurons whose
`|activation|` exceeds the P-th percentile of that token's activation
distribution are marked active.  A threshold of 70 marks the top-30 % of
neurons as active.

**OR is used deliberately**: a false negative (predicting a neuron inactive
when it is actually active) corrupts the MLP output, so the predictor errs
on the side of over-predicting.  Recall is therefore the primary safety
metric; density (fraction of neurons predicted active) is the sparsity cost.

## Usage

### Basic

```bash
.venv/bin/python tools/profiler/O1_predict.py --input ffn_activations.npz
```

Evaluates all layers present in the file with defaults: K=3, threshold=70th
percentile.  Prints a per-layer table to stdout:

```
Layer 0:  512 tokens,  H=2048,  I=8192
     K  thresh%    recall   precision   density
  ----------------------------------------------
     3      70.0     0.943       0.521     0.574
```

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
Layer 0:  512 tokens,  H=2048,  I=8192
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

The two tools form a pipeline:

```
record_ffn_activations.py          O1_predict.py
──────────────────────────         ──────────────────────────
  vLLM inference + hooks    ──▶    offline recall/precision
  → ffn_activations.npz            sweep over K and threshold
```

`record_ffn_activations.py` needs a model and GPU/CPU inference.
`O1_predict.py` only needs NumPy and the `.npz` file — it can run on any
machine, including a Mac without a GPU.
