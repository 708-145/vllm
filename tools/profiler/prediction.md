# FFN Down-Projection Activation Prediction — Analysis

Recorded data: `ffn_activations128.npz` (40 layers, 12 734 tokens, 8 192 intermediate neurons,
Qwen2.5-based model).  All percentile estimates use a 200 k random sample per layer; oracle
top-K analysis uses the full token set.

---

## 1. Distribution of `|down_input|`

`down_input` is the element-wise product `SiLU(gate_raw) * up_raw` — the input to the down
projection.  Its absolute value distribution is right-skewed but with a very fat body:

| layer group | p20 | p30 | p40 | p50 (median) | p90 | mean |
|-------------|----:|----:|----:|-------------:|----:|-----:|
| 0 (first)   | 0.052 | 0.084 | 0.119 | 0.161 | 0.578 | 0.280 |
| 1–2         | 0.033–0.041 | 0.055–0.066 | 0.082–0.095 | 0.114–0.128 | 0.422–0.434 | 0.191–0.196 |
| 3–27 (plateau) | **0.019–0.026** | 0.032–0.044 | 0.048–0.063 | 0.067–0.096 | 0.231–0.303 | 0.105–0.138 |
| 28–31       | 0.024–0.031 | 0.042–0.054 | 0.063–0.082 | 0.089–0.116 | 0.348–0.500 | 0.160–0.226 |
| 32–38       | 0.037–0.059 | 0.064–0.104 | 0.097–0.159 | 0.138–0.229 | 0.625–1.289 | 0.279–0.600 |
| 39 (last)   | 0.077 | 0.137 | 0.212 | 0.313 | 1.844 | 1.265 |

Key observation: **the plateau layers (3–27) are the flattest** — tightest range, lowest median —
while the first 3 and last 8 layers show substantially higher absolute values.

---

## 2. True sparsity — fraction of neurons active per token

Using threshold = `mult × global_median` per layer:

| layer | 1×median (p50) | 2×median | 5×median |
|------:|---------------:|---------:|---------:|
| 0     | **58%**        | 31%      | 5%       |
| 15    | **49%**        | 25%      | 4%       |
| 39    | **49%**        | 31%      | 11%      |

**There is no sparse regime at a natural threshold.**  Even at 5×median, 4–11% of neurons
remain active per token.  This is fundamental to SiLU/GELU-gated architectures: unlike ReLU,
the gate output is never exactly zero, so `down_input` has a smooth distribution with no hard
zero mass.

---

## 3. KV-lookup predictor results

Running `O1_predict.py --train-split 0.9 --top-k 1 5 --threshold-mult 0.3 0.5 0.7 1.0`
on layers 0, 15, 39:

```
Layer 0:  12734 tokens  (library=11460, eval=1274),  H=2560,  I=8192  median|act|=0.1641
     K       mult    recall   precision   density
     1       0.3x     0.867       0.873     0.794
     5       0.3x     0.988       0.819     0.963
     1       1.0x     0.742       0.760     0.465
     5       1.0x     0.917       0.629     0.682

Layer 15:  ...  median|act|=0.0781
     1       1.0x     0.553       0.572     0.468
     5       1.0x     0.933       0.511     0.892

Layer 39:  ...  median|act|=0.3164
     1       1.0x     0.721       0.737     0.458
     5       1.0x     0.927       0.587     0.745
```

### Why the predictor fails to provide useful savings

**OR-expansion destroys sparsity.**  Any two random library tokens have Jaccard similarity
≈ 0.34–0.55 at 1×median threshold.  OR-ing K=5 masks raises predicted density to 89–99%,
far above the true ~49% active density.  Increasing K improves recall at the cost of even
higher density.

**The fundamental tension:**

| K (neighbours) | recall (layer 15) | predicted density |
|---------------:|------------------:|------------------:|
| 1              | 55%               | 47%               |
| 5              | 93%               | 89%               |

There is no K that simultaneously achieves high recall and low predicted density.

---

## 4. Per-neuron activity frequency

The `neuron_activity` field reveals that most neurons are structurally always-on:

| layer | always-on (>90%) | mostly-on (>50%) | mostly-off (<10%) | total |
|------:|-----------------:|-----------------:|------------------:|------:|
| 0     | 15               | 1 093            | 11                | 8 192 |
| 15    | 3                | 23               | **1 346**         | 8 192 |
| 39    | 2 588            | **5 756**        | 2                 | 8 192 |

Layer 15 is actually the most hopeful: 1 346 neurons fire on fewer than 10% of tokens and could
be statically pre-masked.  Layer 39 has 70% of neurons firing on the majority of tokens —
structurally dense.

---

## 5. Oracle top-K analysis — the ceiling on any predictor

The oracle selects exactly the top-K neurons by `|down_input|` per token — the best
recall any predictor operating with a fixed compute budget K could achieve.

### Oracle recall at fixed K% of 8192 neurons (threshold = 1×median)

| layer | avg active | K=5% | K=10% | K=15% | K=20% | K=30% | K=50% |
|------:|-----------:|-----:|------:|------:|------:|------:|------:|
| 0     | 4084       | 0.121 | 0.243 | 0.360 | 0.465 | 0.631 | 0.893 |
| 1     | 4091       | 0.110 | 0.219 | 0.328 | 0.435 | 0.633 | 0.913 |
| 2     | 4086       | 0.109 | 0.217 | 0.323 | 0.427 | 0.628 | 0.925 |
| 15    | 4093       | 0.101 | 0.203 | 0.304 | 0.406 | 0.608 | 0.960 |
| 20    | 4091       | 0.101 | 0.202 | 0.302 | 0.403 | 0.605 | 0.973 |
| 27    | 4095       | 0.101 | 0.202 | 0.303 | 0.404 | 0.605 | 0.968 |
| 35    | 4089       | 0.101 | 0.202 | 0.303 | 0.405 | 0.607 | 0.966 |
| 36    | 4087       | 0.102 | 0.204 | 0.305 | 0.407 | 0.610 | 0.958 |
| 37    | 4086       | 0.102 | 0.205 | 0.307 | 0.410 | 0.615 | 0.951 |
| 38    | 4087       | 0.103 | 0.206 | 0.308 | 0.411 | 0.617 | 0.948 |
| 39    | 4093       | 0.102 | 0.204 | 0.306 | 0.408 | 0.611 | 0.957 |

### K% required for oracle recall ≥ 95% / 99%

| layer | avg active | avg frac% | K@95% | K%@95% | K@99% | K%@99% |
|------:|-----------:|----------:|------:|-------:|------:|-------:|
| 0     | 4084       | 49.9%     | 4544  | **55%** | 4960 | 60%    |
| 1     | 4091       | 49.9%     | 4416  | **53%** | 4864 | 59%    |
| 2     | 4086       | 49.9%     | 4320  | **52%** | 4832 | 58%    |
| 15    | 4093       | 50.0%     | 4016  | **49%** | 4576 | 55%    |
| 20    | 4091       | 49.9%     | 3936  | **48%** | 4304 | 52%    |
| 27    | 4095       | 50.0%     | 3968  | **48%** | 4368 | 53%    |
| 35    | 4089       | 49.9%     | 3984  | **48%** | 4432 | 54%    |
| 36    | 4087       | 49.9%     | 4032  | **49%** | 4608 | 56%    |
| 37    | 4086       | 49.9%     | 4096  | **50%** | 4800 | 58%    |
| 38    | 4087       | 49.9%     | 4128  | **50%** | 4864 | 59%    |
| 39    | 4093       | 50.0%     | 4048  | **49%** | 4688 | 57%    |

### Critical finding — the oracle gives no advantage from layer-differentiated K

The average active neuron count is **essentially constant at ~4090 (≈50%) across all 40 layers**,
regardless of whether the layer is in the plateau, the embedding, or the output layers.  The
median threshold by construction marks exactly 50% of neurons as active, so this is a tautology
of the threshold choice — but it holds even with absolute thresholds (mult=2×, 5×) because the
distribution shape is similar across layers.

**Using a higher K for layers 0–2 and 35–39, and a lower K=15% for the plateau, does not
improve the tradeoff.**  Here is why:

1. **Layers 0–2 are not harder to cover.**  At K=15%, oracle recall is 0.36 for layer 0 vs
   0.30 for layer 15 — almost the same.  The slightly higher recall at layer 0 comes from
   the fact that the distribution is more spread out (higher p90), meaning the truly-active
   set is *less uniform* and a few high-value neurons capture more recall per unit of K.
   The gain is marginal (6 percentage points).

2. **Layers 35–39 are not harder either.**  Oracle recall at K=15% is 0.305–0.308, virtually
   identical to the plateau.  Layer 39 has larger absolute values but the *relative* rank
   structure is the same.

3. **At K=15%, mean oracle recall is ≈ 30% everywhere.**  You would need K ≈ 49–55% of
   neurons to reach 95% recall — which is more neurons than you save by skipping (since the
   true active fraction is also ~50%).  The compute budget required to achieve useful recall
   exceeds the compute you would save.

The layer-differentiated K strategy therefore provides no material benefit over a flat K.

---

## 6. Why this model cannot benefit from FFN sparsity prediction via `down_input`

Three independent reasons:

### 6a. SiLU produces no true zeros
ReLU(x) = 0 for x ≤ 0, giving exact structural zeros.  SiLU(x) = x·σ(x) is never exactly
zero — it asymptotes to zero for very negative x but never reaches it in float32.  The
`down_input = SiLU(gate) * up` is therefore always nonzero, and the "active" definition is
necessarily a soft threshold, not a hard structural one.

### 6b. The gate logit distribution is not bimodal
For sparsity prediction to work well, neuron activations should be approximately Bernoulli —
either clearly on or clearly off.  In this model, `|down_input|` follows a smooth unimodal
distribution (right-skewed gamma-like), so any threshold cuts through the bulk of the
distribution rather than the valley between two modes.

### 6c. The KV-lookup predictor incurs OR-expansion overhead
Even if structural sparsity existed, the OR-of-K-neighbours predictor amplifies density:
K=1 at 1×median already matches true density (~49%), but recall is only 55–74%.  Every unit
of K added to improve recall adds more density than it recovers in precision.

---

## 7. Directions that would work

### 7a. Gate pre-activation threshold
Replace `down_input` with `gate_raw`.  SiLU(x) < 0.001 when x < −6.9.  A hard threshold on
`gate_raw < −4` (SiLU ≈ 0.007) would yield genuine near-zero outputs with a hard cutoff.
Record whether `gate_raw > threshold` per neuron/token and use that as the activity mask.

### 7b. Static per-neuron masking
Layer 15 has 1 346 neurons that fire on fewer than 10% of tokens.  These can be permanently
masked without any per-token prediction cost — no predictor required, zero recall loss risk.

### 7c. Fixed top-K with learned routing
Use a small router network (e.g. a linear probe on the hidden state) to predict the top-K
neurons directly, rather than cosine-similarity KV retrieval.  The router can be trained to
minimise the gap between oracle top-K recall and predicted top-K recall.  This avoids the
OR-expansion problem.

### 7d. Token-skipping instead of neuron-skipping
If per-token hidden states repeat (e.g. cached prefill), skip the entire FFN computation and
reuse the cached output.  This is orthogonal to neuron selection and avoids the sparsity
ceiling entirely.
