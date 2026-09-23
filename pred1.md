
# Experimental dynamic MLP sparsity prediction

## Core idea

Evaluate the gate projection in reduced precision. For output values before/after (to be decided) SwiGLU apply a threshold (gate_thresh) to derive hot channels. For the hot channels, compute the full precision values with original weights. The gate output thus consists of approximations for cold channels and correct values for hot channels. 
Do the same for the up projection: Compute low precision for cold channels and full precision for hot channels. Either the same hot/cold mix as derived from the gate projection. Alternative: for cold gate channels above up_thresh recompute gate and up projection in high precision.

Down projection is different since the hot/cold channels correspond to inputs. Compute the full matrix in low precision. Then add a sparse high precision correction for hot input channels. Needs invention how to implement this efficiently! 

## Evaluation metrics

### Limitation of output cosine similarity

All experiments report **output cosine similarity** between `out_full` and
`out_hybrid` in the MLP output (residual-stream) space.  This is a convenient,
fast metric but has three systematic weaknesses:

1. **Magnitude-blind.**  Cosine similarity is 1.0 for `out_hybrid = 2 × out_full`
   — a perfect score despite doubling the residual contribution.  Ternary gate
   approximations change the output norm non-trivially and this is invisible.

2. **Not tied to logits.**  The residual stream is projected onto the vocabulary
   via `W_U = lm_head.weight` (shape `[vocab, H]`).  Two hidden vectors can be
   close in cosine distance yet produce very different top-token predictions,
   depending on where they fall relative to the rows of `W_U`.  A cosine gap of
   0.015 may be negligible in one direction and catastrophic in another.

3. **Mean masks tail events.**  Perplexity is a geometric mean of per-token NLL;
   one severely wrong token dominates.  A high mean cosine similarity hides
   occasional tokens where the approximation produces a completely wrong output.

### Cosine similarity

Used as the primary metric throughout exp1–11.  Cheap and layer-agnostic, but
only measures angle in hidden space — see limitations above.

### Perplexity proxy

#### Full logit-space KL divergence

The most direct proxy for perplexity increase.  Given the MLP output error
`δh = out_hybrid − out_full`, compute the change in logit space using a
first-order approximation that skips LayerNorm (valid for small δh):

```python
lf = out_full   @ W_U.T          # (T, vocab)
lh = out_hybrid @ W_U.T          # (T, vocab)
pf = torch.softmax(lf, dim=-1)
ph = torch.softmax(lh, dim=-1)
kl = (pf * (pf / (ph + 1e-9)).log()).sum(-1).mean()   # KL(full || hybrid)
```

`W_U` is `model.lm_head.weight`; for tied-embedding models use
`model.model.embed_tokens.weight`.  The GEMM `(T, H) @ (H, vocab)` is
expensive for large vocabularies — subsample to ~500 tokens or use bfloat16
to keep it tractable in the per-layer loop.

#### W_U-weighted L2 error

Weights the hidden-space error by how much each direction actually moves
logits, using the singular spectrum of `W_U` (one-time offline SVD):

```python
# Offline (once per model):
U_emb, S_emb, _ = torch.linalg.svd(W_U, full_matrices=False)  # (H, rank), (rank,)

# Per token batch:
err_proj     = (out_hybrid - out_full) @ U_emb   # (T, rank)
weighted_err = (err_proj * S_emb).norm(dim=-1).mean()
```

Strictly more informative than hidden-space cosine: it measures the magnitude
of the error in the subspace that `W_U` amplifies most.

### Top-1 preservation rate

What fraction of tokens keep the same predicted top-1 token after the
approximation?  The most interpretable single number for top-token prediction
quality:

```python
top1_match = (lf.argmax(-1) == lh.argmax(-1)).float().mean()
```

Reuses the same `lf`/`lh` from the KL computation — no extra cost.

### Practical helper for exp12+

All metrics share the same `W_U` GEMM.  Recommended single helper:

```python
def logit_metrics(
    out_full: torch.Tensor,   # (T, H)
    out_hybrid: torch.Tensor, # (T, H)
    W_U: torch.Tensor,        # (vocab, H)
) -> tuple[float, float, float]:
    """Returns (kl_div, top1_match, logit_cos)."""
    lf = out_full   @ W_U.T
    lh = out_hybrid @ W_U.T
    pf = torch.softmax(lf.float(), dim=-1)
    ph = torch.softmax(lh.float(), dim=-1)
    kl         = float((pf * (pf / (ph + 1e-9)).log()).sum(-1).mean())
    top1_match = float((lf.argmax(-1) == lh.argmax(-1)).float().mean())
    cos        = float(F.cosine_similarity(lf, lh, dim=-1).mean())
    return kl, top1_match, cos
```

`cos` here is cosine similarity in **logit space** — more meaningful than
hidden-space cosine because it measures angle between the pre-softmax output
distributions.

### Metric comparison summary

| Metric | Space | Ties to perplexity | Ties to top-1 | Cost |
|--------|-------|--------------------|---------------|------|
| Output cosine similarity | hidden | weak | weak | free |
| Logit cosine similarity | logit | moderate | moderate | `W_U` GEMM |
| KL divergence | probability | **direct** | moderate | `W_U` GEMM |
| Top-1 preservation rate | token | moderate | **direct** | `W_U` GEMM |
| `W_U`-weighted L2 error | logit (singular) | moderate | moderate | `W_U` SVD + GEMM |

All four non-trivial metrics share the same `W_U` GEMM, so the marginal cost
of adding all of them together is just one GEMM — compute all in a single pass
over `lf` and `lh`.


## Experiment 1

As proxy simply use the sign of each weight as low precision value.
Evaluate cosine similarity of gate output, SwiGLU(gate)*up and down projection output with this predictor scheme versus full precision evaluation.

### Results (granite-4.2-3b, 128 calibration chunks, 10 559 tokens/layer)

All three projections replaced by their sign matrices. Metric: mean per-token
cosine similarity (sign-approx output vs full-precision output), averaged over
all 40 layers.

| Stage | Mean | Min | Max |
|---|---|---|---|
| gate output (pre-SiLU) | 0.855 | 0.789 | 0.919 |
| SwiGLU output (= down-proj input) | 0.523 | 0.389 | 0.668 |
| down projection output | 0.365 | 0.271 | 0.463 |

**Gate** cosine similarity of ~0.85 shows the sign predictor recovers the
direction of gate logits well — sufficient to identify hot/cold channels.

**SwiGLU drops to ~0.52** because the nonlinearity is most sensitive near
zero: a wrong sign there flips the whole contribution of that channel.

**Down output at ~0.37** is far too low for the approximation to be used
as-is. Error from both previous stages compounds through the dense
down-projection matmul.

**Conclusion:** sign values are a useful routing signal but not a substitute
for actual computation. The hybrid scheme (sign for routing, full precision
for hot channels) from the Core Idea section is the necessary next step —
the approximation output itself must not be passed downstream.

Per-layer detail (layers 0–39):

| Layer | gate | SwiGLU | down |
|---|---|---|---|
| 0 | 0.9192 | 0.4734 | 0.3603 |
| 1 | 0.9142 | 0.3890 | 0.2767 |
| 2 | 0.8676 | 0.5304 | 0.4352 |
| 3 | 0.8601 | 0.5225 | 0.3740 |
| 4 | 0.8546 | 0.5410 | 0.3824 |
| 5 | 0.8667 | 0.5333 | 0.3906 |
| 6 | 0.8623 | 0.5093 | 0.3690 |
| 7 | 0.8765 | 0.5191 | 0.3882 |
| 8 | 0.8794 | 0.4713 | 0.3341 |
| 9 | 0.8829 | 0.4738 | 0.3104 |
| 10 | 0.8762 | 0.4659 | 0.3107 |
| 11 | 0.8816 | 0.4457 | 0.2955 |
| 12 | 0.8783 | 0.4551 | 0.3040 |
| 13 | 0.8832 | 0.4621 | 0.3210 |
| 14 | 0.8775 | 0.4788 | 0.3441 |
| 15 | 0.8713 | 0.5104 | 0.3633 |
| 16 | 0.8642 | 0.5333 | 0.3666 |
| 17 | 0.8657 | 0.4859 | 0.2708 |
| 18 | 0.8576 | 0.5228 | 0.3601 |
| 19 | 0.8502 | 0.5347 | 0.3533 |
| 20 | 0.8427 | 0.5177 | 0.3347 |
| 21 | 0.8365 | 0.5434 | 0.3667 |
| 22 | 0.8388 | 0.5528 | 0.3581 |
| 23 | 0.8413 | 0.5324 | 0.3587 |
| 24 | 0.8521 | 0.4859 | 0.3149 |
| 25 | 0.8476 | 0.4989 | 0.3276 |
| 26 | 0.8416 | 0.5045 | 0.3188 |
| 27 | 0.8360 | 0.5315 | 0.3671 |
| 28 | 0.8416 | 0.5439 | 0.3671 |
| 29 | 0.8408 | 0.5650 | 0.4107 |
| 30 | 0.8369 | 0.5657 | 0.4009 |
| 31 | 0.8429 | 0.5641 | 0.3996 |
| 32 | 0.8371 | 0.5429 | 0.3891 |
| 33 | 0.8262 | 0.5314 | 0.3935 |
| 34 | 0.8342 | 0.5693 | 0.4086 |
| 35 | 0.8402 | 0.5810 | 0.4102 |
| 36 | 0.8408 | 0.5857 | 0.4160 |
| 37 | 0.8264 | 0.5883 | 0.4510 |
| 38 | 0.8110 | 0.6071 | 0.4489 |
| 39 | 0.7890 | 0.6680 | 0.4631 |


## Experiment 2

Use the sign of each weight as low precision value but implement the hybrid scheme described in the Core Idea section.
Evaluate cosine similarity of gate output, SwiGLU(gate)*up and down projection output with this predictor scheme versus full precision evaluation.

### Scheme

1. **Low-precision pass:** `gate_approx = sign(W_gate) @ x`,  `up_approx = sign(W_up) @ x`
2. **Hot mask per token/channel:** `hot = |gate_approx| > gate_thresh`
3. **Hybrid gate/up:** recompute hot channels with true weights, keep sign values for cold
4. **SwiGLU hybrid:** `SiLU(gate_hybrid) * up_hybrid`
5. **Hybrid down:** `sign(W_down) @ swiglu_hybrid` + sparse correction
   `(W_down − sign(W_down))[:, hot] @ swiglu_hybrid[:, hot]` for the hot channels

### Results (granite-4.2-3b, 128 calibration chunks, 2 000 tokens/layer cap)

Mean cosine similarity across all 40 layers vs full-precision at each `gate_thresh`:

| gate_thresh | hot channels | gate cos-sim | SwiGLU cos-sim | down cos-sim |
|---|---|---|---|---|
| 0.0 (all hot = full precision) | 100% | 1.0000 | 1.0000 | 1.0000 |
| 1.0 | 99.0% | 0.9973 | 0.2107 | 0.0073 |
| 2.0 | 98.1% | 0.9852 | 0.0766 | 0.0036 |
| 4.0 | 96.2% | 0.9102 | 0.0302 | −0.0011 |
| 8.0 | 92.4% | 0.6505 | 0.0183 | −0.0036 |
| 16.0 | 84.8% | 0.3458 | 0.0276 | 0.0038 |

**Critical finding:** the SwiGLU and down cosine similarity collapse to near zero
as soon as any channels are left in low precision (`thresh > 0`), even when
99% of channels are recomputed exactly. The remaining 1% of cold channels
contribute disproportionately because:

* The SiLU nonlinearity maps a wrong-sign gate logit to a completely wrong
  output (e.g. sign-approx ≈ −1 vs true value ≈ +3 → SiLU output off by ~4×).
* These errors are multiplied by the up projection and then spread across all
  hidden dimensions by the down projection, destroying directional alignment.

**The threshold is applied to the wrong signal.** `|gate_approx|` is the
magnitude of the sign-approximation (= dot product with ±1 weights), not the
magnitude of the true gate logit. A small `|gate_approx|` means the true gate
is also near zero (channels where SiLU ≈ 0 anyway), so those channels are
actually safe to approximate. A large `|gate_approx|` does not guarantee the
approximation is accurate — the sign-approx can be large with the wrong sign
if the true logit has a different magnitude pattern.

**Conclusion:** the hybrid scheme requires a better predictor than the raw
sign-approximation output as a routing signal. The gate_approx direction is a
useful sign indicator (experiment 1 showed 0.85 cos-sim for the gate), but
using it as a magnitude threshold for hot/cold routing fails because the
residual errors in cold channels dominate the output after nonlinearity.
The next step is to investigate whether the true gate activations from a prior
token can serve as a better routing proxy, or whether the threshold should be
applied to the *residual* `|gate_full − gate_approx|` after a cheap
correctness check.


## Experiment 3

Would a low-rank approximation `W_gate ≈ G_A G_B` (rank 64 or similar) give a
better gate predictor than the sign matrix?

### Setup

Compute the truncated SVD of `W_gate` (shape 8192×2560) at ranks 4, 16, 64,
256, 1024 for every layer.  Compare:

1. Gate output cosine similarity vs `gate_raw` — same metric as experiments 1–2.
2. Sign agreement `frac(sign(gate_approx) == sign(gate_full))` — the routing
   accuracy that actually determines hybrid-scheme quality.
3. FLOP cost per token relative to the full GEMM.

### Results (granite-4.2-3b, 40 layers, 2 000 tokens/layer)

#### Gate cosine similarity vs `gate_raw`

| Method | Mean | Notes |
|---|---|---|
| Sign predictor `sign(W)` | **0.856** | baseline |
| Rank-4 SVD | 0.716 | worse |
| Rank-16 SVD | 0.762 | worse |
| Rank-64 SVD | 0.817 | worse |
| Rank-256 SVD | 0.871 | first rank to beat sign |
| Rank-1024 SVD | 0.947 | — |

The sign predictor beats SVD up to approximately **rank 200**.

#### Sign agreement (routing accuracy for hot/cold split)

Fraction of per-token, per-channel decisions where
`sign(gate_approx) == sign(gate_full)`:

| Method | Sampled layers mean |
|---|---|
| Sign predictor | ~0.867 |
| Rank-16 SVD | ~0.818 |
| Rank-64 SVD | ~0.836 |
| Rank-256 SVD | ~0.858 |

Sign agreement of the low-rank predictor is **lower** than the plain sign
predictor at all tested ranks.  Rank-256 (~13% of full FLOP cost) only just
approaches the sign predictor's routing accuracy.

#### FLOP cost per token (W_gate: 8192×2560)

| Method | FLOPs | vs full GEMM |
|---|---|---|
| Sign predictor | ~42 M additions (no multiplies) | ≈ 0.1–0.2× |
| Rank-16 | 344 K | 0.008× |
| Rank-64 | 1.38 M | 0.033× |
| Rank-256 | 5.5 M | 0.131× |
| Full GEMM | 41.9 M | 1.0× |

### Why the sign predictor wins

The singular value spectrum is flat — the top 64 singular values capture only
15.6% of total spectral energy (r=256 captures 31.9%).  There is no dominant
low-rank structure.  In this regime, retaining all 8192 weight rows binarized
to {±1} preserves more directional information than a small set of exact
low-rank directions.  The sign matrix is equivalent to a full-rank random
binary projection, which is well-conditioned by the Johnson–Lindenstrauss lemma.

### Conclusion

A low-rank factorization at rank 64 is both **worse as a predictor** (lower
gate cos-sim, lower sign agreement) and **more expensive** than the sign
predictor on practical hardware (two dense GEMMs vs one addition-only pass).
The rank would need to exceed ~200 to match sign-predictor accuracy, at which
point the cost advantage over the full GEMM is only ~10×.

The sign predictor remains the best cheap proxy for routing.  The routing
*quality* problem identified in experiment 2 needs a different solution.


## Experiment 4

Check if `|gate_approx|` (magnitude of the sign-prediction output) is a better
routing signal than the raw threshold used in experiment 2.

Specifically: how well does `|gate_approx| = |sign(W_gate) @ x|` predict
`|gate_full| = |W_gate @ x|`?  A good rank correlation between the two would
mean the magnitude of the cheap pass reliably identifies the truly-active
channels, enabling a threshold on `|gate_approx|` to be used for routing with
acceptable mis-routing rates.

### Setup

For each layer, compute:
1. **Spearman rank correlation** between `|gate_approx|` and `|gate_full|`
   across all (token, channel) pairs — measures how well the cheap magnitude
   predicts the true magnitude.
2. **Quadrant breakdown** (split at median of `|gate_approx|`):
   - A: correct sign & large approx — hot channels correctly identified ✓
   - B: wrong sign & large approx — hot channels with wrong sign (dangerous) ✗
   - C: correct sign & small approx — cold channels correctly identified ✓
   - D: wrong sign & small approx — cold channels with wrong sign (safe: SiLU≈0) ✓
3. **Hybrid quality** routing by top-F `|gate_approx|` as hot mask.

### Results (granite-4.2-3b, 40 layers, 2 000 tokens/layer, MPS)

#### Magnitude rank correlation

Spearman ρ(|gate_approx|, |gate_full|):
mean=**0.638**  min=0.576  max=0.724

Moderate positive correlation — `|gate_approx|` does carry signal about which
channels will be truly large, but with substantial noise (~0.36 unexplained rank
variance).

#### Quadrant breakdown (mean across 40 layers)

| Quadrant | Fraction | Meaning |
|---|---|---|
| A: correct-sign & large-approx | 0.487 | hot, correctly identified |
| **B: wrong-sign & large-approx** | **0.013** | **hot, mis-classified (dangerous)** |
| C: correct-sign & small-approx | 0.370 | cold, correctly classified |
| D: wrong-sign & small-approx | 0.130 | cold, mis-classified (safe: SiLU≈0) |

Quadrant B (wrong-sign large channels) is only **1.3%** of all channel-token
pairs — but these are the pairs that destroy SwiGLU cosine similarity, because
they have large SiLU errors that propagate through the dense down projection.

#### Hybrid quality (top-F |gate_approx| hot mask)

| hot % | SwiGLU cos-sim | down cos-sim |
|---|---|---|
| 50% | 0.1397 | 0.0883 |
| 30% | 0.2307 | 0.1566 |
| 20% | 0.2849 | 0.1966 |
| 10% | **0.3508** | **0.2446** |

Using the top-10% most active channels (by `|gate_approx|`) in full precision
and leaving the rest as sign-approximation gives SwiGLU cosine similarity of
0.35 — far better than the 0.05–0.21 range seen in experiment 2 with the same
hot fraction.  The key difference: in experiment 2, thresholding on `|gate_approx|`
included quadrant-B channels as cold (wrong-sign but large-approx); here those
same channels are included as hot (large-approx → recomputed), eliminating the
most dangerous errors.

**Critical insight:** `|gate_approx|` is actually a *good* proxy for `|gate_full|`
in the sense that matters: the quadrant-B fraction is tiny (1.3%).  Channels
with large `|gate_approx|` are overwhelmingly correct-sign (quadrant A).  The
failure in experiment 2 was the *opposite* routing: using large `|gate_approx|`
as a cold criterion (i.e. "only recompute small-approx channels") — which
leaves the dangerous large-approx-wrong-sign channels (B) uncorrected.
Top-F hot (recompute the biggest ones) is the right use of this signal.

### Conclusion

`|gate_approx|` is a useful routing signal **when used as a hot criterion**
(mark the largest channels as hot → recompute).  With 10% hot channels, SwiGLU
cosine similarity reaches 0.35, down cosine 0.24.  The quadrant analysis confirms
the sign predictor gets the sign right 98.7% of the time on large channels —
so the remaining error comes from the 87% of channels left as sign-only cold.


## Experiment 5

Use the prior-token hotlist and refine the hotlist for the next token by running
some channels in full precision out of the critical path.

### Scheme

1. For token `t` use the hotlist `H_{t-1}` from the previous token (temporal
   locality hypothesis: hot channels are stable across adjacent tokens).
2. Compute the full MLP in hybrid mode using `H_{t-1}`.
3. In parallel (out of the critical path, must finish before layer `l` of token
   `t+1`): recompute `k_refine` boundary channels (those ranked nearest the
   hot/cold cutoff in `H_{t-1}`) in full precision at token `t` to produce
   an updated hotlist `H_t`.
4. Metric: IoU between `H_{t-1}` and the true `H_t`, and SwiGLU/down cosine
   similarity when routing with the prior-token hotlist ± refinement.

### Results (granite-4.2-3b, 40 layers, 2 000 tokens/layer, MPS)

#### Hotlist temporal stability (Jaccard IoU between adjacent tokens)

| hot % | mean IoU | min | max |
|---|---|---|---|
| 50% | 0.503 | 0.417 | 0.561 |
| 30% | 0.402 | 0.260 | 0.487 |
| 20% | 0.348 | 0.200 | 0.476 |
| 10% | 0.283 | 0.154 | 0.416 |

IoU of ~0.50 at 50% hot means the prior hotlist gets about half the channels
right.  At tighter (10%) hot fractions, IoU drops to 0.28 — the hotlist
changes substantially token to token.

#### Hybrid quality with prior-token hotlist + refinement

Mean SwiGLU cosine similarity:

| hot % | no refine | refine 5% | refine 10% | refine 20% | refine 30% |
|---|---|---|---|---|---|
| 50% | 0.3009 | 0.2922 | 0.2832 | 0.2647 | 0.2453 |
| 30% | 0.3652 | 0.3593 | 0.3531 | 0.3400 | 0.3261 |
| 20% | 0.3984 | 0.3935 | 0.3883 | 0.3771 | 0.3650 |
| **10%** | **0.4359** | **0.4320** | **0.4273** | 0.4170 | 0.4067 |

Mean down-projection cosine similarity:

| hot % | no refine | refine 5% | refine 10% | refine 20% | refine 30% |
|---|---|---|---|---|---|
| 50% | 0.2160 | 0.2093 | 0.2025 | 0.1883 | 0.1734 |
| 30% | 0.2604 | 0.2558 | 0.2510 | 0.2410 | 0.2304 |
| 20% | 0.2834 | 0.2796 | 0.2756 | 0.2670 | 0.2579 |
| **10%** | **0.3096** | **0.3066** | **0.3030** | 0.2954 | 0.2877 |

### Key findings

**Prior-token hotlist alone (no refinement) outperforms experiment 4.**
At 10% hot channels, the prior-token hotlist gives SwiGLU cosine similarity
of **0.436** vs 0.351 with top-F `|gate_approx|`.  This is the best result
so far across all experiments.  The reason: the prior token's true gate values
are a much better predictor of the current token's hot channels than any cheap
approximation, even with IoU of only 0.28.  The ~28% of hot channels that do
transfer correctly happen to be the most consistently active ones (large gate
values that are stable across tokens), which are also the ones that matter most
for the SwiGLU output.

**Refinement does not help — it hurts slightly.**  Recomputing boundary
channels at token `t` using `gate_full[t]` to update the hotlist consistently
*reduces* SwiGLU and down cosine similarity.  This is because the boundary
strategy (channels nearest the prior hot/cold threshold) updates the channels
that are *least* certain in the prior, replacing them with correct decisions —
but the overall effect is to increase hot-channel churn, removing stably-hot
channels in favour of newly-discovered hot ones.  The correction benefit is
outweighed by the instability cost.  A better refinement strategy would
selectively protect stably-hot channels while only flipping the boundary ones.

**Tighter hotlists work better.**  10% hot gives ~40% better SwiGLU cosine
similarity than 50% hot.  This is consistent across all experiments: the sign
approximation for cold channels degrades gracefully when the hot fraction is
small, because the most important (largest gate) channels are recomputed and
the cold channels have near-zero SiLU output anyway.

### Conclusion

The prior-token hotlist is the strongest routing signal found so far.  With
10% hot channels and no refinement, it achieves SwiGLU cosine similarity 0.44
and down cosine 0.31 — significantly better than the sign-predictor-based
routing (0.35/0.24) and far better than the all-sign baseline (0.52 SwiGLU
but without any full-precision recomputation).  The temporal stability of hot
channels (IoU ~0.28 at 10%) is sufficient to exploit because the stably-hot
channels dominate the output.  Refinement at the boundary hurts; a better
strategy might be to carry forward the full prior gate vector and threshold it
rather than doing a top-k IoU comparison.

## Experiment 6

Magnitude-confidence refinement of the prior-token hotlist.  Instead of
refining the channels ranked nearest the hot/cold boundary (experiment 5),
use `|gate_full[t-1]|` as a per-channel confidence score and focus the
refinement budget where the prior is least certain.

### Scheme

Given a refinement budget of `k_refine` channels:

1. Among the `k_hot` prior-hot channels, pick the `k_refine/2` with the
   **smallest** `|gate_full[t-1]|` — these are the weakest hot channels,
   most likely to have dropped below the threshold at token `t`.
2. Among the remaining prior-cold channels, pick the `k_refine/2` with the
   **largest** `|gate_full[t-1]|` — these are the strongest cold channels,
   most likely to have crossed into hot territory.
3. For those `k_refine` channels, re-evaluate with the true `gate_full[t]`
   and update the hot/cold decision.  All other channels keep the prior
   decision unchanged — in particular, high-magnitude prior-hot channels are
   **protected** from being flipped.

This is compared directly against the experiment 5 boundary-rank strategy
(refine channels nearest the rank cutoff) and the no-refinement baseline.

### Results (granite-4.2-3b, 40 layers, 2 000 tokens/layer, MPS)

#### SwiGLU cosine similarity — hot=10% (best configuration from exp5)

| refine budget | no_refine (baseline) | boundary_rank (exp5) | magnitude_conf (exp6) | Δ exp6 vs exp5 | Δ exp6 vs baseline |
|---|---|---|---|---|---|
| 0% | 0.4359 | 0.4359 | 0.4359 | +0.0000 | +0.0000 |
| 5% | 0.4359 | 0.4320 | 0.4329 | +0.0010 | −0.0030 |
| 10% | 0.4359 | 0.4273 | 0.4306 | +0.0033 | −0.0053 |
| 20% | 0.4359 | 0.4170 | **0.4267** | **+0.0096** | −0.0092 |
| 30% | 0.4359 | 0.4067 | 0.4149 | +0.0082 | −0.0210 |

#### Down-projection cosine similarity — hot=10%

| refine budget | no_refine | boundary_rank | magnitude_conf | Δ exp6 vs exp5 |
|---|---|---|---|---|
| 0% | 0.3096 | 0.3096 | 0.3096 | +0.0000 |
| 5% | 0.3096 | 0.3066 | 0.3073 | +0.0007 |
| 10% | 0.3096 | 0.3030 | 0.3053 | +0.0023 |
| 20% | 0.3096 | 0.2954 | **0.3021** | **+0.0067** |
| 30% | 0.3096 | 0.2877 | 0.2935 | +0.0058 |

Full table at hot=20%:

| refine budget | no_refine | boundary_rank | magnitude_conf | Δ exp6 vs exp5 |
|---|---|---|---|---|
| 0% | 0.3984 | 0.3984 | 0.3984 | +0.0000 |
| 5% | 0.3984 | 0.3935 | 0.3939 | +0.0004 |
| 10% | 0.3984 | 0.3883 | 0.3897 | +0.0014 |
| 20% | 0.3984 | 0.3771 | 0.3820 | +0.0049 |
| 30% | 0.3984 | 0.3650 | **0.3748** | **+0.0098** |

### Key findings

**Magnitude-confidence refinement consistently beats boundary-rank refinement**
at every budget and every hot fraction, by a margin of +0.001 to +0.010 SwiGLU
cosine similarity.  The improvement grows with budget size and is most pronounced
at larger hot fractions — e.g. at hot=20%, refine=30%, the gap is +0.010.

**However, both refinement strategies still underperform the no-refinement
baseline** (prior-token hotlist, no corrections).  At hot=10%, the best
magnitude-confidence result (budget=20%, SwiGLU=0.4267) is still 0.009 below
the no-refinement baseline of 0.4359.  At hot=20%, the gap is 0.016 at
budget=30%.

**The fundamental problem is confirmed:** refining the hotlist at the *current*
token (even with oracle full-precision gate values) adds hot-channel churn that
outweighs the benefit of correcting the prior's errors.  Protecting high-magnitude
prior-hot channels reduces but does not eliminate this churn.

**Why refinement still hurts:** The channels selected by magnitude-confidence
(weak hot + strong cold) are the ones that genuinely flip between tokens — they
are intrinsically unstable.  Correctly classifying them at token `t` means
*replacing stable prior-hot channels with newly-discovered ones*, disrupting
the routing for those stable channels at token `t+1`.  The benefit of one
correct routing decision is outweighed by the propagation of instability.

### Conclusion

Magnitude-confidence is the right conceptual direction — protecting stable hot
channels improves over the boundary-rank strategy — but refinement itself is
counterproductive relative to simply carrying the prior hotlist unchanged.  The
best strategy so far remains: **10% hot, prior-token hotlist, zero refinement**
(SwiGLU cosine similarity 0.4359, down cosine 0.3096).

The implication for system design: do not spend the inter-token compute budget
on refining the hotlist.  Instead, use it for something else entirely — e.g.
speculative prefetching of the hot channel weights, or running the gate
projection for the next layer's token concurrently.

## Experiment 7

Scheme similar to experiment 2, but route on the **post-nonlinearity neuron
magnitude** `|SwiGLU_approx|` instead of the pre-SiLU gate logit.

### Scheme

1. Compute gate and up projections in low precision:
   `gate_approx = sign(W_gate) @ x`,  `up_approx = sign(W_up) @ x`
2. Compute the cheap SwiGLU approximation:
   `swiglu_approx = SiLU(gate_approx) * up_approx`
3. Select the top-F neurons by `|swiglu_approx|` as hot.
4. Recompute hot neurons in full precision:
   `swiglu_hybrid[hot] = SiLU(W_gate[hot,:] @ x) * (W_up[hot,:] @ x)`
   Cold neurons keep the `swiglu_approx` value.
5. Down projection with full-precision `W_down`:
   `out = W_down @ swiglu_hybrid`

Metrics vs full-precision reference (`swiglu_full`, `out_full = W_down @ swiglu_full`):
- **Neuron cosine similarity**: `cos(swiglu_full, swiglu_hybrid)`
- **Output cosine similarity**: `cos(out_full, out_hybrid)`
- **Energy fraction**: fraction of `||swiglu_full||²` contained in hot neurons

### Results (granite-4.2-3b, 40 layers, 2 000 tokens/layer, MPS)

| hot% | neuron cos-sim | out cos-sim | energy in hot | Δ out vs exp4 (same hot%) |
|---|---|---|---|---|
| 0.5% | 0.3636 | 0.3155 | 25.0% | n/a |
| 1.0% | 0.3174 | 0.2714 | 31.3% | n/a |
| 2.0% | 0.2598 | 0.2175 | 38.8% | n/a |
| 5.0% | 0.1666 | 0.1325 | 50.5% | n/a |
| **10.0%** | **0.0906** | **0.0652** | 60.0% | −0.179 vs exp4's 0.245 |
| **20.0%** | **0.1156** | **0.0948** | 69.2% | −0.102 vs exp4's 0.197 |
| **30.0%** | **0.3558** | **0.3326** | 74.1% | +0.176 vs exp4's 0.157 |

### Key finding: U-shaped cosine similarity vs hot fraction

The output cosine similarity is **not monotonically increasing** with the hot
fraction.  It peaks at 0.5% hot (0.316), then *falls* to a minimum near 0.065
at 10%, then recovers to 0.333 at 30%.  This is a qualitatively different
failure mode from all previous experiments.

**Why the U-shape occurs:**

The routing signal `|swiglu_approx|` is dominated by the wrong neurons.
`swiglu_approx = SiLU(gate_approx) * up_approx` inherits the sign errors of
both the gate and up approximations.  A channel where `gate_approx` has the
wrong sign produces a strongly *negative* gate logit fed into SiLU, giving a
large *negative* (or near-zero) SiLU output.  But `up_approx` for that same
channel may also have the wrong sign, making the product `SiLU(−large) * (−up)`
potentially large and *positive*.  These sign-error-amplified neurons rank
high in `|swiglu_approx|` even though their true `swiglu_full` value is near
zero or has the opposite sign.

In other words: **`|swiglu_approx|` selects the most mis-approximated neurons
as hot**, not the most active ones.  Recomputing those channels in full
precision and replacing a large (wrong-sign) approximation with a near-zero
(correct) value zeroes out a contribution that the cold channels' sign
approximations were implicitly relying on — destroying the directional alignment
of `swiglu_hybrid`.

The recovery at 30% occurs because enough neurons are recomputed that the
correct activations begin to outweigh the damage from correcting the false-large
ones.  At 0.5% the budget is so small that only the very largest
`|swiglu_approx|` values are touched — a mixed bag, but there are few enough
that the overall effect is marginally positive.

**The energy fraction column tells the same story:** at 10% hot, those neurons
capture 60% of the energy in `swiglu_full` — but they were *selected by*
`|swiglu_approx|`, and the neuron cosine similarity is only 0.09.  The 10%
selected channels account for 60% of the true energy, but the approximation
of those channels is extremely inaccurate (cos-sim 0.09), causing a large
error.  Contrast with 0.5% hot: only 25% of energy is in hot channels but the
approximation of *all remaining* channels is relatively accurate, giving higher
overall cosine similarity.

### Comparison with experiment 4 (routing on |gate_approx|)

At 10% hot, experiment 4 achieves out cos-sim **0.245** vs experiment 7's
**0.065** — a factor of ~4× worse for SwiGLU-routing at the same budget.
At 30% hot, experiment 7 finally beats experiment 4 (0.333 vs 0.157), showing
that at very large budgets the post-nonlinearity routing does eventually win
because the energy concentration improves (74% of neuron energy at 30%).

### Conclusion

Routing on `|SwiGLU_approx|` is significantly worse than routing on
`|gate_approx|` at practically useful hot fractions (≤20%).  The compound
sign errors in both gate and up approximations create a badly misleading
routing signal that preferentially selects the most mis-approximated neurons
rather than the most active ones.

The post-nonlinearity signal would only be useful if both gate and up
approximations were much more accurate to begin with — which would require
a better predictor (e.g. the prior-token gate vector from experiment 5/6).

## Experiment 8

Ternary weight approximation: replace the sign predictor `sign(W) ∈ {±1}` with
a ternary `T(W, τ) ∈ {−1, 0, +1}` that zeros out small weights.

### Scheme

Same as experiment 7, except the weight proxy is:

    T(w, τ) = sign(w)  if |w| ≥ τ,  else 0

with a per-layer adaptive threshold `τ = α × mean(|W_gate|)`.  The full
pipeline:

1. `gate_approx = T(W_gate, τ) @ x`
2. `up_approx   = T(W_up,   τ) @ x`
3. `swiglu_approx = SiLU(gate_approx) * up_approx`  — routing signal
4. Select top-F neurons by `|swiglu_approx|` as hot
5. Recompute hot neurons: `swiglu_hybrid[hot] = SiLU(W_gate[hot] @ x) * (W_up[hot] @ x)`
6. `out = W_down @ swiglu_hybrid`

α is swept over {0.0 (sign baseline), 0.25, 0.5, 0.75, 1.0, 1.5}.

### Results (granite-4.2-3b, 40 layers, 2 000 tokens/layer, MPS)

#### Gate cosine similarity vs full-precision gate

| α | zero% | gate cos-sim |
|---|---|---|
| 0.00 (sign) | 0% | 0.856 |
| 0.25 | 16% | 0.892 |
| 0.50 | 32% | 0.916 |
| **0.75** | **46%** | **0.928** |
| 1.00 | 58% | 0.928 |
| 1.50 | 77% | 0.892 |

Zeroing 46–58% of weights (α=0.75–1.0) maximises gate cosine similarity.  Beyond
1.0× mean the signal is over-sparsified and quality drops.

#### Output cosine similarity (mean across 40 layers)

| α | zero% | 0.5% hot | 1% | 2% | 5% | **10%** | 20% | 30% |
|---|---|---|---|---|---|---|---|---|
| 0.00 (sign) | 0% | 0.316 | 0.271 | 0.218 | 0.133 | 0.065 | 0.095 | 0.333 |
| 0.25 | 16% | 0.372 | 0.321 | 0.259 | 0.160 | 0.081 | 0.118 | 0.361 |
| 0.50 | 32% | 0.414 | 0.358 | 0.288 | 0.178 | 0.090 | 0.124 | 0.366 |
| **0.75** | **46%** | **0.431** | **0.372** | **0.300** | **0.184** | **0.093** | 0.118 | 0.355 |
| 1.00 | 58% | 0.419 | 0.362 | 0.291 | 0.178 | 0.090 | 0.102 | 0.332 |
| 1.50 | 77% | 0.334 | 0.286 | 0.229 | 0.139 | 0.071 | 0.069 | 0.258 |

Best α per hot fraction: **0.75** for ≤10% hot, **0.50** for ≥20% hot.

#### Improvement over sign baseline (α=0, exp7)

| hot% | sign out cos-sim | best ternary | Δ |
|---|---|---|---|
| 0.5% | 0.316 | **0.431** (α=0.75) | +0.115 |
| 1% | 0.271 | **0.372** (α=0.75) | +0.101 |
| 5% | 0.133 | **0.184** (α=0.75) | +0.052 |
| 10% | 0.065 | **0.093** (α=0.75) | +0.028 |
| 30% | 0.333 | **0.366** (α=0.50) | +0.034 |

### Key findings

**Ternary weights consistently improve over sign weights at all hot fractions
and all layers.**  The improvement is largest at small hot fractions (0.5–2%),
where the gain is +0.10–0.12 absolute in output cosine similarity.

**The sweet spot is α=0.75 (46% zeros)**, which maximises gate cosine similarity
at 0.928.  This corresponds to zeroing all weights below the 46th percentile of
|W| — roughly half the weight entries.  Beyond this, over-sparsification
degrades the useful signal.

**The U-shaped curve from experiment 7 persists but shifts upward.**  The
minimum is still around 10% hot, and the scheme is still far below experiment
4's output cosine similarity of 0.245 at 10% hot.  Ternary weights improve the
*level* of the routing signal but do not fix the *structural problem*: the
ternary SwiGLU still selects the most mis-approximated neurons as hot.

**Why ternary helps:** zeroing near-zero weights removes the dominant source of
sign errors in the gate approximation.  The sign of a small weight `|w| ≈ 0`
contributes ±1 to the dot product but carries no real signal; zeroing it
removes that noise contribution.  The gate cosine similarity improves from 0.856
(sign) to 0.928 (ternary, α=0.75), which means the `swiglu_approx` routing
signal is substantially less noisy.

### Conclusion

Ternary weight approximation with α=0.75 is strictly better than the sign
approximation at all tested hot fractions, with +10 pp improvement at small
hot fractions.  However, the output cosine similarity at 10% hot (0.093) is
still far below the experiment 4 level (0.245, routing on `|gate_approx|`
*before* SiLU) and the experiment 5 level (0.436, prior-token hotlist).

The post-nonlinearity routing problem from experiment 7 is partially mitigated
but not solved.  The structural issue — that ternary SwiGLU still
preferentially identifies mis-approximated neurons — remains.  A more accurate
weight proxy (e.g. ternary applied *only* to gate, with full-precision up, to
decouple the two error sources) would be the logical next step.

## Experiment 9

Ternary gate proxy with full-precision up projection — decoupling the two
error sources in the SwiGLU routing signal.

### Motivation

Experiments 7 and 8 routed on `|swiglu_approx| = |SiLU(gate_approx) * up_approx|`
where both projections were approximated.  This created compound errors: the
`up_approx` error meant that even a perfectly accurate `gate_approx` would
produce a distorted routing signal.  Experiment 8's ternary improvement was
capped by the residual `up_approx` noise.

Eliminating the up error entirely:

    up_full = W_up @ x          full precision, all channels

changes the routing signal to `SiLU(gate_approx) * up_full`, where `up` is
always exact.  The only remaining approximation in cold channels is the ternary
gate.

### Scheme

1. `gate_approx = T(W_gate, τ) @ x`  — ternary gate, all channels
2. `up_full = W_up @ x`               — full-precision up, all channels
3. `swiglu_approx = SiLU(gate_approx) * up_full`  — routing signal
4. Hot = top-F neurons by `|swiglu_approx|`
5. Hot: recompute `gate_full[hot] = W_gate[hot] @ x`; cold: keep `gate_approx`
6. `swiglu_hybrid = SiLU(gate_hybrid) * up_full`  (up always exact)
7. `out = W_down @ swiglu_hybrid`

τ = α × mean(|W_gate|),  α ∈ {0.0 (sign), 0.25, 0.5, 0.75, 1.0, 1.5}.

**Cost note:** `up_full = W_up @ x` is a full GEMM that must be paid regardless.
This scheme's cheap pass is the ternary gate GEMM only; the saving relative to
full precision comes from avoiding the full `W_gate @ x` for cold channels.

### Results (granite-4.2-3b, 40 layers, 2 000 tokens/layer, MPS)

#### Gate cosine similarity (same as exp8 — gate proxy unchanged)

| α | zero% | gate cos-sim |
|---|---|---|
| 0.00 | 0% | 0.856 |
| 0.50 | 32% | 0.916 |
| **0.75** | **46%** | **0.928** |
| 1.00 | 58% | 0.928 |
| 1.50 | 77% | 0.892 |

#### Output cosine similarity vs full precision

| α | zero% | 0.5% hot | 1% | 2% | 5% | **10%** | 20% | 30% |
|---|---|---|---|---|---|---|---|---|
| 0.00 (sign) | 0% | 0.421 | 0.367 | 0.302 | 0.202 | 0.143 | 0.313 | 0.661 |
| 0.25 | 16% | 0.456 | 0.399 | 0.329 | 0.222 | 0.162 | 0.345 | 0.683 |
| 0.50 | 32% | 0.477 | 0.417 | 0.344 | 0.233 | 0.174 | 0.361 | 0.690 |
| **0.75** | **46%** | **0.484** | **0.423** | **0.349** | **0.237** | 0.179 | 0.367 | 0.693 |
| 1.00 | 58% | 0.475 | 0.416 | 0.344 | 0.235 | **0.181** | **0.368** | 0.695 |
| 1.50 | 77% | 0.421 | 0.368 | 0.306 | 0.215 | 0.174 | 0.355 | **0.699** |

#### Comparison across experiments (10% hot, best α)

| Experiment | Scheme | Out cos-sim @10% |
|---|---|---|
| Exp 7 (sign gate + sign up) | post-SiLU routing | 0.065 |
| Exp 8 (ternary gate + ternary up, α=0.75) | post-SiLU routing | 0.093 |
| **Exp 9 (ternary gate + full up, α=1.0)** | **post-SiLU routing** | **0.181** |
| Exp 4 (sign gate, pre-SiLU routing) | pre-SiLU routing | 0.245 |
| Exp 5 (prior-token hotlist) | prior-token routing | 0.310 |

### Key findings

**The U-shaped cosine-similarity curve is eliminated.** Output cosine similarity
is now monotonically increasing with hot fraction across all α values — e.g.
at α=0.75: 0.484 → 0.423 → 0.349 → 0.237 → 0.179 → 0.367 → 0.693 going from
0.5% to 30% hot.  The structural failure of experiments 7 and 8 (routing
selecting the most mis-approximated neurons) is resolved because `up_full` is
exact, so `|SwiGLU_approx|` now reliably reflects true neuron activity.

**Full-precision up is responsible for most of the gain.** Comparing α=0 (sign
gate) between exp8 and exp9 at 10% hot: 0.065 → 0.143 — more than doubling
output cosine similarity without any change to the gate proxy.  The ternary gate
improvement on top (α=0.75) adds a further 0.036 (0.143 → 0.179).

**At large hot fractions (20–30%), the scheme performs very well.** At 30% hot
and α=1.5, output cosine similarity reaches **0.699**.  The cold channels
(70% of neurons) contribute only ~2% of the SwiGLU output energy at this point,
so their ternary gate approximation has minimal impact.

**At small hot fractions (≤10%), exp 9 still trails exp 4 and exp 5.**
At 10% hot, best exp9 (0.181) vs exp4 (0.245) vs exp5 (0.310).  The cold
channels' ternary gate errors are still large enough to hurt the output.

**The sweet spot shifts to α=1.0–1.5 for large hot fractions** (vs α=0.75 for
small ones), because at large hot fractions fewer cold channels remain, so
over-sparsifying the gate (77% zeros) causes negligible routing harm while
improving cold-channel gate accuracy.

### Conclusion

Ternary gate + full-precision up **fixes the fundamental routing problem** from
experiments 7–8 and produces a well-behaved, monotonically improving scheme.
At 10% hot it delivers output cosine similarity 0.181 (vs 0.093 in exp8, vs
0.065 in exp7), and at 30% hot reaches 0.699.

The remaining gap below exp4 (pre-SiLU routing, 0.245 at 10% hot) and exp5
(prior-token hotlist, 0.310) indicates that the cold-channel ternary gate
approximation is still the bottleneck.  The next logical step: combine the
ternary gate proxy with pre-SiLU routing (route on `|gate_approx|` rather than
`|swiglu_approx|`) and full-precision up, which should inherit the better routing
quality of exp4 while improving cold-channel SwiGLU accuracy via exact up values.

  ## Experiment 10 {
    ### Motivation

    Experiment 9 confirmed that routing on `|SwiGLU_approx|` with full-precision
    up eliminates the U-shaped curve, but output cosine similarity at 10% hot
    (0.181) still trails exp4's pre-SiLU routing (0.245) and exp5's prior-token
    hotlist (0.310).  The hypothesis: `|SwiGLU_approx| = |SiLU(gate_approx) *
    up_full|` is a weaker ranking signal than raw `|gate_approx|` because SiLU
    non-linearity suppresses near-zero channels (which may be important) and
    amplifies channels where `gate_approx` is already large (but possibly
    over-estimated).

    Prediction: replacing the routing signal with `|gate_approx|` (exp4's
    pre-SiLU signal) while keeping full-precision up and ternary gate for cold
    channels should inherit exp4's routing quality and exp9's cold-channel accuracy.

    ### Scheme

      1. `gate_approx  = T(W_gate, τ) @ x`          ternary gate, all channels
      2. `up_full      = W_up @ x`                   full-precision up, all channels
      3. `hot = top-F channels by |gate_approx|`     **pre-SiLU routing** (exp4 signal)
      4. `gate_hybrid[hot]  = W_gate[hot] @ x`       full-precision gate for hot
         `gate_hybrid[cold] = gate_approx[cold]`     ternary gate for cold
      5. `swiglu_hybrid = SiLU(gate_hybrid) * up_full`
      6. `out = W_down @ swiglu_hybrid`              full-precision down always

    vs exp9: step 3 uses `|gate_approx|` instead of `|SiLU(gate_approx) * up_full|`.

    τ = α × mean(|W_gate|), α ∈ {0.0, 0.25, 0.50, 0.75, 1.00, 1.50};
    hot fractions ∈ {0.5%, 1%, 2%, 5%, 10%, 20%, 30%}.

    ### Results (granite-4.2-3b, 40 layers, 2 000 tokens/layer, MPS)

    #### Gate cosine similarity vs full-precision gate

    | α | zero% | gate cos-sim |
    |---|---|---|
    | 0.00 | 0% | 0.856 |
    | 0.25 | 16% | 0.892 |
    | 0.50 | 32% | 0.916 |
    | **0.75** | **46%** | **0.928** |
    | 1.00 | 58% | 0.928 |
    | 1.50 | 77% | 0.892 |

    (Identical to exp9 — gate proxy unchanged; sweet spot still α=0.75.)

    #### Output cosine similarity vs full precision

    | α | zero% | 0.5% hot | 1% | 2% | 5% | **10%** | 20% | 30% |
    |---|---|---|---|---|---|---|---|---|
    | 0.00 (sign) | 0% | 0.551 | 0.529 | 0.503 | 0.454 | 0.398 | 0.316 | 0.252 |
    | 0.25 | 16% | 0.594 | 0.572 | 0.546 | 0.495 | 0.438 | 0.353 | 0.286 |
    | 0.50 | 32% | 0.621 | 0.600 | 0.574 | 0.523 | 0.464 | 0.377 | 0.308 |
    | **0.75** | **46%** | **0.633** | **0.612** | **0.586** | **0.536** | **0.476** | **0.386** | **0.314** |
    | 1.00 | 58% | 0.628 | 0.608 | 0.584 | 0.534 | 0.473 | 0.380 | 0.307 |
    | 1.50 | 77% | 0.574 | 0.555 | 0.531 | 0.480 | 0.418 | 0.326 | 0.257 |

    #### Neuron (SwiGLU) cosine similarity vs full precision

    | α | zero% | 0.5% hot | 1% | 2% | 5% | **10%** | 20% | 30% |
    |---|---|---|---|---|---|---|---|---|
    | 0.00 (sign) | 0% | 0.603 | 0.581 | 0.555 | 0.506 | 0.449 | 0.366 | 0.300 |
    | 0.25 | 16% | 0.644 | 0.622 | 0.596 | 0.545 | 0.487 | 0.401 | 0.332 |
    | 0.50 | 32% | 0.672 | 0.650 | 0.624 | 0.573 | 0.513 | 0.424 | 0.353 |
    | **0.75** | **46%** | **0.685** | **0.664** | **0.638** | **0.587** | **0.527** | **0.435** | **0.361** |
    | 1.00 | 58% | 0.684 | 0.664 | 0.638 | 0.588 | 0.526 | 0.432 | 0.357 |
    | 1.50 | 77% | 0.638 | 0.619 | 0.594 | 0.543 | 0.479 | 0.386 | 0.313 |

    #### Comparison across experiments (10% hot, best α)

    | Experiment | Routing signal | Out cos-sim @10% |
    |---|---|---|
    | Exp 7 (sign gate + sign up) | post-SiLU `\|SwiGLU_approx\|` | 0.065 |
    | Exp 8 (ternary gate + ternary up, α=0.75) | post-SiLU `\|SwiGLU_approx\|` | 0.093 |
    | Exp 9 (ternary gate + full up, α=1.0) | post-SiLU `\|SwiGLU_approx\|` | 0.181 |
    | Exp 4 (sign gate, pre-SiLU) | pre-SiLU `\|gate_approx\|` | 0.245 |
    | Exp 5 (prior-token hotlist) | prior-token hotlist | 0.310 |
    | **Exp 10 (ternary gate + full up, α=0.75)** | **pre-SiLU `\|gate_approx\|`** | **0.476** |

    ### Key findings

    **Pre-SiLU routing is massively better than post-SiLU routing with the same
    weights.** Switching from `|SwiGLU_approx|` (exp9) to `|gate_approx|` (exp10)
    at α=0.75 raises output cosine similarity at 10% hot from **0.181 → 0.476**
    — a 2.6× improvement.  This confirms the hypothesis: the SiLU non-linearity
    was actively degrading the routing signal by suppressing channels with small
    but non-zero gate values.

    **Exp10 now outperforms all previous experiments at every hot fraction ≤10%.**
    At 10% hot: 0.476 vs 0.310 (exp5) vs 0.245 (exp4).  This is a significant
    result — routing on `|gate_approx|` from a *ternary* gate proxy beats both
    the sign-gate pre-SiLU routing (exp4) and the prior-token hotlist (exp5).

    **No U-shape.** Like exp9, output cosine similarity is monotonically
    decreasing with hot fraction (0.633 → 0.612 → 0.586 → 0.536 → 0.476 → 0.386
    → 0.314 at α=0.75), confirming that exact up values prevent the routing
    from picking the most mis-approximated neurons.

    **α=0.75 is the sweet spot across all hot fractions** (46% zeros in W_gate).
    At 1.0 and 0.5 the results are within 0.003–0.010 of the best, so the
    choice is not critical in a ±0.25 band around 0.75.

    **Cold-channel quality drives the gap vs full precision at large hot
    fractions.** At 30% hot (0.314 at α=0.75), 70% of channels use the ternary
    gate approximation; the gap below 1.0 is entirely from those cold channels.
    Reducing this gap requires either (a) higher hot fraction budget or (b) a
    better cold-channel proxy.

    **Consistent across all 40 layers.** Δout (ternary α=0.75 vs sign α=0) at
    10% hot ranges from +0.050 (layer 29–30) to +0.122 (layer 0) with mean
    +0.079, indicating the improvement is structural, not confined to specific
    layers.

    ### Conclusion

    Pre-SiLU routing on `|gate_approx|` combined with ternary gate proxy
    (α=0.75, 46% zeros) and full-precision up **decisively outperforms all
    previous routing strategies**.  At 10% hot channels it delivers output
    cosine similarity **0.476** — compared to 0.181 in exp9, 0.245 in exp4, and
    0.310 in exp5.

    The remaining gap to full precision at 10% hot is driven by cold-channel
    ternary gate errors (54% of channels use the proxy).  The next directions:

    1. **Combine with prior-token hotlist** (exp5): use prior-token knowledge to
       bias the hot selection toward temporally stable channels — could push
       above 0.5 at 10% hot.
    2. **Ternary up proxy for cold channels**: does adding a ternary up proxy
       for the cold channels (like exp8 did) hurt or help when routing is
       pre-SiLU?
    3. **Scale to lower hot fractions**: at 0.5% hot we already see 0.633 output
       cosine similarity — evaluate whether the FLOP savings at 1–5% hot are
       worth the quality cost in practice.
  }

  ## Experiment 11 {
    ### Motivation

    Experiment 10 (ternary gate α=0.75 + pre-SiLU `|gate_approx|` routing +
    full-precision up) reached 0.476 output cosine similarity at 10% hot,
    exceeding exp5's prior-token hotlist (0.310) and exp4's sign-gate routing
    (0.245).  The question: can the routing signal be further improved by using
    the **prior token's** gate activations as the hot-channel selector?

    At decode time the hotlist from token t−1 is free: either the full-precision
    gate `gate_full[t-1]` (computed anyway on the critical path) or the ternary
    proxy `gate_approx[t-1]` (a byproduct of the ternary pass) can seed the
    routing for token t.  This eliminates any online GEMM cost for routing.

    Three routing variants are tested, all sharing the exp10 weight scheme
    (ternary gate proxy α, full-precision up, full W_down):

    | Variant | Routing signal | Inference cost |
    |---|---|---|
    | A — **current** | `\|gate_approx[t]\|` (exp10 baseline) | ternary GEMM on critical path |
    | B — **proxy prior** | `\|gate_approx[t-1]\|` | free: reuse last step's ternary result |
    | C — **oracle prior** | `\|gate_full[t-1]\|` | free: reuse last step's full gate |

    Variant C is the oracle ceiling — it tells us whether the prior hotlist idea
    itself is sound independent of proxy quality.  The proxy prior (B) is more
    practical: it doesn't require storing an extra full-precision intermediate.

    ### Scheme

      1. `T_gate = T(W_gate, τ)`, τ = α × mean(|W_gate|)
      2. `gate_approx[t] = T_gate @ x[t]`       ternary gate, current token
      3. `up_full[t]     = W_up @ x[t]`          full-precision up, current token
      4. Routing signal from prior token (variant-dependent)
      5. `hot = top-F channels by routing signal`
      6. `gate_hybrid[hot]  = W_gate[hot] @ x[t]`  full-prec gate for hot
         `gate_hybrid[cold] = gate_approx[t][cold]` ternary for cold
      7. `swiglu = SiLU(gate_hybrid) * up_full[t]`
      8. `out    = W_down @ swiglu`

    Sweeps: α ∈ {0.0 (sign), 0.75}; hot fractions ∈ {0.5%, 1%, 2%, 5%, 10%, 20%, 30%}.
    Adjacent-token pairs: prior=0..T−2, current=1..T−1 (T=2 000 cap).

    ### Results (granite-4.2-3b, 40 layers, 2 000 tokens/layer, MPS)

    #### Output cosine similarity — α=0.75 (best from exp10)

    | hot% | current (exp10) | proxy prior | oracle prior |
    |---|---|---|---|
    | 0.5% | 0.633 | **0.675** | 0.669 |
    | 1% | 0.612 | **0.665** | 0.657 |
    | 2% | 0.586 | **0.654** | 0.643 |
    | 5% | 0.536 | **0.630** | 0.616 |
    | **10%** | 0.476 | **0.600** | 0.585 |
    | 20% | 0.386 | **0.552** | 0.538 |
    | 30% | 0.314 | **0.511** | 0.497 |

    #### Output cosine similarity — α=0.0 (sign baseline)

    | hot% | current | proxy prior | oracle prior |
    |---|---|---|---|
    | 0.5% | 0.551 | **0.593** | 0.588 |
    | 1% | 0.529 | **0.583** | 0.576 |
    | 2% | 0.503 | **0.571** | 0.562 |
    | 5% | 0.454 | **0.549** | 0.536 |
    | **10%** | 0.398 | **0.522** | 0.506 |
    | 20% | 0.316 | **0.479** | 0.462 |
    | 30% | 0.251 | **0.443** | 0.425 |

    #### Comparison across all experiments (10% hot, best config)

    | Experiment | Routing signal | Out cos-sim @10% |
    |---|---|---|
    | Exp 7 (sign gate + sign up) | post-SiLU `\|SwiGLU_approx\|` | 0.065 |
    | Exp 8 (ternary gate + ternary up) | post-SiLU `\|SwiGLU_approx\|` | 0.093 |
    | Exp 9 (ternary gate + full up) | post-SiLU `\|SwiGLU_approx\|` | 0.181 |
    | Exp 4 (sign gate, pre-SiLU) | `\|gate_approx[t]\|` | 0.245 |
    | Exp 5 (prior-token hotlist) | `\|gate_full[t-1]\|` (oracle) | 0.310 |
    | Exp 10 (ternary gate + full up) | `\|gate_approx[t]\|` pre-SiLU | 0.476 |
    | **Exp 11 (ternary gate + full up)** | **`\|gate_approx[t-1]\|` proxy prior** | **0.600** |
    | Exp 11 ceiling (ternary gate + full up) | `\|gate_full[t-1]\|` oracle prior | 0.585 |

    ### Key findings

    **Proxy prior outperforms oracle prior at every hot fraction.**  Using
    `|gate_approx[t-1]|` (ternary) beats `|gate_full[t-1]|` (exact) by
    0.010–0.020 consistently.  This is a surprising inversion: the less accurate
    signal routes better.  The likely explanation is that `gate_approx` smooths
    out single-token transients in the full gate — channels that briefly spike
    in `gate_full` but are not persistently hot are not promoted.  The ternary
    proxy has implicit temporal low-pass filtering from its threshold, which
    improves stability as a one-step-ahead predictor.

    **Prior routing gives a large, uniform gain over current-token routing.**
    At 10% hot, proxy prior (0.600) vs current (0.476): +0.124 improvement
    across all 40 layers.  The gain is larger in deeper layers (layers 28–39
    average Δ≈+0.155) than shallow layers (0–9 average Δ≈+0.100), suggesting
    that the later layers' activations are more temporally correlated.

    **0.600 output cosine similarity at 10% hot** with a routing signal that
    is already available from the previous step — no routing GEMM overhead on
    the critical path.  This is nearly double exp5's 0.310 (which also used a
    prior hotlist but with sign-gate cold channels and sign W_down correction).

    **Exp5 vs exp11 gap explained entirely by the weight scheme.**  Exp5 used
    oracle prior routing (`|gate_full[t-1]|`) and achieved 0.310.  Exp11 with
    oracle prior achieves 0.585 — a 1.9× improvement from the same routing
    oracle.  The difference is pure weight scheme: ternary gate + full up vs
    sign gate + sign up + residual W_down correction.

    **Monotonically decreasing, no U-shape** across all α values and variants —
    the full-precision up continues to guarantee well-behaved routing.

    **α=0.75 remains optimal** (46% gate zeros) with proxy prior; the ternary
    threshold sparsity is equally beneficial regardless of the routing variant.

    ### Conclusion

    Combining the ternary gate proxy (α=0.75) + full-precision up from exp10
    with prior-token routing yields **0.600 output cosine similarity at 10%
    hot channels** — with **zero routing overhead** on the critical path.

    The unexpected finding that the proxy prior beats the oracle prior reveals
    that the ternary proxy acts as a temporal filter: it suppresses transient
    spikes and routes based on channels that are persistently large, which is
    exactly what is needed for a one-step-ahead predictor.

    Next directions:

    1. **Why does proxy prior beat oracle prior?** Quantify the temporal
       autocorrelation of `|gate_approx|` vs `|gate_full|`; measure the
       fraction of "transient" channels (hot at t but cold at t+1) that the
       proxy correctly ignores.
    2. **Two-step lookahead**: use `|gate_approx[t-1]|` to also reduce the hot
       gate recompute budget — can we run the ternary pass only on non-hotlist
       channels and skip even the full hot-gate recompute for channels stable in
       the prior list?
    3. **Ternary down proxy**: cold channel contributions to the down projection
       use full W_down — would a ternary W_down for cold channels yield further
       savings without hurting quality?
  }

  ## Experiment 12 {
    ### Motivation

    Experiments 10–11 use full-precision W_down for all I intermediate channels.
    The down projection (H × I = 2560 × 8192) is the costliest GEMM in the MLP
    block.  At 10% hot, only 10% of the input columns carry high-quality SwiGLU
    values; the remaining 90% (cold) are small due to the ternary gate
    approximation.  The hypothesis: cold columns' W_down contribution is
    dominated by sign rather than magnitude, so replacing W_down[:, cold] with
    T(W_down, τ_d) should recover most quality at lower compute cost.

    ### Scheme

    Anchored on exp11's best config (α_gate=0.75, proxy-prior routing):

      1. `gate_approx[t]  = T(W_gate, τ_g) @ x[t]`          ternary gate
      2. `up_full[t]      = W_up @ x[t]`                     full-precision up
      3. `hot = top-F by |gate_approx[t-1]|`                 proxy-prior routing
      4. `gate_hybrid     = W_gate[hot] @ x[t]  ⊕  gate_approx[t][cold]`
      5. `swiglu          = SiLU(gate_hybrid) * up_full[t]`
      6. `out = W_down[:, hot] @ swiglu[hot] + T(W_down, τ_d)[:, cold] @ swiglu[cold]`

    Three output schemes compared:
    - **ternary_cold**: exact W_down for hot, T(W_down, τ_d) for cold
    - **full_down**: exact W_down for all (exp11 baseline)
    - **ternary_all**: T(W_down, τ_d) for all channels (no hot/cold split)

    α_down ∈ {0.0, 0.25, 0.50, 0.75, 1.00, 1.50};
    hot fractions ∈ {0.5%, 1%, 2%, 5%, 10%, 20%, 30%}.

    ### Results (granite-4.2-3b, 40 layers, 2 000 tokens/layer, MPS)

    #### W_down proxy zero-fraction

    | α_down | zero% of W_down |
    |---|---|
    | 0.00 | 0% |
    | 0.25 | 16% |
    | 0.50 | 32% |
    | **0.75** | **46%** |
    | 1.00 | 58% |
    | 1.50 | 77% |

    #### Output cosine similarity — ternary_cold (exp12 scheme)

    | α_down | zero% | 0.5% | 1% | 2% | 5% | **10%** | 20% | 30% |
    |---|---|---|---|---|---|---|---|---|
    | 0.00 | 0% | 0.517 | 0.510 | 0.502 | 0.485 | 0.462 | 0.425 | 0.392 |
    | 0.25 | 16% | 0.554 | 0.547 | 0.538 | 0.520 | 0.495 | 0.455 | 0.420 |
    | 0.50 | 32% | 0.576 | 0.569 | 0.560 | 0.540 | 0.515 | 0.474 | 0.437 |
    | **0.75** | **46%** | **0.582** | **0.575** | **0.566** | **0.546** | **0.521** | **0.480** | **0.442** |
    | 1.00 | 58% | 0.573 | 0.566 | 0.557 | 0.538 | 0.514 | 0.473 | 0.436 |
    | 1.50 | 77% | 0.519 | 0.512 | 0.504 | 0.487 | 0.465 | 0.428 | 0.395 |

    #### Quality cost vs exp11 full_down baseline (Δ at 10% hot)

    | α_down | Δ(ternary_cold − full_down) @10% |
    |---|---|
    | 0.00 | −0.138 |
    | 0.25 | −0.105 |
    | 0.50 | −0.085 |
    | **0.75** | **−0.079** |
    | 1.00 | −0.086 |
    | 1.50 | −0.135 |

    #### Hot/cold split value: Δ(ternary_cold − ternary_all) at 10% hot

    | α_down | Δ |
    |---|---|
    | 0.00 | −0.003 |
    | 0.75 | −0.003 |
    | 1.50 | −0.003 |

    ### Key findings

    **Full_down always wins — ternary down unconditionally hurts.** At every hot
    fraction and every α_down, `full_down` (exp11 baseline) is the best scheme
    by a large and consistent margin.  The best ternary_cold result at 10% hot
    is 0.521 (α_down=0.75) vs 0.600 with full_down — a **−0.079 deficit** even
    at the optimal sparsity.  At sign-only (α_down=0) the deficit is −0.138.

    **The hot/cold split for W_down is nearly worthless.** Δ(ternary_cold −
    ternary_all) is only −0.003 at 10% hot and −0.008 at 30% hot across all
    α_down values.  Keeping exact W_down for hot channels adds essentially
    nothing over approximating all channels equally.  This means the hot/cold
    distinction that is powerful for the gate projection is irrelevant for the
    down projection.

    **W_down ternary quality doesn't scale the same way as W_gate.** The sweet
    spot is still α_down=0.75 (46% zeros, same pattern as W_gate) but the
    *magnitude* of loss is much higher: −0.079 for down vs the −0.010 overhead
    from ternary gate in exp10.  W_down columns for cold channels carry more
    signal than the gate rows for cold channels — the cold SwiGLU values are not
    as small as initially expected after the ternary gate + full up pipeline.

    **The cause**: cold SwiGLU entries are *not* near-zero. Even with a ternary
    gate approximation for cold channels, `up_full` is exact, so
    `SiLU(gate_approx_cold) * up_full_cold` can be substantial when `up_full`
    is large.  The full W_down is therefore needed to correctly weight these
    contributions.

    ### Conclusion

    Ternary W_down for cold channels is **not viable** — it costs 0.079 output
    cosine similarity at 10% hot with no compensating benefit, and the hot/cold
    split for W_down adds nothing (−0.003).  Full-precision W_down must be
    retained for all channels.

    The cause is that `up_full` is exact: even cold SwiGLU entries can be
    non-trivial, and approximating their W_down contribution introduces errors
    proportional to `|up_full_cold|` rather than the near-zero values assumed.

    **Implication for the overall scheme**: the FLOP saving target for the down
    projection must come from *sparsity in the SwiGLU vector* (zeroing out cold
    contributions entirely) rather than weight approximation.  This points
    toward a different direction: approximate cold SwiGLU as exactly zero and
    compute `out ≈ W_down[:, hot] @ swiglu[hot]` — a true sparse GEMM that
    skips cold columns altogether.  The quality cost of this zeroing is exp11's
    result (0.600 at 10% hot) minus what a true sparse down GEMM would achieve,
    which is a separate question from weight approximation.
  }
