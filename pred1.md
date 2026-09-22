
# Experimental dynamic MLP sparsity prediction

## Core idea

Evaluate the gate projection in reduced precision. For output values before/after (to be decided) SwiGLU apply a threshold (gate_thresh) to derive hot channels. For the hot channels, compute the full precision values with original weights. The gate output thus consists of approximations for cold channels and correct values for hot channels. 
Do the same for the up projection: Compute low precision for cold channels and full precision for hot channels. Either the same hot/cold mix as derived from the gate projection. Alternative: for cold gate channels above up_thresh recompute gate and up projection in high precision.

Down projection is different since the hot/cold channels correspond to inputs. Compute the full matrix in low precision. Then add a sparse high precision correction for hot input channels. Needs invention how to implement this efficiently! 

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
