# MLP Weight Compression Analysis

Analysis of weight compression opportunities in the FFN layers of
`ibm-granite/granite-4.2-3b` (40 layers, residual stream = 2560,
hidden dim = 8192).

---

## Terminology

| Term | Meaning |
|---|---|
| **residual stream** | The 2560-dimensional hidden state passed between layers |
| **hidden dimension** | The 8192-dimensional intermediate neuron space inside the FFN |
| `gate_proj` / `up_proj` | `[8192, 2560]` — project residual stream → hidden dim |
| `down_proj` | `[2560, 8192]` — project hidden dim → residual stream |

---

## 1. Activation sparsity (gate-threshold)

The FFN computes:
```
gate_raw  = x @ gate_w.T          # [T, 8192]
up_out    = x @ up_w.T            # [T, 8192]
down_in   = silu(gate_raw) * up_out
out       = down_in @ down_proj.T  # [T, 2560]
```

When `gate_raw[i] < threshold`, neuron `i` contributes near-zero to the
output regardless of weight magnitude. Skipping those neurons entirely
(zeroing their contribution to `down_proj`) saves compute without touching
weights.

- **~31% of neurons** can be skipped per token while keeping cosim_p5 ≥ 0.95
  (per-layer binary-searched threshold, see `gate_threshold.py`)
- Compute saving: `2 × inactive_frac / 3` (gate always runs; up+down optional)
- This is **activation sparsity**, not weight sparsity — the weights are dense

---

## 2. Weight sparsity: per-neuron input masking (`gate_proj` / `up_proj`)

Each of the 8192 neurons reads from 2560 residual-stream input dims.
Ranking input dims by `|gate_w[i,:]| + |up_w[i,:]|` per neuron and zeroing
the least-important ones:

| Drop% (per neuron) | cosim_mean | cosim_p5 |
|---|---|---|
| 20% (512 dims) | **0.986** | 0.981 |
| 31% (binary-searched to cosim_p5 ≥ 0.95) | ~0.963 | 0.950 |

- Per-neuron masks are **static** (weight-based, input-independent)
- Global input masking (same 512 dims dropped for all neurons) gives only 0.910 — the per-neuron approach is far superior
- The masks are statistically random (pairwise Jaccard ≈ random baseline) — no grouping structure exists to exploit for dense matmul

---

## 3. `down_proj` weight structure

### 3a. Receptive field (column grouping)

Each of the 2560 residual-stream output rows reads from all 8192 hidden dims.
Attempting to cluster output rows by shared hidden-dim receptive field:

- Weight energy is **uniformly distributed** — keeping top 80% of hidden dims
  per row already loses cosim ~0.02, and the union mask across any group of
  rows covers nearly all 8192 dims
- No grouping structure exists at any block size (B=4…2560): DCT spectrum CV
  < 0.02, DC/AC ratio = 1.000±0.004 — the weight columns are spatially i.i.d.

### 3b. Frequency domain (DCT)

DCT of `down_proj` weight columns (full-column or block-64):
- Energy spectrum is **perfectly flat** — each DCT coefficient carries exactly
  1/N of total energy regardless of block size
- DCT is a **homogenising** transform only when the input is non-white;
  these weights are already white noise in DCT space
- Video-codec bit schedules (more bits for DC, fewer for AC) fail catastrophically:
  full-column `1×8b + 63×4b + 256×2b + rest×1b` at 1.23 b/w → cosim ~0.84
- **Conclusion: DCT / block transforms provide no benefit for `down_proj`**

### 3c. Why activation sparsity still works

Gate-threshold skipping drops **entire columns** of `down_proj` when the
corresponding neuron's activation is near zero. This exploits **activation
sparsity** (data-dependent, per-token), not weight sparsity. The weights
themselves are dense and unstructured.

---

## 4. Quantization analysis (`down_proj`)

### 4a. Weight distribution

- Near-Gaussian: kurtosis ≈ 3.4–3.6 (Gaussian = 3)
- std ≈ 0.0036, **uniform energy** across all weight positions
- Top 1% of weights carry only ~3.7% of energy (no outliers)
- Per-column energy: 23% of weights capture 50% of energy (linear = white noise)

### 4b. Rate-distortion bound (Shannon, Gaussian source)

| Target relative MSE | Shannon minimum | Approx cosim |
|---|---|---|
| 1% of variance | 4.98 b/w | 0.9995 |
| 3% | 4.19 b/w | 0.9985 |
| **10%** | **3.32 b/w** | **0.9950** |
| 30% | 2.53 b/w | 0.9850 |

### 4c. Uniform int-N quantization (per-column scale)

| bits | cosim_mean | cosim_p5 |
|---|---|---|
| 2 | 0.57–0.76 | — |
| 3 | 0.88–0.96 | — |
| **4** | **0.967–0.992** | **0.959–0.988** |
| 8 | ≈1.000 | ≈1.000 |

Per-tensor int4 empirical entropy: **0.85 bits/symbol** (most indices cluster
in central bins — 3.15 bits wasted per weight with per-tensor scale).

---

## 5. Lloyd-Max LUT quantization

### 5a. Scale granularity

With N=10 Lloyd-Max levels (optimal for Gaussian):

| Scale granularity | n_scales | overhead (b/w) | cosim (layer 0) | cosim (layer 17) |
|---|---|---|---|---|
| None | 0 | 0 | 0.105 ❌ | 0.063 ❌ |
| **Per-tensor** | **1** | **~0** | **0.992** | 0.968 |
| Per-row (residual dim) | 2560 | 0.002 | 0.992 | **0.984** |
| Per-column (neuron) | 8192 | 0.006 | 0.992 | 0.967 |
| Per-group G=128 | 163K | 0.125 | 0.993 | 0.972 |

**Per-row scale** (one fp16 per residual-stream output dim) is the sweet spot:
near-zero overhead (0.002 b/w), best quality on layers with higher row-norm
variance (cv 0.07–0.13 across layers).

### 5b. Fixed global LUT (Gaussian Lloyd-Max) with per-row scale

| N | entropy (H) | cosim_mean | cosim_p5 |
|---|---|---|---|
| 8 | 2.81 b | 0.983–0.989 | 0.975–0.985 |
| **10** | **3.11 b** | **0.988–0.992** | **0.981–0.988** |
| 12 | 3.36 b | 0.990–0.994 | 0.984–0.991 |
| 16 | 3.72 b | 0.992–0.996 | 0.987–0.993 |

N=10 matches int4 quality at 3.11 b/w (vs 4.00 b/w) using a single shared
10-entry LUT + one fp16 scale per row.

### 5c. Per-row adaptive LUT (own N + own levels per row)

Each row gets its own Lloyd-Max levels fitted to its actual weight distribution.
Minimum N selected per row to meet a cosim floor:

| Floor | Dominant N | % rows | idx b/w (naive) | **idx b/w (packed)** | LUT oh | **total b/w** | out cosim |
|---|---|---|---|---|---|---|---|
| 0.980 | 8 | ~80–96% | 3.23 | 3.05 | 0.016 | **3.07** | 0.984–0.991 |
| 0.970 | 7 | ~87–97% | 3.01 | 2.82 | 0.014 | **2.84** | 0.975–0.986 |
| **0.960** | **6** | **~94–98%** | 3.00 | **2.61** | 0.012 | **2.62** | **0.966–0.982** |
| 0.950 | 5 | ~93–97% | 3.00 | 2.34 | 0.010 | **2.36** | 0.954–0.976 |

---

## 6. Optimal joint packing

Standard `ceil(log2(N))` bits/index wastes bits. Joint encoding of G values
into W bits achieves near-Shannon efficiency:

| N | log₂(N) | Packing | b/index | waste |
|---|---|---|---|---|
| 5 | 2.322 | 3 → 7 bits | 2.333 | 0.5% |
| **6** | **2.585** | **5 → 13 bits** | **2.600** | **0.6%** |
| 7 | 2.807 | 11 → 31 bits | 2.818 | 0.4% |
| 8 | 3.000 | 1 → 3 bits | 3.000 | 0.0% |
| 10 | 3.322 | 3 → 10 bits | 3.333 | 0.3% |

The practical encoding for N=6: pack 5 indices into 13 bits (6⁵=7776 ≤ 8192),
or 10 indices into 26 bits. LUT decode is a single table lookup.

---

## 7. Recommended operating points

### `down_proj`

| Target quality | Method | Total b/w | Notes |
|---|---|---|---|
| cosim ≥ 0.99 | int4 uniform, per-col scale | 4.00 | Baseline |
| cosim ≥ 0.99 | N=10 Lloyd-Max, per-row scale | 3.11 | Shared 10-entry LUT |
| **cosim ≥ 0.97** | **Per-row 6-6-7, per-row scale** | **2.667** | 1 byte per 3 values |
| cosim ≥ 0.95 | Per-row N=5..7, per-row scale | 2.36 | 3-into-7-bit packing |

### `gate_proj` / `up_proj`

See Section 8 below.

---

## 8. `gate_proj` and `up_proj`

Shape: `[8192, 2560]` — 8192 rows (neurons), each reading from 2560 residual-stream
dims. The per-row orientation is the same as `down_proj` (rows = the unit being
quantized), but H=2560 instead of 8192, which raises LUT overhead slightly.

### 8a. Weight distribution

| Projection | std (mean) | kurtosis | row_norm_cv | col_norm_cv |
|---|---|---|---|---|
| `gate_proj` | 0.01230 | 4.42 | 0.141 | 0.036 |
| `up_proj` | 0.01371 | 3.88 | 0.116 | 0.050 |
| `down_proj` | 0.00309 | 4.23 | 0.090 | 0.099 |

Both are near-Gaussian (kurtosis 3.9–4.9 vs Gaussian=3). Higher kurtosis than
`down_proj` means slightly heavier tails — more outlier rows will need larger N.
Row-norm CV is higher (0.12–0.15 vs 0.07–0.09), confirming per-row scale remains
the right granularity.

### 8b. LUT overhead at H=2560

With H=2560 (vs H=8192 for `down_proj`), the LUT storage cost per weight is
**3.2× higher**: `n×16 / 2560` vs `n×16 / 8192`. At N=6: 0.044 b/w overhead
(vs 0.012 for `down_proj`). Still small relative to the index bits.

### 8c. Fixed-N results (gate + up both quantized, down_proj exact)

Quality measured at **FFN output** — errors from gate and up propagate
multiplicatively through the SiLU gating.

| N | idx b/w | lut oh | total b/w | out cosim (mean) | out cosim p5 |
|---|---|---|---|---|---|
| 6 | 2.600 | 0.044 | **2.644** | 0.941–0.962 | 0.872–0.946 |
| 7 | 2.818 | 0.050 | **2.868** | 0.953–0.976 | 0.880–0.959 |
| 8 | 3.000 | 0.056 | **3.056** | 0.961–0.981 | 0.895–0.968 |
| 10 | 3.333 | 0.069 | **3.402** | 0.975–0.989 | 0.946–0.979 |

(Ranges across layers 0, 8, 17, 31, 39.)

### 8d. Adaptive per-row N (gate and up must both satisfy floor)

| Floor | Dom N | idx b/w | total b/w | out cosim | out p5 |
|---|---|---|---|---|---|
| 0.980 | 8 | 3.01–3.06 | **3.07–3.11** | 0.975–0.985 | 0.954–0.969 |
| 0.970 | 7 | 2.72–2.79 | **2.76–2.84** | 0.963–0.977 | 0.943–0.958 |
| **0.960** | **6** | **2.60–2.62** | **2.65–2.67** | **0.947–0.972** | **0.902–0.945** |
| 0.950 | 6 | 2.60–2.61 | 2.65–2.65 | 0.942–0.958 | 0.871–0.946 |

Distribution at floor 0.960: ~93–99% of rows use N=6, ~2–5% need N=7 or 8.

### 8e. Comparison to `down_proj`

| Projection | Floor 0.960 total b/w | out cosim range |
|---|---|---|
| `down_proj` | **2.62** | 0.966–0.982 |
| `gate_proj`+`up_proj` | **2.65–2.67** | 0.947–0.972 |

Gate/up cosim is slightly lower because errors in gate and up **multiply**
(SiLU gate × up activation) — the two quantization errors compound rather
than add. Higher kurtosis also means more outlier rows requiring N>6.

### 8f. Combined model compression (all three projections)

At the N=6 / floor 0.960 operating point across all three projections:

| Projection | b/w | notes |
|---|---|---|
| `gate_proj` LUT | ~2.66 | N=6 dominant, per-row scale |
| `up_proj` LUT | ~2.66 | same |
| `down_proj` LUT | ~2.62 | N=6 dominant, per-row scale |
| **MLP average** | **~2.65** | vs int4 = 4.00 b/w |
| **Saving vs int4** | **−1.35 b/w (−34%)** | at cosim ≥ 0.95–0.97 |

The LUT approach is equally applicable to all three MLP projections.
The same N=6 operating point and 5-into-13-bit packing applies throughout.

### 8g. Side-by-side structural comparison

| Property | `down_proj` [2560, 8192] | `gate_proj`/`up_proj` [8192, 2560] |
|---|---|---|
| Distribution | near-Gaussian, kurt 4.2 | near-Gaussian, kurt 3.9–4.4 |
| Per-row scale meaning | one scale per residual-stream output dim | one scale per neuron |
| H per row | 8192 | 2560 |
| LUT overhead at N=6 | 0.012 b/w | 0.044 b/w (3.2× more, still small) |
| Floor 0.960 total b/w | **2.62** | **2.65–2.67** |
| Floor 0.960 cosim range | 0.966–0.982 | 0.947–0.972 |
| Dominant N at floor 0.960 | 6 | 6 |
| Error propagation | additive to residual stream | multiplicative (gate × up) |

The smaller H=2560 raises the per-weight LUT overhead but it remains under
0.05 b/w. The main quality gap comes from the multiplicative error: quantizing
both gate and up independently means their errors multiply through
`silu(gate) × up` before entering `down_proj`, whereas `down_proj` quantization
error is purely additive to the residual stream.

---

## 9. Per-row 6-6-7 encoding

### 9a. Byte-packing insight

Both pure 6-6-6 and mixed 6-6-7 pack exactly into **one byte per 3 values**:

| Scheme | codewords needed | fits in 1 byte? | bpw |
|---|---|---|---|
| 6-6-6 (pure N=6) | 6³ = 216 | ✅ ≤ 256 | **2.667** |
| **6-6-7** | 6×6×7 = 252 | ✅ ≤ 256 | **2.667** |
| 7-7-7 (pure N=7) | 7³ = 343 | ❌ > 256 | needs 9 bits |

The 6-6-7 scheme costs **nothing extra** vs pure 6-6-6. It simply uses 36 of
the 40 spare codewords (256−216) to give every third position one additional
reconstruction level — strictly better quality at identical storage cost.

Per row: two LUTs are fitted to the row's actual weight distribution — a 6-level
LUT6 and a 7-level LUT7 (both Lloyd-Max on that row's values). Positions 0,1
of each triplet use LUT6; position 2 uses LUT7.

An adaptive fallback chooses N=12 for rows where 6-6-7 weight cosim falls below
a threshold.

### 9b. Metadata layout — no separate scale needed

LUT values are stored as fp16 directly in the weight's original scale, so a
separate per-row scale factor is **redundant** — the LUT entries are the
reconstruction values already in the right units.

Per-row metadata is a fixed **13 × fp16 = 208 bits** block for both formats:

```
6-6-7 row:  [fp16 × 6: LUT6 levels] [fp16 × 7: LUT7 levels]   = 208 bits
N=12 row:   [fp16 × 12: LUT levels] [fp16: sentinel = NaN]     = 208 bits
```

The 13th fp16 is the format flag: if it is NaN → N=12 mode (entries 1–12 are
the levels); otherwise it is the 7th LUT7 entry → 6-6-7 mode. No separate
flag table, no side-channel, uniform metadata stride of 208 bits per row.

Note: N=12 fallback rows use only 12 entries and get the NaN sentinel "for
free" — their metadata is actually 16 bits smaller in information content than
a 6-6-7 row.

### 9c. Corrected bpw accounting

Removing the redundant scale field:

| Component | old (with scale) | new (no scale) |
|---|---|---|
| Index bytes | 8/3 b/w | 8/3 b/w |
| LUT metadata at H=8192 | (6+7+1)×16/8192 | (6+7)×16/8192 |
| LUT metadata at H=2560 | (6+7+1)×16/2560 | (6+7)×16/2560 |
| **Total bpw (H=8192)** | **2.694** | **2.669** |
| **Total bpw (H=2560)** | **2.754** | **2.719** |

### 9d. Quality gain over pure 6-6-6 (per-row LUT6)

Evaluated on 10 representative layers (0,4,8,12,17,22,27,31,35,39).
"floor" = per-row weight cosim threshold below which the row falls back to N=12.

**`down_proj` (H=8192):**

| floor | % rows use 6-6-7 | out cosim | Δ vs pure LUT6 | Δ vs pure N=12 |
|---|---|---|---|---|
| 0.990 | 0% | 0.982–0.995 | +0.011–+0.030 | ±0.000 (= N=12) |
| 0.980 | 0% | 0.982–0.995 | +0.011–+0.030 | ±0.000 (= N=12) |
| **0.970** | **41–99%** | **0.965–0.991** | **+0.001–+0.019** | **−0.004–−0.016** |
| 0.960 | 92–99% | 0.965–0.985 | +0.000–+0.013 | −0.010–−0.018 |

**`gate_proj` + `up_proj` (H=2560):**

| floor | % rows use 6-6-7 | out cosim | Δ vs pure LUT6 | Δ vs pure N=12 |
|---|---|---|---|---|
| 0.990 | 0% | 0.975–0.991 | +0.016–+0.039 | ±0.000 (= N=12) |
| 0.980 | 0% | 0.975–0.991 | +0.016–+0.039 | ±0.000 (= N=12) |
| **0.970** | **78–96%** | **0.955–0.979** | **+0.003–+0.036** | **−0.006–−0.024** |
| 0.960 | 96–99% | 0.945–0.977 | +0.001–+0.015 | −0.013–−0.027 |

### 9e. Interpretation

- **At floors 0.990/0.980**: no row qualifies for 6-6-7 (per-row weight cosim is
  too tight). All rows fall back to N=12. The adaptive scheme equals pure N=12.

- **At floor 0.970**: the sweet spot. ~80–96% of rows use 6-6-7, gaining
  +0.001–+0.036 cosim over pure LUT6 at **zero extra bits** (still 2.667 bpw).
  Fallback N=12 rows use 12×fp16 + NaN sentinel = 208 bits metadata, same
  stride as 6-6-7 rows. Index encoding: 5 values into 18 bits = 3.60 bpw
  for the minority fallback rows.

- **6-6-7 is always strictly better than pure 6-6-6** — the only question is
  how much better. The gain is largest for gate/up (multiplicative error path)
  where even a small improvement in per-row LUT quality has compounded benefit.

- The 4 unused byte codewords (256−252=4) are available for future use
  (e.g. special tokens, escape codes) without any format change.

### 9f. Recommended operating point

For all three projections at **2.667 bpw** (1 byte per 3 values, byte-aligned,
no packing overhead):

| Projection | Scheme | bpw | cosim range |
|---|---|---|---|
| `down_proj` | per-row 6-6-7, floor 0.970 | 2.667* | 0.965–0.991 |
| `gate_proj` | per-row 6-6-7, floor 0.970 | 2.667* | 0.955–0.979 |
| `up_proj` | per-row 6-6-7, floor 0.970 | 2.667* | 0.955–0.979 |

\* Minority of rows fall back to N=12 (3.60 bpw index + 208-bit metadata, same
stride). Weighted average bpw stays ≤ 2.72 (H=8192) / ≤ 2.78 (H=2560).

---

## 10. Applicability to attention and embedding tensors

The 6-6-7 + N=12 adaptive scheme was evaluated on all non-MLP weight tensors
at layers 0, 2, 15, 17, 22, 31, 39 using per-row weight cosim as the quality
metric (no attention activations captured).

### 10a. Tensor inventory

| Tensor | Shape | R | H |
|---|---|---|---|
| `self_attn.q_proj` | [2560, 2560] | 2560 | 2560 |
| `self_attn.k_proj` | [512, 2560] | 512 | 2560 |
| `self_attn.v_proj` | [512, 2560] | 512 | 2560 |
| `self_attn.o_proj` | [2560, 2560] | 2560 | 2560 |
| `lm_head` | [100352, 2560] | 100352 | 2560 |
| `model.embed_tokens` | [100352, 2560] | 100352 | 2560 |

LayerNorm weights (1D) are excluded — too small to benefit.

### 10b. Results

Format: `% rows using 6-6-7 / weighted bpw` at each floor.
bpw formula: `p×(8/3 + 13×16/H) + (1−p)×(18/5 + 13×16/H)` where p = fraction
using 6-6-7. (Both formats share the 13×fp16 = 208-bit metadata block.)

| Tensor | L | R | H | std | kurt | f=0.99 | f=0.98 | f=0.97 | f=0.96 |
|---|---|---|---|---|---|---|---|---|---|
| q_proj | 0 | 2560 | 2560 | 0.01644 | 10.3 | 0%/3.68 | 0%/3.68 | 86%/2.88 | 97%/2.78 |
| q_proj | 2 | 2560 | 2560 | 0.01422 | 4.1 | 0%/3.68 | 0%/3.68 | 77%/2.97 | 94%/2.81 |
| q_proj | 15 | 2560 | 2560 | 0.01511 | 4.7 | 0%/3.68 | 0%/3.68 | 71%/3.02 | 93%/2.81 |
| q_proj | 17 | 2560 | 2560 | 0.01567 | 4.4 | 0%/3.68 | 0%/3.68 | 81%/2.93 | 98%/2.77 |
| q_proj | 22 | 2560 | 2560 | 0.01458 | 4.6 | 0%/3.68 | 0%/3.68 | 77%/2.97 | 98%/2.77 |
| q_proj | 31 | 2560 | 2560 | 0.01399 | 5.9 | 0%/3.68 | 0%/3.68 | 76%/2.97 | 94%/2.81 |
| q_proj | 39 | 2560 | 2560 | 0.01151 | 3.9 | 0%/3.68 | 0%/3.68 | 68%/3.04 | 95%/2.80 |
| k_proj | 0 | 512 | 2560 | 0.01937 | 13.5 | 0%/3.68 | 0%/3.68 | 89%/2.85 | 96%/2.79 |
| k_proj | 2 | 512 | 2560 | 0.01692 | 4.8 | 0%/3.68 | 0%/3.68 | 78%/2.96 | 93%/2.82 |
| k_proj | 15 | 512 | 2560 | 0.01806 | 6.3 | 0%/3.68 | 0%/3.68 | 71%/3.02 | 88%/2.86 |
| k_proj | 17 | 512 | 2560 | 0.01886 | 4.6 | 0%/3.68 | 0%/3.68 | 75%/2.98 | 97%/2.78 |
| k_proj | 22 | 512 | 2560 | 0.01610 | 4.6 | 0%/3.68 | 0%/3.68 | 58%/3.14 | 95%/2.79 |
| k_proj | 31 | 512 | 2560 | 0.01452 | 6.8 | 0%/3.68 | 0%/3.68 | 72%/3.01 | 96%/2.78 |
| k_proj | 39 | 512 | 2560 | 0.00983 | 4.0 | 0%/3.68 | 0%/3.68 | 79%/2.95 | 94%/2.80 |
| v_proj | 0 | 512 | 2560 | 0.01082 | 4.3 | 0%/3.68 | 0%/3.68 | 70%/3.03 | 94%/2.80 |
| v_proj | 2 | 512 | 2560 | 0.01059 | 3.3 | 0%/3.68 | 0%/3.68 | 88%/2.86 | 99%/2.76 |
| v_proj | 15 | 512 | 2560 | 0.01048 | 3.7 | 0%/3.68 | 0%/3.68 | 76%/2.97 | 99%/2.75 |
| v_proj | 17 | 512 | 2560 | 0.01043 | 3.8 | 0%/3.68 | 0%/3.68 | 74%/2.99 | 94%/2.80 |
| v_proj | 22 | 512 | 2560 | 0.01077 | 3.7 | 0%/3.68 | 0%/3.68 | 66%/3.07 | 100%/2.75 |
| v_proj | 31 | 512 | 2560 | 0.01120 | 3.8 | 0%/3.68 | 0%/3.68 | 59%/3.13 | 91%/2.83 |
| v_proj | 39 | 512 | 2560 | 0.01928 | 3.4 | 0%/3.68 | 0%/3.68 | 80%/2.93 | 99%/2.75 |
| o_proj | 0 | 2560 | 2560 | 0.00237 | 6.2 | 0%/3.68 | 0%/3.68 | 20%/3.49 | 92%/2.82 |
| o_proj | 2 | 2560 | 2560 | 0.00271 | 4.5 | 0%/3.68 | 0%/3.68 | 85%/2.89 | 98%/2.76 |
| o_proj | 15 | 2560 | 2560 | 0.00262 | 10.4 | 0%/3.68 | 0%/3.68 | 70%/3.03 | 97%/2.78 |
| o_proj | 17 | 2560 | 2560 | 0.00257 | 8.8 | 0%/3.68 | 0%/3.68 | 65%/3.07 | 94%/2.80 |
| o_proj | 22 | 2560 | 2560 | 0.00258 | 5.3 | 0%/3.68 | 0%/3.68 | 54%/3.17 | 86%/2.88 |
| o_proj | 31 | 2560 | 2560 | 0.00267 | 6.0 | 0%/3.68 | 0%/3.68 | 66%/3.06 | 98%/2.77 |
| o_proj | 39 | 2560 | 2560 | 0.00332 | 8.2 | 0%/3.68 | 0%/3.68 | 57%/3.15 | 99%/2.76 |
| lm_head | — | 100352 | 2560 | 0.00399 | 3.1 | 0%/3.68 | 0%/3.68 | **99%/2.76** | 99%/2.75 |
| embed_tokens | — | 100352 | 2560 | 0.48143 | 3.1 | 0%/3.68 | 0%/3.68 | **99%/2.76** | 99%/2.75 |

### 10c. Key findings

**The scheme is universally applicable across all tensor types.**

1. **Floor 0.980 gives 0% 6-6-7 rows everywhere** — same as MLP. The per-row
   weight cosim threshold is too tight for 6-6-7 at this quality level across
   all tensor types. All rows fall back to N=12 at 3.68 bpw.

2. **Floor 0.960 is the universal sweet spot**: 86–100% of rows use 6-6-7
   across all tensors and layers. Effective bpw 2.75–2.88 everywhere.

3. **`lm_head` and `embed_tokens` are the easiest** (99% at floor 0.970,
   bpw ≈ 2.76) — near-Gaussian distribution (kurtosis ≈ 3.1) and large
   R=100352 give excellent per-row LUT coverage.

4. **`o_proj` layer 0 is the hardest** — only 20% at floor 0.970 due to high
   kurtosis (6.2) and very small std (0.00237). Still 92% at floor 0.960.

5. **Higher kurtosis → more N=12 fallback**. `q_proj`/`k_proj`/`o_proj` have
   kurtosis 4–13 vs `v_proj`'s 3.3–4.3 — `v_proj` is the cleanest attention
   tensor.

### 10d. Whole-model operating point

At **floor = 0.960** across all tensors:

| Tensor type | Typical bpw | vs int4 (4.00) |
|---|---|---|
| MLP projections (×3 per layer) | 2.67–2.72 | −1.30 |
| Attention projections (×4 per layer) | 2.75–2.88 | −1.15 |
| `lm_head` / `embed_tokens` | ~2.75 | −1.25 |
| **Whole-model average** | **~2.75** | **−1.25 (−31%)** |

The 4 unused byte codewords (253–255) are available for future extension
(escape codes, special tokens) without breaking the format.
