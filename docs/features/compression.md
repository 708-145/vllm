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
| **cosim ≥ 0.97** | **Per-row N=6..8, per-row scale** | **2.62** | 5-into-13-bit packing |
| cosim ≥ 0.95 | Per-row N=5..7, per-row scale | 2.36 | 3-into-7-bit packing |

### `gate_proj` / `up_proj`

See Section 8 below.

---

## 8. `gate_proj` and `up_proj` — TODO

Applicability of the above LUT compression to `gate_proj` [8192, 2560] and
`up_proj` [8192, 2560] is evaluated in the next section.
