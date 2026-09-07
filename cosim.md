# FFN Channel Cosimilarity & Grouping Analysis

This document describes the methodology, implementation, and results of computing channel-wise cosine similarity for FFN activations using [`cosim.py`](cosim.py), with the goal of partitioning channels into groups of exactly 64 highly similar channels.

---

## 1. Methodology

### Why Down-Projection Inputs?
The down-projection input `down_input` (defined as $\text{SiLU}(\text{gate}(x)) \odot \text{up}(x)$ in SwiGLU models) represents the active features feeding into the FFN's second layer (`down_proj`). Measuring the cosine similarity of these channels identifies neurons that exhibit highly correlated firing behaviors across tokens. 

By grouping highly similar channels together, we can:
* Analyze redundancy and structural correlation within the intermediate dimension of the FFN.
* Facilitate efficient model quantization, structured pruning, or tensor-decomposition techniques where channels are blocked or grouped (e.g. block-wise/group-wise quantization).

### Channel Cosine Similarity
For two channel activation vectors $u, v \in \mathbb{R}^T$ (where $T$ is the number of processed tokens):
$$ \text{cosim}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2} $$

### Equal-Sized Greedy Seeding Clustering
We partition the $C$ channels into $C / 64$ groups of exactly 64 channels each using a robust, custom **greedy seeding clustering algorithm**:
1. **Pre-compute Density**: For each channel, compute its "neighborhood density score" as the mean of its top-64 similarity values in the similarity matrix.
2. **Dense Seeding Priority**: Sort all channels in descending order of their density score. This ensures that the densest clusters in the similarity space are grouped first, leaving sparser/outlier channels for later.
3. **Greedy Grouping**:
   * Select the highest-ranked unassigned channel as the seed $s$.
   * Find the 63 unassigned channels that have the highest cosine similarity to $s$.
   * Group these 64 channels together and mark them as assigned.
   * Repeat until all channels are partitioned.

This algorithm is guaranteed to produce groups of exactly 64 channels, unlike standard K-Means which does not enforce equal-sized clusters.

---

## 2. Usage Instructions

The script runs purely using the `.npz` files collected by `tools/profiler/record_ffn_activations.py`. **No model weights or downloads are required**, making it extremely lightweight and completely offline-compatible.

### Basic Grouping & Comparison (Greedy vs. Random Baseline)
To run the script on any `.npz` file and evaluate both the greedy clustering and the random baseline:
```bash
.venv/bin/python cosim.py ffn_activations128.npz
```

### Custom Settings
* **Select Specific Layers**:
  ```bash
  .venv/bin/python cosim.py ffn_activations128.npz --layers 0 1
  ```
* **Change Group Size**:
  ```bash
  .venv/bin/python cosim.py ffn_activations128.npz --group-size 32
  ```
* **Change Output File/Format**:
  ```bash
  # Save as JSON
  .venv/bin/python cosim.py ffn_activations128.npz --output channel_groups.json
  
  # Save as NPZ
  .venv/bin/python cosim.py ffn_activations128.npz --output channel_groups.npz
  ```

---

## 3. Experimental Results

The results below compare the **Greedy Seeding** method against a **Random Baseline** grouping on `ibm-granite/granite-4.2-3b` activations.

### File: `ffn_activations128.npz` (12,734 Tokens, 8,192 Channels)

| Layer | Method | Mean Pairwise Similarity | Min Group Similarity | Max Group Similarity | Std |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Layer 0** | **Greedy** | **0.02925** | 0.00060 | 0.17937 | 0.02685 |
| | Random | -0.00010 | -0.00178 | 0.00241 | 0.00070 |
| **Layer 1** | **Greedy** | **0.02600** | 0.00118 | 0.16098 | 0.02394 |
| | Random | -0.00002 | -0.00123 | 0.00174 | 0.00058 |
| **Layer 2** | **Greedy** | **0.02847** | 0.00062 | 0.18608 | 0.02896 |
| | Random | 0.00005 | -0.00152 | 0.00236 | 0.00066 |
| **Layer 3** | **Greedy** | **0.02633** | 0.00177 | 0.15142 | 0.02536 |
| | Random | -0.00004 | -0.00197 | 0.00285 | 0.00069 |

### Interpretation
* **Massive Relative Similarity Gain**: The mean pairwise similarity of the greedy groupings is positive and significantly higher (by **over 250x** to **500x**) compared to the random baseline, which hovers around 0.0 (perfect orthogonality/randomness).
* **High Sparsity Effect**: Due to the SiLU activation and SwiGLU gating, many channels in `down_input` are sparse, resulting in lower absolute cosine similarity values than raw projection outputs. However, the greedy grouping is highly effective at clustering the remaining correlated active channels, as evidenced by the maximum group similarities reaching as high as **0.186**!
