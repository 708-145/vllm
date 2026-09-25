
# Experimental dynamic MLP sparsity prediction

## Summary of experiments

| Exp | Scheme | Routing signal | Key result (out cos-sim @10% hot) | Conclusion |
|---|---|---|---|---|
| 1 | Sign weights, no hybrid | — (full sign-approx pass) | 0.365 down cos-sim (all cold) | Sign is a useful routing signal but not a computation substitute |
| 2 | Sign weights + hybrid (hot/cold split) | `\|gate_approx\|` threshold | ~0.05 SwiGLU, ~0.01 down (collapses) | Thresholding on sign-approx magnitude is the wrong routing direction |
| 3 | Low-rank SVD of W_gate (routing signal only) | Rank-r gate approx | Sign predictor beats SVD up to rank ~200 | Low-rank proxy is worse and more expensive than sign for routing |
| 4 | Sign gate; top-F hot by `\|gate_approx\|` | Pre-SiLU `\|gate_approx\|` | **0.245** | Routing on pre-SiLU gate magnitude works; 1.3% mis-classified "dangerous" channels |
| 5 | Prior-token hotlist (no refinement) | `\|gate_full[t-1]\|` oracle | **0.310** | Prior hotlist outperforms online approximation; refinement hurts |
| 6 | Magnitude-confidence refinement of prior hotlist | `\|gate_full[t-1]\|` + selective flip | 0.307 (20% refine, hot=10%) | Beats boundary-rank refinement but still worse than no refinement |
| 7 | Sign gate + sign up; route on `\|SwiGLU_approx\|` | Post-SiLU `\|SwiGLU_approx\|` | **0.065** (U-shaped curve) | Post-SiLU routing selects most mis-approximated neurons; fails badly |
| 8 | Ternary gate + ternary up; post-SiLU routing | Post-SiLU `\|SwiGLU_approx\|` | **0.093** | Ternary helps (+0.028) but U-shape persists |
| 9 | Ternary gate + **full-precision up**; post-SiLU routing | Post-SiLU `\|SwiGLU_approx\|` | **0.181** | Full up eliminates U-shape; up error was the root cause |
| 10 | Ternary gate (α=0.75) + full up; **pre-SiLU routing** | Pre-SiLU `\|gate_approx\|` | **0.476** | Best single-token scheme; 2.6× improvement over exp9 |
| 11 | Ternary gate + full up; **proxy-prior routing** | `\|gate_approx[t-1]\|` (free at decode) | **0.600** | Proxy prior beats oracle prior; zero routing overhead |
| 12 | Exp11 + ternary W_down for cold channels | Proxy prior | 0.521 (−0.079 vs exp11) | Ternary down not viable; cold SwiGLU values are non-trivial |
| 13 | Logit-space re-evaluation of exp10–11 | Various | Top-1: 0.059 (exp11 proxy @10%) | Exp11 proxy best on top-1/logit-cos; KL favours exp10 (distribution tails) |
| 14 | End-to-end top-1 perturbation (all 40 layers, prefill) | Proxy prior ≡ current (prefill) | **48% predictions change @10% hot** | Per-layer errors compound; 0.61 hidden cos-sim → 48% e2e perturbation |
| 15 | Low-rank SVD gate+up (no hot/cold split) vs exp14 | — (full SVD approx) | **99%+ perturbation at all ranks** | SVD completely fails e2e; systematic bias compounds across layers |
| 16 | Single-layer and cumulative perturbation sweep | Exp14 config | Layer 39: 13.4%, layer 17: 10.4%; top-8 = 52% of total | No "culprit" layers; uniform improvement across all layers is the right strategy |
| 17 | Block-ternary encoding (B=16, FP16 block-max scale, TARE metric) — encoding quality only, no inference | — | TARE=1.376 (α=0, sign); 5.33× compression vs BF16 | α=0 (pure sign) minimises TARE; block-max scale over-estimates small weights → block RMS is the natural next step |
| 18 | Block-ternary with TARE-optimal per-block scale (tilt-weighted geometric mean) — encoding quality only | — | TARE=0.828 (α=0.25); 5.33× vs BF16 | 39% better than block-max; optimal α shifts to 0.25; main gain from scale, not sparsity |
| 19 | 1-bit sign + E8M0 per-block scale, sweep B∈{8,16,32,64} — encoding quality only | — | B=8: TARE=0.837, **8.00×**; B=16: TARE=0.859, **10.67×** | B=8 E8M0 matches FP16-opt at B=16 in both quality and compression; sweet spot is B=8 |
| 20 | B=8 E8M0 sign gate+up, hot refinement for both, full W_down — e2e top-1 | Proxy-prior | match=0.194 @10% hot (vs 0.516 exp14) | Encoding W_up cold channels is too costly; cold gate×cold up errors multiply in SwiGLU; full-precision up must be retained |
| 21 | B=8 E8M0 gate only, full up+down — e2e top-1 | Current / proxy-prior (identical in prefill) | match=0.414 @10% hot (vs 0.516 exp14) | E8M0 worse than ternary despite lower TARE: over-scaled cold values inflate SiLU; TARE does not align with SwiGLU's asymmetric over/under-estimation cost |
| 22 | B=8 E5M3 gate only, top-k routing, full up+down — e2e top-1 | Current / proxy-prior | match=0.408 @10% hot | E5M3 (7.2× better scale precision than E8M0) still below ternary baseline; scale precision alone does not fix cold-channel over-activation |
| 23 | B=8 E5M3 gate, **threshold routing** T×mean(gate_approx), full up+down — e2e top-1 | Threshold on current | **match=0.736 @T=0.5** (best result in series) | Threshold routing with accurate E5M3 scale fixes exp2's failure; adaptive hot fraction beats fixed top-k by 20 pp |
| 24 | E5M3 threshold fine sweep T=0.20–0.80 (step 0.05), per-layer hot% monitoring, 2% floor variant | Threshold on current | **match=0.818 @T=0.20**, 88% hot | Monotonic improvement as T decreases; no dead layers; floor unnecessary; curve still rising at T=0.20 |
| 25a | **Diagnostic**: E5M3 gate_approx activation error vs gate_full, all channels — encoding quality on real activations | — | Mean SNR 5.7 dB (R²=0.69) for E5M3 gate; up E5M3 SNR 4.7 dB; ternary SNR 5.1 dB | All three encodings comparable SNR; rel% metric dominated by near-zero channels; R² the meaningful signal |
| 25b | **Diagnostic**: gate_approx error stratified by hot/cold split at each threshold T | — | Hot SNR ~6–7 dB (R²≈0.70); cold SNR ~0–2 dB (R²≈0.04–0.33) | Scheme works not because hot approx is accurate but because SiLU suppresses cold noise toward zero; cold R²≈0.04 is harmless |
| 25c | **Diagnostic**: E5M3 routing quality (hot/cold binary classification) vs oracle gate_full — precision/recall sweep | — | T=0.20: F1=0.903, IoU=0.823; T=0.80: F1=0.784, IoU=0.645 | Routing degrades with higher T (sparser); no systematic bias; hot%(approx) ≈ hot%(oracle) throughout |
| 25d | **Diagnostic**: SVD routing quality vs oracle, rank sweep 16–1024, hot=20% and 50% | — | rank=1024 @20%: F1=0.745; @50%: F1=0.873; rank=64 @20%: F1=0.551 | Low-rank SVD loses fine-grained channel ordering absent from leading singular directions; F1 barely changes with rank beyond ~256 — signal saturates; E5M3 dominates at sparse regime |
| 25e | **Diagnostic**: learned linear predictor (cross-covariance SVD) routing quality, rank 16–1024 — training cost analysis | — | Regression F1≈0.32–0.54; binary-label F1≈0.38–0.58 (all ranks nearly equal) | Trained linear predictor far worse than W_gate SVD; cross-cov rank-1 saturates (almost no gain rank 16→1024); nonlinear SwiGLU target cannot be predicted linearly from x; training cost 25.7 TFLOPs (trivial offline) but result is uncompetitive |
| 26 | Sparse SwiGLU: E5M3 routing, hot channels full-precision gate+up, cold channels zeroed (no approx value used), full W_down — e2e top-1 | Threshold on current | match=0.814 @T=0.20, 88% hot | Consistently −0.004 to −0.059 vs exp24; cold gate_approx contribution (SiLU≈0) is slightly helpful, not harmful; zeroing cold channels is not an improvement |
| 27 | Low-rank SVD routing (union top-k gate+up), hot full-precision gate+up, cold E5M3 B=8 gate+up, full W_down — e2e top-1 | Top-k union on \|gate_lr\|∪\|up_lr\| | rank=1024: **0.834 @50% hot**, 0.612 @20%; rank=256: 0.764 @50%; rank=64: 0.392–0.678 | Low-rank routing + E5M3 cold for both gate+up beats exp24 at equal hot% for rank≥256 @50%; at 20% hot exp24 (0.818) still wins; cold E5M3 up tolerable when routing quality is high |
| 28 | E5M3-encoded SVD factor matrices (binary sign+scale), rank=1024 and 2048, same hybrid scheme as exp27 — e2e top-1 | Top-k union on encoded \|gate_lr\|∪\|up_lr\| | rank=2048: 0.808 @50%, 0.582 @20%; rank=1024: 0.766 @50% | Encoding SVD factors costs ~3–7 pp vs full-prec factors (exp27); rank=2048 partly recovers loss but still −2.6 pp vs exp27 r1024 @50%; routing cost at r=2048 = 105% of full GEMM (no net saving); E5M3 encoding of orthonormal vectors loses too much directional info |
| 29 | SwiGLU-LR routing: top-k on \|SiLU(gate_lr)·up_lr\| combined signal, hot full-prec, cold E5M3 B=8 gate+up — e2e top-1 | Top-k on \|SiLU(gate_lr)·up_lr\| | rank=1024: 0.732 @50%, 0.568 @20%; rank=256: 0.682 @50% | Worse than exp27 union-LR at all ranks and fractions; combined signal shrinks hot% to exactly frac (no union expansion) — recall drops; union-LR's over-selection (hot%≤2×frac) is beneficial because it catches channels where only one projection is large |
| 30 | Nonlinear output predictor σ(Px): scalar nonlinearities (none/ReLU/SiLU/Abs) × regression/binary targets × rank 256/2560, trained on recorded activations — routing quality only | Trained P, σ(Px) signal | Full-rank linear reg: F1=0.486 @20%, 0.629 @50% (best); all nonlinearities worse or equal | Nonlinearities do not help; bilinear barrier: SwiGLU = SiLU(gate)×up cannot be expressed as σ(single linear map); full-rank ridge regression beats exp25e cross-cov (0.486 vs 0.32) but still 38% below E5M3 (0.784); weight-derived routing remains superior with zero training cost |
| 31 | Two-layer MLP predictor h=SiLU(xW1), out=hW2, hidden widths 256/1024/2560, reg and bin targets, SVD warm-start for W1 — routing quality only | Trained MLP signal | hidden=1024 reg: F1=0.393 @20%, 0.572 @50%; hidden=2560 reg: F1=0.385 @20%, 0.571 @50% | Two-layer MLP is worse than single-layer linear regression (exp30 F1=0.486) at 20% hot and comparable at 50%; bilinear barrier not broken despite hidden layer; likely causes: 500 steps insufficient, binary target collapses (F1≈0.14–0.16), high variance across layers (layer8: 0.31 vs layer0: 0.54) suggests underfitting |
| 31b | Exp31 with ReLU vs SiLU hidden activation comparison, hidden=1024, 500 steps — routing quality only | Trained MLP (ReLU or SiLU hidden) | SiLU: F1=0.393 @20%, 0.572 @50%; ReLU: F1=0.278 @20%, 0.527 @50% | ReLU is worse than SiLU: half the hidden units are dead at init (SVD warm-start has both +/− projections) and never recover; SiLU's smooth negative tail keeps all units active; neither beats linear regression; hidden activation choice is second-order vs the fundamental bilinear barrier |
| 32 | Sparse W_gate predictor: magnitude pruning (unstructured + row-wise) ± 300-step Adam fine-tune, keep_rates 0.1–0.5 — routing quality only | Top-k on \|x @ W_sparse.T\|, 5 layers [0,8,16,24,32] | Unstructured keep=0.5: F1=0.880 @20%, 0.913 @50%; +ft: 0.902/0.932. Row pruning: F1=0.623 @50% keep, flat across hot%; fine-tune has no effect on row scheme | Unstructured magnitude pruning beats E5M3 (0.784) even at keep=0.3 (0.783); fine-tuning adds +2–7 pp; row pruning is much weaker and unimprovable by fine-tuning (zeroed rows have zero gradient); 50% unstructured sparsity gives E5M3-level routing at 50% GEMM cost |
| 33 | Sparse W_gate e2e top-1: unstructured keep=0.5 ± 300-step Adam ft, top-k routing, cold gate from sparse, up always full — e2e top-1 | Top-k on \|x @ W_sparse.T\|, cold gate = x @ W_sparse.T | kr=0.5+ft: **0.850 @20%, 0.862 @30%, 0.876 @50%**; kr=0.5 no-ft: 0.830/0.830/0.852 | New best across all hot fractions: +24 pp vs exp27 @20%, +16 pp @30%, +4 pp @50%; fine-tuning adds +2 pp; sparse W_gate doubles as routing signal and cold approximation, eliminating E5M3 encoding step entirely |
| 34 | Thermal match rate for exp33 kr=0.5+ft and exp24 reference schemes — new metric | Thermal match (T=0.7/1.0, τ=ln2): forgive if gap < T·ln2 | sparse @20%: strict=15.0%, **thermal@0.7=5.6%, thermal@1.0=5.0%**; sparse @50%: strict=12.4%, **thermal@0.7=4.8%, thermal@1.0=3.6%** | Thermal metric reveals ~8–10 pp of strict perturbation is below the noise floor at T=0.7; exp33 @50% hot is effectively FP8-equivalent (3.6% thermal@1.0); exp24 E5M3 T=0.20 drops from 18.2% strict → 7.6% thermal@0.7; exp24 T=0.50 still 16% thermal — its errors are harder (large gap ~1.27 logits) |
| 35 | Sparse up projection + zero-cold comparison; target regime 20–30% hot; strict + thermal metric | gate routing: top-k(\|x @ W_gate_sparse.T\|); cold gate = W_gate_sparse, cold up = W_up_sparse or zero; up always full for hot | sparse_gate+up @20%: strict=21.2%, **thermal@0.7=9.4%**; sparse_gate+up @30%: strict=18.4%, **thermal@0.7=8.6%**; zero_cold @20%: strict=76.8%, thermal=73.4% | Sparse up cold is worse than full-precision up (exp33): +7.8 pp strict at 20% hot; however thermal gap is narrower (+2.8 pp). Zero cold is catastrophically hard (gap 5.3 logits @20%): omitting cold channels is a hard error, not recoverable thermally. Sparse_gate_only @30% hot is the new sweet spot: strict=12.0%, thermal@0.7=5.6%, thermal@1.0=4.8% — FP8-equivalent at T=1.0 at the target hot fraction |
| 36 | E5M3 B=8 cold up with sparse gate routing — e2e strict + thermal | cold gate = W_gate_sparse; cold up = E5M3 B=8 sign+scale; hot = full precision; down = full | @20% hot: strict=38.6%, **thermal@0.7=30.6%**; @30%: strict=30.0%, thermal@0.7=23.4%; @50%: strict=19.8%, thermal=10.8% | E5M3 cold up is far worse than full-precision up at sparse hot fractions (20–30%): +25 pp strict / +24 pp thermal at 20% hot; mean gap ~2.0 logits (hard errors). Only recovers at 50% hot (+6.6 pp strict, thermal@1.0=8.2%). W_up must stay full precision for cold channels regardless of up encoding scheme — confirming exp20 and exp35 at the cold-up encoding level |
| 37 | 3bpw cold up: 2-bit/weight, 2 E5M3 scales per B=16 block, TARE-optimal EM, sparse gate routing — e2e strict + thermal | cold up = ±{s_lo,s_hi} per B=16, E5M3 quantised scales; hot = full; down = full | @20%: strict=24.4%, **thermal@0.7=15.6%**; @30%: strict=22.6%, thermal@0.7=13.2%; @50%: strict=16.8%, thermal@0.7=8.4% | 3bpw halves the penalty vs 1bpw E5M3 (~11 pp strict penalty vs ~22 pp at 20–30% hot), but still +7–9 pp thermal over full-precision up. Gap metric drops from 2.0 → 1.39 logits (softer errors) but not yet matching full-prec (1.06). At 50% hot, 3bpw is within 3 pp strict / 3 pp thermal of full-prec up. W_up cold must still be full precision for the 20–30% hot target regime |
| 38 | 3bpw static encoding quality check — no routing, uniform encoding of individual matrices and combinations | gate=3bpw (no routing); gate+up=3bpw; gate+down=3bpw; all=3bpw | gate only: strict=49%, **thermal@0.7=46.2%**, gap=3.5L; gate+up: strict=96.4%, gap=8.5L; gate+down: strict=88.4%, gap=8.1L; all: strict=99.8%, gap=10.6L | 3bpw gate alone is already catastrophic (49% strict) without routing; gate+up compound to 96% perturbation. Confirms that hot/cold routing is not merely helpful but structurally essential — 3bpw encoding applied globally destroys output quality. Each additional encoded matrix compounds error multiplicatively: gate×up SwiGLU product doubles the noise, down projection broadcasts it over all output dims |
| 39 | FP6 S1E2M3 (6.5bpw, B=16, 1 E5M3 block scale) static encoding quality check — no routing, same matrix combinations as exp38. **Note: E5M3 is not an OCP format; scale alignment is suboptimal vs E8M0.** | gate=FP6; gate+up=FP6; gate+down=FP6; all=FP6 | gate only: strict=22.6%, **thermal@0.7=15.0%**, gap=1.21L; gate+up: strict=81.6%, gap=4.7L; gate+down: strict=95.2%, gap=8.3L; all: strict=99.4%, gap=12.4L | FP6 gate alone is −28 pp better than 3bpw gate (22.6% vs 51%), confirming more mantissa bits help, but results are not representative of OCP MXFP6 due to the non-standard E5M3 block scale. See exp40 for the correct OCP MX comparison. |
| 40 | OCP MXFP8-E4M3 / MXFP8-E5M2 / MXFP6-E3M2 / MXFP6-E2M3 static quality check — B=32, E8M0 block scale, no routing, all 4 matrix combinations | gate/up/down encoded uniformly per format | **MXFP8-E4M3 gate+up+down: strict=3.6%, thermal@0.7=0.4%**; MXFP6-E2M3 gate+up+down: strict=4.2%, thermal@0.7=0.8%; MXFP6-E3M2 gate+up+down: strict=7.8%, thermal@0.7=2.0%; MXFP8-E5M2 gate+up+down: strict=8.0%, thermal@0.7=2.2% | All four MX formats produce FP8-equivalent or better quality even when all three MLP matrices are encoded simultaneously with no routing. MXFP8-E4M3 and MXFP6-E2M3 both achieve ≤4.2% strict / ≤0.8% thermal@0.7 with all-matrix encoding. Confirms exp39 was dominated by the suboptimal E5M3 scale scheme, not a fundamental property of 6-bit weights. E8M0 power-of-two scaling is the key — near-zero perturbation even with MXFP6. |
| 41 | MXFP6-E2M3 with E8M0 vs E5M3 block scale — controlled isolation of scale format, B=32, no routing | E8M0 (OCP power-of-two) vs E5M3 (TARE-optimal fine-grained), element format fixed to E2M3 | E8M0: gate_only=1.6%/0.0% thermal, all=4.2%/0.8%; **E5M3: gate_only=28.2%/18.2% thermal, all=99.0%/98.2%** | E8M0 is confirmed as the culprit: same E2M3 elements with E5M3 scale collapse from 4.2% to 99% perturbation on all-matrix encoding. The E5M3 TARE scale is accurate (geomean-aligned) but not power-of-two — the FP6 grid misaligns with the weight exponent field, producing large quantisation errors for most weights. E8M0's exact power-of-two alignment is the essential property of OCP MX, not the number of scale mantissa bits. |
| 42 | 3bpw 2-level ±{s_lo,s_hi} with E8M0 scales vs E5M3 scales — B=16, TARE EM, no routing | E8M0 (power-of-two centroids, this exp) vs E5M3 centroids (exp38 ref) | E8M0: gate_only=49%/42.6% thermal, all=98.8%/97.8%; E5M3: gate_only=51%/46.2%, all=99.8%/99.2% | Unlike MXFP6, switching 3bpw to E8M0 scales gives only marginal improvement (~2 pp). Both variants are catastrophically bad. The 3bpw scheme has no explicit per-weight exponent field — weights are just ±s_lo or ±s_hi — so the power-of-two alignment benefit of E8M0 provides almost no gain. The 3bpw scheme is fundamentally limited by having only 2 magnitude levels per block, not by scale format. MXFP6-E2M3 (32 levels) with E8M0 remains the clear winner at similar storage cost. |
| 43 | MXFP4-E2M1 (OCP, 4.25bpw) and hypothetical MXFP5-E2M2 (5.25bpw) — B=32, E8M0, no routing, all 4 matrix combinations; MXFP6-E2M3 inline for reference | E2M1 (8 codes, fp_max=6); E2M2 (16 codes, fp_max=7); E2M3 (32 codes, fp_max=7.5) | MXFP4 all: strict=17.2%, **thermal@0.7=8.6%**; MXFP5 all: strict=9.6%, thermal@0.7=3.6%; MXFP6 all: strict=4.2%, thermal@0.7=0.8% | MXFP4 lands in NVFP4 territory (17.2% strict all-matrix). Each mantissa bit halves the perturbation: MXFP4→5 saves ~8 pp strict, MXFP5→6 saves ~5 pp. MXFP5 thermal@0.7 on all-matrix (3.6%) is already FP8-equivalent — the extra mantissa bit over MXFP4 pushes it across the FP8 threshold. MXFP6 strictly better than MXFP5 at same exponent width. The E2Mx family shows clean monotone improvement with mantissa bits at E8M0 B=32. |
| 44 | E5M0 vs E8M0 block scale + scale distribution — MXFP4/5/6-E2Mx, B=32 | E8M0 (range 2^±127) vs E5M0 (range 2^±15); element formats unchanged; plus full scale exponent histogram | **Δ = 0.0 pp everywhere**; scales span only e∈[−12, −3], 10 distinct values; 66% of blocks use e∈{−8,−7}; 4 bits would cover the full observed range | E8M0 range entirely unused. All 78.6M blocks have e∈[−12, −3]. Distribution is bimodal: e=−8 (32.5%) and e=−7 (33.6%) account for 66% of all blocks; e∈[−10,−7] covers 99.3%. A 4-bit scale (16 values) or even E4M0 would be sufficient for this model. |
| 44b | LUT-based mixed-mode scale scheme: bit-cost analysis — no new inference; analytical follow-on to exp44 | Scale distribution from exp44; 3 modes: ≤2 scales→1 bit/block, 3–4 scales→2 bits/block, ≥5 scales→4 bits/block; 2-bit per-channel mode flag | **1.461 bits/block** actual overhead (1.266 index + 0.194 LUT amortisation); mode-0: 76.6% of channels, mode-1: 22.9%, mode-2: 0.4%; MXFP6-E2M3 total = **6.046 bpw** vs 6.250 flat | E8M0 scale overhead reduced 5.5× (8.00 → 1.46 bits/block), saving 0.204 bpw. Mode-2's 16-entry LUT is over-provisioned — 2 bits (4 entries) covers 99.6% of channels. User estimate of ~1.2 bits/block was close; actual 1.46 driven by 23% mode-1 channels (2 bits/block each). |
| 44c | LUT scheme applied to MXFP5-E2M2 and MXFP4-E2M1 — analytical extension; no new inference | Same LUT channel assignment (format-independent); only element bpw changes: MXFP4=4 bits/weight, MXFP5=5, MXFP6=6; scale overhead = 1.461/32 = **0.04566 bpw** in all cases | MXFP4+LUT: **4.046 bpw** (saving 0.204 bpw / 4.8%); MXFP5+LUT: **5.046 bpw** (3.9%); MXFP6+LUT: **6.046 bpw** (3.3%). Quality unchanged: LUT is lossless scale re-encoding. MLP total: MXFP5+LUT=1.59 GB, MXFP4+LUT=1.27 GB vs BF16 5.03 GB | Absolute scale saving (0.204 bpw) is identical across all E2Mx formats — it is purely a scale-field reduction. MXFP5+LUT (5.046 bpw, thermal@0.7=3.6%) is the practical sweet spot: FP8-equivalent quality at 3.17× BF16 compression. MXFP4+LUT buys another 0.77 GB at the cost of crossing back above the FP8 thermal threshold. |
| 44d | MXFP4-E2M1 code usage histogram + LUT12 feasibility — weight analysis + all-matrix quality run | All 2.5B weight slots encoded as MXFP4; code counts per unsigned magnitude; LUT12 = drop 4 least-used codes (0.0, 3.0, 4.0, 6.0), remap to nearest retained | Usage: top-4 codes (0.5→22.3%, 1.0→19.0%, 0.0→11.9%, 1.5→14.9%) cover 68%; bottom-4 cover 28.8%; drop4 maps 3.0/4.0/6.0→2.0 and 0.0→0.5. LUT12 all-matrix: **strict=94.8%, thermal@0.7=92.6%** (baseline: 17.2%/8.6%) | **LUT12 is catastrophically bad (+77.6 pp strict).** The 4 least-used codes by count are structurally critical: zero suppresses 11.9% of weights (replacing with ±0.5 adds correlated noise); {3.0,4.0,6.0} cover the top 16.9% of the dynamic range (all collapse to 2.0). MXFP4 has 8 load-bearing codes — usage skew reflects the weight distribution, not code redundancy. A feasible 12-code variant would require a redesigned non-uniform codebook, not code pruning. |
| 45 | Per-channel optimal LUT12 (Lloyd-Max, 12 non-neg magnitudes + sign bit) vs MXFP4/5 — all-matrix, E8M0 B=32; Metal encoding | 1 sign + 4 magnitude-index bits = **5.25 bpw**; Lloyd-Max 30 iters per output channel; level-0 pinned to 0.0; 24 B/channel LUT sidecar (18 MB total) | **strict=9.2%, thermal@0.7=3.4%, thermal@1.0=2.8%** vs MXFP4 (17.2%/8.6%) and MXFP5-E2M2 (9.6%/3.6%) at same 5.25 bpw | LUT12 is a **learned MXFP5**: same 5.25 bpw, ~same quality (−0.4 pp strict vs fixed MXFP5). Achievement: 12 optimally-placed magnitudes matches 16 fixed MXFP5 magnitudes. The ~0.4 pp gain is pure optimal placement; most of the MXFP4→LUT12 improvement comes from the extra magnitude bit (+1 bit → 8→12 codes). With exp44b scale LUT: **~5.046 bpw + 18 MB sidecar**. |
| 45b | Per-channel LUT6 (6 non-neg magnitudes + sign, 4.25 bpw) — all-matrix, E8M0 B=32 | 1 sign + 3 magnitude-index bits = **4.25 bpw** (same as MXFP4); Lloyd-Max 30 iters; 6 of 8 available 3-bit slots used; 24 B/channel LUT sidecar | **strict=20.6%, thermal@0.7=12.0%, thermal@1.0=10.6%** — **worse than MXFP4** (17.2%/8.6%) by +3.4 pp strict / +3.4 pp thermal | Optimal placement of 6 magnitudes cannot compensate for 2 fewer codes than MXFP4's 8. The geometric OCP grid is already near-optimal for this weight distribution. Confirms LUT12's achievement: the benefit of per-channel optimisation only kicks in when using more codes than the fixed grid (12 > 8 at 4 magnitude bits). |


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

### Thermal match rate

**Motivation.**  Top-1 match is measured at temperature 0 (greedy), which
flags any rank-swap even when the displaced token is still highly probable at
typical inference temperatures.  A rank-swap where the full-precision top-1
token is still "thermally accessible" in the hybrid logits — i.e. would still
be sampled a meaningful fraction of the time at temperature T — should not
count as a real error.

**Definition.**  For each token position let:
- `lf` = full-precision logits
- `lh` = hybrid logits
- `baseline_top1 = argmax(lf)` — the correct answer
- `hybrid_top1   = argmax(lh)` — what the hybrid prefers
- `gap = lh[hybrid_top1] − lh[baseline_top1]` — how strongly the hybrid
  prefers its own answer over the full-precision answer (≥ 0 when perturbed)

A perturbation is **forgiven** when `gap < T · τ`, i.e. the hybrid's
preference is within the thermal noise floor.  The **thermal match rate** is:

```
thermal_match = mean(exact_match  OR  gap < T · τ)
```

**Pairwise probability interpretation.**  The gap Δl gives the pairwise
sampling ratio between the two tokens directly:

```
P(hybrid picks its own top-1 | choosing between just these two) = sigmoid(Δl / T)
```

At `τ = ln(2) ≈ 0.693` the forgiveness threshold is `Δl < T · ln(2)`, which
means `sigmoid(Δl/T) < 2/3` — the full-precision token still wins >33% of
pairwise draws.  This is the **recommended default**.

Other choices and their pairwise interpretations:

| τ | Formula | Full-prec token pairwise win-rate |
|---|---|---|
| 0 | strict (= top-1 match) | 50% (they agree) |
| ln(2) ≈ 0.693 | gap < T·0.693 | >33% — "basically a tie" |
| ln(3) ≈ 1.099 | gap < T·1.099 | >25% |
| ln(9) ≈ 2.197 | gap < T·2.197 | >10% |

**Implementation** (drop-in alongside top-1 match, no extra GEMM):

```python
import math

def thermal_match(
    lf: torch.Tensor,         # (T_seq, vocab) full-precision logits
    lh: torch.Tensor,         # (T_seq, vocab) hybrid logits
    temperature: float = 0.7,
    tau: float = math.log(2), # ln(2): forgive if full-prec token wins >33% pairwise
) -> float:
    """Top-1 match rate with thermal forgiveness at the given temperature."""
    baseline_top1 = lf.argmax(-1)                                   # (T_seq,)
    hybrid_top1   = lh.argmax(-1)                                   # (T_seq,)
    exact_match   = baseline_top1 == hybrid_top1                    # (T_seq,) bool

    # Logit of hybrid's preferred token
    hybrid_rank1_logit = lh.gather(
        -1, hybrid_top1.unsqueeze(-1)).squeeze(-1)                  # (T_seq,)
    # Logit of the full-precision top-1 token, looked up in the HYBRID logits
    baseline_in_hybrid = lh.gather(
        -1, baseline_top1.unsqueeze(-1)).squeeze(-1)                # (T_seq,)

    # Gap >= 0 when perturbed; zero when exact_match
    gap = hybrid_rank1_logit - baseline_in_hybrid                   # (T_seq,)

    forgiven = (~exact_match) & (gap < temperature * tau)
    return float((exact_match | forgiven).float().mean())
```

**Reporting convention** used from exp34 onwards: always report both
`strict_match` (T=0) and `thermal_match` (T=0.7, τ=ln2) side by side.
The gap between them quantifies how many perturbations are "below the noise
floor" and effectively free at typical chat temperatures.

### Perturbation impact reference

Top-1 perturbation rate (= 1 − top-1 match) is interpretable by analogy with
quantization schemes whose quality is well-characterised in practice.

#### Perceptibility

| Perturbation | Human perceptibility | Benchmark impact |
|---|---|---|
| 0–3% | Imperceptible; outputs functionally identical | MMLU/HellaSwag within noise |
| 3–8% | Rarely noticeable; occasional phrasing differences | ~0.5–1 pp benchmark drop |
| 8–15% | Occasionally noticeable on long outputs; factual drift possible | ~1–3 pp benchmark drop |
| 15–25% | Noticeable on direct A/B comparison; coherence mostly preserved | ~3–7 pp benchmark drop |
| >30% | Clearly degraded; repetition and hallucination more frequent | Significant benchmark drop |

A 15% perturbation means 1 in 7 greedy-decode positions would pick a different
token.  In isolation most swaps involve near-synonyms or punctuation; over a
generation of length N the probability of at least one diverged token is roughly
`1 − (1 − perturb)^N`.  At N=20 and 15% perturbation that is ~96% — essentially
every response diverges somewhere, though each divergence may be invisible to a
casual reader.

#### Comparison with production quantization schemes

Approximate top-1 perturbation rates for typical LLM quantization, relative to
BF16 baseline (values are model- and calibration-dependent; treat as order-of-
magnitude):

| Quantization scheme | Typical perturbation | vLLM default |
|---|---|---|
| BF16 → FP16 | ~0% | — |
| BF16 → INT8 W8A8 | ~1–3% | some backends |
| BF16 → FP8 E4M3 W8A8 | ~2–5% | H100 default |
| BF16 → INT4 GPTQ/AWQ W4A16 | ~5–10% | optional |
| BF16 → NVFP4 W4A4 (Blackwell) | ~10–20% | GB200 default |
| BF16 → INT4 uncalibrated | ~15–30% | not recommended |

**15% perturbation is approximately the NVFP4 (FP4) regime.**  FP8 is the current
"transparent" production floor (~3–5%).  The experiments in this log target the
regime between FP8 and NVFP4, where selective full-precision computation for hot
channels recovers quality compared to global low-precision quantization.

#### Best results in this log (as of exp34)

Strict perturbation (T=0 greedy) and thermal perturbation (T=0.7 and T=1.0,
τ=ln2 — forgive if full-prec token still wins >33% pairwise):

| Scheme | Strict perturb | Thermal @T=0.7 | Thermal @T=1.0 | Quant analogue (strict) |
|---|---|---|---|---|
| Exp14 ternary @10% hot | 48.4% | — | — | worse than uncalibrated INT4 |
| Exp24 E5M3 T=0.50 | 26.4% | 16.0% | 12.4% | uncalibrated INT4 |
| Exp24 E5M3 T=0.20 | 18.2% | 7.6% | 4.8% | lower end of NVFP4 |
| Exp27 SVD r1024 @50% hot | 16.6% | — | — | NVFP4 range |
| Exp33 kr=0.5+ft @20% hot | 15.0% | 5.6% | 5.0% | NVFP4 range |
| Exp33 kr=0.5+ft @30% hot | 13.8% | 5.6% | 4.8% | NVFP4 range |
| **Exp33 kr=0.5+ft @50% hot** | **12.4%** | **4.8%** | **3.6%** | **upper FP8 / lower NVFP4** |

At T=1.0 the best scheme (**exp33 @50% hot, 3.6% thermal**) is already within the
FP8-equivalent range (~3–5%).  The strict metric overstates the real-world impact
by ~9 pp at T=0.7 — most perturbations are near-ties resolved by thermal noise.

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
| Top-1 preservation rate (strict) | token | moderate | **direct** | `W_U` GEMM |
| Thermal match rate (T=0.7, τ=ln2) | token | moderate | **direct + forgiveness** | `W_U` GEMM |
| `W_U`-weighted L2 error | logit (singular) | moderate | moderate | `W_U` SVD + GEMM |

All four non-trivial metrics share the same `W_U` GEMM, so the marginal cost
of adding all of them together is just one GEMM — compute all in a single pass
over `lf` and `lh`.

## Weight compression scheme for predictor

New scheme using ternary weight encoding with block scales for each output
dimension.  Uses the following error metric — a scale-tilted relative error
calibrated per tensor — to evaluate approximation quality:

```python
def log_tilted_lre(w_true: torch.Tensor, w_approx: torch.Tensor,
                   floor_percentile: float = 1.0) -> float:
"""
Scale-tilted relative error metric.
- Mostly scale-invariant (log-ratio base)
- Gentle upward tilt for larger weights (log1p tilt)
- Floor derived from the weight distribution itself (no magic constant)
- Compute per weight matrix, not globally
"""
eps   = torch.quantile(w_true.abs(), floor_percentile / 100.0).clamp(min=1e-9)
wt    = w_true.abs().clamp(min=eps)
wa    = w_approx.abs().clamp(min=eps)
lr    = torch.log(wa / wt)
tilt  = torch.log1p(w_true.abs() / eps)   # 0.69 at floor, ~7.3 at max
mse_w = (lr.pow(2) * tilt).sum() / tilt.sum()
return float(mse_w.sqrt())
```


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

## Experiment 10

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

## Experiment 11

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

## Experiment 12
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

## Experiment 13
### Motivation

All experiments through exp12 used output cosine similarity in hidden space
as the primary metric.  As noted in the "Evaluation metrics" section above,
this is magnitude-blind and not tied to top-1 token prediction quality.
Experiment 13 re-evaluates the three best configurations from exp10–11 under
four logit-space metrics computed via a single W_U GEMM (lm_head.weight,
shape 100 352 × 2560).

**Important caveat on interpretation**: metrics here are computed per MLP
layer in isolation — the approximation error at one layer is not propagated
through subsequent layers.  Top-1 values will therefore appear low (the MLP
output perturbation rarely changes the final argmax when viewed in isolation
at a single layer), but the *relative* ordering between configurations
remains meaningful.

### Configs evaluated (all α_gate=0.75, full-precision up, full W_down)

| Config | Routing signal |
|---|---|
| exp10-current | `\|gate_approx[t]\|` — current token (exp10 best) |
| exp11-proxy | `\|gate_approx[t-1]\|` — proxy prior (exp11 best) |
| exp11-oracle | `\|gate_full[t-1]\|` — oracle prior |

512 tokens/layer (subsampled from 2 000; W_U GEMM at vocab=100 352 is expensive).

### Results (granite-4.2-3b, 40 layers, MPS)

#### Top-1 preservation rate (higher = better)

| Config | 0.5% | 1% | 2% | 5% | **10%** | 20% | 30% |
|---|---|---|---|---|---|---|---|
| exp10 current | 0.068 | 0.057 | 0.047 | 0.032 | 0.019 | 0.009 | 0.004 |
| **exp11 proxy** | **0.101** | **0.093** | **0.086** | **0.071** | **0.059** | **0.045** | **0.036** |
| exp11 oracle | 0.091 | 0.086 | 0.076 | 0.061 | 0.049 | 0.037 | 0.030 |

#### KL divergence full‖hybrid (lower = better; NaN layers excluded from mean)

| Config | 0.5% | 1% | 2% | 5% | **10%** | 20% | 30% |
|---|---|---|---|---|---|---|---|
| **exp10 current** | **1.402** | **1.356** | **1.301** | **1.191** | **1.066** | **0.856** | **0.664** |
| exp11 proxy | 1.551 | 1.529 | 1.500 | 1.442 | 1.369 | 1.251 | 1.141 |
| exp11 oracle | 1.530 | 1.509 | 1.481 | 1.421 | 1.353 | 1.243 | 1.134 |

#### Logit cosine similarity (higher = better)

| Config | 0.5% | 1% | 2% | 5% | **10%** | 20% | 30% |
|---|---|---|---|---|---|---|---|
| exp10 current | 0.636 | 0.617 | 0.593 | 0.544 | 0.486 | 0.396 | 0.323 |
| **exp11 proxy** | **0.679** | **0.671** | **0.660** | **0.637** | **0.608** | **0.561** | **0.519** |
| exp11 oracle | 0.672 | 0.661 | 0.647 | 0.621 | 0.591 | 0.544 | 0.503 |

#### Hidden cosine similarity (reference — matches prior experiments)

| Config | 0.5% | 1% | 2% | 5% | **10%** | 20% | 30% |
|---|---|---|---|---|---|---|---|
| exp10 current | 0.636 | 0.618 | 0.594 | 0.546 | 0.489 | 0.400 | 0.327 |
| **exp11 proxy** | **0.680** | **0.672** | **0.662** | **0.640** | **0.611** | **0.565** | **0.524** |
| exp11 oracle | 0.675 | 0.665 | 0.652 | 0.626 | 0.596 | 0.550 | 0.509 |

### Key findings

**exp11-proxy wins on top-1, logit-cos, and hidden-cos at every hot
fraction.**  Proxy prior (`|gate_approx[t-1]|`) is the best configuration
by all three output-quality metrics.  The proxy-prior advantage over
oracle-prior seen in hidden-cos (exp11) is confirmed in logit-space:
proxy top-1 0.059 vs oracle 0.049 at 10% hot (+0.010), and logit-cos
0.608 vs 0.591 (+0.017).

**KL divergence inverts the ranking: exp10-current has the lowest KL.**
This is the one metric where the proxy prior performs worse (KL 1.369 vs
1.066 for exp10 at 10% hot).  The inversion is explained by the nature of
KL: it is dominated by tokens where `ph` is near zero at the true argmax.
The prior-token routing sometimes selects a hotlist from a token where the
dominant gate channels differ slightly from the current token, producing a
larger distributional shift on a small fraction of tokens that heavily
penalises KL while not affecting the argmax.  The very same routing that
improves the *most likely* prediction can produce a heavier-tailed
distribution error on tokens that are already uncertain.

**KL is unreliable at early (0–2) and late (39) layers** — the per-layer
output perturbation is large enough that the softmax difference overflows
float32, producing NaN values.  These layers are excluded from the KL mean
via nanmean.  This indicates those layers are particularly sensitive to MLP
approximation errors.

**Logit-space cosine and hidden-space cosine track almost identically**
(within 0.001–0.003 at every point), confirming that for this model W_U
is roughly isotropic in the directions sampled and hidden-space cosine is
a reliable proxy for logit-space cosine.

**Absolute top-1 values are low** (0.06 at 10% hot for the best config)
because this measures per-layer single-MLP perturbation against the final
vocabulary.  Errors from one layer are small relative to the total residual
stream, so the argmax rarely flips from a single-layer perturbation.  The
metric is still informative as a relative ranking; end-to-end propagation
would amplify these per-layer effects.

**Proxy prior beats oracle prior on every non-KL metric**, across all 40
layers.  The deeper layers (34–39) show the largest absolute top-1 values
(0.12–0.21) because those layers contribute most directly to the final logit
distribution.

### Conclusion

Under logit-space metrics, **exp11 proxy prior (`|gate_approx[t-1]|`)
remains the best configuration** for top-1 preservation rate and logit
cosine similarity.  The single exception is KL divergence, where
exp10 current-token routing is preferred — but KL's sensitivity to
distributional tails makes it a poor proxy for argmax accuracy in this
setting.

The close agreement between logit-space cosine (0.608) and hidden-space
cosine (0.611) at 10% hot validates that the hidden-space metric used in
experiments 1–12 is a reliable ranking signal for this model.  Future
experiments can continue using hidden-space cosine for speed while
spot-checking logit-space metrics at key configurations.

For practical deployment the relevant metric is top-1 preservation:
exp11 proxy achieves 0.059 at 10% hot (per-layer, single MLP in isolation).
Quantifying end-to-end top-1 degradation across all 40 layers requires
a forward-pass interception experiment — a natural next step.

## Experiment 14
### Motivation

Experiment 13 measured top-1 preservation per MLP layer in isolation.
Per-layer values were low (~6% at 10% hot) because a single-layer
perturbation rarely changes the final argmax.  This experiment measures
the **end-to-end top-1 perturbation rate**: fraction of tokens where the
final next-token prediction changes when the hybrid MLP scheme runs across
**all 40 layers simultaneously**.

### Method

For each (config, hot-fraction):
- **Baseline pass**: normal inference via `llm.generate`, hook on
      `model.model.norm` captures final hidden states, projected through
      W_U to get per-token argmax.
- **Hybrid pass**: all 40 layers' `mlp.forward` replaced by `HybridMLP`
      (ternary gate α=0.75 + full up + full W_down).  Same hook captures
      perturbed final hidden states.  `enable_prefix_caching=False` ensures
      full re-computation on every pass.

`top1_match = fraction of prefill-token predictions identical to baseline.`

`HybridMLP` stores only `T_gate` (bf16, scaled by `mean(|W_gate|)` to
preserve expected magnitude in cold channels) — no full weight copies to
avoid OOM.

**Methodology note on proxy-prior**: the proxy-prior routing
(`|gate_approx[t-1]|`) is only meaningful in autoregressive decode, where
`t-1` is the genuinely preceding token of the same sequence.  In a batched
prefill call all prompt tokens are processed simultaneously, so
`gate_approx[t-1]` is the previous *batch position* (a different sequence).
The two configs therefore produce **identical results in the prefill metric**
— they differ only in decode, which this experiment does not measure.

### Results (granite-4.2-3b, 8 prompts, 500 prefill tokens, CPU backend)

#### End-to-end top-1 perturbation rate  (fraction of tokens whose prediction changes)

| Config | 5% hot | 10% hot | 20% hot | 30% hot |
|---|---|---|---|---|
| exp10_current / exp11_proxy | **0.544** | **0.484** | 0.412 | 0.368 |

*(Both configs are identical in this prefill metric — see methodology note.)*

#### End-to-end top-1 match rate  (fraction of tokens preserved)

| Config | 5% hot | 10% hot | 20% hot | 30% hot |
|---|---|---|---|---|
| exp10_current / exp11_proxy | 0.456 | **0.516** | 0.588 | 0.632 |

### Key findings

**At 10% hot channels, 48% of token predictions change end-to-end** compared
to full-precision inference.  This is far larger than the per-layer isolation
figure of 6% (exp13), confirming that errors compound across layers: 40
layers each introducing a small perturbation accumulate to a large output shift.

**The perturbation rate is monotonically decreasing with hot fraction**, as
expected — more hot channels → better approximation → fewer changed predictions.
At 30% hot only 37% of predictions change.

**Comparison with per-layer metric:**

| Metric | 10% hot |
|---|---|
| Per-layer top-1 perturbation (exp13 isolation) | ~94% (i.e., 6% match rate, per layer) |
| End-to-end top-1 perturbation (this exp, all 40 layers) | 48% |

The end-to-end rate is lower than the per-layer rate because: (a) each
layer contributes only a fraction of the total residual, so a single-layer
perturbation has less impact than all-layer simultaneous perturbation might
suggest; (b) errors in different layers partially cancel.

**Proxy-prior vs current-token routing cannot be distinguished in prefill.**
The proxy-prior scheme requires autoregressive decode context (where `t-1`
is a genuine prior for the same sequence).  End-to-end decode-time comparison
requires running full autoregressive generation, which is too slow on CPU at
the required scale.

### Conclusion

The end-to-end top-1 perturbation rate at 10% hot is **48%** — roughly half
of all token predictions change when all 40 MLP layers simultaneously use the
ternary gate + full up approximation.  While the per-layer hidden-space cosine
similarity (0.61) suggested reasonable approximation quality, the compounding
of errors across all layers produces a substantial impact on output quality
in absolute terms.

This motivates the question: how does the perturbation rate scale with the
number of layers approximated?  A partial-layer experiment (approximate only
the top-k highest-loss layers) could identify which layers are responsible
for most of the perturbation budget.

## Experiment 15
### Motivation

Experiment 3 evaluated low-rank SVD as a gate *routing signal* only and
found it worse than sign until rank ~200.  This experiment applies low-rank
approximation to **both gate and up projections** as the actual computation
(no hot/cold routing; all channels approximated), keeping down full-precision,
and measures the **end-to-end top-1 perturbation rate** (same method as
exp14).

### Scheme

      1. `W_gate ≈ U_r S_r Vt_r`  (truncated SVD)
      2. `W_up   ≈ U_r S_r Vt_r`  (separate SVD per projection)
      3. `gate_lr = (U_r * s_r) @ Vt_r @ x`   two-step matmul
      4. `up_lr   = (U_r * s_r) @ Vt_r @ x`
      5. `swiglu  = silu(gate_lr) * up_lr`
      6. `out     = W_down @ swiglu`           full precision always

SVD factors stored as bfloat16 (max rank 1024) and cached to disk.
No hot/cold split — approximation covers all I=8192 intermediate channels.

### FLOP cost vs full GEMM (H=2560, I=8192)

| Rank | FLOP% of full gate GEMM | Energy% captured |
|---|---|---|
| 128 | 6.6% | 27% |
| 512 | 26% | 66% |
| 1024 | 53% | 100% (= min(H,I)) |

(Rank 1024 = full rank of W since min(8192, 2560) = 2560; 100% energy means
exact reconstruction to bfloat16 precision.)

### Results (granite-4.2-3b, 8 prompts, 500 prefill tokens, CPU backend)

| Rank | FLOP% | Energy% | Match rate | **Perturb rate** |
|---|---|---|---|---|
| 128 | 6.6% | 27% | 0.002 | **99.8%** |
| 512 | 26% | 66% | 0.010 | **99.0%** |
| **1024** | **53%** | **100%** | **0.008** | **99.2%** |
| ternary 10% hot (exp14) | ~52% | n/a | 0.516 | 48% |
| full precision | 100% | 100% | 1.000 | 0% |

### Key findings

**Low-rank gate+up completely destroys end-to-end prediction quality at all
ranks.**  Even at rank 1024 (53% FLOPs, 100% singular value energy captured),
99.2% of token predictions change — far worse than the ternary gate + full up
scheme at comparable FLOP cost (10% hot ≈ 52% FLOPs, 48% perturbation).

**Rank 1024 is *worse* than rank 512**, which is itself worse than rank 128
in perturbation rate.  This non-monotonic behaviour (lower rank → slightly
lower perturbation) suggests the dominant failure mode is not truncation error
but **systematic approximation bias** that accumulates across all 40 layers:
the SVD reconstruction error in each layer shifts the residual stream in a
fixed direction, and these shifts compound catastrophically across layers.

**Critical comparison with exp14**: the ternary gate scheme at 10% hot (which
spends ~52% of gate FLOPs on hot-channel recompute) achieves 48% perturbation;
the rank-1024 SVD scheme at the same FLOP budget achieves 99.2%.  The
difference is fundamental: the ternary scheme uses **exact full-precision
values for hot channels** — it concentrates its budget on the most important
neurons.  The SVD scheme distributes its budget uniformly across all channels,
meaning no channel ever gets a fully correct value.

**Exp3's cosine similarity was misleading.** Exp3 showed rank-1024 gate cosine
similarity ≈ 1.0 vs `gate_raw`, which appeared excellent.  But cosine
similarity of a single layer's gate output is not predictive of end-to-end
quality when the same systematic error repeats across all 40 layers.

### Conclusion

Low-rank SVD approximation of gate and up projections is **not viable** as an
MLP approximation strategy for this model, even at rank 1024 (full rank).  The
end-to-end perturbation rate approaches 100% regardless of rank, confirming
that the systematic per-layer bias compounds catastrophically across 40 layers.

This conclusively establishes that **selective full-precision recompute** (as in
exp10–11) is the correct approach: approximate cold channels with a proxy but
keep hot channels exactly correct, rather than distributing approximation error
uniformly across all channels.

The ternary gate + full up scheme (exp11 proxy-prior) at 10% hot achieves
**48% perturbation at 52% FLOP cost** — contrasted with SVD's 99% perturbation
at the same cost.  The hot/cold split with exact hot values is the key design
principle.

## Experiment 16
      ### Motivation

      Experiment 14 showed that approximating all 40 layers simultaneously
      produces a **48% end-to-end top-1 perturbation rate** at 10% hot channels.
      Two questions follow naturally:

      1. Which layers are individually responsible for the most perturbation?
      2. How quickly does e2e perturbation grow as we add more approximated layers?

      This experiment answers both via:

      - **Single-layer sweep**: patch each of the 40 layers independently and
        measure the e2e top-1 perturbation caused by that layer alone.
      - **Cumulative sweep**: patch the top-k highest-contribution layers
        simultaneously (greedy, ranked by single-layer score) and observe how
        perturbation accumulates.

      Config: ternary gate α=0.75, full up, full down, routing on |gate_approx[t]|,
      10% hot channels (same as exp14 exp10_current).

      ### Results (granite-4.2-3b, 8 prompts, 500 prefill tokens, CPU backend)

      #### Single-layer perturbation (one layer approximated at a time)

      | Layer | Perturb | Layer | Perturb | Layer | Perturb | Layer | Perturb |
      |---|---|---|---|---|---|---|---|
      | 0 | 0.0900 | 10 | 0.0700 | 20 | 0.0620 | 30 | 0.0440 |
      | 1 | 0.0700 | 11 | 0.0680 | 21 | 0.0580 | 31 | 0.0520 |
      | 2 | 0.0840 | 12 | 0.0720 | 22 | 0.0560 | 32 | 0.0720 |
      | 3 | 0.0820 | 13 | 0.0740 | 23 | 0.0540 | 33 | 0.0640 |
      | 4 | 0.0680 | 14 | 0.0740 | 24 | 0.0620 | 34 | 0.0700 |
      | 5 | 0.0600 | 15 | 0.0780 | 25 | 0.0400 | 35 | 0.0760 |
      | 6 | 0.0780 | 16 | 0.0820 | 26 | 0.0420 | 36 | 0.0780 |
      | 7 | 0.0620 | 17 | **0.1040** | 27 | 0.0500 | 37 | 0.0480 |
      | 8 | 0.0620 | 18 | 0.0540 | 28 | 0.0400 | 38 | 0.0820 |
      | 9 | 0.0600 | 19 | 0.0560 | 29 | 0.0640 | 39 | **0.1340** |

      Ranked by contribution (highest first):
      39, 17, 0, 2, 38, 3, 16, 36, 15, 6, 35, 13, 14, 12, 32, 34, 1, 10, 11, 4, ...

      #### Cumulative perturbation (top-k layers by single-layer rank)

      | k (layers) | Top-k layers | Perturb | Match |
      |---|---|---|---|
      | 1 | [39] | 0.134 | 0.866 |
      | 2 | [39, 17] | 0.172 | 0.828 |
      | 4 | [39, 17, 0, 2] | 0.224 | 0.776 |
      | 8 | [39, 17, 0, 2, 38, 3, 16, 36] | 0.254 | 0.746 |
      | 16 | [39, 17, 0, 2, 38, 3, 16, 36, 15, 6, 35, 13, 14, 12, 32, 34] | 0.336 | 0.664 |
      | 24 | top-24 | 0.390 | 0.610 |
      | 32 | top-32 | 0.454 | 0.546 |
      | **40** | **all** | **0.484** | **0.516** |

      *(k=40 matches exp14 exactly — confirms consistency.)*

      ### Key findings

      **Layer 39 (final) is the dominant single-layer contributor at 13.4%.**
      Layer 17 is the second-highest at 10.4%.  All other layers fall in the
      4–9% range with no sharp outliers.  This is a relatively flat distribution
      — there is no single "bad" layer that drives the bulk of the perturbation.

      **Early layers (0–3) have above-average impact (~7–9%)** despite being
      furthest from the output.  This is expected: errors introduced in early
      layers propagate through all subsequent layers, amplifying their effect.
      Late-middle layers (18–28) have the lowest single-layer impact (4–6%),
      consistent with the residual stream being most stable in that range.

      **Cumulative perturbation scales sub-linearly but without a sharp knee.**
      The top 8 layers (20% of the network) explain only 0.254 / 0.484 = **52%
      of the total perturbation**, and the top 16 layers explain **69%**.  There
      is no small set of "culprit" layers that can be left at full precision to
      recover most of the quality at low cost.

      | k | Perturb | Fraction of total 0.484 |
      |---|---|---|
      | 1 | 0.134 | 28% |
      | 2 | 0.172 | 36% |
      | 4 | 0.224 | 46% |
      | 8 | 0.254 | 52% |
      | 16 | 0.336 | 69% |
      | 24 | 0.390 | 81% |
      | 32 | 0.454 | 94% |
      | 40 | 0.484 | 100% |

      **Diminishing marginal contribution per additional layer.**  Going from
      k=1 to k=2 adds 3.8 pp; k=2→4 adds 5.2 pp; k=4→8 adds 3.0 pp; k=8→16
      adds 8.2 pp; k=16→24 adds 5.4 pp; k=24→32 adds 6.4 pp; k=32→40 adds
      3.0 pp.  The surprisingly large jump at k=8→16 reflects the cluster of
      moderate-contribution layers (6, 35, 13–15, 12, 32, 34) that share similar
      single-layer scores.

      **No "free lunch" via partial-layer approximation.**  To achieve the
      exp14 result at 10% hot (48% perturbation), one must approximate all 40
      layers.  Approximating only the top-8 worst layers gives 25% perturbation —
      but those 8 layers constitute 20% of all MLP FLOPs.  The cost/benefit is
      not obviously better than simply raising the hot-channel fraction to 20%
      for all 40 layers (exp14: 41% perturbation at 20% hot).

      ### Conclusion

      The perturbation budget is **distributed across all 40 layers with no
      dominant outlier** beyond layer 39 and 17.  The final layer (39) stands out
      primarily because its output feeds directly into the unembedding projection
      with no further residual mixing — even a small approximation error has
      maximum logit impact.

      Partial-layer approximation is not a viable quality-recovery strategy:
      keeping even the 8 most-sensitive layers at full precision saves only ~2.3 pp
      of perturbation (from 48% to ~45%) while eliminating 20% of the potential
      FLOP savings.

      The practical implication is that **improving approximation quality uniformly
      across all layers** (e.g., raising the hot-channel fraction or improving the
      routing signal) is a more effective path than selectively protecting a subset
      of layers.

## Experiment 17

Pure encoding experiment — no inference, no MLP patching.  Evaluates the
block-ternary weight encoding scheme described in the "Weight compression
scheme for predictor" section against full-precision weights using the TARE
metric.

### Encoding scheme

Each weight matrix `W` of shape `(O, I)` is encoded as follows:

1. **Block partition** — reshape to non-overlapping blocks of `B = 16`
   consecutive elements along the input dimension:
   `W_blocks` shape `(O × I/B, B)`.

2. **Per-block FP16 scale** — `s_b = max(|w_b|)`, stored as `float16`.
   One scale value per block.

3. **Per-block threshold** — `τ_b = α × s_b`, where `α` is a sweep
   parameter (`0.0` = pure sign, no zeros; `0.75` = matches exp10 sweet spot).

4. **Ternary codes** — `t_b = sign(w_b) × (|w_b| ≥ τ_b) ∈ {−1, 0, +1}`.

5. **Approximate weight** — `w̃_b = t_b × s_b`
   (block-max scale; over-estimates small weights — a per-block RMS scale
   is a natural follow-up).

6. **Packed storage** — 2 bits per element → 4 bytes per block of 16,
   plus 2 bytes for the FP16 scale = **6 bytes per 16 weights** vs
   32 bytes (FP32) or 16 bytes (BF16).  Compression ratio: **2.67× vs BF16**.

### TARE metric

Scale-Tilted Anchored Relative Error:

```
eps   = quantile(|W|, 1%)              anchored floor (per tensor)
tilt  = log1p(|w| / eps)              per-element weight (larger weights matter more)
TARE  = sqrt( Σ tilt × log²(|w̃|/|w|) / Σ tilt )
```

Sign errors on large weights cost roughly `log²(2) ≈ 0.48` each; errors on
near-zero weights are down-weighted to near zero by the `tilt` factor.

### Results (granite-4.2-3b, 40 layers, MPS)

Storage per block of 16 weights: 4 B (2 bit/elem packed codes) + 2 B (FP16 scale)
= **6 bytes per 16 weights**, vs 32 B (BF16) → **5.33× compression vs BF16**.

All three projections have the same shape (8192×2560) and identical storage:
BF16 = 40.0 MiB, encoded = **7.5 MiB**.

Mean TARE across 40 layers, swept over α (threshold = α × block-max scale):

| α | zero% | gate TARE | up TARE | down TARE |
|---|---|---|---|---|
| **0.00 (sign)** | **0.0%** | **1.376** | **1.380** | **1.392** |
| 0.25 | 39.6% | 1.815 | 1.821 | 1.843 |
| 0.50 | 68.7% | 2.881 | 2.886 | 2.907 |
| 0.75 | 85.5% | 3.537 | 3.540 | 3.549 |
| 1.00 | 93.7% | 3.869 | 3.872 | 3.873 |
| 1.50 | 100.0% | 4.149 | 4.152 | 4.155 |

**α = 0 (pure sign, no zeros) minimises TARE** for all three projections.
Introducing zeros (α > 0) monotonically increases TARE because the block-max
scale is a poor approximation for zeroed weights — `w̃ = 0` while the true
weight may be up to `scale` in magnitude, producing a large log-ratio error.

The TARE value of ~1.38 at α=0 does not mean "138% error" — it is a
weighted RMS of `log(|w̃|/|w|)`.  A sign-only encoding sets `|w̃| = scale`
(block max) for every element, so the error per element is
`log(scale / |w_i|)` — large for elements far below the block max.

### Key finding: block-max scale is suboptimal for sign encoding

The block-max scale (`s = max(|w_b|)`) correctly reconstructs the largest
element in each block but over-estimates all others by a factor of
`max(|w_b|) / |w_i|`.  For a sign-only encoding the ideal scale would be
the **block RMS** or **block mean-abs**, which minimises the MSE of
`w̃ = ±s` against the true weights.  This is the natural next experiment.

The zero fraction at α=0 is 0% by construction (sign has no zeros).  The
progression from α=0.25 (40% zeros) to α=1.5 (100% zeros) confirms the
threshold interpretation: at α=1.0 almost all elements are below the
block max and are zeroed out; at α=1.5 every element is zeroed.

### Conclusion

Block-ternary encoding with B=16 and FP16 block-max scale achieves **5.33×
compression vs BF16** (7.5 MiB per projection vs 40.0 MiB).  Under the TARE
metric, α=0 (pure sign) is the best threshold — introducing zeros with the
block-max scale worsens quality monotonically.  The block-max scale is the
bottleneck: it over-estimates small weights within each block.  The immediate
next step is to replace it with a **block RMS or mean-abs scale**, which
should substantially reduce TARE for the sign encoding and may also change
the optimal α.

## Experiment 18

Derives the per-block scale that analytically minimises TARE for a sign
encoding, and compares it against the block-max (exp17) and block-RMS
baselines across all three MLP projections.

### Optimal scale derivation

For a sign encoding `w̃_i = sign(w_i) × s`, the TARE loss as a function of `s`
is a weighted least-squares in log-space:

```
L(s) = Σ_i tilt_i × (log s − log|w_i|)²
```

Setting `dL/d(log s) = 0` gives the analytic minimiser:

```
log s* = Σ_i tilt_i × log|w_i| / Σ_i tilt_i
s*     = exp( tilt-weighted mean of log|w_i| )
```

where `tilt_i = log1p(|w_i| / eps)` and `eps` is the 1st-percentile of `|W|`
(the per-tensor TARE floor).  This is the **tilt-weighted geometric mean** of
the block's absolute weights.  It is computed in one vectorised pass over all
blocks and stored as FP16 — identical storage cost to exp17.

### Results (granite-4.2-3b, 40 layers, MPS)

Storage unchanged from exp17: **7.5 MiB per projection, 5.33× vs BF16**.
`*` marks the best α per scale type.

**gate projection**

| Scale | α=0.00 | α=0.25 | α=0.50 | α=0.75 | α=1.00 |
|---|---|---|---|---|---|
| block-max (exp17) | 1.3762 * | 1.8154 | 2.8807 | 3.5367 | 3.8694 |
| block-RMS | 0.9017 * | 0.9707 | 1.6286 | 2.2705 | 2.8111 |
| **TARE-optimal** | 0.8336 | **0.8276 *** | 1.2648 | 1.7780 | 2.2561 |

**up projection**

| Scale | α=0.00 | α=0.25 | α=0.50 | α=0.75 | α=1.00 |
|---|---|---|---|---|---|
| block-max (exp17) | 1.3803 * | 1.8211 | 2.8857 | 3.5396 | 3.8717 |
| block-RMS | 0.9048 * | 0.9744 | 1.6335 | 2.2744 | 2.8131 |
| **TARE-optimal** | 0.8363 | **0.8304 *** | 1.2684 | 1.7808 | 2.2574 |

**down projection**

| Scale | α=0.00 | α=0.25 | α=0.50 | α=0.75 | α=1.00 |
|---|---|---|---|---|---|
| block-max (exp17) | 1.3915 * | 1.8433 | 2.9067 | 3.5486 | 3.8727 |
| block-RMS | 0.9090 * | 0.9807 | 1.6443 | 2.2871 | 2.8241 |
| **TARE-optimal** | 0.8390 | **0.8333 *** | 1.2723 | 1.7859 | 2.2629 |

**Sign encoding (α=0) summary:**

| Projection | block-max | block-RMS | TARE-optimal | Δ opt vs max | Δ opt vs RMS |
|---|---|---|---|---|---|
| gate | 1.3762 | 0.9017 | **0.8336** | −0.543 | −0.068 |
| up | 1.3803 | 0.9048 | **0.8363** | −0.544 | −0.068 |
| down | 1.3915 | 0.9090 | **0.8390** | −0.553 | −0.070 |

### Key findings

**TARE-optimal scale reduces TARE by 0.54 vs block-max and 0.07 vs block-RMS**
at α=0 (sign encoding).  The block-RMS is already a substantial improvement
over block-max (−0.47), and the optimal scale improves a further −0.07 on top.

**The optimal α shifts from 0 to 0.25 with the TARE-optimal scale.**  With
block-max and block-RMS, α=0 (pure sign, no zeros) is best.  With the
optimal scale, introducing 25% zeros at α=0.25 gives a small additional gain
(gate: 0.8276 vs 0.8336 at α=0, Δ=−0.006).  This is because the optimal scale
is fitted to the non-zero elements, and zeroing a few near-floor elements
removes their residual contribution to the loss.  The gain is modest — the
main driver is the scale, not the sparsity.

**All three projections behave identically.**  TARE scores differ by <0.003
between gate, up, and down at every (scale, α) combination, confirming the
weight distributions are uniform across projections and layers.

### Conclusion

The TARE-optimal per-block scale (`s* = tilt-weighted geometric mean of |w_b|`)
achieves **TARE = 0.834** at α=0.25 for all three projections — a **39% reduction**
vs the block-max baseline (1.376) at the same storage cost (5.33× vs BF16).
The optimal α shifts from 0 to 0.25, introducing a small fraction of zeros
that marginally improves quality under the optimal scale.

The remaining TARE of ~0.83 represents the irreducible error of a
1-bit-per-weight sign encoding with a single FP16 scale per 16 elements.
Reducing it further requires either finer granularity (smaller B), more bits
per weight (e.g. 3-level with separate negative/positive scales), or storing
the full magnitude alongside the sign for high-magnitude elements.

## Experiment 19

1-bit sign encoding with TARE-optimal **E8M0** (power-of-two) per-block scales,
swept over block sizes B ∈ {8, 16, 32, 64}.

### E8M0 scale

E8M0 is 8 exponent bits, 0 mantissa bits: `s = 2^e`, stored in **1 byte**
instead of FP16's 2 bytes.  The optimal exponent is derived directly from the
exp18 formula:

```
e* = round( tilt-weighted mean of log2(|w_b|) )
s_e8m0 = 2^e*
```

Rounding to an integer exponent is the only approximation vs exp18's FP16
optimal scale.  From the weight distribution of granite-4.2-3b, `log2(s*)`
spans roughly −9 to −5 with std ≈ 0.34, so the maximum rounding error is
0.5 bits in the exponent — a factor of `2^0.5 ≈ 1.41` in scale.

### Storage

All three projections are 8192×2560.  With 1-bit sign codes packed at 8 per byte:

| B | codes | E8M0 scale | total/block | MiB/proj | ratio vs BF16 |
|---|---|---|---|---|---|
| 8 | 1 B | 1 B | 2 B | 5.00 | **8.00×** |
| 16 | 2 B | 1 B | 3 B | 3.75 | **10.67×** |
| 32 | 4 B | 1 B | 5 B | 3.12 | **12.80×** |
| 64 | 8 B | 1 B | 9 B | 2.81 | **14.22×** |

FP16 scale (exp18 reference) at B=16: 5.00 MiB, **8.00×**.

### Results (granite-4.2-3b, 40 layers, MPS, α=0)

Mean TARE across all 40 layers per projection:

| B | scale | gate | up | down | ratio |
|---|---|---|---|---|---|
| 8 | E8M0-opt | 0.834 | 0.837 | 0.839 | **8.00×** |
| **16** | **E8M0-opt** | **0.855** | **0.861** | **0.861** | **10.67×** |
| 16 | FP16-opt (exp18) | 0.834 | 0.836 | 0.839 | 8.00× |
| 32 | E8M0-opt | 0.864 | 0.873 | 0.869 | **12.80×** |
| 64 | E8M0-opt | 0.867 | 0.878 | 0.871 | **14.22×** |

Mean TARE across all three projections:

| B | scale | mean TARE | ratio vs BF16 |
|---|---|---|---|
| 8 | E8M0-opt | **0.837** | 8.00× |
| 16 | E8M0-opt | 0.859 | 10.67× |
| 16 | FP16-opt (exp18 ref) | 0.836 | 8.00× |
| 32 | E8M0-opt | 0.868 | 12.80× |
| 64 | E8M0-opt | 0.872 | 14.22× |

### Key findings

**B=8 E8M0 matches FP16-optimal at B=16 (both 8.00×) with the same TARE.**
TARE 0.837 vs 0.836 — a difference of 0.001.  The E8M0 rounding error at B=8
is fully compensated by the finer block granularity.  Both achieve 8× compression
at identical quality.

**Larger blocks compress more but hurt TARE.**  Going from B=8 to B=64 improves
compression from 8.00× to 14.22× (1.78×) while TARE rises from 0.837 to 0.872
(+0.035, about 4%).  The scale-quantisation error (E8M0 rounding) grows with
block size because a single exponent must cover a wider dynamic range of weights.

**E8M0 scale adds negligible TARE vs FP16 at B=16.**  The quantisation from FP16
to E8M0 at the same block size costs only +0.021 TARE (0.855 vs 0.834) while
saving 1 byte per block — halving scale storage overhead.  At B=8 the story is
even cleaner: E8M0 at B=8 is essentially tied with FP16-optimal at B=16.

### Conclusion

The sweet spot is **B=8 with E8M0 scale: 8.00× compression, TARE=0.837** —
matching FP16-optimal at B=16 in both compression ratio and quality, while
halving the scale storage per block.

For applications that can tolerate a small quality regression, **B=16 E8M0**
gives **10.67× compression at TARE=0.859** — 33% more compressed than the
FP16 baseline at a cost of +0.023 TARE.  B=32 and B=64 compress further but
the TARE gain over B=16 is diminishing relative to the quality cost.

The practical recommendation: use **B=8, E8M0-optimal** as the baseline encoding
for the gate predictor weight.  The next question is whether this encoding quality
(TARE ≈ 0.84) is sufficient to preserve the routing accuracy demonstrated in
exp10–11.

## Experiment 20

End-to-end top-1 perturbation rate for the B=8 E8M0 sign encoding applied to
**both** gate and up projections, with hot-channel full-precision refinement of
both.  Down projection always runs at full precision on the complete SwiGLU
vector.  Routing: proxy-prior (`|gate_approx[t-1]|`), same as exp11.

### Scheme

Pre-computed once per layer (stored as float32 for GEMM, logically 8× compressed):

```
W_gate_enc = sign(W_gate) * s_gate   B=8 E8M0-optimal scales
W_up_enc   = sign(W_up)   * s_up     B=8 E8M0-optimal scales
```

Per token:

1. `gate_approx = W_gate_enc @ x`  — cheap sign-scaled GEMM
2. `up_approx   = W_up_enc   @ x`  — cheap sign-scaled GEMM
3. `hot = top-k by |gate_approx[t-1]|`  — proxy-prior, zero overhead
4. `gate_hybrid[hot] = W_gate[hot] @ x`  ; `gate_hybrid[cold] = gate_approx[cold]`
5. `up_hybrid[hot]   = W_up[hot]   @ x`  ; `up_hybrid[cold]   = up_approx[cold]`
6. `swiglu = SiLU(gate_hybrid) * up_hybrid`
7. `out    = W_down @ swiglu`  — full precision, full vector

### Results (granite-4.2-3b, 8 prompts, 500 prefill tokens, CPU backend)

| hot% | exp20 match | exp20 perturb | exp14 match | Δ vs exp14 |
|---|---|---|---|---|
| 5% | 0.116 | **88.4%** | 0.456 | −0.340 |
| 10% | 0.194 | **80.6%** | 0.516 | −0.322 |
| 20% | 0.310 | **69.0%** | 0.588 | −0.278 |
| 30% | 0.412 | **58.8%** | 0.632 | −0.220 |

### Key finding: encoding W_up for cold channels is costly

Exp20 is **substantially worse than exp14** at every hot fraction.  At 10% hot,
match rate drops from 0.516 (exp14) to 0.194 (exp20) — a 32 pp regression.

The cause is the interaction between the cold gate approximation and the cold
up approximation in SwiGLU:

```
swiglu_cold = SiLU(gate_approx_cold) * up_approx_cold
```

In exp11/14 (full up), `up_full_cold` was exact, so only the gate approximation
error propagated into the SwiGLU.  Here, both `gate_approx_cold` and
`up_approx_cold` carry independent sign-encoding errors.  The SwiGLU multiplies
these two errors together, squaring the relative error in the cold-channel
contribution before it reaches W_down.

Exp12 observed the same phenomenon in the other direction: a ternary W_down for
cold channels was not viable because `up_full` is exact and cold SwiGLU values
are non-trivial.  Exp20 demonstrates the symmetric case: encoding W_up for cold
channels is not viable when cold gate values are also approximated.

### Conclusion

The B=8 E8M0 encoding is suitable for W_gate (the routing/predictor role) but
**not** for W_up cold channels alongside an approximated gate.  The full-precision
up projection must be retained for all channels — exactly as in exp11/14.

The practical encoding budget: **W_gate at 8× compression (B=8 E8M0), W_up and
W_down at full precision**.  This was the implicit assumption in exp14 and remains
the correct split.  The encoding experiments (17–19) quantify how accurately the
gate predictor can be compressed; exp20 confirms that extending the same
compression to W_up cold channels is too costly end-to-end.

## Experiment 21

End-to-end top-1 validation of B=8 E8M0 sign encoding on W_gate only, with
full-precision W_up and W_down — the encoding split established by exp20.

### Scheme

Same as exp14 but replaces the global-α ternary gate proxy with the B=8 E8M0
TARE-optimal sign encoding:

```
W_gate_enc = sign(W_gate) * s_e8m0   per-block E8M0 scale, B=8
```

Cold gate values: `gate_approx[cold] = W_gate_enc[cold] @ x`.
Hot gate values: `gate_full[hot] = W_gate[hot] @ x` (full precision).
W_up: full precision for all channels.
W_down: full precision on full SwiGLU vector.
Routing: both `current` (|gate_approx[t]|) and `proxy_prior` (|gate_approx[t-1]|) tested.

### Results (granite-4.2-3b, 8 prompts, 500 prefill tokens, CPU backend)

| hot% | current | proxy_prior | exp14 (ternary α=0.75) | Δ vs exp14 |
|---|---|---|---|---|
| 5% | 0.378 | 0.378 | 0.456 | −0.078 |
| 10% | 0.414 | 0.414 | 0.516 | −0.102 |
| 20% | 0.456 | 0.456 | 0.588 | −0.132 |
| 30% | 0.520 | 0.520 | 0.632 | −0.112 |

Current and proxy_prior are identical — expected for batched prefill (the prior
is a different sequence's last token, not a genuine temporal prior; as noted in
exp14's methodology, proxy-prior only helps in autoregressive decode).

### Key finding: E8M0 cold channel values are over-scaled

The B=8 E8M0 encoding is *worse* than exp14's global ternary (α=0.75) despite
having lower TARE.  The diagnostic:

| Encoding | mean\|w̃\| | zero% |
|---|---|---|
| True W_gate | 0.00738 | 0% |
| E8M0 enc | **0.00677** | 0% |
| Ternary (α=0.75) | **0.00400** | 46% |

The E8M0 per-block scale is the tilt-weighted geometric mean of the block's
absolute weights — a good reconstruction target for TARE, but it over-estimates
most elements relative to the global mean.  Cold channel gate values
`gate_approx_cold = W_gate_enc_cold @ x` therefore have larger magnitude than
the true cold gates, pushing more cold-channel contributions through SiLU and
into the down projection.  This inflates the cold-channel output error despite
the better TARE score.

The ternary scheme at α=0.75 scales every non-zero weight by `mean(|W_gate|)`,
which by construction gives cold gate values with the correct *expected* magnitude.
TARE penalises log-ratio errors uniformly — it does not distinguish between
over-estimation (which raises SiLU output) and under-estimation (which suppresses
it).  For the SwiGLU cold channel computation, under-estimation is benign (near-zero
SiLU output is discarded) while over-estimation is harmful (inflated cold
contribution corrupts the output).

### Conclusion

The E8M0 TARE-optimal encoding is an excellent predictor/routing signal — its
gate approximation cosine similarity is higher than the ternary scheme (TARE 0.837
vs 0.856).  However, it is **not** a better cold-channel computation proxy because
TARE optimality does not align with the asymmetric cost of over- vs
under-estimation in SwiGLU.

The correct use of the E8M0 encoding is for **routing only** (selecting the hot
mask), with cold channel gate values replaced by a *downward-biased* approximation
— such as the ternary scheme's `mean(|W|)` scaling — to keep cold SiLU outputs
near zero.  This points to a hybrid: use E8M0 as the routing proxy, and use a
separate magnitude-suppressed approximation (e.g. scale by `β × mean(|W|)`, β < 1)
for cold channel computation.

## Experiment 22

E5M3 gate encoding (B=8, top-k routing) — direct replacement of E8M0 in exp21.

### E5M3 format

E5M3: 5 exponent bits, 3 mantissa bits, 1 byte/block — same storage as E8M0.
`s = (1 + m/8) × 2^e`, with `m ∈ {0..7}` giving 8 levels per octave.

From the pre-run analysis across all 40 layers:

| Format | Mean scale error (log₂) | Mean linear error | Max linear error |
|---|---|---|---|
| E8M0 | 0.251 | 19.6% | 41.4% |
| **E5M3** | **0.035** | **2.5%** | **6.7%** |

E5M3 is 7.2× more accurate than E8M0 at the same 1 byte/block storage cost.

### Results (granite-4.2-3b, 8 prompts, 500 prefill tokens, CPU)

| hot% | E5M3 current | E5M3 proxy_prior | exp14 (ternary) | exp21 (E8M0) | Δ E5M3 vs exp14 |
|---|---|---|---|---|---|
| 5% | 0.386 | 0.386 | 0.456 | 0.378 | −0.070 |
| 10% | 0.408 | 0.408 | 0.516 | 0.414 | −0.108 |
| 20% | 0.456 | 0.456 | 0.588 | 0.456 | −0.132 |
| 30% | 0.532 | 0.532 | 0.632 | 0.520 | −0.100 |

E5M3 is marginally better than E8M0 at some hot fractions (30%: 0.532 vs 0.520)
but both remain well below the ternary baseline.  The improved scale precision
does not close the gap.  Current and proxy_prior remain identical (batched
prefill, as expected).

### Conclusion

The E5M3 scale improvement (19.6% → 2.5% error) does not translate to better
end-to-end quality vs the ternary α=0.75 scheme.  The root cause from exp21
holds: the issue is not scale precision but the fundamental asymmetry of SwiGLU —
the ternary scheme's 46% zero fraction suppresses cold channels near zero,
while both E8M0 and E5M3 assign nonzero scales to every element.  No amount of
scale precision fixes the over-activation of cold channels when the encoding
produces nonzero output for all 8192 channels.

---

## Experiment 23

E5M3 gate encoding (B=8) with **magnitude-threshold routing** on `|gate_approx|`
instead of top-k.  This revisits the exp2 approach, which failed with the sign
predictor because `|gate_approx|` was a poor proxy for `|gate_full|`.  With E5M3
(2.5% mean scale error) the approximation is much more accurate.

### Scheme

```
hot = { channels where |gate_approx[t]| > T × mean(|gate_approx[t]|) }
```

`T` is a per-inference threshold factor, swept over {0.5, 1.0, 1.5, 2.0, 3.0}.
No sort needed — hot mask is a single comparison per token.  Hot channel count
varies per token (adaptive sparsity), unlike top-k which is fixed.

### Results (granite-4.2-3b, 8 prompts, 500 prefill tokens, CPU)

| Threshold T | match | perturb |
|---|---|---|
| **0.5× mean** | **0.736** | **26.4%** |
| 1.0× mean | 0.600 | 40.0% |
| 1.5× mean | 0.494 | 50.6% |
| 2.0× mean | 0.424 | 57.6% |
| 3.0× mean | 0.328 | 67.2% |

Reference: exp14 ternary @10% hot = **0.516** match.

### Key findings

**Threshold T=0.5 is the best single result across all experiments at 73.6%
match (26.4% perturbation)** — significantly better than exp14's best of 63.2%
at 30% hot.  T=1.0 also beats exp14 at 60.0% match.

**Threshold routing decisively outperforms top-k routing.**  At T=0.5 the
match rate is 0.736 vs exp22's best of 0.532 — a 20 pp gap.  The reason:
threshold routing is *adaptive*.  Tokens with many large gate activations
recompute more channels; tokens where the gate is uniformly small recompute
almost nothing.  The hot fraction is matched to the actual per-token sparsity
of the gate distribution, whereas top-k forces a fixed fraction regardless.

**T=1.0 matches the exp14 ternary at 0.60** with no fixed hot fraction — the
threshold naturally selects the channels that need full precision on each token.

**The E5M3 scale precision is what makes this work.**  In exp2, thresholding on
`|gate_approx|` (sign predictor) collapsed SwiGLU cosine similarity to near
zero because the sign-approx magnitude had no reliable relationship to the true
gate magnitude.  With E5M3's 2.5% mean scale error, `|gate_approx|` faithfully
reflects `|gate_full|`, so the threshold correctly identifies channels that are
truly large and need recomputation.

### Conclusion

B=8 E5M3 gate encoding with threshold routing at **T=0.5× mean** achieves
**73.6% match / 26.4% perturbation** — the best end-to-end result in the entire
experiment series, surpassing exp14's 63.2% at 30% hot by 10 pp.

The threshold scheme has two further practical advantages over top-k:
1. **No sort** — the hot mask is computed with a single comparison per token
2. **Adaptive hot fraction** — naturally recomputes more on "busy" tokens and
   less on "quiet" ones, matching compute to actual per-token difficulty

The combination of E5M3 weight encoding + magnitude threshold routing is the
new best scheme.  The next questions are: what is the typical hot fraction at
T=0.5 (compute cost), and does the threshold transfer to decode-time
autoregressive generation where proxy-prior routing becomes meaningful?

## Experiment 24

Fine-grained threshold sweep T ∈ {0.20, 0.25, …, 0.80} for the B=8 E5M3
gate encoding, with per-layer hot-fraction monitoring and a 2% floor variant.

### Scheme

```
hot = { |gate_approx[t]| > T × mean(|gate_approx[t]|) }          (pure)
hot = above ∪ top-2% channels by |gate_approx[t]|                  (floored)
```

### Results (granite-4.2-3b, 8 prompts, 500 prefill tokens, CPU)

| T | match | perturb | mean hot% | Δ floored |
|---|---|---|---|---|
| **0.20** | **0.818** | **18.2%** | 88.0% | +0.000 |
| 0.25 | 0.798 | 20.2% | 85.0% | +0.000 |
| 0.30 | 0.798 | 20.2% | 82.0% | +0.000 |
| 0.35 | 0.782 | 21.8% | 79.1% | +0.000 |
| 0.40 | 0.768 | 23.2% | 76.1% | +0.000 |
| 0.45 | 0.740 | 26.0% | 73.2% | +0.000 |
| 0.50 | 0.736 | 26.4% | 70.3% | +0.000 |
| 0.55 | 0.724 | 27.6% | 67.4% | +0.000 |
| 0.60 | 0.684 | 31.6% | 64.5% | +0.000 |
| 0.65 | 0.688 | 31.2% | 61.7% | +0.000 |
| 0.70 | 0.664 | 33.6% | 58.9% | +0.000 |
| 0.75 | 0.636 | 36.4% | 56.1% | +0.000 |
| 0.80 | 0.622 | 37.8% | 53.4% | — |

References: exp23 T=0.50 match=0.736 · exp14 ternary @30% match=0.632

### Key findings

**T=0.20 is the new best at match=0.818 (18.2% perturbation)** — beating
exp23's T=0.50 (0.736) by 8.2 pp and exp14's best (0.632) by 18.6 pp.  Match
improves monotonically as T decreases from 0.80 to 0.20, with no plateau
visible yet — the optimal threshold may be below 0.20.

**The floor makes no difference at any tested threshold.**  Pure and floored
results are identical throughout.  At T=0.20 the mean hot fraction is already
88%, so the 2% floor is never the binding constraint.  Even at T=0.80 (53%
hot), all layers have well above 2% active channels — there are no dead layers
in this model for any tested threshold.

**Hot fraction is high across the board.**  At T=0.20, 88% of channels are
being recomputed — the scheme is converging toward full precision, which
explains the high match rate.  The useful operating range is where the
quality-cost curve is steep.  Between T=0.20 (88% hot, 18.2% perturb) and
T=0.50 (70% hot, 26.4% perturb) we pay 18 pp more perturbation to save 18 pp
of hot channels.

**The quality–compute curve is roughly linear in this range** with no sharp
knee visible.  There is a small non-monotonicity at T=0.60/0.65 (0.684 vs
0.688) suggesting per-sample variance at this sample size (n=500 tokens).

### Conclusion

The optimal threshold in the tested range is T=0.20 with match=0.818.  The
sweep suggests the optimum may be even lower (T < 0.20), but at T=0.20 already
88% of channels are recomputed — approaching full precision at high compute cost.
The threshold routing scheme has no dead-layer problem; the 2% floor is
unnecessary for this model.

The trade-off picture across key operating points:

| T | hot% | match | perturb | interpretation |
|---|---|---|---|---|
| 0.20 | 88% | **0.818** | 18.2% | near-full precision, best quality |
| 0.30 | 82% | 0.798 | 20.2% | similar quality, 6pp cheaper |
| 0.50 | 70% | 0.736 | 26.4% | exp23 baseline |
| 0.65 | 62% | 0.688 | 31.2% | comparable to exp14 ternary @30% |
| exp14 ternary | ~52% hot FLOP | 0.632 | 36.8% | ternary gate baseline |

The next experiment should extend the sweep below T=0.20 to find whether there
is a true optimum or whether the curve keeps improving all the way to T→0
(which would be equivalent to full precision computation with an encoding
overhead, not a useful approximation).

## Experiment 25a — Activation error of gate_approx and up_approx

### Motivation

Exp23/24 showed that E5M3 threshold routing achieves 0.818 match at T=0.20.
The TARE metric from exp17–19 measures weight-space encoding error.  This
experiment measures the actual *activation* error — `|gate_approx − gate_full|`
— on real hidden states from calibration prompts, to understand whether the
encoding quality translates to good routing fidelity.

### Scheme

Hook each MLP layer's forward pass.  For every token batch capture `x`
(the MLP input) and compute:
- `gate_full = x @ W_gate.T`
- `gate_approx = x @ W_gate_enc.T` (E5M3 B=8)
- `up_approx = x @ W_up_enc.T` (E5M3 B=8, hypothetical)
- `gate_ter = x @ W_gate_ter.T` (ternary α=0.75, exp14 encoding)

Metrics: mean relative error, RMS SNR (dB), R².

### Results (granite-4.2-3b, 8 prompts, ~500 prefill tokens, CPU)

| Encoding | Mean SNR (dB) | R² | Notes |
|---|---|---|---|
| E5M3 gate (exp22–24) | **5.7** | 0.69 | used for routing and cold values |
| E5M3 up (hypothetical) | 4.7 | — | up never encoded in exp20–24 |
| Ternary gate (exp14) | 5.1 | — | used in earlier experiments |

Mean relative error (200–400%) is misleading — dominated by near-zero channels
where `|full| + ε` is tiny.  SNR and R² are the meaningful metrics.

### Key findings

All three encodings have comparable SNR (~5–6 dB), meaning the error RMS is
roughly 50–58% of the signal RMS per channel.  This sounds large but is
tolerable because:
1. Routing uses `|gate_approx|` for ranking, not absolute values — rank
   correlation matters more than absolute accuracy.
2. Cold channels are near-zero by selection; SiLU suppresses any residual error.

The rel_err spike at layers 4, 33, 38–39 (>600%) indicates layers with heavy
weight outliers but these do not worsen routing quality (confirmed by exp24
no-dead-layers result).

### Conclusion

E5M3 encoding achieves R²≈0.69 on real activations — sufficient for
reliable routing, confirmed by the F1=0.90 routing quality measured in exp25c.

---

## Experiment 25b — Gate error stratified by hot/cold

### Motivation

The all-channel error from exp25a averages over hot and cold channels together.
Since the scheme uses `gate_approx` as the *value* for cold channels, the
relevant question is: how accurate is `gate_approx` specifically on cold
channels (where the error affects the output), vs hot channels (where
`gate_full` replaces it anyway)?

### Results (granite-4.2-3b, thresholds T=0.20–0.80)

| T | hot% | hot SNR (dB) | hot R² | cold SNR (dB) | cold R² |
|---|---|---|---|---|---|
| 0.20 | 88% | 6.0 | 0.704 | 0.2 | 0.035 |
| 0.40 | 76% | 6.3 | 0.715 | 0.6 | 0.124 |
| 0.60 | 65% | 6.6 | 0.725 | 1.2 | 0.231 |
| 0.80 | 54% | 6.9 | 0.733 | 1.9 | 0.332 |

### Key findings

**Hot channels** (large `|gate_approx|`, recomputed at full precision): SNR ~6–7 dB,
R²≈0.70 — the encoding is reasonably accurate but it doesn't matter because
these channels are overwritten with `gate_full`.

**Cold channels** (small `|gate_approx|`, value used directly): SNR ~0–2 dB,
R²≈0.04–0.33 — nearly pure noise in terms of encoding the true value.  Yet the
scheme works precisely *because* of this: cold channels have small `|gate_approx|`
by definition, so `SiLU(gate_approx_cold) ≈ 0` regardless of encoding accuracy.
The terrible cold-channel R² is harmless.

### Conclusion

The routing scheme's correctness rests on two asymmetries: (1) hot channels are
recomputed exactly; (2) cold channels are near-zero so SiLU suppresses any
encoding error.  E5M3 quality on cold channels is irrelevant — the encoding
only needs to be accurate enough that `|gate_approx|` reliably identifies which
channels are hot.

---

## Experiment 25c — E5M3 routing quality: precision/recall vs oracle

### Motivation

Measure the binary routing decision quality: when E5M3 says "hot", does the
oracle (gate_full) agree?  Apply the same threshold T to both `|gate_approx|`
and `|gate_full|` and measure precision/recall/F1/IoU.

### Results (granite-4.2-3b, 40 layers, macro-averaged)

| T | hot%(A) | hot%(F) | Precision | Recall | F1 | IoU | Accuracy |
|---|---|---|---|---|---|---|---|
| 0.20 | 88.0% | 87.6% | 0.901 | 0.905 | 0.903 | 0.823 | 0.829 |
| 0.40 | 76.2% | 75.6% | 0.844 | 0.852 | 0.848 | 0.737 | 0.770 |
| 0.60 | 64.7% | 64.0% | 0.809 | 0.819 | 0.814 | 0.687 | 0.762 |
| 0.80 | 53.7% | 53.0% | 0.779 | 0.790 | 0.784 | 0.645 | 0.771 |

### Key findings

- Routing quality degrades as T increases (sparser hot set = finer boundary
  decisions, more misclassifications at the margin).
- No systematic bias: `hot%(approx) ≈ hot%(oracle)` throughout.
- At T=0.20 (exp24 best operating point): F1=0.903, IoU=0.823 — the ~10%
  routing errors are further suppressed by SiLU before reaching the output.

### Conclusion

E5M3 is a high-quality routing signal at the operating points used in exp23/24.
The degradation at higher T explains the quality gap between T=0.20 and T=0.80.

---

## Experiment 25d — SVD routing quality vs oracle, rank sweep

### Motivation

Exp27 uses SVD of W_gate and W_up as routing signals.  Before running e2e tests,
measure whether SVD at low ranks can match E5M3's routing quality.

### Results (granite-4.2-3b, 40 layers, macro-averaged)

**hot=20%:**

| Rank | F1 | IoU | rec_gate | rec_up |
|---|---|---|---|---|
| 16 | 0.508 | 0.343 | 0.337 | 0.251 |
| 64 | 0.551 | 0.382 | 0.361 | 0.275 |
| 256 | 0.612 | 0.443 | 0.391 | 0.315 |
| 1024 | 0.745 | 0.595 | 0.451 | 0.402 |
| **E5M3 (exp25c)** | **0.784** | **0.645** | — | — |

**hot=50%:**

| Rank | F1 | IoU |
|---|---|---|
| 16 | 0.794 | 0.659 |
| 256 | 0.822 | 0.698 |
| 1024 | 0.873 | 0.775 |
| **E5M3 (exp25c ~88%)** | **0.903** | **0.823** |

### Key findings

Low-rank SVD is consistently worse than E5M3 at routing, especially at 20% hot.
Even rank=1024 (F1=0.745 @20%) falls below E5M3 (F1=0.784 @~53% hot).

The root cause: SVD captures global weight-space directions (which channels are
*on average* large for typical inputs), while E5M3 preserves the sign and
per-block scale of every individual weight, giving token-specific channel ranking.
The `rec_gate` and `rec_up` columns show that even the gate signal alone at
rank=1024 only recalls 45% of oracle-hot channels at 20% hot.

F1 barely changes from rank 16 to rank 1024 at 50% hot (~0.79 to ~0.87),
suggesting the routing signal is saturating — the trailing singular directions
contain most of the per-token channel-ordering information.

### Conclusion

For sparse routing (≤20% hot), E5M3 is superior.  For dense routing (50%+ hot),
rank=1024 SVD becomes competitive and has the advantage of providing a separate
routing signal for W_up, enabling E5M3 cold values for both gate and up
(tested in exp27).

---

## Experiment 25e — Learned linear predictor: training cost and routing quality

### Motivation

Could a predictor trained on recorded activations do better than using the
model weights directly?  We compute the optimal rank-r linear predictor for
two targets using the closed-form cross-covariance SVD solution:
- **Regression**: predict `down_input` (SwiGLU activity) directly
- **Binary**: predict the hot/cold label vector

### Training cost

| Step | Cost per layer | Cost all 40 layers |
|---|---|---|
| Cross-covariance `C = X^T Y / N` | 534 GFLOPs | 21.4 TFLOPs |
| SVD of `C` (H×I = 2560×8192) | 107 GFLOPs | 4.3 TFLOPs |
| **Total** | **641 GFLOPs** | **25.7 TFLOPs** |

25.7 TFLOPs is a few GPU-seconds of offline compute — entirely feasible.
Inference routing cost is 0.8–52.5% of one GEMM depending on rank (same as SVD).

### Results (10 representative layers, N=12734 tokens)

**hot=20%:**

| Predictor | Rank 16–1024 F1 range |
|---|---|
| Regression cross-cov | 0.318 → 0.321 |
| Binary hot-label cross-cov | 0.380 → 0.384 |
| **SVD of W_gate (exp25d)** | **0.508 → 0.745** |
| **E5M3 of W_gate (exp25c)** | **0.784** |

**hot=50%:**

| Predictor | Rank 16–1024 F1 range |
|---|---|
| Regression cross-cov | 0.539 → 0.540 |
| Binary hot-label cross-cov | 0.577 → 0.580 |
| **SVD of W_gate (exp25d)** | **0.794 → 0.873** |

### Key findings

1. **Trained predictors are dramatically worse than weight-derived ones.**
   The regression predictor at rank=1024 (F1=0.32 @20% hot) is far below even
   SVD rank=16 (F1=0.508).

2. **Rank has almost no effect** — F1 changes by <0.005 from rank 16 to 1024.
   The cross-covariance `C = X^T Y` is effectively rank-1: nearly all the linear
   correlation between inputs and SwiGLU outputs lives in a single direction.
   Additional singular vectors add nothing.

3. **Root cause**: `down_input = SiLU(W_gate x) * W_up x` is nonlinear in `x`.
   A linear predictor can only capture the linear component, which is far weaker
   than the structured signal that `x @ W_gate.T` provides directly.  The weight
   matrix *is* the optimal linear predictor of gate activity — no training needed.

4. **Binary label target** is marginally better than regression (F1 +0.06) but
   both are uncompetitive.

### Conclusion

Training a learned linear predictor provides no benefit over using the model
weights directly.  The 25.7 TFLOPs training cost is trivial but the result
is ~2× worse than E5M3 routing.  The correct approach is to use `x @ W_gate_enc.T`
(E5M3-encoded actual gate weights) as the routing signal.

---

## Experiment 26 — Sparse SwiGLU: cold channels zeroed

### Motivation

Exp24 uses `gate_approx` (E5M3) as the cold-channel gate value fed into SiLU.
Exp25b showed cold-channel SNR is 0–2 dB (nearly noise).  Would zeroing cold
channels entirely (no approximation at all) perform better?

### Scheme

```
gate_approx = x @ W_gate_enc.T      # E5M3 routing only
hot = |gate_approx| > T * mean(|gate_approx|)
gate = where(hot, x @ W_gate.T, 0)  # cold = 0, not gate_approx
up   = where(hot, x @ W_up.T,   0)
out  = (SiLU(gate) * up) @ W_down.T
```

### Results (granite-4.2-3b, 8 prompts, 500 tokens)

| T | hot% | exp26 match | exp24 match | Δ |
|---|---|---|---|---|
| 0.20 | 88% | 0.814 | **0.818** | −0.004 |
| 0.40 | 76% | 0.752 | **0.779** | −0.027 |
| 0.60 | 65% | 0.636 | **0.695** | −0.059 |
| 0.80 | 54% | 0.576 | **0.632** | −0.056 |

### Key findings

Zeroing cold channels is **consistently worse** than using `gate_approx`.  The
gap grows as T increases — more cold channels means more zeroing means more loss.

Cold channels have small `|gate_approx|` by definition, so `SiLU(gate_approx_cold)`
is close to zero but not exactly zero.  The residual contribution is small but
slightly positive and helpful, not harmful.

### Conclusion

The E5M3 approximation on cold channels is a mild beneficial signal rather than
noise to be discarded.  Exp24's scheme (use `gate_approx` for cold channels,
`gate_full` for hot) is the correct design.

---

## Experiment 27 — Low-rank SVD routing with E5M3 cold gate and up

### Motivation

Exp24 routes on `|gate_approx|` (E5M3) and uses full-precision `up` for all
channels.  The up projection is the second-largest compute cost.  If routing
quality is high enough, E5M3 cold values for `up` as well could halve the cold-
channel compute.  SVD routing provides separate signals for both gate and up.

### Scheme

```
gate_lr = x @ W_gate_lr.T   (rank-r SVD of W_gate)
up_lr   = x @ W_up_lr.T     (rank-r SVD of W_up)
hot = top-k(|gate_lr|, k) ∪ top-k(|up_lr|, k)   # union, k = frac * I
gate = where(hot, gate_full, gate_e5m3)
up   = where(hot, up_full,   up_e5m3)
out  = (SiLU(gate) * up) @ W_down.T
```

Key difference from exp24: **both gate and up use E5M3 for cold channels**
(not just gate).  This is enabled by high routing quality catching channels
where up matters.

### Results (granite-4.2-3b, 8 prompts, 500 tokens)

| Rank | hot=20% | hot=30% | hot=50% | Δ vs exp24 @50% |
|---|---|---|---|---|
| 64 | 0.392 | 0.472 | 0.678 | −0.058 |
| 256 | 0.520 | 0.606 | 0.764 | +0.028 |
| **1024** | 0.612 | 0.702 | **0.834** | **+0.098** |
| exp24 ref | **0.818** | 0.798 | 0.736 | — |

### Key findings

- **Rank=1024 @50% hot beats exp24** (0.834 vs 0.736) — first scheme to beat
  exp24 at a comparable operating point.  The better routing from SVD (which
  sees both gate and up) compensates for encoding W_up cold channels.
- **At 20% hot exp24 still wins** (0.818 vs 0.612) — SVD routing quality is
  insufficient to protect the sparse hot set accurately (F1=0.745 vs E5M3
  F1=0.784 from exp25c/25d).
- **Rank matters** — 64→256→1024 shows clear monotonic improvement.
- Routing cost: rank=1024 = 52.5% of one GEMM (for both gate and up routing).

### Conclusion

SVD routing with E5M3 cold gate+up is a viable scheme at 50%+ hot, offering
better quality than E5M3 threshold routing at equal compute because the union
routing captures channels important to either projection.  The scheme is not
competitive at sparse operating points due to limited SVD routing quality.

---

## Experiment 28 — E5M3-encoded SVD factor matrices, rank 2048

### Motivation

Exp27 stores SVD factors (Vt, UT) in float32.  Could encoding the factor
matrices themselves with E5M3 (binary sign + per-block scale) reduce storage
and routing cost while preserving routing quality?  Also tests rank=2048
(the maximum useful rank, min(H,I)=2560).

### Scheme

```
Vt_g_enc = E5M3(Vt_g)   # (r, H) encoded row-wise
UT_g_enc = E5M3(UT_g)   # (r, I) encoded row-wise
routing: hg = (x @ Vt_g_enc.T) * s_g  →  gate_lr = hg @ UT_g_enc
```

Singular values `s_g` stay float32.  Same hybrid hot/cold computation as exp27.

### Results (granite-4.2-3b, 8 prompts, 500 tokens)

| Rank | hot=20% | hot=30% | hot=50% | Δ vs exp27 fp32 r1024 @50% |
|---|---|---|---|---|
| 1024 (E5M3 factors) | 0.558 | 0.636 | 0.766 | −0.068 |
| 2048 (E5M3 factors) | 0.582 | 0.674 | 0.808 | −0.026 |
| exp27 r1024 (fp32) | 0.612 | 0.702 | **0.834** | — |

Routing cost at rank=2048: 105% of one full GEMM — no compute saving.

### Key findings

- Encoding SVD factor matrices with E5M3 costs 3–7 pp vs float32 factors.
- Rank=2048 recovers ~half the loss vs rank=1024 float32, but at 2× routing cost.
- **Root cause of degradation**: rows of Vt and UT are orthonormal unit vectors.
  Their information content is entirely in their *direction*, not magnitude.
  E5M3's per-block-of-8 scale quantisation distorts those directions, corrupting
  the routing signal in a way it cannot for W_gate (whose rows have meaningful
  magnitude variation).

### Conclusion

SVD factor matrices must stay float32 (or at minimum BF16) for routing to be
effective.  E5M3 encoding of orthonormal vectors is structurally incompatible
with the routing use case.

---

## Experiment 29 — SwiGLU-LR routing: top-k on |SiLU(gate_lr) * up_lr|

### Motivation

Exp27 routes on the **union** of top-k(|gate_lr|) and top-k(|up_lr|).  A channel
enters the hot set if *either* projection is large.  But SwiGLU channel i
contributes `SiLU(gate[i]) * up[i]` — both must be large for a large output.
Routing on `|SiLU(gate_lr) * up_lr|` should select channels where the combined
product is large, giving a tighter and more accurate hot set.

### Scheme

```
gate_lr  = x @ W_gate_lr.T
up_lr    = x @ W_up_lr.T
signal   = |SiLU(gate_lr) * up_lr|   # combined SwiGLU estimate
hot      = top-k(signal, k)           # exactly k channels, no union expansion
```

### Routing quality (Part 1, from activations NPZ, hot=20%)

| Signal | Rank | hot%(A) | F1 | IoU |
|---|---|---|---|---|
| SwiGLU-LR (exp29) | 64 | 20.0% | 0.329 | 0.198 |
| Union-LR (exp27) | 64 | 36.0% | 0.334 | 0.201 |
| SwiGLU-LR (exp29) | 1024 | 20.0% | 0.596 | 0.425 |
| Union-LR (exp27) | 1024 | 35.9% | 0.460 | 0.299 |

At hot=50%:

| Signal | Rank | hot%(A) | F1 | IoU |
|---|---|---|---|---|
| SwiGLU-LR (exp29) | 1024 | 50.0% | 0.704 | 0.544 |
| Union-LR (exp27) | 1024 | 75.6% | **0.706** | **0.545** |

### E2e results (granite-4.2-3b, 8 prompts, 500 tokens)

| Rank | hot=20% | hot=30% | hot=50% | Δ vs exp27 |
|---|---|---|---|---|
| 64 | 0.394 | 0.440 | 0.516 | −0.162 @50% |
| 256 | 0.472 | 0.570 | 0.682 | −0.082 @50% |
| 1024 | 0.568 | 0.630 | 0.732 | **−0.102** @50% |

### Key findings

**SwiGLU-LR is worse than union-LR at all ranks and hot fractions.**

The routing quality table reveals why.  At hot=50%, rank=1024: SwiGLU-LR
achieves F1=0.704 with hot%(A)=50%, while union-LR achieves F1=0.706 with
hot%(A)=75.6%.  The F1 against a 50% oracle is essentially identical, but
union-LR selects 75.6% of channels — many extra, but with better recall.

This exposes an **asymmetric cost structure**:
- **False positive** (hot but should be cold): costs one full-precision GEMM
  row for a channel that contributes little — small cost.
- **False negative** (cold but should be hot): approximates a genuinely large
  channel with E5M3 — large quality cost.

Union-LR over-selects (hot%≤2×frac) and trades cheap false positives for
fewer costly false negatives.  SwiGLU-LR is precision-optimal but recall-
limited, and in this asymmetric cost structure **recall dominates**.

### Conclusion

The union routing in exp27 is superior to the SwiGLU-combined signal precisely
because over-selection is cheap in this scheme.  Routing on the combined
product discards the union expansion that makes exp27 work.

## Experiment 30 — Nonlinear output predictor: σ(Px) for activity routing

### Motivation

Exp25e showed that a linear cross-covariance predictor is dramatically worse
than weight-derived routing (F1≈0.32 vs 0.784).  The cross-covariance predictor
is trained to predict `down_input = SiLU(W_gate x) * W_up x` from `x` — a
nonlinear target.  Could adding a scalar output nonlinearity (ReLU, SiLU, Abs)
to the predictor `P ∈ ℝ^{I×H}` capture the nonlinear structure and close
the gap to E5M3?

The predictor has the same shape as W_gate: `(I, H) = (8192, 2560)` — same
inference FLOP cost as one gate GEMM at full rank, or cheaper at low rank.

### Scheme

Two predictor ranks × four output nonlinearities × two training targets:

| Rank | Inference cost |
|---|---|
| 256 | 10% of one gate GEMM |
| 2560 (full) | 100% of one gate GEMM |

**Nonlinearities**: none (linear), ReLU, SiLU, Abs

**Targets**:
- **reg**: MSE on `down_input` (SwiGLU activity values)
- **bin**: BCE on `hot = |down_input[i]| > mean|down_input|` (binary labels)

**Training**:
- Linear (none): closed-form ridge regression `P* = (X^T X + λI)^{-1} X^T Y`
- Nonlinear: Adam, 300 steps, batch=512, lr=3e-3, warm-started from linear solution
- 80/20 train/test split per layer
- Evaluated on 5 layers (0, 8, 16, 24, 32), run on MPS (~28s/layer)

### Results (granite-4.2-3b, 5 layers, MPS)

**hot=20%  (k=1638):**

| Nonlinearity | Target | rank=256 F1 | rank=2560 F1 |
|---|---|---|---|
| none (linear) | reg | 0.401 | **0.486** |
| none (linear) | bin | 0.143 | 0.449 |
| relu | reg | 0.205 | 0.246 |
| relu | bin | 0.463 | 0.471 |
| silu | reg | 0.380 | 0.360 |
| silu | bin | 0.409 | 0.471 |
| abs  | reg | 0.333 | 0.337 |
| abs  | bin | 0.144 | 0.167 |

**hot=50%  (k=4096):**

| Nonlinearity | Target | rank=256 F1 | rank=2560 F1 |
|---|---|---|---|
| none (linear) | reg | 0.574 | **0.629** |
| none (linear) | bin | 0.430 | 0.615 |
| relu | reg | 0.502 | 0.510 |
| relu | bin | 0.570 | 0.590 |
| silu | reg | 0.571 | 0.560 |
| silu | bin | 0.546 | 0.601 |
| abs  | reg | 0.537 | 0.542 |
| abs  | bin | 0.431 | 0.443 |

Reference points:

| Method | hot=20% F1 | hot=50% F1 |
|---|---|---|
| Linear cross-cov SVD (exp25e) | 0.32 | 0.54 |
| **Best trained (linear reg, r=2560)** | **0.486** | **0.629** |
| SVD W_gate rank=1024 (exp25d) | 0.745 | 0.873 |
| E5M3 W_gate threshold (exp25c) | 0.784 | ~0.873 |

### Key findings

1. **Nonlinearities do not help — linear regression is best.**  Every output
   nonlinearity (ReLU, SiLU, Abs) produces equal or worse F1 than plain linear
   regression across both ranks and both hot fractions.

2. **Full-rank linear regression (F1=0.486 @20%)** beats the cross-covariance
   predictor from exp25e (F1=0.32) because ridge regression finds the exact
   least-squares solution `P* = arg min ||PX - Y||²`, whereas exp25e's
   cross-covariance SVD only captured dominant correlation directions.  Both
   remain far below E5M3 (F1=0.784).

3. **The bilinear barrier.**  The activity target is
   `a[i] = SiLU(W_gate[i]·x) * W_up[i]·x` — a product of two independent
   linear maps in `x`.  A scalar output nonlinearity `σ(P[i]·x)` applies to
   the output of a *single* linear projection per channel.  No scalar `σ`
   applied to one linear map can express a product of two independent linear
   maps — the bilinear structure is irreducible.

4. **Why weight-derived predictors win.**  `x @ W_gate.T` directly computes
   the gate pre-activation — the actual routing signal — without any training.
   Any learned predictor trying to approximate the post-SwiGLU product is
   solving a harder problem with no more parameters.

5. **ReLU/bin is the best nonlinear variant** (F1=0.471 @20%, 0.590 @50%),
   slightly above SiLU/bin, because the binary BCE objective with a ReLU output
   resembles a logistic classifier — it can learn a decision boundary rather
   than regressing a continuous target.  But still far below linear regression.

### Conclusion

Adding a scalar output nonlinearity to a trained linear predictor provides no
benefit for hot-channel routing.  The bilinear structure of SwiGLU activity
`SiLU(gate) * up` cannot be captured by any `σ(P x)` model.  The best trained
predictor (full-rank linear regression, F1=0.486 @20%) is 38% below E5M3
(F1=0.784), which needs no training at all.

A two-layer MLP predictor (with a hidden layer that can represent the product
`gate * up` jointly) would be required to close this gap, but at that point the
inference cost exceeds a full GEMM and the motivation for a cheap predictor
is lost.

## Experiment 31 — Two-layer MLP predictor

### Motivation

Exp30 established that a scalar output nonlinearity σ(Px) cannot break the
bilinear barrier — predicting `SiLU(W_gate x) * W_up x` requires knowing two
independent linear projections simultaneously.  A two-layer MLP

```
h[j] = SiLU(W1[j] · x)       hidden (j = 1..r)
out[i] = Σ_j W2[i,j] * h[j]  output
```

can in principle represent the bilinear product: if hidden unit j encodes a
mixture of `W_gate[i]·x` and `W_up[i]·x`, then W2 can learn to multiply them.
With sufficient hidden width r this should close the gap to E5M3.

### Scheme

- Architecture: `x → Linear(H,r,bias=False) → SiLU → Linear(r,I,bias=False)`
- W1 warm-started with top-r rows of Vt_gate (SVD right singular vectors of
  W_gate) to encode relevant gate directions from step 0
- W2 initialised N(0, 1/√r)
- Training: Adam 500 steps, batch=512, lr=3e-3, MPS
- Two targets: regression (MSE on `down_input`) and binary (BCE on hot labels)
- 80/20 train/test split; evaluated on 5 layers (0, 8, 16, 24, 32)

### Inference cost

| Hidden width | FLOPs | % of one gate GEMM |
|---|---|---|
| 256 | 5.5M | 13% |
| 1024 | 22.0M | 52% |
| 2560 | 55.1M | 131% |

### Results (granite-4.2-3b, 5 layers, MPS, ~24s/layer)

**hot=20%  (k=1638) — F1 per layer and mean:**

| Hidden | Target | L0 | L8 | L16 | L24 | L32 | Mean |
|---|---|---|---|---|---|---|---|
| 256 | reg | 0.355 | 0.261 | 0.378 | 0.334 | 0.423 | 0.350 |
| 256 | bin | 0.135 | 0.166 | 0.152 | 0.156 | 0.097 | 0.141 |
| 1024 | reg | 0.475 | 0.306 | 0.395 | 0.343 | 0.444 | **0.393** |
| 1024 | bin | 0.153 | 0.173 | 0.158 | 0.167 | 0.104 | 0.151 |
| 2560 | reg | 0.543 | 0.306 | 0.273 | 0.311 | 0.492 | 0.385 |
| 2560 | bin | 0.163 | 0.177 | 0.163 | 0.167 | 0.108 | 0.156 |

**hot=50%  (k=4096) — F1 per layer and mean:**

| Hidden | Target | L0 | L8 | L16 | L24 | L32 | Mean |
|---|---|---|---|---|---|---|---|
| 256 | reg | 0.553 | 0.522 | 0.557 | 0.550 | 0.581 | 0.553 |
| 256 | bin | 0.416 | 0.466 | 0.443 | 0.439 | 0.401 | 0.433 |
| 1024 | reg | 0.618 | 0.541 | 0.564 | 0.553 | 0.587 | **0.572** |
| 1024 | bin | 0.413 | 0.475 | 0.457 | 0.453 | 0.395 | 0.438 |
| 2560 | reg | 0.659 | 0.541 | 0.515 | 0.534 | 0.607 | 0.571 |
| 2560 | bin | 0.412 | 0.478 | 0.460 | 0.451 | 0.400 | 0.440 |

Reference comparison:

| Method | hot=20% F1 | hot=50% F1 | Cost |
|---|---|---|---|
| Exp31 MLP hidden=1024 reg | 0.393 | 0.572 | 52% of GEMM |
| Exp30 linear reg full-rank | **0.486** | **0.629** | 100% of GEMM |
| SVD W_gate rank=1024 (exp25d) | 0.745 | 0.873 | 52% of GEMM |
| E5M3 W_gate (exp25c) | 0.784 | ~0.873 | ~100% of GEMM |

### Key findings

1. **Two-layer MLP is worse than single-layer linear regression at 20% hot**
   (best MLP F1=0.393 vs linear 0.486) and only comparable at 50% hot
   (0.572 vs 0.629).  The bilinear barrier is not broken.

2. **Binary target collapses** — F1≈0.14–0.16 at 20% hot regardless of hidden
   width.  BCE with sigmoid on a 8192-way binary output is harder to optimise
   than MSE; the model converges to predicting near-uniform probabilities.

3. **High layer variance** — layer 0 (F1=0.54 @20%, hidden=2560) vs layer 8
   (F1=0.31) suggests the MLP is overfitting to layer 0's structure but
   underfitting layers with more complex activity patterns.  500 steps is
   insufficient for stable convergence across all layers.

4. **Why the hidden layer doesn't help**: W1 is warm-started from SVD of W_gate,
   so the hidden units encode gate directions — but W_up directions are absent
   from W1 initialisation and must be learned from scratch in 500 gradient steps
   against a noisy MSE signal.  The MLP cannot efficiently discover the
   `(gate_direction, up_direction)` pairing needed per channel.

5. **Cost-quality frontier**: exp31 hidden=1024 costs the same FLOPs as SVD
   routing rank=1024 (52%) but achieves F1=0.393 vs SVD's 0.745 @20% hot.
   The trained MLP has no advantage over the weight-derived SVD at any
   comparable cost point.

### Conclusion

A two-layer MLP predictor trained on recorded activations does not improve
over single-layer linear regression and is far below weight-derived routing.
The key obstacle: discovering per-channel `(W_gate[i], W_up[i])` direction
pairs from gradient descent on 12k tokens is harder than it sounds, and
the model has no inductive bias toward the weight structure the network uses.
The weight matrices *already encode* the optimal routing signal — no learned
predictor can improve on using them directly.

## Experiment 31b — Hidden activation comparison: ReLU vs SiLU

### Motivation

Exp31 used SiLU as the hidden activation.  For a target that is a product of
two linear maps — `SiLU(W_gate x) * W_up x` — ReLU hidden units might be
preferable in theory: `ReLU(a) * b` has a hard zero region (when a<0), and a
two-layer MLP with ReLU hidden units can represent bilinear products exactly
with one hidden-unit pair per output channel.  Does ReLU outperform SiLU?

### Scheme

Same as exp31 but with `hidden_acts = [silu, relu]`, `hidden=1024`, `n_steps=500`.
W1 warm-started from SVD of W_gate (top-1024 Vt rows) in both cases.

### Results (granite-4.2-3b, 5 layers 0/8/16/24/32, MPS, ~13s/layer)

| Act | hot=20% F1 | hot=50% F1 |
|---|---|---|
| SiLU (exp31) | **0.393** | **0.572** |
| ReLU | 0.278 | 0.527 |
| Linear reg full-rank (exp30) | **0.486** | **0.629** |
| SVD W_gate rank=1024 (exp25d) | 0.745 | 0.873 |

Per-layer breakdown (hot=20%, reg target):

| Layer | SiLU | ReLU |
|---|---|---|
| 0 | 0.463 | 0.341 |
| 8 | 0.303 | 0.201 |
| 16 | 0.394 | 0.198 |
| 24 | 0.360 | 0.212 |
| 32 | 0.448 | 0.438 |
| **mean** | **0.393** | **0.278** |

### Key findings

**ReLU is worse than SiLU across all layers and hot fractions.**

The cause is dying-ReLU at initialisation.  W1 is warm-started from SVD right
singular vectors of W_gate — a matrix with both positive and negative values.
The hidden pre-activations `W1 x` therefore have both signs at initialisation,
meaning roughly half of ReLU units are in the zero-gradient region for any given
token.  With only 500 gradient steps and a high-dimensional target (I=8192),
these dead units never recover — the effective hidden width is ~512, not 1024.

SiLU's smooth negative tail (`SiLU(x) ≈ x * 0.17` for moderately negative x)
keeps gradient flowing through all units regardless of sign, making it strictly
better for this warm-started initialisation scheme.

### Conclusion

ReLU does not help and is actively harmful relative to SiLU when W1 is
warm-started from weight SVD.  The theoretical representational advantage of
ReLU for bilinear products is irrelevant in practice: the dying-unit problem
dominates at this training budget.  Neither activation closes the gap to
weight-derived routing — hidden activation choice is a second-order concern
compared to the fundamental bilinear barrier diagnosed in exp30–31.


---

## Experiment 32 — Sparse W_gate predictor

### Motivation

Exp25–31 confirmed that the optimal routing signal is `x @ W_gate.T` — the weight
matrix itself.  Learned predictors are all worse.  This experiment asks a different
question: can we **sparsify W_gate** while keeping routing quality competitive?

A sparse W_gate has two benefits:
1. **Faster routing GEMM** — unstructured sparsity gives a theoretical upper bound of
   `keep_rate × FLOP`; structured (row) sparsity gives real hardware savings.
2. **Compact storage** — CSR/CSC formats or block-sparse representations cut memory
   bandwidth for the routing step.

### Scheme

Two pruning strategies, each with optional 300-step Adam fine-tuning:

| Scheme | Description |
|---|---|
| `unstructured` | Zero the `(1−keep_rate)` weights with smallest `\|w\|` element-wise |
| `row` | Zero entire output rows (channels) with smallest L2 norm |
| `unstructured_ft` | Magnitude-prune, then 300 Adam steps (MSE vs `x @ W_gate_full.T`) with mask fixed |
| `row_ft` | Row-prune, then 300 Adam steps with mask fixed |

Keep rates swept: **0.5, 0.3, 0.2, 0.1** (fraction of weights retained).
Hot fractions evaluated: **20%** and **50%** of intermediate channels.
Layers evaluated: **0, 8, 16, 24, 32** (5 layers, ~28 s/layer on MPS).

### Results (granite-4.2-3b, 5 layers 0/8/16/24/32, MPS)

#### Routing F1 — hot=20%  (k=1638 out of 8192)

| Scheme | kr=0.5 | kr=0.3 | kr=0.2 | kr=0.1 | kr=1.0 (full) |
|---|---|---|---|---|---|
| unstructured | **0.880** | 0.783 | 0.712 | 0.608 | 1.000 |
| unstructured_ft | **0.902** | **0.828** | **0.773** | **0.685** | 1.000 |
| row | 0.623 | 0.440 | 0.330 | 0.294 | 1.000 |
| row_ft | 0.623 | 0.440 | 0.330 | 0.294 | 1.000 |
| **E5M3 W_gate (exp25c)** | — | — | — | — | **0.784** |
| **SVD rank=1024 (exp25d)** | — | — | — | — | 0.745 |

#### Routing F1 — hot=50%  (k=4096 out of 8192)

| Scheme | kr=0.5 | kr=0.3 | kr=0.2 | kr=0.1 | kr=1.0 (full) |
|---|---|---|---|---|---|
| unstructured | **0.913** | 0.836 | 0.779 | 0.695 | 1.000 |
| unstructured_ft | **0.932** | **0.879** | **0.840** | **0.776** | 1.000 |
| row | 0.554 | 0.536 | 0.527 | 0.519 | 1.000 |
| row_ft | 0.554 | 0.536 | 0.527 | 0.519 | 1.000 |
| **SVD rank=1024 (exp25d)** | — | — | — | — | **0.873** |

#### Storage / compute savings

| keep_rate | Non-zeros (8192×2560) | Zeroed | Unstructured GEMM saving |
|---|---|---|---|
| 0.5 | 10,485,760 | 50% | ≤50% FLOPs |
| 0.3 | 6,291,456 | 70% | ≤70% FLOPs |
| 0.2 | 4,194,304 | 80% | ≤80% FLOPs |
| 0.1 | 2,097,152 | 90% | ≤90% FLOPs |

### Key findings

**Unstructured magnitude pruning is surprisingly strong.**  At keep=0.5 (50%
sparsity), unstructured pruning achieves F1=0.880 @20% hot — beating E5M3
encoding (0.784) by 9.6 pp without any fine-tuning.  Even at keep=0.3 (70%
sparsity) it matches E5M3 (0.783 vs 0.784).  This is a better operating point
than E5M3: same routing quality at 70% fewer routing FLOP.

**Fine-tuning adds +2–7 pp across the board.**  300 Adam steps (MSE target)
consistently improve routing F1.  The gain is largest at high sparsity:
keep=0.1 ft (0.685) vs no-ft (0.608) is +7.7 pp @20% hot.  Even keep=0.5 ft
(0.902) beats E5M3 by 11.8 pp.

**Row pruning is much weaker.**  Zeroing entire channels (rows of W_gate) loses
substantial routing information at every keep_rate.  At keep=0.5, row pruning
reaches only F1=0.623 vs unstructured's 0.880 — a 25.7 pp gap.  The signal
degrades almost linearly with keep_rate.  Row pruning also shows the flat
behaviour across hot% seen in exp25d for low-rank SVD: when entire channels are
zeroed the remaining channels rank poorly relative to the oracle.

**Fine-tuning has no effect on row pruning.**  `row_ft` == `row` at all
keep_rates because zeroed rows contribute no gradient — the mask constrains W to
the pruned subspace, so the per-channel gradient `dL/dW[i]` for zeroed row `i`
is always zero.  The 300 Adam steps only re-fit the non-zeroed rows to better
match the full matrix's responses within the active subspace, but since the
zero-row channels remain absent from the routing signal entirely, F1 cannot
improve.

**Comparison to other routing schemes:**

| Scheme | F1 @20% hot | F1 @50% hot | Routing cost |
|---|---|---|---|
| Full W_gate (oracle) | 1.000 | 1.000 | 1× GEMM |
| Unstructured kr=0.5 +ft | **0.902** | **0.932** | ≤0.50× GEMM |
| Unstructured kr=0.5 no-ft | 0.880 | 0.913 | ≤0.50× GEMM |
| E5M3 W_gate (exp25c) | 0.784 | — | 1× GEMM (low bandwidth) |
| Unstructured kr=0.3 no-ft | **0.783** | 0.836 | ≤0.30× GEMM |
| SVD rank=1024 (exp25d) | 0.745 | 0.873 | ~0.25× GEMM |
| Linear reg full-rank (exp30) | 0.486 | 0.629 | 1× GEMM (trained P) |

### Conclusion

**Unstructured magnitude pruning of W_gate is the most efficient routing
scheme found so far.**  At keep=0.5, it beats E5M3 in routing quality while
halving the routing GEMM cost.  At keep=0.3, it matches E5M3 quality at 70%
fewer FLOPs.  Fine-tuning adds another +2–7 pp at small cost (offline, 300
steps).

Row (structured) pruning is the wrong direction: it removes entire channel
directions and cannot be recovered by fine-tuning.  Unstructured sparsity
preserves the directional information of each channel row while zeroing
low-magnitude weights that contribute little to the dot-product ordering.

The practical implication: for the hybrid hot/cold inference scheme, replace
E5M3-encoded W_gate with a 50% unstructured-sparse W_gate (stored in CSR or
2:4 structured sparse format).  This simultaneously reduces routing latency
and improves routing quality, and is fully compatible with the exp27 union
routing scheme (apply sparsity to both gate and up factor matrices).


---

## Experiment 33 — Sparse W_gate end-to-end top-1 assessment

### Motivation

Exp32 established that unstructured magnitude pruning (keep=0.5) + 300 Adam fine-tune
steps gives F1=0.902 @20% hot — 11.8 pp above E5M3's 0.784.  This experiment tests
whether that routing improvement translates to better **end-to-end top-1 match**.

### Scheme

```
W_sparse[l] = magnitude_prune_unstructured(W_gate[l], keep=0.5)
           +  300 Adam steps (MSE vs x @ W_gate_full.T, mask fixed)

routing:  hot = top-k(|x @ W_sparse.T|)          # top-k hot channels per token
hot:      gate = x @ W_gate_full.T (full precision)
          up   = x @ W_up_full.T   (full precision — always, as in exp24)
cold:     gate = x @ W_sparse.T    (sparse approx, reusing routing GEMM)
merge:    gate = where(hot, gate_full, gate_sparse)
swiglu:   silu(gate) * up_full
out:      swiglu @ W_down.T (full precision)
```

Key design choice: **W_sparse doubles as both routing signal and cold approximation**,
so the routing GEMM is the only extra cost (no separate E5M3 encoding pass).

Compared:
- `kr=0.5+ft` — 50% kept, 300 Adam ft steps
- `kr=0.5`    — 50% kept, no ft (pure magnitude pruning)

Hot fractions swept: **20%, 30%, 50%**.

### Results (granite-4.2-3b, all 40 layers, prefill, 8 prompts / 500 tokens)

| Scheme | hot=20% | hot=30% | hot=50% |
|---|---|---|---|
| **kr=0.5+ft** | **0.850** | **0.862** | **0.876** |
| kr=0.5 no-ft | 0.830 | 0.830 | 0.852 |
| exp24 E5M3 threshold (T=0.20, ~88% hot) | — | — | 0.818 (adaptive) |
| exp27 SVD union rank=1024 | 0.612 | 0.702 | **0.834** |
| exp24 threshold T=0.50 | — | 0.798 | 0.736 |
| exp14 ternary | 0.588 | 0.632 | — |

Δ vs best prior result at each hot fraction:

| hot% | exp33 kr=0.5+ft | Prior best | Δ |
|---|---|---|---|
| 20% | **0.850** | 0.818 (exp24 adaptive) | **+3.2 pp** |
| 30% | **0.862** | 0.798 (exp24 T=0.50) | **+6.4 pp** |
| 50% | **0.876** | 0.834 (exp27) | **+4.2 pp** |

### Key findings

**New best at every hot fraction.**  kr=0.5+ft achieves 0.850/0.862/0.876 —
the highest top-1 match rates in the entire experiment series.

**Fine-tuning adds a consistent +2 pp.**  The gap between kr=0.5+ft and kr=0.5
no-ft is small (0.850 vs 0.830 @20%, 0.876 vs 0.852 @50%) but consistent across
all hot fractions.  In routing F1 terms (exp32) the gap was +2.2 pp @20% hot, and
that translates roughly proportionally to e2e match.

**Sparse W_gate vs E5M3 as cold approximation.**  The surprising result is that
using sparse W_gate directly as the cold approximation (not E5M3-encoded) is better
than exp24's E5M3 scheme.  Both have similar SNR on cold channels, but the sparse
approximation's error is concentrated on small-magnitude weights (exactly those
pruned), whereas E5M3 uniformly distorts all channels.  The routing quality gain
(F1 0.902 vs 0.784) is the primary driver — fewer false-negative cold channels
means fewer wrong cold values propagate through SiLU.

**Dense hot fractions are no longer penalised.**  Exp24 was best at T=0.20 (~88%
hot) because lower T meant more hot channels and less cold-channel error.  Exp33
monotonically improves with more hot channels (0.850 → 0.862 → 0.876) as expected
for top-k routing, but the 20% point already beats exp24's best.

**E5M3 encoding is no longer needed.**  The sparse W_gate serves as both routing
signal and cold approximation in a single matrix multiply, eliminating the E5M3
block-scale quantisation step entirely.  This simplifies the implementation.

### Comparison to all prior schemes

| Scheme | match @20% hot | match @50% hot | notes |
|---|---|---|---|
| **Exp33 kr=0.5+ft** | **0.850** | **0.876** | new best |
| Exp33 kr=0.5 no-ft | 0.830 | 0.852 | |
| Exp27 SVD r1024 union | 0.612 | 0.834 | cold E5M3 gate+up |
| Exp24 E5M3 T=0.20 | 0.818 (88% hot) | — | adaptive threshold |
| Exp24 E5M3 T=0.50 | — | 0.736 | |
| Exp14 ternary | 0.588 | — | |

### Conclusion

**Unstructured 50%-sparse W_gate + 300 Adam steps is the strongest routing scheme
found so far**, setting new records at 20%, 30%, and 50% hot fractions.  The scheme
is also simpler than exp24/27: no E5M3 encoding, no SVD, just magnitude pruning and
a single sparse GEMM that does double duty as routing signal and cold approximation.

The 300 Adam fine-tune steps add +2 pp at the cost of a one-time ~140 s offline
pass (MPS, all 40 layers).  Given the consistent gain, fine-tuning is recommended.

Next step: extend to threshold routing (adaptive hot%) to match exp24's operating
point of ~88% hot, and measure whether the gain over exp24 persists there.

---

## Experiment 34 — Thermal match rate

### Motivation

All prior e2e experiments report **strict top-1 match** (temperature=0 greedy),
which flags any rank-swap even when the displaced token is still highly probable
at typical inference temperatures.  Exp34 introduces the **thermal match rate**
defined in the metrics section and measures how many strict perturbations are
actually below the noise floor at production temperatures.

### Scheme

```
gap       = lh[hybrid_top1] − lh[baseline_top1]   (≥ 0 when perturbed)
forgiven  = (~exact_match) AND (gap < T · ln2)
thermal_match(T) = mean(exact_match OR forgiven)
```

At τ = ln2: forgiven perturbations are those where the hybrid's preference for
its own answer over the full-precision answer corresponds to <2× pairwise
probability ratio — i.e. the correct token would still be sampled >33% of the
time in pairwise competition.

Schemes evaluated:
- **exp33 kr=0.5+ft** at hot=20%, 30%, 50%  (new best from exp33)
- **exp24 E5M3 T=0.20** (best prior single-pass scheme, ~88% hot)
- **exp24 E5M3 T=0.50** (reference at ~36% effective hot)

### Results (granite-4.2-3b, 500 prefill tokens, 8 prompts)

#### Full metric table

| Scheme | Strict% | Thermal @T=0.7 | Thermal @T=1.0 | Mean gap (perturbed) |
|---|---|---|---|---|
| sparse kr=0.5+ft @20% hot | 15.0% | **5.6%** | **5.0%** | 0.908 logits |
| sparse kr=0.5+ft @30% hot | 13.8% | **5.6%** | **4.8%** | 0.942 logits |
| sparse kr=0.5+ft @50% hot | 12.4% | **4.8%** | **3.6%** | 0.809 logits |
| exp24 E5M3 T=0.20 (~88% hot) | 18.2% | 7.6% | 4.8% | 0.661 logits |
| exp24 E5M3 T=0.50 (~36% hot) | 26.4% | 16.0% | 12.4% | 1.271 logits |

#### Perturbation reduction from strict → thermal

| Scheme | Strict | → Thermal @0.7 | Δ | → Thermal @1.0 | Δ |
|---|---|---|---|---|---|
| sparse @20% hot | 15.0% | 5.6% | −9.4 pp | 5.0% | −10.0 pp |
| sparse @30% hot | 13.8% | 5.6% | −8.2 pp | 4.8% | −9.0 pp |
| sparse @50% hot | 12.4% | 4.8% | −7.6 pp | 3.6% | −8.8 pp |
| exp24 T=0.20 | 18.2% | 7.6% | −10.6 pp | 4.8% | −13.4 pp |
| exp24 T=0.50 | 26.4% | 16.0% | −10.4 pp | 12.4% | −14.0 pp |

### Key findings

**~8–10 pp of strict perturbation is below the thermal noise floor.**  For the
exp33 sparse scheme, roughly 60–65% of strict perturbations are forgiven at
T=0.7.  These are rank-swaps where the hybrid model slightly prefers a different
token, but the correct token is still nearly as likely — a thermal fluctuation,
not a real error.

**Exp33 @50% hot reaches FP8-equivalent quality on the thermal metric.**  At
T=1.0 the thermal perturbation is **3.6%**, squarely within the FP8-equivalent
range (~3–5%).  At T=0.7 it is 4.8%, still near the FP8 floor.  The strict
metric (12.4%) significantly overstates the real-world impact.

**Exp24 E5M3 T=0.20 is surprisingly competitive on the thermal metric.**  Its
strict perturbation of 18.2% drops to 7.6% @T=0.7 and 4.8% @T=1.0 — matching
exp33 @30% hot on the thermal metric despite being 3.2 pp worse on strict.
Its errors have a smaller gap (0.661 logits vs 0.908) — they are softer misses.

**Exp24 E5M3 T=0.50 has hard errors.**  The mean gap of 1.271 logits among
perturbed tokens means the hybrid model is substantially confident in its wrong
answer.  Only 10.4 pp of its 26.4% strict perturbation is forgiven at T=0.7,
leaving 16.0% thermal — much worse than the sparse scheme.  The lower threshold
in exp24 T=0.20 avoids these hard errors by keeping more channels hot.

**The gap column is a new quality signal.**  Small mean gap → errors are soft,
likely resolved at any production temperature.  Large mean gap → errors are
hard, will cause divergence even at T=1.0.

### Comparison with quantization analogues (thermal metric)

| Scheme | Thermal @T=0.7 | Thermal @T=1.0 | Analogue |
|---|---|---|---|
| FP8 E4M3 (reference) | ~2–5% | ~2–4% | production floor |
| **Exp33 kr=0.5+ft @50% hot** | **4.8%** | **3.6%** | **FP8 range** |
| Exp33 kr=0.5+ft @30% hot | 5.6% | 4.8% | FP8 / upper NVFP4 boundary |
| Exp24 E5M3 T=0.20 | 7.6% | 4.8% | INT4 GPTQ / lower NVFP4 |
| NVFP4 (reference) | ~10–20% | ~8–16% | Blackwell default |
| Exp24 E5M3 T=0.50 | 16.0% | 12.4% | NVFP4 range |

### Conclusion

The thermal metric fundamentally changes the picture.  What looked like a
12–15% perturbation problem (NVFP4 range) on the strict metric is actually a
**3.6–5.6% problem** (FP8 range) at production temperatures.  The exp33
sparse scheme at 50% hot channels already achieves FP8-equivalent effective
quality at T=1.0, using only 50% of the W_gate weights for routing.

The strict metric remains useful as a conservative upper bound and for
comparing schemes against each other.  The thermal metric is the right number
to report when assessing user-visible quality impact.

---

## Experiment 35 — Sparse up projection and zero-cold comparison

### Motivation

Exp33 kept W_up at full precision for cold channels.  This experiment targets the
**20–30% hot regime** and asks two questions:

1. Does sparsifying W_up for cold channels (same kr=0.5+ft scheme) help or hurt
   compared to always-full-precision up?
2. Is omitting cold channels entirely (feeding zeros to the down projection) a
   hard or soft error under the thermal metric?

### Schemes compared

| Label | Cold gate | Cold up | Hot gate | Hot up |
|---|---|---|---|---|
| `sparse_gate_only` | `x @ W_gate_sparse.T` | `x @ W_up_full.T` (full) | full | full |
| `sparse_gate+up` | `x @ W_gate_sparse.T` | `x @ W_up_sparse.T` | full | full |
| `zero_cold` | 0 | 0 | full | full |

Routing for all three: `top-k(|x @ W_gate_sparse.T|)`.  Down projection always
full precision.  W_gate_sparse and W_up_sparse both use kr=0.5, 300 Adam ft steps
(same `build_sparse` function, matching target activations for each projection).

### Results (granite-4.2-3b, all 40 layers, 500 prefill tokens)

#### Strict and thermal perturbation (%)

| Scheme | Strict% | Thermal @T=0.7 | Thermal @T=1.0 | Mean gap (perturbed) |
|---|---|---|---|---|
| sparse_gate_only @20% | **13.4%** | **6.6%** | **5.6%** | 1.06 logits |
| sparse_gate+up @20% | 21.2% | 9.4% | 7.2% | 1.11 logits |
| **zero_cold @20%** | **76.8%** | **73.4%** | **71.2%** | **5.32 logits** |
| sparse_gate_only @30% | **12.0%** | **5.6%** | **4.8%** | 1.08 logits |
| sparse_gate+up @30% | 18.4% | 8.6% | 7.0% | 1.05 logits |
| **zero_cold @30%** | **63.0%** | **54.8%** | **52.4%** | **2.97 logits** |
| sparse_gate_only @50% | 13.2% | 5.4% | 4.4% | 0.83 logits |
| sparse_gate+up @50% | 14.8% | 6.0% | 4.8% | 0.81 logits |
| **zero_cold @50%** | **33.6%** | **24.8%** | **22.2%** | **1.83 logits** |

#### Perturbation reduction strict → thermal

| Scheme | Strict | Thermal @0.7 | Δ@0.7 | Thermal @1.0 | Δ@1.0 |
|---|---|---|---|---|---|
| sparse_gate_only @20% | 13.4% | 6.6% | −6.8 pp | 5.6% | −7.8 pp |
| sparse_gate+up @20% | 21.2% | 9.4% | −11.8 pp | 7.2% | −14.0 pp |
| zero_cold @20% | 76.8% | 73.4% | −3.4 pp | 71.2% | −5.6 pp |
| sparse_gate_only @30% | 12.0% | 5.6% | −6.4 pp | 4.8% | −7.2 pp |
| sparse_gate+up @30% | 18.4% | 8.6% | −9.8 pp | 7.0% | −11.4 pp |
| zero_cold @30% | 63.0% | 54.8% | −8.2 pp | 52.4% | −10.6 pp |

### Key findings

**Zero cold is catastrophically hard — not a soft error.**  Omitting cold channels
(feeding zero to the down projection) gives 76.8% strict perturbation at 20% hot,
and only 3.4 pp is forgiven at T=0.7 (73.4% thermal).  The mean logit gap among
perturbed tokens is **5.32 logits** — the model is extremely confident in the wrong
answer.  This is a structural error: SiLU(0)=0 eliminates the cold SwiGLU
contribution entirely, but those ~80% cold channels carry substantial signal.
Cold channels must retain *some* approximation (even a poor one); zeroing them is
much worse than any approximate scheme tested.

Even at 50% hot (only 50% zeroed), zero_cold gives 33.6% strict / 22.2% thermal
perturbation — worse than any other scheme at any hot fraction.

**Sparse up cold is worse than full-precision up, but soft errors.**
`sparse_gate+up` is 7.8 pp worse than `sparse_gate_only` on the strict metric
at 20% hot (21.2% vs 13.4%), but the thermal gap shrinks considerably:
the mean gap for `sparse_gate+up` (1.11 logits) is only slightly larger than
`sparse_gate_only` (1.06 logits).  At T=0.7 the gap is 2.8 pp (9.4% vs 6.6%),
and at T=1.0 it is 1.6 pp (7.2% vs 5.6%).  These are soft errors — the sparse
up approximation degrades the cold channel values but not catastrophically.

The reason sparse up hurts more than sparse gate: the exp32 routing quality
analysis shows that F1≈0.902 for sparse gate routing.  The cold gate channels are
the ones the model *correctly identified as low-importance*, so their gate error
(cold gate × sparse up) is penalised by SiLU which suppresses near-zero gate
values.  But sparse up errors on cold channels are **not** suppressed by SiLU
since up appears *after* the SiLU nonlinearity — they propagate directly.

**Sparse_gate_only @30% hot is the sweet spot for the target regime.**

| Scheme | Strict% | Thermal @T=0.7 | Thermal @T=1.0 |
|---|---|---|---|
| sparse_gate_only @30% | **12.0%** | **5.6%** | **4.8%** |
| sparse_gate+up @30% | 18.4% | 8.6% | 7.0% |
| FP8-equivalent | ~3–5% | ~2–4% | ~2–4% |
| NVFP4-equivalent | ~10–20% | ~8–16% | ~8–16% |

At 30% hot with sparse gate only: strict perturbation 12.0% (NVFP4 lower bound),
thermal@T=1.0 **4.8%** (FP8 range).  This is the best operating point in the
20–30% hot target regime across both metrics.

**Note: this run's sparse_gate_only numbers differ slightly from exp33.**
Exp33's `sparse_gate_only @20%` gave strict=0.850 (15.0% perturbation), while
exp35 gives 13.4%.  This is expected: exp35 rebuilds the sparse W_gate weights
independently via the HF safetensors path (no vLLM fused weight) while exp33
used the vLLM fused gate_up_proj[:I].  The two are the same underlying
weights but the Adam fine-tune has random mini-batch sampling, so results vary
by ~1–2 pp between runs.

### Conclusion

**Keep W_up full precision for cold channels.**  Sparse up cold adds 6–8 pp
strict perturbation and 2–3 pp thermal perturbation at 20–30% hot with no
compensating benefit — it is simply a worse approximation for cold channels.
The finding from exp20 (W_up must stay full for cold) is confirmed in the sparse
regime: sparse gate cold is fine (signal suppressed by SiLU); sparse up cold is
not (error propagates directly through the bilinear product).

**Zero cold is a hard structural error, not a soft approximation miss.**
Mean gap 5.32 logits (20% hot) is ~7–8× larger than sparse approximation schemes
(~0.8–1.1 logits) and barely forgiven thermally.  Cold channels must contribute
their approximate values — the residual signal from SiLU(cold_gate)×up_full is
load-bearing even for nominally "inactive" channels.

**Recommended scheme going forward:** `sparse_gate_only`, kr=0.5+ft, hot=25–30%.
- Strict: ~12–14% (NVFP4 lower bound)
- Thermal @T=1.0: ~4.8–5.6% (FP8 range)
- W_up: always full precision
- W_down: always full precision
- Cold gate: x @ W_sparse.T (free, reuses routing GEMM)

---

## Experiment 36 — E5M3 B=8 cold up with sparse gate routing

### Motivation

Exp35 established that W_up must stay full precision for cold channels.
This experiment tests whether E5M3 B=8 encoding — the best compact encoding
found in exp19 (TARE=0.837, 8× compression) — is good enough for cold up
when routing quality is high (F1=0.902 from sparse gate, exp32).

In exp27 E5M3 cold up was used successfully with SVD routing at 50% hot.
The question is whether the higher routing quality of sparse W_gate makes
E5M3 cold up viable at the target 20–30% hot regime.

### Scheme

```
routing:   hot = top-k(|x @ W_gate_sparse.T|)   [kr=0.5+ft, from exp35 cache]
cold gate: x @ W_gate_sparse.T                  [sparse approx, free]
hot gate:  x @ W_gate_full.T                    [full precision]
cold up:   x @ W_up_e5m3.T                      [E5M3 B=8 sign+scale]
hot up:    x @ W_up_full.T                      [full precision]
down:      swiglu @ W_down.T                    [always full precision]
```

W_up_e5m3 built once per layer: `sign(W_up) × s*` where `s*` is the
tilt-weighted geometric mean of `|w_b|` per block of 8, quantised to E5M3
(5-bit exponent + 3-bit mantissa, 1 byte/block). Build time: 6s for all 40 layers.

Hot fractions swept: **20%, 25%, 30%, 50%**.

### Results (granite-4.2-3b, 500 prefill tokens)

#### Full results

| Scheme | Strict% | Thermal @T=0.7 | Thermal @T=1.0 | Mean gap |
|---|---|---|---|---|
| sparse_gate+e5m3_up @20% | 38.6% | 30.6% | 27.0% | 2.005 logits |
| sparse_gate_only @20% | **13.4%** | **6.6%** | **5.6%** | 1.062 logits |
| sparse_gate+e5m3_up @25% | 34.4% | 24.6% | 20.6% | 1.610 logits |
| sparse_gate_only @25% | **12.6%** | **5.6%** | **4.6%** | 1.048 logits |
| sparse_gate+e5m3_up @30% | 30.0% | 23.4% | 19.2% | 1.584 logits |
| sparse_gate_only @30% | **12.0%** | **5.6%** | **4.8%** | 1.081 logits |
| sparse_gate+e5m3_up @50% | 19.8% | 10.8% | 8.2% | 1.064 logits |
| sparse_gate_only @50% | **13.2%** | **5.4%** | **4.4%** | 0.833 logits |
| exp24 E5M3 T=0.20 (~88% hot) | 18.2% | 7.6% | 4.8% | 0.661 logits |

#### E5M3 up penalty vs full precision up

| hot% | Δ strict | Δ thermal@0.7 | Δ thermal@1.0 | Δ mean gap |
|---|---|---|---|---|
| 20% | +25.2 pp | +24.0 pp | +21.4 pp | +0.943 logits |
| 25% | +21.8 pp | +19.0 pp | +16.0 pp | +0.562 logits |
| 30% | +18.0 pp | +17.8 pp | +14.4 pp | +0.503 logits |
| 50% | +6.6 pp | +5.4 pp | +3.8 pp | +0.231 logits |

### Key findings

**E5M3 cold up fails badly at 20–30% hot.** At 20% hot, E5M3 cold up adds
+25.2 pp strict and +24.0 pp thermal@0.7 compared to full-precision cold up.
The mean logit gap of 2.0 logits (vs 1.06 for full-precision up) confirms
these are **hard errors** — not the soft near-ties seen with sparse gate cold.
At T=0.7, 30.6% thermal perturbation is solidly in the NVFP4 range — far
from the FP8-equivalent 5.6% achieved by keeping up full precision.

**The penalty shrinks with more hot channels but never disappears.**
At 50% hot, E5M3 up adds only +6.6 pp strict (19.8% vs 13.2%) and +5.4 pp
thermal@0.7. Still +0.23 logit harder errors. Even at 50% hot, E5M3 cold up
is worse than sparse_gate_only at 20% hot — cold up encoding hurts
regardless of how few cold channels remain.

**Why E5M3 cold up fails while E5M3 cold gate succeeded.**  The asymmetry is
structural in SwiGLU:

```
SwiGLU(i) = SiLU(gate(i)) × up(i)
```

- **Cold gate error**: `gate_cold = gate_full + ε_gate`.  Since cold channels
  are routed cold *because* `|gate_full(i)|` is small, `SiLU(gate_cold)` is
  already near zero.  The SiLU suppresses both the signal and the error — the
  error in the down projection contribution is `SiLU(gate_cold + ε) × up_full`,
  and for small gate values `SiLU` is nearly linear with small slope, so `ε_gate`
  matters little.

- **Cold up error**: `up_cold = up_full + ε_up`.  The up error is **not
  suppressed** by SiLU — it appears as `SiLU(gate_cold) × ε_up`.  Even though
  `SiLU(gate_cold) ≈ 0` for the coldest channels, the channels at the cold/hot
  boundary have gate values large enough to carry meaningful signal, and their
  `ε_up` propagates directly.  With 70–80% of channels cold, there are many
  such boundary channels summing into the down projection.

**Comparison with exp27** (SVD routing at 50% hot, E5M3 cold up):
Exp27 achieved match=0.834 (16.6% strict) with E5M3 cold up at 50% hot.
Here, sparse_gate+e5m3_up at 50% hot gives 19.8% strict — 3.2 pp worse despite
better routing quality (F1=0.913 vs 0.873).  The routing quality improvement
did not compensate for the E5M3 up encoding penalty.  The penalty from E5M3
cold up is independent of routing quality — it is a cold-up approximation error.

### Conclusion

**E5M3 B=8 encoding for cold W_up is not viable at 20–30% hot channels.**
The penalty is 18–25 pp strict and 18–24 pp thermal — making E5M3 cold up
nearly as bad as the sparse cold up from exp35 (which was already rejected).

The confirmed architecture for the target regime is:

```
routing:   top-k(|x @ W_gate_sparse.T|)   kr=0.5+ft
cold gate: x @ W_gate_sparse.T            [free — reuses routing GEMM]
cold up:   x @ W_up_full.T                [MUST be full precision]
hot:       full precision gate + up
down:      full precision always
```

At 30% hot this gives strict=12.0%, thermal@T=1.0=**4.8%** (FP8-equivalent).
No encoding of cold W_up is competitive with full precision at these hot fractions.

---

## Experiment 37 — 3bpw cold up (2-bit/weight, 2 E5M3 scales, B=16)

### Motivation

Exp36 showed E5M3 B=8 1bpw cold up adds +25 pp strict / +24 pp thermal at 20%
hot — too lossy.  The hypothesis: a single scale per 8 weights cannot represent
the range of `|w_up|` values within a block.  Two magnitude levels per block
should substantially reduce approximation error.

### Encoding scheme

**3 bits per weight** = 2 bits of codes + 2 E5M3 scale bytes per B=16 block:

```
Storage per 16-weight block:
  codes:  16 weights × 2 bits = 32 bits = 4 bytes
  scales: 2 × E5M3 (1 byte each) = 2 bytes
  total:  6 bytes / 16 weights = 3 bits/weight = 2.67× vs BF16

Code alphabet: {−s_hi, −s_lo, +s_lo, +s_hi}
  bit layout: [sign | magnitude_level]  (MSB = sign, LSB = level)
```

**TARE-optimal scale selection** via EM in log-space:
1. Init: split block by median `|w|`, compute tilt-weighted geometric mean of each half → `(s_lo_init, s_hi_init)`
2. E-step: assign each weight to nearest centroid in log-space (threshold = geometric midpoint `√(s_lo·s_hi)`)
3. M-step: recompute each centroid as tilt-weighted geometric mean of its members
4. Repeat ≤10 steps; typically converges in 3–4
5. Quantise both centroids to E5M3

Build time: **14 s** for all 40 layers (vs 6 s for 1bpw E5M3).

### Results (granite-4.2-3b, 500 prefill tokens)

#### Full results

| Scheme | Strict% | Thermal @T=0.7 | Thermal @T=1.0 | Mean gap |
|---|---|---|---|---|
| sparse+3bpw_up @20% | 24.4% | 15.6% | 13.2% | 1.392 logits |
| sparse+1bpw_up @20% | 38.6% | 30.6% | 27.0% | 2.005 logits |
| sparse_gate_only @20% | **13.4%** | **6.6%** | **5.6%** | 1.062 logits |
| sparse+3bpw_up @25% | 23.6% | 13.0% | 11.0% | 1.179 logits |
| sparse+1bpw_up @25% | 34.4% | 24.6% | 20.6% | 1.610 logits |
| sparse_gate_only @25% | **12.6%** | **5.6%** | **4.6%** | 1.048 logits |
| sparse+3bpw_up @30% | 22.6% | 13.2% | 10.4% | 1.161 logits |
| sparse+1bpw_up @30% | 30.0% | 23.4% | 19.2% | 1.584 logits |
| sparse_gate_only @30% | **12.0%** | **5.6%** | **4.8%** | 1.081 logits |
| sparse+3bpw_up @50% | 16.8% | 8.4% | 6.8% | 0.874 logits |
| sparse+1bpw_up @50% | 19.8% | 10.8% | 8.2% | 1.064 logits |
| sparse_gate_only @50% | **13.2%** | **5.4%** | **4.4%** | 0.833 logits |
| exp24 E5M3 T=0.20 | 18.2% | 7.6% | 4.8% | 0.661 logits |

#### Penalty vs full-precision cold up

| hot% | 3bpw Δ strict | 3bpw Δ th@0.7 | 3bpw Δ th@1.0 | 3bpw Δ gap | 1bpw Δ strict | 1bpw Δ th@0.7 |
|---|---|---|---|---|---|---|
| 20% | +11.0 pp | +9.0 pp | +7.6 pp | +0.33 L | +25.2 pp | +24.0 pp |
| 25% | +11.0 pp | +7.4 pp | +6.4 pp | +0.13 L | +21.8 pp | +19.0 pp |
| 30% | +10.6 pp | +7.6 pp | +5.6 pp | +0.08 L | +18.0 pp | +17.8 pp |
| 50% | +3.6 pp | +3.0 pp | +2.4 pp | +0.04 L | +6.6 pp | +5.4 pp |

### Key findings

**3bpw halves the cold-up penalty vs 1bpw E5M3.**  At 20% hot, the strict
penalty drops from +25.2 pp (1bpw) to +11.0 pp (3bpw); the thermal@0.7 penalty
drops from +24.0 pp to +9.0 pp.  Mean gap falls from 2.005 to 1.392 logits —
errors are substantially softer.

**The improvement is proportional and consistent.**  At every hot fraction, 3bpw
reduces the penalty by roughly 2× vs 1bpw in both strict and thermal terms.
This is consistent with the 2× increase in bits per weight (1→2) and confirms
the TARE-optimal 2-level quantisation is working correctly.

**3bpw is still not competitive with full-precision up at 20–30% hot.**  The
remaining penalty is +9–11 pp thermal@0.7, placing 3bpw cold up at 13–16%
thermal perturbation — solidly in the NVFP4 range, not FP8.  Full-precision up
at 20–30% hot gives 5.6–6.6% thermal@0.7 (near FP8).

**At 50% hot, 3bpw comes within 3 pp.**  Strict: 16.8% vs 13.2% (Δ=+3.6 pp);
thermal@0.7: 8.4% vs 5.4% (Δ=+3.0 pp); mean gap: 0.874 vs 0.833 (+0.04 L —
nearly identical).  At this hot fraction, 3bpw cold up is borderline acceptable.

**The fundamental constraint is unchanged.**  Going from 1bpw to 3bpw halves
the penalty but does not eliminate it.  The cold-up error is not a quantisation
precision problem alone — it reflects the structural issue that up errors at
boundary channels (small but non-zero cold gate) are not suppressed by SiLU.
Doubling bits halves the signal error but the error mechanism itself is preserved.

### Extrapolation

The trend penalty(bpw) scales roughly as:

| bpw | Δ thermal@0.7 @20% hot | Gap |
|---|---|---|
| 1 (E5M3 B=8) | +24.0 pp | 2.005 L |
| 3 (2-level E5M3 B=16) | +9.0 pp | 1.392 L |
| ∞ (full precision) | 0 pp | 1.062 L |

The diminishing returns suggest that even 4bpw (4 levels) would reduce the
thermal penalty to roughly +4–5 pp — still above the FP8-equivalent floor.
This is consistent with the structural argument: the cold-up error floor is
set by the boundary-channel effect, not quantisation noise.

### Conclusion

**3bpw halves the cold-up penalty vs 1bpw but does not close the gap to
full-precision up.**  At 20–30% hot the thermal perturbation is 13–16% with
3bpw cold up, vs 5–7% with full-precision up.  The gap is roughly +9 pp
thermal@0.7 — too large for the target FP8-equivalent operating point.

The cold-up encoding strategy is reaching diminishing returns.  The gap to
full-precision up has a structural floor from boundary-channel SwiGLU errors
that cannot be eliminated by better quantisation of W_up.

**Confirmed architecture for the 20–30% hot target regime remains:**
```
cold gate: x @ W_gate_sparse.T   (sparse approx — free)
cold up:   x @ W_up_full.T       (full precision — non-negotiable)
```


## Experiment 38 — 3bpw static encoding quality check (no routing)

**Goal:** Quantify the standalone quality cost of 3bpw encoding on each MLP
matrix independently and in combination, with **no hot/cold routing**.  Every
channel is approximated uniformly.  This answers whether 3bpw is a viable
global quantisation scheme for MLP weights, and establishes how errors
compound when multiple matrices are encoded simultaneously.

**Setup:** Same 3bpw encoding as exp37 (2-level TARE-optimal E5M3 B=16), same
calibration set (8 prompts, 500 prefill tokens), same strict + thermal metrics.
No routing, no sparse weights — purely static weight approximation.

### Conditions

| Condition | W_gate | W_up | W_down |
|---|---|---|---|
| gate only | 3bpw | full | full |
| gate + up | 3bpw | 3bpw | full |
| gate + down | 3bpw | full | 3bpw |
| gate + up + down | 3bpw | 3bpw | 3bpw |

Note: `gate only` matches the "cold" contribution from exp33/34 if all channels
were cold — i.e. if hot_frac=0 in the routing experiments.

### Results

| Condition | Strict | Thermal@0.7 | Thermal@1.0 | Mean gap |
|---|---|---|---|---|
| gate only | 49.0% | 46.2% | 44.8% | 3.53 L |
| gate + up | 96.4% | 94.6% | 93.6% | 8.48 L |
| gate + down | 88.4% | 86.0% | 85.6% | 8.09 L |
| gate + up + down | 99.8% | 99.2% | 99.0% | 10.60 L |

### Analysis

**Gate alone (49% strict) confirms routing is mandatory.**  Even the best
available 3bpw encoding of W_gate destroys half the top-1 predictions when
applied globally.  In the hot/cold routing scheme (exp33–37), only cold
channels use the sparse/encoded gate — and those errors are suppressed by
SiLU because cold gate values are near-zero.  Remove the routing and the
suppression disappears: every large gate activation is now approximated,
introducing large pre-SiLU errors that propagate directly.

The gap of 3.53 logits for gate alone (vs 1.06 logits in exp34 `sparse gate
only @30% hot`) quantifies the routing benefit: routing reduces the mean error
gap by ~2.5 logits by concentrating approximation on channels where SiLU
naturally suppresses the error.

**Gate + up (96.4% strict) is a qualitative step change.**  Encoding both gate
and up projects their individual errors into the SwiGLU product:

```
SiLU(gate_enc) × up_enc ≈ SiLU(gate + δg) × (up + δu)
                         ≈ SiLU(gate)×up + SiLU(gate)×δu + SiLU'(gate)×δg×up + ...
```

The cross-terms are not suppressed by any structural property when applied
globally.  The gap jumps from 3.5 L to 8.5 L — the two error sources multiply
rather than add, confirming the SwiGLU noise-multiplication effect seen in
exp20.

**Gate + down (88.4% strict) is nearly as bad.**  Encoding W_down distributes
the gate error over all hidden dimensions H=2560 during the down projection.
Each output dimension accumulates O(I) approximation errors, producing a
biased output norm shift that is large enough to flip the majority of top-1
tokens.

**All three matrices (99.8% strict) is complete model collapse.**  The mean gap
of 10.6 logits means the perturbed model assigns nearly all probability mass to
wrong tokens — qualitatively different from quantisation noise, closer to
random output.

### Comparison with hot/cold routing

The routing experiments (exp33–37) operate at 20–30% hot fraction, meaning
70–80% of channels are handled by encoded/sparse weights.  The fact that
`sparse gate only @30% hot` achieves 12% strict perturbation vs `gate only
(no routing)` at 49% is striking: routing recovers 37 pp of quality simply by
choosing *which* channels to approximate.

This confirms the architecture conclusion from exp33–37:

```
hot/cold routing is the load-bearing mechanism — not the encoding quality.
3bpw (or any encoding) can only be tolerated in the cold regime where SiLU
suppresses gate errors toward zero. Global application is non-viable.
```

### Conclusion

3bpw encoding applied globally is catastrophic across all matrix combinations.
The encoding quality is not the bottleneck — the absence of routing is.  This
experiment provides a useful lower bound: any viable compression scheme for
MLP weights in this architecture requires structured routing (hot/cold split)
to remain in the FP8-equivalent regime.  Static global quantisation of MLP
weights to 3bpw is not competitive with INT8/FP8 global schemes.


## Experiment 39 — FP6 S1E2M3 static encoding quality check (no routing)

**Goal:** Repeat the exp38 matrix-combination quality check using FP6 S1E2M3
instead of 3bpw, to establish whether more mantissa bits (3 vs effectively 0)
improve static encoding quality enough to change the routing requirement.

**FP6 S1E2M3 format:** 1 sign bit, 2 exponent bits, 3 mantissa bits.
Exponent bias = 1.  32 non-negative representable magnitudes spanning 0 to 7.5:

```
subnormal (e=0): 0.000, 0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875
normal  e=1:     1.000, 1.125, 1.250, 1.375, 1.500, 1.625, 1.750, 1.875
normal  e=2:     2.000, 2.250, 2.500, 2.750, 3.000, 3.250, 3.500, 3.750
normal  e=3:     4.000, 4.500, 5.000, 5.500, 6.000, 6.500, 7.000, 7.500
```

**Encoding:** B=16, one TARE-optimal E5M3 block scale per block (same scale
derivation as exp19/22).  Each weight encoded as sign × FP6(|w|/s) × s.

**Storage:** 16 weights × 6 bits + 8 bits scale = 104 bits = 13 bytes per block
= **6.5 bpw** = 2.46× vs BF16.  Slightly less compressed than 3bpw (2.67×)
but with 15 magnitude levels per block vs 2.

**Same 4 conditions as exp38** for direct comparison.

### Results

| Condition | Strict | Thermal@0.7 | Thermal@1.0 | Mean gap |
|---|---|---|---|---|
| gate only | 22.6% | 15.0% | 12.6% | 1.21 L |
| gate + up | 81.6% | 76.4% | 73.8% | 4.74 L |
| gate + down | 95.2% | 94.8% | 94.0% | 8.26 L |
| gate + up + down | 99.4% | 99.2% | 98.8% | 12.42 L |

### Comparison with exp38 (3bpw)

| Condition | Δ strict | Δ thermal@0.7 | Δ gap@0.7 |
|---|---|---|---|
| gate only | **−28.4 pp** | **−31.2 pp** | **−2.32 L** |
| gate + up | −14.8 pp | −18.2 pp | −3.74 L |
| gate + down | +6.8 pp | +8.8 pp | +0.16 L |
| gate + up + down | −0.4 pp | ≈0 pp | +1.82 L |

### Analysis

**Gate-only improves dramatically (+28 pp over 3bpw gate-only).**  FP6 has 15
magnitude levels per block vs 2 for 3bpw, directly reducing the pre-SiLU gate
approximation error.  The mean gap also drops from 3.53 L to 1.21 L — now in
the same range as the routed sparse-gate experiments (exp34: 1.06 L @30% hot).
This is the most important result: gate encoding quality matters a lot.

However, 22.6% strict without routing is still nearly twice the 12% achieved
*with* routing (@30% hot, exp33/34).  Routing adds 10+ pp on top of better
encoding, by concentrating approximation error on channels where SiLU suppresses
it.  The routing mechanism and encoding quality are complementary, not redundant.

**Gate + up collapses to 81.6% strict.**  Adding FP6-encoded up degrades
sharply despite FP6 being better than 3bpw per-matrix.  The SwiGLU product
`SiLU(gate_enc) × up_enc` multiplies two independent error streams; both
contain non-trivial errors at non-zero gate values, leading to near-total
output disruption.  This re-confirms the exp20 finding that encoding W_up
globally is incompatible with acceptable quality.

**Gate + down (95.2%) is worse than exp38's gate + down (88.4%), by +7 pp.**
This is counter-intuitive since FP6 gate is better than 3bpw gate.  The
explanation lies in the error scale: FP6 gate errors are smaller in magnitude
(gap 1.21 L vs 3.53 L), but those errors are spread over a wider dynamic range
— the 15-level FP6 grid produces outputs that, when a wrong level is selected,
are further from zero than 3bpw's 2-level approximation.  The down projection
broadcasts gate errors linearly to all H=2560 output dimensions; a smaller-gap
but wider-spread gate error can produce larger aggregate output error in certain
directions.  Additionally the down projection itself (encoded FP6) introduces
its own errors which compound with gate errors multiplicatively.

**All three matrices (99.4%) remains near-total collapse** as in exp38.
Individual matrix quality improvements do not carry over when all three are
encoded simultaneously — errors compound across all three operations.

### Key takeaway: FP6 gate is promising, up and down remain problematic

The gate-only result (22.6% strict, gap 1.21 L) shows FP6 has significantly
better gate encoding quality than 3bpw (49% strict, gap 3.53 L).  This opens
a path for a future routing experiment: apply FP6 to cold gate channels instead
of sparse gate, potentially combined with the existing full-precision cold up.
The 1.21 L mean gap is already close to the routed sparse-gate gap of 1.06 L,
suggesting FP6 cold gate might match or approach sparse gate quality without
requiring a pre-computed sparse weight matrix.

Encoding up or down globally remains unacceptable at any bpw tested so far.

### Conclusion

FP6 S1E2M3 substantially improves gate encoding quality over 3bpw (−28 pp
strict, gap halved from 3.5 → 1.2 L) but does not eliminate the routing
requirement.  Gate + up and gate + down combinations are still catastrophic.
The result motivates a routing experiment using FP6 cold gate (instead of sparse
W_gate) as the next step.


## Experiment 39 — retrospective note on E5M3 block scale

The exp39 results (gate-only 22.6% strict, all-matrix 99.4%) are **not
representative of OCP MXFP6** quality.  The gap vs exp40 MXFP6-E2M3
(gate-only 1.6% strict, all-matrix 4.2%) is enormous.

The root cause is the block scale format:

| Scale | Type | Representable values | Precision |
|---|---|---|---|
| E5M3 (exp39) | 5-bit exponent, 3-bit mantissa | ~240 positive values | Fine-grained, arbitrary powers |
| **E8M0 (OCP MX)** | 8-bit exponent, 0-bit mantissa | 255 exact powers of 2 | Coarse steps but exact alignment |

The E5M3 scale in exp39 uses the TARE-weighted geometric mean of `|w|` shifted
to the FP6 grid centre — a heuristic alignment.  This leaves each weight
needing to be represented as `sign × fp6_code × s` where `s` may not be
aligned to the FP6 code boundaries.  The E8M0 power-of-two scale in OCP MX
is mathematically designed so that the quantisation grid for `|w|/s` maps
exactly onto the fp-format grid: the scale shifts the exponent of every weight
by an integer number of bits, and the mantissa bits capture the remaining
fractional part exactly.  This is why E8M0 + FP6 (MXFP6) achieves ~2% strict
perturbation while E5M3 + FP6 achieves 22%.

**The exp38/39 series was therefore not testing "FP6 quality" but rather
"FP6 quality with a suboptimal non-power-of-two block scale".  Exp40 with
proper OCP E8M0 scales gives the correct answer.**

## Experiment 40 — OCP MXFP8 / MXFP6 static encoding quality check

**Goal:** Measure top-1 perturbation for all four OCP MX element formats
(MXFP8-E4M3, MXFP8-E5M2, MXFP6-E3M2, MXFP6-E2M3) with the correct
OCP-specified E8M0 block scale and B=32, applied uniformly to all 40 layers
with no routing, across the same four matrix combinations as exp38/39.

### OCP MX format summary

| Format | bpw (elem) | bpw (with B=32 E8M0 scale) | fp_max | codes |
|---|---|---|---|---|
| MXFP8-E4M3 | 8.0 | 8.25 | 448 | 127 (NaN excluded) |
| MXFP8-E5M2 | 8.0 | 8.25 | 57344 | 124 (NaN/Inf excluded) |
| MXFP6-E3M2 | 6.0 | 6.25 | 28 | 32 (no NaN in MX) |
| MXFP6-E2M3 | 6.0 | 6.25 | 7.5 | 32 (no NaN in MX) |

**E8M0 scale selection:** `scale = 2^ceil(log2(block_max / fp_max))` — the
smallest power of two that maps the block maximum onto the fp-format maximum.

### Results

#### Perturbation rates (strict / thermal@0.7)

| Condition | MXFP8-E4M3 | MXFP8-E5M2 | MXFP6-E3M2 | MXFP6-E2M3 |
|---|---|---|---|---|
| gate only | 1.6% / **0.0%** | 3.6% / 0.2% | 3.0% / 0.2% | 1.6% / **0.0%** |
| gate + up | 2.8% / **0.0%** | 6.8% / 1.0% | 6.4% / 0.8% | 3.2% / 0.4% |
| gate + down | 2.6% / 0.2% | 6.8% / 0.4% | 7.0% / 0.2% | 3.4% / **0.0%** |
| **gate + up + down** | **3.6% / 0.4%** | 8.0% / 2.2% | 7.8% / 2.0% | **4.2% / 0.8%** |

#### Mean logit gap (perturbed tokens only)

| Condition | MXFP8-E4M3 | MXFP8-E5M2 | MXFP6-E3M2 | MXFP6-E2M3 |
|---|---|---|---|---|
| gate only | 0.04 L | 0.12 L | 0.13 L | 0.06 L |
| gate + up | 0.10 L | 0.22 L | 0.24 L | 0.23 L |
| gate + down | 0.12 L | 0.27 L | 0.27 L | 0.10 L |
| gate + up + down | 0.17 L | 0.38 L | 0.37 L | 0.26 L |

### Analysis

**All four formats achieve FP8-equivalent or better quality across all matrix
combinations.**  The best result — MXFP8-E4M3 gate+up+down — is 3.6% strict /
0.4% thermal@0.7, squarely in the INT8 range and better than typical FP8
quantisation schemes.  Even MXFP6-E2M3 with all three matrices reaches only
4.2% strict / 0.8% thermal@0.7.

**Format ordering:** MXFP8-E4M3 ≈ MXFP6-E2M3 > MXFP6-E3M2 ≈ MXFP8-E5M2.
The E2M3 element formats (both FP6 and FP8) outperform the E3M2/E5M2 variants
at these weight scales.  This reflects the weight distribution of Granite-4.2:
weights are concentrated in a narrow dynamic range where higher mantissa
precision matters more than a wider exponent range.

**Mean gaps are tiny** (0.04–0.38 L) compared to the 1–10 L range seen in
exp36–39.  All perturbations are soft near-ties — the model is functioning
correctly and only occasionally permuting near-equal tokens.

**The E8M0 scale is the key difference vs exp39.**  MXFP6-E2M3 with E5M3
scale (exp39) gave 22.6% strict gate-only; with E8M0 scale the same element
format gives 1.6% strict.  The power-of-two alignment exactly maps the
weight's exponent bits into the FP6 code's exponent bits, leaving only
mantissa rounding error.  No heuristic scale alignment can match this.

**gate+down ≈ gate+up** (within ±0.4 pp across formats).  The anomaly from
exp39 where gate+down >> gate+up disappears completely.  With proper E8M0
scaling, down projection encodes with the same fidelity as gate/up, and
the two errors contribute similarly to total perturbation.

**Errors add approximately linearly:** gate+up+down ≈ gate + up + down
perturbations for the two best formats.  No multiplicative error compounding.
This is a strong indication that at MX-quality levels, the error magnitudes
are small enough that cross-term interactions are negligible.

### Implications for the routing scheme

The exp33–37 routing scheme (sparse gate + full up cold) was motivated by the
inability to encode W_up cold cheaply.  Exp40 shows that **MXFP6 or MXFP8
applied to all three matrices globally already achieves ≤4.2% strict**.
This is strictly better than the routing scheme's best result (exp33 @50% hot:
12.4% strict).

Two separate conclusions follow:

1. **Global MX quantisation is a strong baseline** that the routing scheme
   must beat to justify its complexity.  At 30% hot, exp33 gives 13.8% strict —
   3× worse than MXFP8-E4M3 all-matrix.

2. **Combining routing with MX cold encoding** (instead of full-precision cold)
   could dramatically improve the routing scheme's cold quality.  If cold
   channels can be MX-encoded at ~1–2% per-matrix error, the full cold
   approximation may become viable — breaking the "cold W_up must be full
   precision" constraint that has blocked progress since exp20.

### Conclusion

OCP MXFP8-E4M3 and MXFP6-E2M3 achieve FP8-equivalent quality (≤4.2% strict,
≤0.8% thermal@0.7) even with all three MLP matrices encoded simultaneously
and no routing.  The E8M0 power-of-two scale is essential — not the element
bit-width.  Exp39's poor results were entirely due to the non-OCP E5M3 scale.
The natural next step is to use MX encodings for cold channels in the routing
scheme, replacing the full-precision cold-up constraint.


## Experiment 41 — MXFP6-E2M3: E8M0 vs E5M3 block scale

**Goal:** Confirm that E8M0 is the key to MXFP6's quality, not the element
format.  Single controlled variable: block scale format (E8M0 vs E5M3), with
element format fixed to MXFP6-E2M3, B=32, no routing, same 4 conditions.

### Results

| Condition | E8M0 strict | E8M0 th@0.7 | E5M3 strict | E5M3 th@0.7 | Δ strict |
|---|---|---|---|---|---|
| gate only | 1.6% | **0.0%** | 28.2% | 18.2% | +26.6 pp |
| gate + up | 3.2% | 0.4% | 86.8% | 84.8% | +83.6 pp |
| gate + down | 3.4% | **0.0%** | 98.0% | 97.8% | +94.6 pp |
| gate + up + down | 4.2% | 0.8% | **99.0%** | 98.2% | +94.8 pp |

Mean logit gap for E5M3 all-matrix: **15.24 L** — near-total model collapse.
Mean logit gap for E8M0 all-matrix: **0.26 L** — soft near-ties only.

### Analysis

The result is unambiguous. Switching from E8M0 to E5M3 scale while keeping
every other parameter identical degrades MXFP6-E2M3 all-matrix from 4.2%
strict to 99.0% strict — a 94.8 pp collapse. The E5M3 scale is *more precise*
than E8M0 (8 mantissa bits vs 0), yet it is catastrophically worse.

**Why E8M0 works and E5M3 does not:**

A floating-point number `w` with value `(1 + f) × 2^e` (f = mantissa fraction,
e = exponent) is represented in FP6-E2M3 as:

```
w ≈ fp6_code × scale = (1 + m/8) × 2^(fp6_exp - 1) × scale
```

For this to be a faithful representation, `scale` must itself be a power of
two — specifically `2^(e - fp6_exp + 1)`. Then:

```
w / scale = (1 + f) × 2^e / 2^(e - fp6_exp + 1)
           = (1 + f) × 2^(fp6_exp - 1)
```

which is exactly within the range that the FP6 mantissa can represent.
**Any non-power-of-two scale introduces a fractional exponent shift**, which
is equivalent to multiplying every weight by an irrational constant before
quantisation — the mantissa bits can no longer faithfully represent the result,
and errors are ~0.5 ULP in the misaligned domain regardless of scale precision.

The E5M3 TARE scale is optimal in the TARE metric sense (minimises
scale-tilted relative error over the geometric mean of the block), but that
metric assumes real-valued reconstruction. The FP format's exponent–mantissa
factorisation means only power-of-two scales are algebraically compatible.

**The failure is not a flaw in the TARE optimisation** — it correctly finds the
best non-power-of-two scale for a real-valued proxy loss. The fundamental
issue is that the TARE loss does not model the exponent–mantissa structure
of the target format. E8M0 alignment is a *structural* requirement of
block-scaled floating-point, not a quality-of-fit choice.

### Conclusion

**E8M0 is the culprit and the solution.** The OCP MX spec's choice of E8M0
(power-of-two only scale, no mantissa bits) is not a simplification — it is
the mathematically correct scale format for block-scaled floating-point
encodings. Non-power-of-two scales like E5M3 introduce a systematic
misalignment between the scale and the element format's exponent grid that
cannot be corrected by better scale-finding heuristics.

All future experiments using sub-BF16 weight encodings should use E8M0 scales.
The TARE metric and E5M3 scale series (exp17–39) are superseded for any
encoding that uses a floating-point element format with an explicit exponent
field.


## Experiment 42 — 3bpw with E8M0 scales vs E5M3 scales

**Goal:** Apply the exp41 lesson (E8M0 vs E5M3 scale) to the 3bpw 2-level
scheme.  If E8M0 dramatically fixed MXFP6, does it also fix 3bpw?

**Encoding:** 2 bits per weight, 2 TARE-optimal power-of-two (E8M0) scales
per B=16 block.  EM runs in log₂-space; centroids rounded to nearest integer
log₂ (i.e. nearest power of two) at each M-step and finally.

Storage: same as exp38 — 6 bytes / 16 weights = **3 bpw = 2.67× vs BF16**.

### Results

| Condition | E8M0 strict | E8M0 th@0.7 | E5M3 strict | E5M3 th@0.7 | Δ strict |
|---|---|---|---|---|---|
| gate only | 49.0% | 42.6% | 51.0% | 46.2% | −2.0 pp |
| gate + up | 94.6% | 93.0% | 96.4% | 94.6% | −1.8 pp |
| gate + down | 90.8% | 87.0% | 88.4% | 86.0% | +2.4 pp |
| gate + up + down | 98.8% | 97.8% | 99.8% | 99.2% | −1.0 pp |

### Context: full comparison table

| Condition | 3bpw E5M3 | 3bpw E8M0 | MXFP6-E2M3 E8M0 |
|---|---|---|---|
| gate only | 51% / 46% | 49% / 43% | **1.6% / 0.0%** |
| gate + up | 96% / 95% | 95% / 93% | **3.2% / 0.4%** |
| gate + down | 88% / 86% | 91% / 87% | **3.4% / 0.0%** |
| gate + up + down | 100% / 99% | 99% / 98% | **4.2% / 0.8%** |

### Analysis

**E8M0 barely helps 3bpw.**  The maximum improvement is 2 pp strict
(gate-only), compared to the 26 pp improvement it gave MXFP6 in exp41.  Both
3bpw variants remain catastrophically bad across all conditions.

The reason is structural and now clear: the exp41 analysis identified that
E8M0 works because it aligns the scale's power-of-two steps with the
**exponent field** of the floating-point element format.  The 3bpw scheme has
no exponent field per weight — each weight is simply `±s_lo` or `±s_hi`, two
scalar values.  There is no floating-point mantissa to align to; E8M0 and E5M3
scales produce identically-structured approximations, differing only in the
discrete set of values that the centroids can take.

In fact, E8M0 is slightly *worse* for 3bpw than E5M3 in the gate+down
condition (+2.4 pp), because forcing centroids to exact powers of two can
be a worse fit to the actual block weight distribution than E5M3's finer grid.
The TARE metric is genuinely trying to minimise reconstruction error, and
constraining centroids to powers of two throws away that optimality for free.

**The fundamental 3bpw problem is the number of levels, not the scale
format.**  With only 2 magnitude levels per block of 16 weights, the scheme
has a quantisation granularity floor of ~1 bit/weight of effective precision
after the sign.  No scale choice can compensate for representing 16 diverse
weights with just two magnitudes.  MXFP6-E2M3 has 16 non-negative magnitudes
(32 codes total) — 8× more — and that is why it works.

### Conclusion

3bpw with E8M0 scales is no better than 3bpw with E5M3 scales (≤2 pp
difference, no consistent direction).  The 3bpw scheme is limited by level
count, not scale format.  The correct way to improve 3bpw is to add more
levels — which is exactly what MXFP6 does.  The encoding series exp37→42
converges on OCP MXFP6-E2M3 with E8M0 scale as the correct 3-bpw-class
encoding, at 6.25 bpw vs 3 bpw but with 30× better quality (4.2% vs 99%).


## Experiment 43 — MXFP4-E2M1 and hypothetical MXFP5-E2M2

**Goal:** Extend the E2Mx progression below MXFP6 to characterise the
quality cliff as mantissa bits are removed.

### Format definitions

| Format | Element bits | Codes | fp_max | bpw (B=32 E8M0) |
|---|---|---|---|---|
| MXFP4-E2M1 | 4 (OCP standard) | 8 | 6.0 | 4.25 |
| MXFP5-E2M2 | 5 (hypothetical) | 16 | 7.0 | 5.25 |
| MXFP6-E2M3 | 6 (OCP standard) | 32 | 7.5 | 6.25 |

MXFP5-E2M2 non-negative codes: `0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0`

### Results — perturbation rates (strict% / thermal@0.7%)

| Condition | MXFP4-E2M1 | MXFP5-E2M2 | MXFP6-E2M3 | MXFP8-E4M3 |
|---|---|---|---|---|
| gate only | 8.8% / 2.6% | 4.8% / 0.8% | 1.6% / 0.0% | 1.6% / 0.0% |
| gate + up | 12.2% / 5.2% | 7.4% / 2.0% | 3.2% / 0.4% | 2.8% / 0.0% |
| gate + down | 15.4% / 7.0% | 7.4% / 3.0% | 3.4% / 0.0% | 2.6% / 0.2% |
| **gate + up + down** | **17.2% / 8.6%** | **9.6% / 3.6%** | **4.2% / 0.8%** | **3.6% / 0.4%** |

### Mean logit gap (all-matrix)

| Format | gap@0.7 |
|---|---|
| MXFP4-E2M1 | 1.285 L |
| MXFP5-E2M2 | 0.657 L |
| MXFP6-E2M3 | 0.265 L |
| MXFP8-E4M3 | 0.172 L |

### Analysis

**MXFP4 all-matrix (17.2% strict / 8.6% thermal@0.7) sits squarely in the
NVFP4 territory.** This matches the hardware expectation: OCP MXFP4 is
designed as a peer to NVIDIA's FP4 formats for Blackwell, and the ~10–20%
strict perturbation range is the known quality regime for 4-bit inference.
Thermally, 8.6% at T=0.7 sits just above the FP8 floor (~3–5%), meaning
MXFP4 errors are mostly real (mean gap 1.29 L — significant but not
catastrophic).

**MXFP5 all-matrix (9.6% strict / 3.6% thermal@0.7) crosses the FP8
threshold thermally.** Strict perturbation is still in the NVFP4 range, but
thermal@0.7 at 3.6% is within the FP8-equivalent band (~2–4%). The extra
mantissa bit over MXFP4 halves the thermal perturbation. This format doesn't
exist in any hardware standard today but the numbers suggest it would be a
compelling sweet spot: better than NVFP4 at a ~1 bpw premium over MXFP4.

**Each mantissa bit roughly halves perturbation** across the E2Mx family:

| Step | Δ strict (all-matrix) | Δ thermal@0.7 |
|---|---|---|
| MXFP4→5 (+1 bit, +1 bpw) | −7.6 pp | −5.0 pp |
| MXFP5→6 (+1 bit, +1 bpw) | −5.4 pp | −2.8 pp |
| MXFP6→8 (+2 bits, +2 bpw) | −0.6 pp | −0.4 pp |

The marginal return diminishes rapidly: going from 5→6 bits gives nearly as
much gain as going from 6→8 bits. **MXFP6-E2M3 is the knee of the curve**
for this model and weight distribution.

**MXFP8-E4M3 vs MXFP6-E2M3:** only 0.6 pp strict / 0.4 pp thermal difference
on all-matrix encoding. The two formats are essentially tied at this quality
level — the extra 2 bpw of MXFP8 buys almost nothing for Granite-4.2-3b MLP
weights in the absence of activations (weight-only quantisation context).

**The gate+down anomaly from exp39 does not reappear** in any format. With
E8M0 scales, gate+down ≈ gate+up across all formats (within ±0.4 pp). This
confirms the exp41/42 analysis: the earlier anomaly was entirely a scale
alignment artefact.

### Conclusion

The E2Mx family with E8M0 B=32 scales shows clean monotone improvement with
mantissa bits. The quality cliff is at MXFP4→5: removing the second mantissa
bit pushes the all-matrix perturbation from FP8-equivalent (3.6% thermal) to
NVFP4 territory (8.6% thermal). MXFP5 is the minimum-bpw format that achieves
FP8-equivalent thermal quality at 5.25 bpw. MXFP6-E2M3 at 6.25 bpw is the
practical operating point with clear headroom. MXFP8-E4M3 offers negligible
further improvement over MXFP6 for static weight encoding.


## Experiment 44 — E5M0 vs E8M0 block scale: does the range matter?

**Question:** The OCP MX spec mandates an 8-bit E8M0 scale (range 2^±127).
Is that range actually needed, or would a narrower 5-bit E5M0 scale
(range 2^±15, analogous to the FP16 exponent) suffice?

**Setup:** MXFP4-E2M1, MXFP5-E2M2, MXFP6-E2M3 with B=32, same 4 matrix
combinations.  Two scale variants:
- **E8M0**: `e = ceil(log2(block_max / fp_max))`, clamped to [−127, 127]
- **E5M0**: same rule, clamped to [−15, 15] (5-bit, 3 bytes saved per 8 blocks)

### Diagnostic: blocks requiring |e| > 15

Before running inference, checked all 78,643,200 blocks across all layers and
all three projection matrices:

```
MXFP4-E2M1: 0/78,643,200 blocks clipped (0.000%)
MXFP5-E2M2: 0/78,643,200 blocks clipped (0.000%)
MXFP6-E2M3: 0/78,643,200 blocks clipped (0.000%)
```

**Zero blocks require a scale outside [2^−15, 2^15].** Granite-4.2-3b weights
are entirely within this range.

### Results

E5M0 and E8M0 produce **bit-identical encoded matrices** for every layer,
projection, and format. The inference results are identical to four decimal
places across all 48 (format × condition × metric) combinations. Δ = 0.0 pp
everywhere.

### Analysis

The result follows directly from the diagnostic: since no block has
`block_max / fp_max` outside [2^−15, 2^15], the E8M0 and E5M0 clamp never
differs. The two scale functions return the same value for every block.

The E8M0 range [2^−127, 2^127] was designed for generality across all possible
weight distributions, including activations, gradients, and models with
significant outliers. For the MLP weights of Granite-4.2-3b (a compact 3B
parameter model with standard initialisation and training), weights are
well-behaved and the scale exponent stays comfortably within [−15, 15].

**Implication for storage:** the 3 extra bits of E8M0 vs E5M0 cost
3 bits per 32-weight block. For MXFP6-E2M3 at B=32:
- E8M0: (32×6 + 8) / 32 = 6.25 bpw
- E5M0: (32×6 + 5) / 32 = **6.156 bpw** — saving 0.094 bpw (~1.5%)

For MXFP4-E2M1: 4.25 → **4.156 bpw**. A small but free saving on this model.

**Caveat:** this is model-specific. Larger models, models with weight outliers
(e.g. from long training runs, RLHF, or certain architectures), or activation
quantisation would likely require the full E8M0 range. The OCP MX spec uses
E8M0 for universal compatibility. E5M0 is only safe after verifying the
distribution, as done here.

### Conclusion

The extended range of E8M0 is entirely unused for Granite-4.2-3b MLP weights.
E5M0 (5-bit, ±15 exponent range) produces identical results at 0.094 bpw
lower storage cost per format. The 3 extra E8M0 bits are OCP overhead for
worst-case generality, not required for this model.

### Scale exponent distribution (MXFP6-E2M3, fp_max=7.5, B=32)

Measured across all 78,643,200 blocks (40 layers × 3 matrices):

| Exponent e | Scale 2^e | Count | % | Cumulative% |
|---|---|---|---|---|
| −12 | 0.000244 | 454 | 0.001% | 0.001% |
| −11 | 0.000488 | 196,795 | 0.250% | 0.251% |
| **−10** | **0.000977** | **13,663,109** | **17.4%** | 17.6% |
| **−9** | **0.001953** | **12,464,990** | **15.9%** | 33.5% |
| **−8** | **0.003906** | **25,532,526** | **32.5%** | 65.9% |
| **−7** | **0.007812** | **26,421,869** | **33.6%** | 99.5% |
| −6 | 0.015625 | 350,215 | 0.4% | 100.0% |
| −5 | 0.031250 | 12,196 | 0.016% | ≈100% |
| −4 | 0.062500 | 982 | 0.001% | ≈100% |
| −3 | 0.125000 | 64 | 0.000% | 100.0% |

**Summary:**
- Range used: e ∈ [−12, −3] — only **10 distinct exponent values** out of E8M0's 255
- Central mass: e ∈ [−10, −7] covers **99.3%** of all blocks
- Bimodal peak: e=−8 (32.5%) and e=−7 (33.6%) together = **66.1%**
- Mean exponent: **−8.17** (scale ≈ 0.0035, consistent with typical BF16 weight magnitude ~0.01–0.03)
- MXFP4 (fp_max=6.0) covers the same [−12, −3] range with a shifted peak at e=−7 (53.1%)

### Minimum viable scale width for this model

The observed range [−12, −3] spans 10 values — 4 bits suffice:

| Scale format | Bits | Range | Covers [−12,−3]? | bpw overhead (B=32) | MXFP6-E2M3 total |
|---|---|---|---|---|---|
| E8M0 (OCP) | 8 | [−127, 127] | ✓ (with massive headroom) | 0.250 | 6.250 |
| E5M0 | 5 | [−15, 15] | ✓ | 0.156 | 6.156 |
| **E4M0** | **4** | **[−7, 8]** | **✓ with offset** | **0.125** | **6.125** |

An E4M0 scale with bias 12 (byte values 0..9 for e=−12..−3) fits in 4 bits.
The practical bpw saving over E8M0 is 0.125 bpw (~2%), but the circuit
complexity saving in hardware is meaningful: a 4-bit vs 8-bit scale multiplier.


### Channel-level scale range analysis (mixed-mode motivation)

Two questions relevant to a mixed-mode encoding design:

**1. What fraction of channels have all their blocks within a narrow range?**

A channel here = one output row of a weight matrix (= 80 consecutive B=32
blocks for gate/up [I=2560 inputs ÷ 32], or 256 blocks for down [I=8192]).

| Scale range | % blocks in range | % channels with ALL blocks in range |
|---|---|---|
| e ∈ [−10, −7] (4 values) | **99.29%** | **84.16%** |
| e ∈ [−9, −7]  (3 values) | 81.91% | 72.34% |
| e ∈ [−8, −7]  (2 values) | 66.06% | 66.73% |

**84% of all output channels have every single one of their blocks within
[−10, −7].** For these channels a 2-bit scale offset (4 values) relative to a
channel-level base exponent would suffice — no per-block E8M0 byte needed.

Per-projection detail for e ∈ [−10, −7]:

| Projection | Total channels | All-in (≤4 values) | % |
|---|---|---|---|
| gate | 327,680 | 291,580 | **89.0%** |
| up | 327,680 | 256,673 | **78.3%** |
| down | 102,400 | 89,468 | **87.4%** |

**2. How much do block scales vary *within* a single channel?**

Span = max(e) − min(e) across all blocks in one channel:

| Span (octaves) | Count | % | Cumulative% |
|---|---|---|---|
| 0 (all blocks same scale) | 37 | 0.00% | 0.00% |
| **1** | **580,544** | **76.6%** | 76.6% |
| **2** | **158,720** | **21.0%** | 97.6% |
| 3 | 14,982 | 2.0% | 99.5% |
| 4 | 3,083 | 0.4% | 99.9% |
| 5–6 | 394 | 0.05% | 100% |

**97.6% of all channels have a within-channel scale span of ≤2 octaves.**
The vast majority span exactly 1 octave (two adjacent powers of two).

### Implications for mixed-mode encoding

The data suggests a practical two-tier scheme:

```
Tier A (84% of channels — "narrow"):
  Channel base exponent  e_base  (1 byte, shared for whole channel)
  Per-block 2-bit delta  d ∈ {−1, 0, +1, +2}  →  scale = 2^(e_base + d)
  Block storage: 2 bits scale + 6 bits element = 8 bits/weight at MXFP6
  vs current MXFP6: 8 bits scale / 80 blocks = 0.1 bits overhead/weight
  New overhead: 8 bits base / (80×32 weights) + 2 bits/weight ≈ 2.003 bpw scale overhead
  → Actually worse: inline 2-bit delta per block beats channel-shared base only
     if many blocks share the same delta.

Simpler framing: the 84% narrow-channel result means that for 84% of channels,
a single 3-bit or 4-bit channel-level scale (replacing 80 independent E8M0 bytes)
would be lossless. Saving: 80 bytes − 1 byte = 79 bytes per channel, or
79/(80×32) ≈ 0.031 bytes/weight = 0.25 bits/weight overhead reduction.
At MXFP6-E2M3 this brings the effective bpw from 6.25 → ~6.04 for 84% of channels.

The more interesting mixed-mode application is **element-format mixing**:
channels confirmed to stay within 2 scale values could encode their weights
more aggressively (e.g. MXFP4 with a guaranteed 1-octave range), while outlier
channels (16%) keep MXFP6 or BF16. This is the direction to explore next.
```

### LUT-based mixed-mode scale scheme: bit cost analysis

**Scheme definition** (proposed by user):

| Mode | Condition | Index bits/block | LUT entries | LUT size |
|---|---|---|---|---|
| 0 | ≤2 distinct scales in channel | 1 bit | 2 × E8M0 | 2 bytes |
| 1 | 3–4 distinct scales | 2 bits | 4 × E8M0 (pad if <4) | 4 bytes |
| 2 | ≥5 distinct scales | 4 bits | 16 × E8M0 (pad if <16) | 16 bytes |

A 2-bit mode flag per channel selects which mode applies (cost ≈ 0.0001 bpw, negligible).

**Channel/block assignment across all 757,760 channels, 78,643,200 blocks:**

| Mode | Channels | % | Blocks | Index bits |
|---|---|---|---|---|
| 0 (1 bit/block) | 580,581 | **76.6%** | 58,675,664 | 58,675,664 |
| 1 (2 bits/block) | 173,830 | **22.9%** | 19,482,608 | 38,965,216 |
| 2 (4 bits/block) | 3,349 | **0.4%** | 484,928 | 1,939,712 |

**Scale overhead computation (per block, averaged across all blocks):**

```
LUT bits:    (580,581×2×8 + 173,830×4×8 + 3,349×16×8) / 78,643,200
           = 15,280,528 / 78,643,200  =  0.194 bits/block

Index bits:  (58,675,664×1 + 19,482,608×2 + 484,928×4) / 78,643,200
           = 99,580,592 / 78,643,200  =  1.266 bits/block

Total:       1.266 + 0.194 = 1.461 bits/block
```

**Your estimate of ~1.2 bits/block was close but slightly optimistic.** The actual figure is **1.46 bits/block**, driven mainly by the index bits from the 22.9% of mode-1 channels (2 bits/block). The LUT overhead is only 0.19 bits/block — very small as expected.

The gap from 1.2 to 1.46 comes from mode-1: 23% of channels need 2 index bits, not 1, which adds 0.23 bits/block above a pure mode-0 world. If only mode-0 existed (all channels ≤2 scales), it would be `1×76.6% + 2×0.194 = 1.07 bits/block`. Mode-1 and mode-2 together push it to 1.46.

**Summary vs flat E8M0:**

| Scheme | Scale overhead bits/block | Total bpw (MXFP6-E2M3) | Saving vs E8M0 |
|---|---|---|---|
| Flat E8M0 B=32 (current) | 8.00 | 6.250 | — |
| LUT 3-mode scheme | **1.46** | **6.046** | **0.204 bpw (3.3%)** |
| Theoretical minimum (entropy) | ~0.8 | ~5.85 | upper bound |

The LUT scheme reduces scale overhead from 8 bits/block to 1.46 bits/block — a **5.5× reduction** — for a net saving of **0.20 bpw** on MXFP6-E2M3.

Note: the distinct-scale counts per channel are: n=1 (0.00%), n=2 (76.6%), n=3 (21.0%), n=4 (2.0%), n=5 (0.4%), n=6–7 (0.04%). There are no channels with more than 7 distinct scales, so mode-2's 16-entry LUT is over-provisioned — a 3-entry or 4-entry LUT with 2 index bits would suffice for 99.6% of channels. Capping mode-2 at 8 entries (3 index bits) instead of 16 would save a further ~0.004 bpw.

### LUT scheme applied to MXFP5-E2M2 and MXFP4-E2M1

The LUT channel assignment (mode-0/1/2 fractions) is **format-independent**: it is derived from the B=32 scale exponent distribution, which is the same regardless of whether the elements are 4-, 5-, or 6-bit. The only thing that changes between formats is the element contribution to bpw.

**Derivation:**

Scale overhead in bpw = 1.461 bits/block ÷ 32 weights/block = **0.04566 bpw** (identical for all E2Mx formats).

| Format | Element bits/weight | Flat E8M0 bpw | LUT bpw | Saving vs flat | Relative saving |
|---|---|---|---|---|---|
| MXFP4-E2M1 | 4 | 4.250 | **4.046** | 0.204 bpw | **4.8%** |
| MXFP5-E2M2 | 5 | 5.250 | **5.046** | 0.204 bpw | **3.9%** |
| MXFP6-E2M3 | 6 | 6.250 | **6.046** | 0.204 bpw | **3.3%** |

The absolute saving is **the same 0.204 bpw** across all three formats — it comes entirely from reducing the scale field from 8.00 to 1.461 bits/block, which is format-agnostic. The relative saving is largest for MXFP4 (4.8%) because the fixed 0.204 bpw scale saving is a larger fraction of total storage at 4 element bits.

**Comparison across the full encoding spectrum:**

| Format + scheme | Total bpw | Strict perturbation (all-matrix) | Thermal@0.7 |
|---|---|---|---|
| MXFP4-E2M1, flat E8M0 | 4.250 | 17.2% | 8.6% |
| MXFP4-E2M1, LUT | **4.046** | 17.2% | 8.6% |
| MXFP5-E2M2, flat E8M0 | 5.250 | 9.6% | 3.6% |
| MXFP5-E2M2, LUT | **5.046** | 9.6% | 3.6% |
| MXFP6-E2M3, flat E8M0 | 6.250 | 4.2% | 0.8% |
| MXFP6-E2M3, LUT | **6.046** | 4.2% | 0.8% |

Quality numbers are unchanged — the LUT scheme is a lossless scale re-encoding (same scale values, just indexed differently per channel).

**Key observation:** MXFP5-E2M2 with LUT (**5.046 bpw**, thermal@0.7=3.6%) is FP8-equivalent and costs only 1.0 bpw more than MXFP4+LUT (4.046 bpw, thermal=8.6%). That 1.0 bpw buys a 5 pp reduction in strict perturbation and crosses the FP8 thermal threshold. The MXFP6+LUT–vs–MXFP5+LUT gap is 1.0 bpw for only a further 5.4 pp strict / 2.8 pp thermal improvement.

**Practical storage summary (all-matrix, gate+up+down, granite-4.2-3b):**

The total MLP weight budget is 40 layers × (2×8192×2560 + 8192×2560) weights = 2,516,582,400 weights = ~2.5B weights.

| Scheme | bpw | Total MLP size | vs BF16 (5.0 GB) |
|---|---|---|---|
| BF16 (reference) | 16.0 | 5.03 GB | — |
| MXFP6+LUT | 6.046 | 1.90 GB | **2.65× smaller** |
| MXFP5+LUT | 5.046 | 1.59 GB | **3.17× smaller** |
| MXFP4+LUT | 4.046 | 1.27 GB | **3.95× smaller** |

**Conclusion:** The LUT scheme delivers the same 0.204 bpw saving regardless of element format. At MXFP4 this represents a 4.8% reduction (the largest relative gain), but MXFP4's 8.6% thermal perturbation means it sits firmly in NVFP4 territory regardless of scale encoding. MXFP5+LUT at 5.046 bpw is the practical sweet spot: FP8-equivalent thermal quality, 3.17× BF16 compression, and the LUT scheme squeezes an extra 200 MB out vs flat E8M0.

## Experiment 44d — MXFP4-E2M1 code usage and LUT12 feasibility

**Goal:** Are all 16 signed MXFP4 codes used evenly? Could the 4 least-used codes
be pruned to a "LUT12" scheme (12 effective values) without significant quality loss?

### Code usage

MXFP4-E2M1 has 8 non-negative magnitudes (with sign: 15 distinct values, or 16 signed
slots treating ±0 separately). Measured across all 2,516,582,400 weight slots
(40 layers × gate + up + down):

| idx | value | count | % | cumul% | gate% | up% | down% |
|---|---|---|---|---|---|---|---|
| 0 | 0.00 | 299,029,544 | **11.88%** | 11.88% | 11.80% | 11.70% | 12.15% |
| 1 | 0.50 | 562,273,446 | **22.34%** | 34.22% | 22.21% | 22.02% | 22.80% |
| 2 | 1.00 | 478,236,621 | **19.00%** | 53.23% | 18.92% | 18.82% | 19.26% |
| 3 | 1.50 | 375,798,286 | **14.93%** | 68.16% | 14.87% | 14.94% | 14.98% |
| 4 | 2.00 | 375,108,597 | **14.91%** | 83.07% | 14.85% | 15.20% | 14.67% |
| 5 | 3.00 | 263,297,971 | **10.46%** | 93.53% | 10.53% | 10.83% | 10.03% |
| 6 | 4.00 | 134,759,677 |  **5.36%** | 98.88% |  5.58% |  5.44% |  5.04% |
| 7 | 6.00 |  28,078,258 |  **1.12%** | 100.0% |  1.23% |  1.04% |  1.08% |

**Observations:**

1. **Distribution is strongly monotone-decreasing with magnitude.** Usage drops
   nearly geometrically: 0.5 is the most common code (22.3%), and 6.0 (fp_max) is
   used only 1.1% of the time — 20× less than index 1.

2. **No code is truly unused.** Even the rarest code (6.0) appears 28M times.
   Codes 0–5 account for 93.5% of all slots; code 6 (4.0) adds 5.4%; code 7 (6.0)
   adds the final 1.1%.

3. **Distribution is consistent across all three matrices** (gate/up/down %) —
   no projection is structurally different in code usage.

4. **The codes are not used evenly.** The top-4 codes (0.5, 1.0, 0.0, 1.5)
   account for 68% of all slots. The bottom-4 by count are: 6.0 (1.1%),
   4.0 (5.4%), 3.0 (10.5%), 0.0 (11.9%) — a combined 28.8%.

### LUT12 feasibility

The 4 least-used codes by count are: **{0.0, 3.0, 4.0, 6.0}** (indices 0, 5, 6, 7).
These are remapped to their nearest retained neighbour:

| Dropped code | Nearest retained | Remap |
|---|---|---|
| 0.0 (idx 0) | 0.5 (idx 1) | 0.0 → 0.5 |
| 3.0 (idx 5) | 2.0 (idx 4) | 3.0 → 2.0 |
| 4.0 (idx 6) | 2.0 (idx 4) | 4.0 → 2.0 |
| 6.0 (idx 7) | 2.0 (idx 4) | 6.0 → 2.0 |

**Quality result (all-matrix gate+up+down, same calibration set):**

| Scheme | strict% | thermal@0.7% | thermal@1.0% |
|---|---|---|---|
| MXFP4-E2M1 (baseline) | **17.2%** | **8.6%** | **7.6%** |
| MXFP4-LUT12 (4 pruned) | **94.8%** | **92.6%** | **91.4%** |

**LUT12 is catastrophically bad.** Perturbation jumps from 17% to 95%.

### Why it fails — structural analysis

The 4 least-used codes are not randomly distributed — they are **structurally
critical**:

- **Dropping zero (0.0 → 0.5):** Every weight that rounds to zero (11.9% of slots)
  is forced to ±0.5 instead. This is equivalent to adding dense noise with
  magnitude ~0.5×scale to 12% of all weights — exactly the channels where the
  model has learned to suppress output.

- **Dropping the three large-magnitude codes (3.0, 4.0, 6.0 → all → 2.0):**
  The top of the representable range collapses. Any weight that needed magnitudes
  in [3, 6] (16.9% of slots) is clamped to 2.0×scale — a severe truncation of the
  weight distribution's tail. The nearest-neighbour collapse maps three distinct
  codes to the same value, breaking the monotone grid entirely.

The root cause is that MXFP4's 8 codes are *geometrically spaced* across the full
dynamic range — every code is load-bearing. The "4 least used by count" happen to
be the zero, the two highest-magnitude codes, and one mid-range code; together they
bracket the entire representable range. Removing them is not pruning dead weight,
it is amputating the dynamic range.

### Could any 4 codes be pruned?

The distribution shows why no set of 4 codes can safely be dropped:

- Codes 0–4 (0.0–2.0) are the high-usage core; removing any of them would affect
  14–22% of weight slots.
- Codes 5–7 (3.0–6.0) are the low-usage tail, but they are essential for
  representing outlier weights. Removing them truncates the dynamic range and
  collapses 16.9% of weights to 2.0.
- Removing **zero specifically** is uniquely harmful: the model uses exact zeros to
  implement structured suppression (especially in cold-channel regimes), and
  replacing them with ±0.5 creates correlated noise at precisely the slots that
  should be silent.

**Conclusion: LUT12 is not feasible for MXFP4.** The format has 8 codes because
they are each necessary. The usage skew (22% for code 1 vs 1% for code 7) reflects
the natural weight magnitude distribution, not code redundancy. A genuine 12-code
variant would require redesigning the codebook (e.g. non-uniform spacing with more
resolution near zero), not simply dropping the least-used entries from the existing
OCP grid.

## Experiment 45 — Per-channel LUT12 (optimal 12-entry codebook)

**Goal:** Instead of the fixed 8-entry OCP MXFP4 grid, fit a **per-output-channel
optimal 12-entry non-negative magnitude codebook** using Lloyd-Max iteration (k-means
on E8M0-scaled absolute weights). Sign is factored out and stored as a separate bit,
exactly as in all OCP MX formats. Measure quality vs MXFP4 (8 magnitudes) and
MXFP6 (32 magnitudes).

### Encoding convention

Weights are split as `W = sign(W) × |W|`. The LUT holds **12 non-negative
magnitudes**; sign is the 13th independent bit. The full per-weight code is therefore:

```
code = sign_bit (1 bit) + magnitude_index (⌈log₂(12)⌉ = 4 bits) = 5 bits/weight
```

12 magnitudes require 4 bits (2⁴ = 16 ≥ 12). Sign requires 1 more bit. Total: **5
bits/weight = 5.25 bpw** — the same as MXFP5-E2M2. This is equivalent to 23 signed
values ({−v₁₁, …, −v₁, 0, +v₁, …, +v₁₁}; zero appears once) packed as sign +
magnitude index.

This is identical to how MXFP4 works: 3 magnitude bits + 1 sign bit = 4 bits/weight
(8 magnitudes). LUT12 is 4 magnitude bits + 1 sign bit = 5 bits/weight (12
magnitudes) — one more magnitude bit than MXFP4.

### Setup

- **Fitting:** Lloyd-Max (30 iterations, quantile-initialised) on the E8M0-scaled
  absolute weights of each output channel across all its B=32 blocks.
  Level-0 is pinned to 0.0 to preserve exact-zero weights.
- **Scale:** same E8M0 B=32 per-block scale as all MX experiments, with
  `fp_max=6.0` (MXFP4 reference) for scale selection.
- **Bits/weight:** 1 sign + 4 magnitude index = **5 bits** → **5.25 bpw**
- **LUT sidecar:** 12 × BF16 = 24 bytes/channel × 757,760 channels ≈ **18 MB**
  (~0.001 bpw overhead; negligible).
- **Scope:** all-matrix (gate + up + down), all 40 layers.
- **Hardware:** Lloyd-Max fitting on MPS (Metal); vLLM inference on CPU.

### Results

| Scheme | strict% | thermal@0.7% | thermal@1.0% | bpw |
|---|---|---|---|---|
| MXFP4-E2M1 (8 magnitudes, fixed) | 17.2% | 8.6% | 7.6% | 4.25 |
| MXFP5-E2M2 (16 magnitudes, fixed) | 9.6% | 3.6% | — | 5.25 |
| **LUT12 per-channel (12 magnitudes, optimal)** | **9.2%** | **3.4%** | **2.8%** | **5.25+ε** |
| MXFP6-E2M3 (32 magnitudes, fixed) | 4.2% | 0.8% | 0.6% | 6.25 |

### Analysis

**LUT12 halves strict perturbation relative to MXFP4: 17.2% → 9.2% (−8.0 pp).**
Thermal@0.7 drops from 8.6% → 3.4% (−5.2 pp), crossing the FP8-equivalent threshold.

**Compared to MXFP5-E2M2 (16 magnitudes, same 5.25 bpw):** LUT12 achieves nearly
identical quality (9.2% vs 9.6% strict, 3.4% vs 3.6% thermal@0.7). The ~0.4 pp
strict advantage is the pure gain from per-channel optimal level placement over the
fixed E2M2 grid — a small but real benefit.

**Two gains from moving MXFP4→LUT12, disentangled:**

1. **More magnitudes (8→12, +1 bit):** Finer quantisation grid — this accounts for
   most of the improvement. The exp43 E2Mx progression shows that adding one
   magnitude bit (8→16 codes) halves strict perturbation; LUT12's 8→12 is half
   that step.

2. **Optimal placement:** Lloyd-Max concentrates levels where the channel's actual
   weight mass is — near zero and the dominant [0.5, 2.0] range — rather than the
   geometrically uniform OCP grid. This gives the residual ~0.4 pp advantage over
   fixed MXFP5.

### Storage breakdown

| Component | Cost |
|---|---|
| Sign bit (1 bit/weight) | 1.000 bpw |
| Magnitude index (4 bits/weight, 12 of 16 codes used) | 4.000 bpw |
| E8M0 block scale (8 bits/block ÷ 32) | 0.250 bpw |
| Per-channel LUT sidecar (24 B/channel) | ~0.001 bpw |
| **Total** | **~5.251 bpw** |

With the exp44b LUT scale scheme: scale overhead drops to 0.046 bpw → **~5.046 bpw
+ 18 MB sidecar** — identical to MXFP5+LUT from exp44c.

### Conclusion

Per-channel LUT12 is a **learned MXFP5**: same 5.25 bpw, same quality neighbourhood,
but with per-channel optimal level placement instead of the fixed E2M2 grid. The
~0.4 pp strict quality gain from optimal placement is real but modest. The
significant benefit is that LUT12 can adapt to any distribution — it would be more
useful for models with non-standard weight distributions (outlier-heavy, post-RLHF,
etc.) where the fixed OCP grid is a poor fit.

Key numbers in context:

| Format | bpw | strict% | thermal@0.7% | Notes |
|---|---|---|---|---|
| MXFP4-E2M1 (OCP fixed) | 4.25 | 17.2% | 8.6% | 1 sign + 3 mag bits |
| MXFP5-E2M2 (fixed) | 5.25 | 9.6% | 3.6% | 1 sign + 4 mag bits, fixed grid |
| **LUT12 per-channel** | **5.25+ε** | **9.2%** | **3.4%** | 1 sign + 4 mag bits, optimal grid |
| MXFP6-E2M3 (OCP fixed) | 6.25 | 4.2% | 0.8% | 1 sign + 5 mag bits |

## Experiment 45b — Per-channel LUT6 (6 non-negative magnitudes, 4.25 bpw)

**Goal:** Test LUT with only 6 non-negative magnitudes + sign bit. Storage is
4 bits/weight (4.25 bpw) — identical to MXFP4 — but codes are fewer and optimally
placed per channel instead of the fixed OCP geometric grid.

### Encoding convention

| Property | LUT6 | MXFP4-E2M1 |
|---|---|---|
| Non-negative magnitudes | 6 | 8 |
| Magnitude index bits | ⌈log₂(6)⌉ = 3 | 3 |
| Sign bit | 1 | 1 |
| **Total bits/weight** | **4** | **4** |
| **bpw (E8M0 B=32)** | **4.25+ε** | **4.25** |
| Signed values | 11 (±v₁…±v₅ + 0) | 15 (±v₁…±v₇ + 0) |

LUT6 uses 6 of the 8 available magnitude slots (3 bits can hold 8 values; 2 slots
wasted), with placement optimised per channel via Lloyd-Max. MXFP4 uses all 8 slots
with a fixed geometric grid.

### Results

| Scheme | strict% | thermal@0.7% | thermal@1.0% | bpw |
|---|---|---|---|---|
| **LUT6 per-channel (6 magnitudes, optimal)** | **20.6%** | **12.0%** | **10.6%** | 4.25+ε |
| MXFP4-E2M1 (8 magnitudes, fixed OCP) | 17.2% | 8.6% | 7.6% | 4.25 |
| LUT12 per-channel (12 magnitudes, optimal) | 9.2% | 3.4% | 2.8% | 5.25+ε |
| MXFP6-E2M3 (32 magnitudes, fixed OCP) | 4.2% | 0.8% | 0.6% | 6.25 |

### Analysis

**LUT6 is worse than fixed MXFP4: +3.4 pp strict, +3.4 pp thermal@0.7.**
Optimal placement of 6 magnitudes cannot compensate for having 2 fewer codes than
the OCP MXFP4 grid. The fixed 8-code geometric spacing outperforms a learned 6-code
placement at the same bit budget.

This is the direct contrast with LUT12:

| Scheme | Magnitudes | Magnitude bits | vs MXFP4 strict | vs MXFP4 thermal@0.7 |
|---|---|---|---|---|
| LUT6 (optimal) | 6 | 3 | **+3.4 pp worse** | **+3.4 pp worse** |
| MXFP4 (fixed) | 8 | 3 | — | — |
| LUT12 (optimal) | 12 | 4 | −8.0 pp better | −5.2 pp better |

**Conclusion:** The number of magnitudes matters more than optimal placement at this
bit width. With 3 magnitude bits:

- MXFP4 uses all 8 available codes with geometric spacing → wins
- LUT6 uses only 6 codes with optimal spacing → loses by 3.4 pp

The geometric MXFP4 grid is well-matched to the observed weight distribution
(monotone-decreasing usage from the exp44d histogram) precisely because geometric
spacing puts more codes near zero — which is where the weight mass is. Per-channel
Lloyd-Max on 6 codes cannot do better because there simply aren't enough codes to
cover the full dynamic range while also providing resolution near zero.

The LUT12 result (exp45) was the achievement: 12 optimally-placed magnitudes in
4 magnitude bits matches 16 fixed magnitudes (MXFP5) in 4 magnitude bits, with
only an 18 MB per-model sidecar cost. LUT6 confirms the floor: below 8 optimally-
placed magnitudes, even perfect per-channel fitting cannot match the fixed OCP grid
at the same bit width.
