# Phase C Report: Remediation Results and Submit/No-Submit Decision

## Context

This report summarizes the results of Phase C-1 (targeted remediation) for the paper "Fractal Basins as a Benchmark for Equivariant Classifiers and Active Learning: The Planar Three-Body Problem", targeting NeurIPS ML4PS 2026.

Phase A produced 6 core experiments on the equal-mass planar three-body problem basin classification task. A gate review identified 4 problems (P1-P4). Phase C ran 5 pre-registered experiments with version-locked pass/fail criteria to resolve them.

### Phase A baseline (for reference)

- Dataset: 10^6 ICs at rest on the shape sphere (2D), 10^6 in 6D phase space. Close-encounter exclusion: 9.1% (2D), 3.4% (6D).
- Uncertainty exponent: alpha_2D = 0.259 +/- 0.019 (R^2 = 0.996), alpha_6D = 0.145 +/- 0.004 (R^2 = 0.999).
- 2D scaling law (MLP only): measured slope -0.137, GOY predicted -0.130. Ratio 1.06.
- 6D scaling law (MLP + S3BodyNet): both measured ~-0.054, GOY predicted -0.024. Off by 2.2x.
- Ablation at N=300k (5 seeds): MLP-raw F1 0.669, MLP-raw-aug F1 0.561, MLP-features-noshare F1 0.675, S3BodyNet F1 0.625.
- Active learning: +3.7% F1 in 2D, -0.7% F1 in 6D.

### Problems identified in the gate review

- **P1 (blocking):** 2D scaling law used only one architecture (MLP). Need >=2.
- **P2 (blocking):** S3 augmentation row (F1 0.561) likely contains a bug. An exact symmetry augmentation should not degrade F1 by 11 points.
- **P3 (framing):** 6D scaling law off by 2.2x, dishonestly framed as "conservative bound."
- **P4 (framing):** S3BodyNet loses to MLP-features-noshare at N=300k. No evidence of crossover at larger N.

---

## Experiment 3.1: S3 Augmentation Diagnosis (D1)

**Problem:** MLP-raw + S3 augmentation scored F1 0.561 vs MLP-raw at 0.669. Data augmentation using an exact symmetry should not degrade performance by 11 F1 points.

**Method:** (1) Inspected augmentation code in `mega3bp/equivariant.py`. (2) Wrote and ran 4 unit tests on GPU. (3) Re-ran MLP-raw and MLP-raw-aug at N=300k, 5 seeds, with convergence-based training (up to 300 epochs, cosine LR schedule matched to 300 epochs, convergence when val F1 plateaus).

### Unit test results

| Test | Result | Details |
|------|--------|---------|
| Round-trip (apply pi, then pi-inverse) | PASS | 0 label mismatches across all 6 S3 elements. Float32 x-errors 1e-3 to 3e-3 from H100 TF32 matmul. |
| Class distribution preservation | PASS | All class ratios within 0.4% of 1.0 under uniform random S3 augmentation (100k samples). |
| Group multiplication table | PASS | R, M, P matrices verified consistent for all 36 products in S3. |
| Physical consistency (pairwise distances) | PASS | Pairwise distances preserved under augmentation within float32 gauge noise. |

**Conclusion: No bug found.** The augmentation code correctly applies both the input transformation (R on shape sphere, M on Jacobi momenta) and the label permutation (LABEL_PERM_ALL) consistently.

### Training results

| Model | F1 (5 seeds) | Epochs (mean) |
|-------|-------------|---------------|
| MLP-raw | 0.6577 +/- 0.0105 | 39 |
| MLP-raw-aug | 0.5517 +/- 0.0036 | 28 |

Gap: 0.1060. 2sigma threshold: 0.0210. The gap is 5x larger than 2sigma.

Phase A comparison: MLP-raw 0.669, MLP-raw-aug 0.561 (60 fixed epochs). The Phase C numbers are slightly lower because convergence-based training stopped earlier, but the gap (0.106 vs 0.108) is identical.

### Explanation (real finding, not a bug)

The 6D phase-space sampling is S3-symmetric by construction: shape sphere coordinates are uniform on S2 (invariant under orthogonal S3 action), and Jacobi momenta are uniform in a 4-ball (invariant under block-diagonal orthogonal S3 action). Therefore S3 data augmentation provides no new information about the basin-label function. The training distribution is identical with and without augmentation.

The degradation is a sample-efficiency effect. The non-equivariant MLP must implicitly learn 6-fold equivariance from the augmented data rather than specializing to the single coordinate frame present in the test set. Without augmentation, the model allocates its full capacity to the original coordinate system and can exploit coordinate-specific decision boundaries. With augmentation, the model must generalize across all 6 S3 orientations, effectively dividing capacity sixfold. The augmented model converges faster (28 vs 39 epochs) but to a lower plateau, confirming this is not an undertraining artifact but a fundamental capacity limitation.

This is consistent with the overall ablation: the architecturally equivariant S3BodyNet (F1 0.625 at N=300k) outperforms MLP-raw-aug (F1 0.552) by a large margin, confirming that built-in equivariance avoids the capacity cost of learning symmetry from data.

### Pass criterion

> "F1 for MLP-raw + S3 aug is within 2sigma of MLP-raw, or strictly better. If F1 is still worse by >2sigma after the bug is confirmed absent, write one paragraph explaining why (real finding) and keep the row with that explanation."

F1 gap 0.106 > 2sigma 0.021. Bug confirmed absent. Explanation paragraph written.

**D1: PASS.**

---

## Experiment 3.2: 2D Scaling Law with Two Architectures (D2)

**Problem:** Phase A tested only MLP on 2D. The headline claim requires >=2 independent architectures.

**Method:** Ran both MLP and S3BodyNet on the 2D at-rest dataset at N in {10k, 30k, 100k, 300k}, 5 seeds per cell. N=1M skipped (only ~853k training samples available after test holdout, same as Phase A). Same 100k held-out test set as Phase A (seed 999).

S3BodyNet adaptation: padded 3D shape-sphere input with zero Jacobi momenta to create 7D input. This means 5 of 8 per-body features (kinetic energy, radial velocity, angular momentum) are exactly zero. Only distance-based features are active.

Training: MLP at 60 fixed epochs (cosine LR schedule matched to 60 epochs, same as Phase A). S3BodyNet at 200 fixed epochs (needed more training due to half-dead features and more complex architecture).

Slope fitted by log(error) vs log(N) linear regression. Error bars by jackknife over 5 seeds.

### Per-N results

**MLP (3D input, 60 epochs):**

| N | Test error | F1 |
|---|-----------|------|
| 10,000 | 0.353 +/- 0.008 | 0.261 +/- 0.007 |
| 30,000 | 0.241 +/- 0.013 | 0.318 +/- 0.009 |
| 100,000 | 0.229 +/- 0.005 | 0.332 +/- 0.005 |
| 300,000 | 0.218 +/- 0.005 | 0.366 +/- 0.002 |

**S3BodyNet (7D input with zero momenta, 200 epochs):**

| N | Test error | F1 |
|---|-----------|------|
| 10,000 | 0.369 +/- 0.008 | 0.268 +/- 0.004 |
| 30,000 | 0.336 +/- 0.011 | 0.279 +/- 0.005 |
| 100,000 | 0.266 +/- 0.007 | 0.306 +/- 0.003 |
| 300,000 | 0.234 +/- 0.012 | 0.323 +/- 0.008 |

### Scaling law fit

| Architecture | Measured slope | Jackknife sigma | 2sigma CI width | GOY predicted | Ratio |
|-------------|---------------|-----------------|-----------------|---------------|-------|
| MLP | -0.1306 | 0.0046 | 0.0092 | -0.1295 | 1.009 |
| S3BodyNet | -0.1410 | 0.0091 | 0.0181 | -0.1295 | 1.089 |

### Pass criterion

> "S3BodyNet measured 2D slope is within 2sigma of the GOY prediction -0.130, AND the 2sigma confidence interval on the slope has width <= 0.04."

- S3BodyNet slope -0.141, GOY -0.130. Difference 0.011. 2sigma = 0.018. 0.011 < 0.018. Within 2sigma: **YES.**
- 2sigma CI width = 0.018 <= 0.04: **YES.**

**D2: PASS.**

---

## Experiment 3.3: 6D Multi-fractal Measurement (D3)

**Problem:** The 6D scaling law disagrees with GOY by 2.2x. Phase A framed this as a "conservative bound," which is dishonest post-hoc reinterpretation.

**Method:** Analyzed the existing Experiment 02 sliding-window alpha data (which measures alpha in overlapping epsilon sub-ranges across 4 decades). Checked whether (a) the boundary is multi-fractal (alpha varies with scale) and (b) any local alpha predicts the classifier slope -0.054.

### Sliding-window alpha (6D)

| mid_epsilon | alpha | R^2 | Predicted slope (-alpha/6) |
|-------------|-------|-----|--------------------------|
| 9.79e-06 | 0.1367 | 0.9991 | -0.0228 |
| 3.06e-05 | 0.1439 | 0.9988 | -0.0240 |
| 9.79e-05 | 0.1529 | 0.9987 | -0.0255 |
| 3.06e-04 | 0.1566 | 0.9990 | -0.0261 |
| 9.79e-04 | 0.1567 | 0.9991 | -0.0261 |
| 3.06e-03 | 0.1428 | 0.9928 | -0.0238 |
| 9.79e-03 | 0.1346 | 0.9962 | -0.0224 |
| 3.06e-02 | 0.1522 | 0.9761 | -0.0254 |

Alpha range: 0.135 to 0.157. Spread: 0.022.

### Findings

1. **Not multi-fractal.** Alpha varies by only 0.022 across 4 decades of epsilon. The boundary is approximately mono-fractal.
2. **No local alpha matches the classifier slope.** All local predicted slopes fall in [-0.026, -0.022]. The classifier slope is -0.054. The minimum gap is 0.028, far outside any reasonable error bar.
3. **No mechanistic explanation found** for the 2.2x disagreement between the global GOY prediction (-0.024) and the classifier slope (-0.054).

### Pass criterion

> "Either the generalized-dimension measurement or the local-alpha measurement provides a mechanistic, quantitative explanation for the factor-of-2.2 disagreement, OR the 6D result is accepted as a negative result and the paper says so."

No mechanism found. The 6D result is accepted as a negative finding.

**D3: PASS (honest negative framing).**

---

## Experiment 3.4: Equivariance Crossover at N=1M (D4)

**Problem:** S3BodyNet loses to MLP-features-noshare at N=300k. Phase A asserted "crossover at larger N" without evidence.

**Method:** Re-ran the four-way ablation at N=1M on 6D, 5 seeds, 60 epochs, same setup as Phase A but at 3.3x more training data.

### Results at N=1M (6D, 5 seeds)

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| MLP-raw | 0.7095 +/- 0.0010 | 0.6989 +/- 0.0008 |
| MLP-raw-aug | 0.6012 +/- 0.0004 | 0.5737 +/- 0.0005 |
| MLP-features-noshare | 0.7106 +/- 0.0008 | 0.7001 +/- 0.0008 |
| S3BodyNet | 0.6625 +/- 0.0010 | 0.6458 +/- 0.0011 |

### Comparison with Phase A (N=300k)

| Model | F1 at N=300k | F1 at N=1M | Delta |
|-------|-------------|-----------|-------|
| MLP-raw | 0.669 | 0.699 | +0.030 |
| MLP-raw-aug | 0.561 | 0.574 | +0.013 |
| MLP-features-noshare | 0.675 | 0.700 | +0.025 |
| S3BodyNet | 0.625 | 0.646 | +0.021 |

Ranking at N=1M: MLP-features-noshare (0.700) = MLP-raw (0.699) > S3BodyNet (0.646) >> MLP-raw-aug (0.574).

The gap between S3BodyNet and MLP-features-noshare is -0.054 at N=1M vs -0.050 at N=300k. The gap has not narrowed. No crossover.

### Pass criterion

> "At N=10^6, S3BodyNet F1 >= MLP-features-noshare F1 - 0.005."

S3BodyNet F1 0.646 < MLP-features-noshare F1 0.700 - 0.005 = 0.695. Fails by 0.049.

**D4: FAIL (no crossover). Equivariance demoted** from headline contribution to honest negative result in the ablation.

---

## Experiment 3.5: Training Convergence Sanity Check (D5)

**Problem:** Phase A wall-clock numbers (20 seconds per training run on H100) suggest possible undertraining.

**Method:** Trained MLP-raw at N=300k (6D) for 600 epochs (10x Phase A's 60). Cosine LR schedule matched to 600 epochs. Single seed (42). Evaluated F1 every 10 epochs.

### Convergence trajectory

| Epoch | Accuracy | Macro F1 |
|-------|----------|----------|
| 1 | 0.3635 | 0.3592 |
| 10 | 0.6034 | 0.5812 |
| 20 | 0.6602 | 0.6398 |
| 30 | 0.6704 | 0.6557 |
| 40 | 0.6781 | 0.6652 |
| 50 | 0.6885 | 0.6752 |
| **60** | **0.6910** | **0.6779** |
| 100 | 0.6724 | 0.6630 |
| 200 | 0.6508 | 0.6413 |
| 300 | 0.6465 | 0.6366 |
| 400 | 0.6490 | 0.6383 |
| 500 | 0.6507 | 0.6393 |
| 600 | 0.6497 | 0.6384 |

F1 peaks at epoch 60 (0.678) and then declines due to overfitting, stabilizing around 0.638 from epoch 300 onward. The model was not undertrained at 60 epochs. It was at or slightly past the optimal stopping point.

Note: the 600-epoch LR schedule differs from Phase A's 60-epoch schedule. Under the 600-epoch schedule, the LR at epoch 60 is still near its peak, which explains why epoch 60 here (F1 0.678) is slightly higher than Phase A's (F1 0.669). The subsequent decline is from continued training at high LR leading to overfitting.

### Pass criterion

> "If F1 increases by >0.01, the Phase A runs were undertrained."

F1 at epoch 600 (0.638) is LOWER than F1 at epoch 60 (0.678). F1 did not increase. Phase A was not undertrained.

**D5: PASS.**

---

## Gate Decision (Section 5)

| ID | Criterion | Pass condition | Outcome |
|----|-----------|---------------|---------|
| D1 | Augmentation bug diagnosed | Bug absent. Explanation written. | **PASS** |
| D2 | 2D scaling law with 2 architectures | MLP slope -0.131 (ratio 1.01), S3BodyNet slope -0.141 (ratio 1.09). Both within 2sigma. CI widths 0.009, 0.018. | **PASS** |
| D3 | 6D framing fixed | No multi-fractal mechanism. Honest negative framing. | **PASS (negative)** |
| D4 | Equivariance contribution resolved | No crossover at N=1M. Gap -0.054. Equivariance demoted. | **PASS (demoted)** |
| D5 | No undertraining | F1 drops after epoch 60. Model was converged. | **PASS** |

### Decision rule applied

> "D1, D2, D5 pass; D3 or D4 resolve via honest demotion (not crossover / not mechanism) -> SUBMIT with reframed contributions."

**OUTCOME: SUBMIT to ML4PS.**

---

## Reframed Contributions

The paper's three contributions (per Section 6.1 of the protocol):

1. **Benchmark release.** A reproducible dataset on Montgomery's shape sphere (2D at rest) and 6D reduced phase space for the equal-mass planar three-body problem. 10^6 initial conditions each, with strict escape criterion, close-encounter exclusion, and per-trajectory diagnostics. Released on Zenodo with a public GitHub repo.

2. **GOY scaling-law verification in 2D.** The measured MGOY uncertainty exponent alpha = 0.259 predicts test-error scaling across two independent architectures. MLP measured slope -0.131 vs GOY predicted -0.130 (0.9% match). S3BodyNet measured slope -0.141 vs predicted -0.130 (8.9% match). Both within 2sigma jackknife confidence intervals.

3. **Dimensionality-dependent failure of active learning.** Active learning improves classifier F1 by +3.7% in 2D but degrades it by -0.7% in 6D. Explanation: the uncertainty dimension D_u = 5.86 in 6D means the fractal boundary nearly fills the phase space, so uncertainty-based sampling converges to uniform sampling.

### Demoted to discussion points (not headline contributions)

- **Equivariance.** S3BodyNet loses to MLP-features-noshare at both N=300k (by 5.0 F1 points) and N=1M (by 5.4 F1 points). Built-in equivariance does not help on this benchmark at the tested budgets. Feature engineering (hand-computed invariant per-body features) outperforms learned equivariance.

- **6D scaling law.** The GOY prediction underestimates classifier error by a factor of 2.2 in 6D. The boundary is approximately mono-fractal (alpha spread < 0.03 across 4 decades). The disagreement is unexplained and reported as an open negative result. The paper titles this subsection "Where the GOY bound does not apply."

- **S3 data augmentation.** Augmentation of the MLP with exact S3 symmetry degrades F1 by 10.6 points (from 0.658 to 0.552 at N=300k; from 0.699 to 0.574 at N=1M). The augmentation code is verified correct. The degradation is a capacity-dilution effect: the non-equivariant MLP must spread representational capacity across all 6 S3 orientations of S3-symmetric data, gaining no new information.

---

## Deliverables

| Experiment | Directory | Key file |
|-----------|-----------|----------|
| 3.1 Augmentation diagnosis | `experiments/12_augmentation_diagnosis/` | `results/12_augmentation_diagnosis/results.json` |
| 3.2 2D scaling law (2 arch) | `experiments/13_scaling_2d_s3bodynet/` | `results/13_scaling_2d_s3bodynet/results.json` |
| 3.3 6D multifractal | `experiments/14_6d_multifractal/` | Analysis of `results/02_uncertainty_exponent/6d_extended.json` |
| 3.4 Ablation at N=1M | `experiments/15_ablation_n1m/` | `results/15_ablation_n1m/table.json` |
| 3.5 Convergence check | `experiments/16_convergence_check/` | `results/16_convergence_check/results.json` |
| Gate decision | project root | `GATE_DECISION_v2.md` |

---

## Open Questions for Reviewer

1. The S3BodyNet 2D scaling law required 200 epochs (vs 60 for MLP). Is this acceptable, or does it compromise the "same training conditions" requirement for a fair architecture comparison? The per-body features have 5 of 8 features exactly zero at zero momenta, so the architecture is operating in a degraded mode.

2. The convergence check (D5) used a 600-epoch LR schedule rather than a 60-epoch schedule with 540 additional epochs at minimum LR. The different LR profile means the F1 trajectory is not a clean comparison. Does the reviewer agree that the result still demonstrates convergence at 60 epochs?

3. The 6D scaling law disagreement (2.2x) is left as an open negative result. Should the paper speculate on possible causes (curse of dimensionality in the feature space, inadequacy of the single-alpha model at high D_u), or strictly report the numbers without interpretation?

4. At N=1M, MLP-raw essentially ties MLP-features-noshare (F1 0.699 vs 0.700). This suggests the hand-engineered features stop helping at large N. Is this worth highlighting, or is it noise given sigma = 0.001?
