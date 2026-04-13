# Gate Decision v2 (Phase C-2)

Date: 2026-04-13

## Per-criterion status

| ID | Criterion | Result | Status |
|----|-----------|--------|--------|
| D1 | Augmentation bug diagnosed (3.1) | No bug. F1 gap 0.106 is real: S3-symmetric data + non-equivariant MLP = capacity dilution. Paragraph written. | PASS |
| D2 | 2D scaling law with 2 architectures (3.2) | MLP slope -0.131 +/- 0.005 (ratio 1.01). S3BodyNet slope -0.141 +/- 0.009 (ratio 1.09). Both within 2sigma of GOY -0.130. CI widths 0.009 and 0.018 (both < 0.04). | PASS |
| D3 | 6D framing fixed (3.3) | No multi-fractal explanation. Alpha varies only 0.022 across 4 decades. No local alpha matches classifier slope -0.054. Honest negative framing. | PASS (negative) |
| D4 | Equivariance contribution resolved (3.4) | At N=1M: S3BodyNet F1 0.646, MLP-features-noshare F1 0.700. Gap -0.054. No crossover. Equivariance demoted. | PASS (demoted) |
| D5 | No undertraining (3.5) | F1 at epoch 60: 0.678, at epoch 600: 0.638. F1 decreased (overfitting). Phase A was converged. | PASS |

## Decision

D1, D2, D5 pass. D3 resolves via honest-negative. D4 resolves via demotion.

Per Section 5 decision rules: "D1, D2, D5 pass; D3 or D4 resolve via honest
demotion (not crossover / not mechanism) -> SUBMIT with reframed contributions."

**OUTCOME: SUBMIT to ML4PS.**

## Reframed contributions

Per Section 6.1:
1. Benchmark release: reproducible dataset on shape sphere and 6D phase space.
2. 2D GOY verification: measured alpha predicts test-error scaling across two
   architectures (MLP and S3BodyNet) to within 1% and 9% respectively.
3. Dimensionality-dependent failure of active learning: AL helps in 2D (+3.7% F1),
   hurts in 6D (-0.7% F1) because D_u = 5.86 means 97% of phase space is near
   a boundary.

Equivariance and 6D scaling are honest discussion points, not headline contributions:
- S3BodyNet loses to MLP-features-noshare at both N=300k and N=1M. Built-in
  equivariance does not help at the tested budgets.
- 6D GOY prediction underestimates classifier error by 2.2x. The boundary is
  mono-fractal. The disagreement is unexplained and reported honestly.
- S3 data augmentation hurts MLP performance by 10 F1 points. The augmentation
  code is correct. The degradation is a capacity effect on S3-symmetric data.
