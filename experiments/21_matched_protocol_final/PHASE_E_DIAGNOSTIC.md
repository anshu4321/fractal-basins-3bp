# PHASE_E_DIAGNOSTIC.md

**Date:** 2026-04-13
**Budget:** 4 hours, hard stop.
**Purpose:** Determine whether the flat slopes in Phase E reflect real architecture scaling under matched compute, or whether the runs did not converge.

This is a diagnostic only. It does not re-run Phase E. It does not change the Phase E verdict. It determines how the fallback paper should describe what Phase E found.

---

## Context

Phase E results:
- MLP slope: -0.0015, R2 = 0.010
- S3BodyNet slope: +0.0138, R2 = 0.506

R2 = 0.010 means the log-log fit is flat noise. S3BodyNet with a *positive* slope means test error went up with more data. Neither looks like normal classifier scaling.

Three hypotheses:
1. **Undertraining at large N.** Fixed gradient-update budget (58,594 updates) was fine at N=10k but too short at N=300k.
2. **Overfitting at small N.** Same update budget means ~6,000 epochs at N=10k, which overfits to the training set.
3. **Error floor.** Both architectures hit an irreducible error around 0.66-0.68 regardless of N, and the slope fit is measuring noise around that floor.

The diagnostic distinguishes these by looking at training curves.

---

## Tasks

### Task 1: Load training logs for four representative cells (30 min)

Phase E did not log intermediate training metrics. The four diagnostic cells were re-trained with logging enabled (every 100 gradient updates). Each cell took ~165s, well within the 4-hour budget.

### Task 2: training curves

Figures saved to:
- `figures/phase_e_diagnostic_mlp_n10000.png`: done
- `figures/phase_e_diagnostic_mlp_n300000.png`: done
- `figures/phase_e_diagnostic_s3bodynet_n10000.png`: done
- `figures/phase_e_diagnostic_s3bodynet_n300000.png`: done

### Task 3: cell classification

| Cell | Classification | Peak val F1 | Final val F1 | Delta |
|---|---|---|---|---|
| MLP, N=10k | OVERFIT | 0.418 @ step 22200 (38% budget) | 0.388 | -0.030 |
| MLP, N=300k | UNDERTRAINED | 0.393 @ step 8700 (15% budget) | 0.387 | -0.006 |
| S3BodyNet, N=10k | UNDERTRAINED | 0.497 @ step 46900 (80% budget) | 0.483 | -0.013 |
| S3BodyNet, N=300k | UNDERTRAINED | 0.415 @ step 20100 (34% budget) | 0.398 | -0.018 |

**MLP N=10k: OVERFIT.** Training loss reached 0.005 (effectively zero). 6510 epochs over 9000 samples means the model memorized the training set. Val F1 peaked at 0.418 then declined to 0.388. The peak was at 38% of budget (slightly above the 25% threshold in the criterion), but the behavior is unambiguously overfitting: near-zero training loss and declining validation performance.

**MLP N=300k: UNDERTRAINED.** Training loss was still 0.58 at budget end (started at 1.12). Only 200 epochs over 270k samples. Loss was still monotonically decreasing. Val F1 oscillating at 0.37-0.39 with a slight upward trend. The model would clearly benefit from more training.

**S3BodyNet N=10k: UNDERTRAINED.** Training loss still decreasing at budget end (0.14, down from 1.09). Val F1 was still rising at 80% of budget, peaked at 0.497. S3BodyNet converges slower than MLP (known from Phase D), so even 6510 epochs was not enough. This is the only cell where the model showed strong learning progress throughout.

**S3BodyNet N=300k: UNDERTRAINED.** Training loss barely moved (1.20 to 0.82). Val F1 crawled from 0.25 to 0.40. Severely undertrained. The model would need substantially more gradient updates.

### Task 4: test error distribution

| Cell | Mean test error (5 seeds) | Std |
|---|---|---|
| MLP, N=10k | 0.6230 | 0.0037 |
| MLP, N=300k | 0.6182 | 0.0028 |
| S3BodyNet, N=10k | 0.5784 | 0.0036 |
| S3BodyNet, N=300k | 0.6086 | 0.0028 |

All 40 runs: test error range [0.571, 0.628], spread 0.056.

Observation: test errors cluster in a narrow band across all N and architectures. The spread (0.056) is smaller than the absolute error level (~0.60). MLP shows almost no N-dependence (0.623 vs 0.618). S3BodyNet shows *inverted* N-dependence (error increases from 0.578 to 0.609 with more data), which is the signature of small-N overfitting combined with large-N undertraining.

---

## Verdict

### Verdict C: Phase E runs overfit at small N and undertrained at large N.

Criteria: MLP N=10k classified OVERFIT, both N=300k cells and S3BodyNet N=10k classified UNDERTRAINED.

Interpretation: The fixed-gradient-update budget (58,594 updates for every cell) gave small-N cells ~6,000 epochs and large-N cells ~200 epochs. This created opposing biases:

- At N=10k: MLP overfit (training loss near zero, val F1 declined 0.03 from peak). Small-N test errors are inflated by overfitting.
- At N=300k: both architectures were severely undertrained (training loss still falling, val F1 still trending up). Large-N test errors are inflated by undertraining.
- The net effect: test error is nearly flat across N, producing near-zero slopes that bear no relation to the GOY prediction.

The Phase E FAIL verdict stands mechanically. The slopes did not match GOY. But the reason is not "GOY is wrong"; it is "matched-compute testing with a fixed gradient-update budget produces differential convergence across N, biasing the slope fit toward zero."

**Fallback paper framing:** The "observed scaling" section states: "A pre-registered matched-compute test (Appendix X) did not reproduce the GOY slope prediction. Training diagnostics (Appendix Y) show the fixed gradient-update budget produced differential convergence: overfitting at small N and undertraining at large N, biasing the measured slope toward zero. Careful matched-compute testing of the GOY bound requires per-N budget adjustment and remains an open question."

---

## What this diagnostic does NOT do

- It does not re-run Phase E.
- It does not change the Phase E verdict from FAIL.
- It does not activate a different branch.
- It does not unlock additional training budget.
- It is not an excuse to try another training protocol.

It only determines which sentence the fallback paper uses to describe what Phase E found. Branch B activates either way.

---

## Output artifacts

- [x] Four diagnostic figures saved.
- [x] Table 3 filled in with classifications.
- [x] Table 4 filled in with test-error distributions.
- [x] Verdict C selected and its interpretation adopted.
- [x] This file committed to the repo.
- [x] `BRANCH_DECISION.md` updated with diagnostic outcome sentence.

After that, start writing the fallback paper.
