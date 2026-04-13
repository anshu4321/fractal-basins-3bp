# Phase E Summary

## What was tested

A pre-registered matched-protocol experiment to answer: do MLP and S3BodyNet both produce scaling slopes within 2sigma of the GOY prediction (-0.1295) when trained under identical conditions?

One protocol. One run. Pre-registered pass criteria. 72-hour hard deadline.

## Protocol

- AdamW, batch 1024, 500-step warmup, cosine decay to 0
- Fixed compute budget: 58,594 gradient updates per cell (200 epochs at N=300k)
- Peak LR tuned once per architecture on 8k/2k split of N=10k: both selected 3e-3
- Seeds {0,1,2,3,4}, N in {10k, 30k, 100k, 300k}
- Metric: 1 - macro_F1
- 40 total runs (2 architectures x 4 N values x 5 seeds)

## Results

| Architecture | Slope | 2sigma CI | R2 |
|---|---|---|---|
| MLP | -0.0015 | 0.0029 | 0.010 |
| S3BodyNet | +0.0138 | 0.0017 | 0.506 |
| GOY prediction | -0.1295 | 0.0190 | -- |

## Verdict: FAIL

Three of four pass criteria failed:

1. MLP-GOY agreement: gap 0.128 vs threshold 0.019. **FAIL.**
2. S3BodyNet-GOY agreement: gap 0.143 vs threshold 0.019. **FAIL.**
3. Inter-architecture agreement: gap 0.015 vs threshold 0.003. **FAIL.**
4. CI width floor: both < 0.04. **PASS.**

## Diagnostic: why it failed

Four cells re-trained with logging (every 100 gradient updates). Classification:

| Cell | Status | Training loss | Val F1 |
|---|---|---|---|
| MLP, N=10k | OVERFIT | 0.005 (near zero) | peaked 0.418, declined to 0.388 |
| MLP, N=300k | UNDERTRAINED | 0.58 (still falling) | oscillating 0.37-0.39 |
| S3BodyNet, N=10k | UNDERTRAINED | 0.14 (still falling) | peaked 0.497, still rising at 80% budget |
| S3BodyNet, N=300k | UNDERTRAINED | 0.82 (barely moved) | crawled from 0.25 to 0.40 |

**Root cause:** The fixed gradient-update budget gave N=10k cells ~6,000 epochs and N=300k cells ~200 epochs. Small-N cells overfit; large-N cells undertrained. Both effects inflate test error, flattening the slope to near zero.

## What this means

The FAIL is real in the mechanical sense: the pre-registered protocol did not produce slopes matching GOY. But the diagnostic shows the protocol itself was inadequate. The flat slopes are an artifact of differential convergence, not evidence against the GOY bound.

Phase C's architecture-specific protocols (60 epochs for MLP, 200 for S3BodyNet) produced slopes of -0.131 and -0.141, both within 2sigma of GOY. Phase E's matched-compute protocol erased that signal.

## Branch decision

**Fallback paper (Branch B) activated.**

Two headline contributions:
1. Benchmark release (dataset + Zenodo DOI)
2. Dimensionality-dependent failure of active learning

GOY scaling demoted to "Observed scaling exponents and the GOY prediction" section. Phase E results and diagnostic figures included in appendix. The paper states explicitly that the matched-compute test did not reproduce tight GOY agreement and explains why (differential convergence), framing it as an open methodological question.

## Artifacts

| File | Description |
|---|---|
| `lr_tuning.json` | Peak LR selection (both architectures chose 3e-3) |
| `results.json` | All 40 run results (accuracy, macro_F1, test_error, confusion matrices) |
| `fit.json` | Slope fits, uncertainties, pass criteria evaluation |
| `VERDICT.md` | Formal PASS/FAIL verdict |
| `diagnostic.py` | Re-training script with intermediate logging |
| `diagnostic_logs.json` | Training curves for 4 diagnostic cells (585 log points each) |
| `PHASE_E_DIAGNOSTIC.md` | Diagnostic analysis with cell classifications and verdict |
| `figures/phase_e_diagnostic_*.png` | 4 training curve figures |
| `BRANCH_DECISION.md` | Branch B activation with diagnostic outcome |
| `PHASE_E_START.md` | Clock start timestamp |

## Timeline

- Hour 0: Committed experiment skeleton, started clock (2026-04-13T00:29:42Z)
- Hour 0-0.1: Peak-LR tuning (8 trials, ~1 min total)
- Hour 0.1-1.6: Main sweep (40 runs, ~95 min total)
- Hour 1.6: Slope fitting and verdict (FAIL)
- Hour 6.8-7.5: Diagnostic re-training (4 cells with logging, ~11 min)
- Hour 7.5: Diagnostic verdict (C: differential convergence)
- Deadline: 2026-04-16T00:29:42Z (not needed; completed in <8 hours)
