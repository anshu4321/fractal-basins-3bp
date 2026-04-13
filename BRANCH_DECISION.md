# Branch Decision

Date: 2026-04-13

## Phase E result

Phase E ran a pre-registered matched-protocol test per PROMPT_v4.md. One training protocol (AdamW, batch 1024, 500-step warmup, cosine decay to 0, 58594 total gradient updates), two architectures, 4 N values, 5 seeds, 40 runs total. Peak LR tuned once per architecture on a held-out slice of N=10k (both selected 3e-3).

### Measured slopes (metric: 1 - macro_F1)

| Architecture | Slope | 2sigma CI | R2 |
|---|---|---|---|
| MLP | -0.0015 | 0.0029 | 0.010 |
| S3BodyNet | +0.0138 | 0.0017 | 0.506 |

### GOY prediction

-0.1295, 2sigma CI [-0.149, -0.110].

### Pass criteria (all four required)

1. MLP-GOY agreement: gap 0.128, threshold 0.019. **FAIL.**
2. S3BodyNet-GOY agreement: gap 0.143, threshold 0.019. **FAIL.**
3. Inter-architecture agreement: gap 0.015, threshold 0.003. **FAIL.**
4. CI width floor: MLP 0.003, S3BodyNet 0.002, max 0.04. **PASS.**

### Verdict

**FAIL.** Three of four criteria failed. The matched-protocol test did not reproduce the GOY scaling prediction.

## Branch activated

**Branch B: Fallback paper** (PROMPT_v4 Section 7).

Two contributions:
1. Benchmark release (dataset + Zenodo DOI).
2. Dimensionality-dependent failure of active learning.

GOY scaling demoted to "Observed scaling exponents and the GOY prediction" section. Phase E matched-protocol table included in appendix. No claim that GOY is verified.

## What changes from Gate Decision v2

- D2 (2D scaling law with 2 architectures) is no longer a headline contribution. Phase C measured slopes under architecture-specific protocols. Phase E's matched-protocol test shows the result does not survive when both architectures train under identical conditions.
- The "2D GOY verification" contribution is dropped from the headline.
- S3BodyNet is demoted to appendix ablation.
- "Capacity dilution" language for augmentation finding demoted to "consistent with."

## What remains

- Benchmark dataset description and Zenodo DOI.
- Uncertainty exponent measurement (alpha_2D = 0.259, alpha_6D = 0.145).
- Wada basin test and basin entropy.
- 2D vs 6D active-learning contrast (headline contribution).
- N=300k and N=1M ablation (appendix).
- 6D scaling law disagreement with GOY (part of "observed range" section).
- Phase E matched-protocol results (appendix table).
