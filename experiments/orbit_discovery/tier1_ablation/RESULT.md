# Tier 1 ablation: 10-seed reliability study

Compares `label_disagreement` vs `classifier_entropy` candidate priors for 3-body periodic-orbit discovery, holding all other pipeline parameters at the paper baseline (k=6, MLP hidden=128, n_layers=4, N_CANDIDATES=5000, N_REFINE_TOP=50).

Seeds present: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Total compute: 8109 s (2.25 h)

## Per-method summary

| Method | n_verified (mean ± 95% CI) | hit_rate_verify (mean ± 95% CI) | Wall / run (mean, s) |
|---|---|---|---|
| label_disagreement | 44.000  (95% CI [42.988, 45.012], std 1.414, n=10) | 0.009  (95% CI [0.009, 0.009], std 0.000, n=10) | 405 |
| classifier_entropy | 18.200  (95% CI [17.543, 18.857], std 0.919, n=10) | 0.004  (95% CI [0.004, 0.004], std 0.000, n=10) | 406 |

## Paired comparison: label_disagreement − classifier_entropy

Paired seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

### n_verified
- mean(ld − ce) = +25.800  (std 2.150)
- paired t = 37.948
- two-sided p = 0.00000
- one-sided p (ld > ce) = 0.00000
- Cohen's d (paired) = +12.000
- Wilcoxon signed-rank: stat=0.00, two-sided p=0.00195
- per-seed diffs: [24.0, 31.0, 24.0, 25.0, 27.0, 25.0, 25.0, 26.0, 27.0, 24.0]

### hit_rate_verify
- mean(ld − ce) = +0.00516  (std 0.00043)
- paired t = 37.948
- two-sided p = 0.00000
- one-sided p (ld > ce) = 0.00000
- Cohen's d (paired) = +12.000
- Wilcoxon signed-rank: stat=0.00, two-sided p=0.00195

## Interpretation

label_disagreement finds ~+141.8% more verified orbits than classifier_entropy (mean Δ = +25.80, one-sided paired-t p = 0.0000, Cohen's d = +12.00). The paper's claim is supported at n=10.
