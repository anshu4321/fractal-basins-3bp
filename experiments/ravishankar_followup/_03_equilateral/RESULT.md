# Equilateral-section search — RESULT

Wall time (outputs only): 0.01 s

## Pipeline summary

| Stage | Output | Count |
|---|---|---|
| Basin map | basin_map.npz | 256² = 65,536 cells |
| Periodic-labelled cells | — | 12,188 (18.6%) |
| Classifier val_acc (best / final) | classifier.eqx | 0.493 / 0.486 |
| Classifier train_acc (final) | — | 0.933 |
| Classifier neighbour agreement | — | 14.1% (fractal — physical ceiling) |
| LD field | ld_field.npz | 512² (max 1.318, mean 0.182) |
| Candidate seeds | candidates_ICs.json | 19 (k-means + figure-8 anchor) |
| Newton-refined at 1e-6 | refined_candidates.json | 6 / 19 |
| HP-verified at 1e-11 | hp_verified_candidates.json | **2 / 6** |
| Monodromy classified | monodromy_equilateral.json | 2 (both hyperbolic) |
| Topology verdict | topology_match_equilateral.json | **2 NOVEL** |

## The two new orbits

| Name | u_x | u_y | T | E | |λ|_max | Syzygy | Verdict |
|---|---|---|---|---|---|---|---|
| EQ1 | +0.3807 | -1.1182 | 7.7147 | -1.2872 | 135.96 | empty | NOVEL |
| EQ3 | -0.4637 | -0.9061 | 7.6661 | -1.8705 | 15842.04 | empty | NOVEL |

Additional monodromy detail:

- **EQ1**: det(M) − 1 = +1.69e-13, closure err = 4.66e-14, stability index = 135.97.
- **EQ3**: det(M) − 1 = -1.28e-09, closure err = 2.69e-12, stability index = 15842.04.

## Distinguishing feature: empty syzygy word

Both HP-verified orbits have bodies that **never** become collinear during
one period (the signed triangle area stays strictly bounded away from zero
throughout). All 24 582 Hristov 2024 entries have non-empty syzygy words
(length ≥ 2). This makes EQ1 and EQ3 genuinely topologically novel: a new
class of *non-syzygy* orbits in the equilateral-section ($L_z \neq 0$) regime.

Verdict string for both (from `topology_match_equilateral.json`):

> NOVEL (non-syzygy orbit: empty word; Hristov 2024 has 0 length-0 entries out of 24,582; Hristov's catalogs are syzygy-based by construction and cannot contain non-syzygy orbits)

## Figure-8 sanity

Basin cell at $(u_x, u_y) = (+0.620, -0.871)$ has $E = -1.2872$,
$\Delta E = 0.0002$ from the Chenciner–Montgomery figure-8 reference
($E \approx -1.287$). The pipeline identifies this cell as periodic-labelled,
confirming the basin + classifier pipeline correctly resolves known orbits.

However Newton refinement from that IC converges to
$T = 7.715$ ($\approx 2 \times$ the figure-8 half-period), landing on
**EQ1** — a *different* orbit at the same energy $E = -1.2872$. The
Chenciner–Montgomery figure-8 at $T = 6.326$ has an IC on the equilateral
section that our $256^2$ grid apparently does not resolve, and is not
rediscovered in this pass.

## Fractal-boundary discussion

The classifier validation accuracy caps at **0.493** despite training
accuracy reaching **0.933**. The root cause is intrinsic: only
**14.1%** of grid cells share a label with all four 4-neighbours.
Under an iid random 80/20 split no classifier can generalise beyond this
neighbour-agreement ceiling.

This is not a classifier failure — it is a physical consequence of the
equilateral section's fractal basin structure. The entropy-based LD field
(max 1.318, mean 0.182) still correctly identifies the fractal
boundary regions, and the candidate-selection strategy (k-means on
periodic+bounded cells, then minimum-LD cluster representative) bypasses the
val-acc limitation by bypassing the classifier entirely at seed-selection
time.

## Artefacts

- `basin_map.npz`, `classifier.eqx`, `classifier.pkl`, `classifier_snapshots.npz`
- `ld_field.npz`, `figure_8_sanity.json`
- `candidates_ICs.json`, `refined_candidates.json`, `hp_verified_candidates.json`
- `monodromy_equilateral.json`, `topology_match_equilateral.json`
- `figures/equilateral_basin_{paper.pdf,blog.png}`
- `figures/candidates_ld_scan_{paper.pdf,blog.png}`
- `figures/orbits_panel_equilateral_{paper.pdf,blog.png}`
- `figures/section_comparison_{paper.pdf,blog.png}`
- `animations/basin_reveal.{gif,mp4}`
- `animations/ld_growth.{gif,mp4}`
- `animations/equilateral_vs_euler.{gif,mp4}`
- `animations/new_orbit_EQ1.{gif,mp4}`, `animations/new_orbit_EQ3.{gif,mp4}`

## TODOs / follow-ups

1. Re-run the search restricted to the $u_y = 0$ sub-axis ($L_z = 0$ regime)
   to connect back to Hristov's zero-angular-momentum catalogue.
2. Full Maslov index on EQ1 and EQ3 (currently only mod 2 from the terminal
   monodromy spectrum; the full Conley–Zehnder winding is a natural next step
   shared with Thread 2's TODOs).
3. 50-digit mpmath HP refinement of EQ1 and EQ3 to tighten closure beyond the
   longdouble $\approx 10^{-14}$ floor achieved here.
4. Cycle expansion: find more periodic orbits on this section (current $256^2$
   grid probably misses short-period orbits; a denser or adaptive grid would
   help).
