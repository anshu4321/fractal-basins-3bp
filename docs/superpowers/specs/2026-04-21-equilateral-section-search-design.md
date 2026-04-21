# Equilateral-triangle section search for new periodic orbits

**Date:** 2026-04-21
**Thread:** 3 of 3 in the Ravishankar follow-up paper
**Target audience:** v2 / follow-up paper Section 4 ("Extension to the fully-symmetric section")

---

## Context

All four orbits A, B, C, D live on the **Euler velocity section**:

r₁ = (−1, 0),  r₂ = (+1, 0),  r₃ = (0, 0),  p₁ = p₂ = (v₁, v₂),  p₃ = −2(v₁, v₂)

which has **Klein 4-group D₂ = {I, σ_x, σ_y, σ_x σ_y}** stabilising it (verified in `15_scaling_and_invariants/symmetry_results.json`, label-agreement 99.2%-99.4% on the basin). Ravishankar's note #3/#4 of 2026-04-15 asked about extending the search to a **fully S₃-permutation-symmetric section** where all three bodies sit on the vertices of an equilateral triangle.

Under S₃ × O(2)_rotational × ℤ₂_reflection × ℤ₂_time-reversal × ℝ₊_scaling, the fully-symmetric section has a much smaller stabiliser (O(2) continuous rotations break the section; scaling is handled analytically; time-reversal identifies ICs pairwise). What's left: a 2-parameter (u_x, u_y) family of velocities, with L = 0 automatic, exactly as on the Euler section but with a different spatial configuration.

This section is essentially unexplored by the Sukava-Dmitrasinovic / Li-Liao / Hristov pipelines (which privilege the Euler section for equal-mass studies). Applying our label-disagreement ML pipeline here is a direct test of the method on a new manifold.

## Goal (one sentence)

Run the full basin-classifier → label-disagreement → candidate → Newton-refine → HP-verify → monodromy → catalog-comparison pipeline on the equilateral-triangle velocity section, producing a list of verified periodic orbits with novelty verdicts.

## Scope

**In:**
- Section definition and closure verification (S₃ symmetry + L = 0 check).
- Basin map on 256×256 (u_x, u_y) grid via forward integration with the existing Yoshida-6 or heyoka pipeline. Label set: {periodic, body-1-escapes, body-2-escapes, body-3-escapes} (4 classes, with S₃-induced relabelling tracked).
- GPU training of classifier (reuse `S3BodyNet` architecture from the 3BP basin project; fine-tune from scratch or from existing checkpoint — see Methodology).
- Label-disagreement map from classifier uncertainty.
- Top-K (K = 20) candidate ICs from the low-LD regions.
- Newton refinement at float64 of each candidate.
- heyoka HP Taylor verification at tol = 10⁻¹⁸ per candidate → residual < 10⁻³⁰.
- Monodromy + stability classification per surviving candidate.
- Syzygy-word topology computation per candidate + match against Hristov 2024 (24,582) and Hristov 2025 stable (971) catalogs.
- Novelty verdict per surviving candidate.
- Figure panel: basin map, LD map, candidate scatter, orbit trajectories.
- `RESULT.md` + LaTeX section fragment.

**Out:**
- No full 50-digit precision refinement (10⁻³⁰ residual is sufficient for publication and keeps this thread bounded).
- No second ML pipeline architecture (reuse S3BodyNet; ablations of classifier choice are out of scope).
- No basin resolution sweep (single 256×256 grid).
- No iterative multi-pass discovery (single pass; if nothing found, document and report negative result).

## Data inputs

| Artefact | Path | Status |
|---|---|---|
| Existing S3BodyNet / MLP classifier code | `experiments/orbit_discovery/...` (to locate during plan) | present |
| Yoshida-6 integrator | `experiments/orbit_discovery/...` | present |
| heyoka HP integrator + Newton refiner | `experiments/orbit_verification/06_high_precision/run_hp_verification.py` | present, reusable |
| Monodromy pipeline | `experiments/orbit_verification/11_monodromy/compute_monodromy.py` | present, reusable |
| Syzygy-word computation | `experiments/orbit_verification/09_syzygy_words/` | present, reusable |
| Hristov 2024 catalog | `experiments/orbit_verification/00_catalogs/hristov_2024_sol_80.txt` | present |
| Hristov 2025 stable catalog | `experiments/orbit_verification/00_catalogs/hristov_2025_stable.txt` | present |
| Topology-sweep catalog (HP-Taylor-integrated Hristov reference words) | `experiments/orbit_verification/14_topology_sweep_hp_full/` | present, reusable |

## Methodology

### Step 1 — Section definition and closure sanity

Define the equilateral-triangle section by:

r₁ = R_∆ (cos(0),     sin(0))    = R_∆ ( 1,  0)
r₂ = R_∆ (cos(2π/3), sin(2π/3))  = R_∆ (−1/2,  √3/2)
r₃ = R_∆ (cos(4π/3), sin(4π/3))  = R_∆ (−1/2, −√3/2)

with R_∆ set so that pairwise distance is 1 (i.e., R_∆ = 1/√3). Center of mass at origin.

S₃-invariant velocity field:
v_k = R(2πk/3) · (u_x, u_y),  k = 0, 1, 2.

One checks Σ v_k = 0 identically (momentum), and L_z = Σ (r_k × v_k) = 0 identically (since each (r_k, v_k) pair has the same angle between them — S₃-symmetric data).

**Closure verification**: a quick Poincaré-return test at (u_x, u_y) = (1, 0.5) and (0.3, 0.7) for T in [10, 30] confirms the S₃ symmetry of the flow — body 1 returns to where body 2 started at t = T/3 (permutation closure, if it were a 3-periodic point). Sanity, not production.

### Step 2 — Basin map

Grid: 256 × 256 on (u_x, u_y) ∈ [−2, 2]² (or data-driven bounds — confirm by computing V(0), T_kin(0), E and ensuring we have bound orbits E < 0 in the interior).

Per grid point:
1. Integrate with Yoshida-6 for N = 10⁴ steps of dt tuned to energy conservation.
2. Label: periodic (returns to within ε in position and velocity within T_max), or "body-k-escapes" for k = 1, 2, 3 based on which body separates first.

Compute: ~1 hour on A100 or ~6 hours on CPU with vectorised jax/heyoka. Prefer GPU with jax-lax.

Output: `basin_map.npz` — labels array, energies array, per-point diagnostics.

### Step 3 — Classifier training

Train on 80% of labelled grid, validate on 20%. Use S3BodyNet architecture if the existing weights initialize well; otherwise fresh MLP.

Target: ≥ 95% validation accuracy on 4-class prediction. 50 epochs with cosine decay; early-stop on validation loss plateau.

Output: `classifier.pt` + `training_curves.pdf`.

### Step 4 — Label-disagreement field

For each grid point (including unlabelled fine-grid points), compute entropy of the classifier's softmax output. Label-disagreement = classifier_entropy between the two top classes (tracks the decision-boundary ambiguity).

Produce a 512×512 fine-grid LD field (forward-pass only, cheap).

Output: `ld_field.npz`, `figures/basin_map.pdf`, `figures/ld_field.pdf`.

### Step 5 — Candidate selection

Top-K = 20 local maxima of LD away from class-boundary artifacts. De-duplicate using the expected Euler-section D₂ analogue: the equilateral section has S₃-rotation-by-2π/3 symmetry, so identify candidates equivalent under cyclic relabelling.

Output: `candidates_ICs.json`.

### Step 6 — Newton refinement at float64

For each candidate IC (u_x, u_y):

1. Numerical period estimate from basin diagnostic (first Poincaré return).
2. Newton–Raphson on the section-return map: zero the map (u_x, u_y, T) → (u_x', u_y', P(u_x, u_y) − 0).
3. Gauss–Newton with step restriction; 20 iterations max.

Survivors: candidates with Newton residual < 10⁻⁹.

Output: `refined_candidates.json`.

### Step 7 — HP Taylor verification

For each surviving candidate, run heyoka adaptive Taylor with fp_type = longdouble at tol = 10⁻¹⁸ for one period. Verify:

- Closure ||x(T) − x(0)|| < 10⁻³⁰ after a final Newton step at longdouble precision.
- Energy conservation ≤ 10⁻²⁵ across the orbit.

Survivors: HP-verified.

Output: `hp_verified_candidates.json`.

### Step 8 — Monodromy + stability

For each HP-verified candidate, run the variational-monodromy pipeline (same as for A, B earlier). Record eigenvalues and classify.

Output: `monodromy_equilateral.json`.

### Step 9 — Syzygy word topology + catalog comparison

For each HP-verified candidate:

1. Compute the syzygy sequence by forward-integrating (all three pairwise collinearity events).
2. Reduce to a free-group word.
3. Compare word lengths and words against Hristov 2024 and Hristov 2025 (word-length sweep, then full-word match on ties).

Verdict per candidate: **NOVEL** (no match) / **REDISCOVERY of Hristov entry #XXXX** / **UNCERTAIN**.

Output: `topology_match_equilateral.json`.

### Step 10 — Figures + LaTeX fragment

`figures/equilateral_basin.pdf`: 4-panel basin + LD + candidate overlay + sample orbit trajectories (2-3 most interesting).

`section_text.tex`: methods, results, verdict.

## Outputs (artefacts)

All figures and animations follow the shared aesthetics spec (`docs/superpowers/specs/2026-04-21-figure-aesthetics-design.md`). Every figure script imports from `experiments/ravishankar_followup/00_style/`.

```
experiments/ravishankar_followup/03_equilateral/
├── section_definition.py          # Step 1 verification
├── run_basin.py                   # Step 2
├── train_classifier.py            # Step 3
├── compute_ld.py                  # Step 4
├── select_candidates.py           # Step 5
├── newton_refine.py               # Step 6
├── hp_verify.py                   # Step 7
├── compute_monodromy.py           # Step 8 (adapted from 11_monodromy)
├── topology_match.py              # Step 9
├── make_figures.py
├── basin_map.npz
├── ld_field.npz
├── classifier.pt
├── candidates_ICs.json
├── refined_candidates.json
├── hp_verified_candidates.json
├── monodromy_equilateral.json
├── topology_match_equilateral.json
├── figures/
│   ├── equilateral_basin_paper.pdf
│   ├── equilateral_basin_blog.png
│   ├── candidates_ld_scan_paper.pdf
│   ├── candidates_ld_scan_blog.png
│   ├── orbits_panel_equilateral_paper.pdf
│   ├── orbits_panel_equilateral_blog.png
│   ├── section_comparison_paper.pdf
│   └── section_comparison_blog.png
├── animations/
│   ├── basin_reveal.gif / .mp4
│   ├── ld_growth.gif / .mp4
│   ├── equilateral_vs_euler.gif / .mp4
│   └── new_orbit_<N>.gif / .mp4    # one per surviving HP-verified orbit
├── training_curves.pdf
├── RESULT.md
└── section_text.tex
```

**Animations** (per aesthetics spec, Thread-3 inventory):
- `basin_reveal.gif` — 10-second column-by-column painting of the 256² basin, deck-friendly.
- `ld_growth.gif` — 8-second snapshot loop of LD field at epochs 1/10/25/40/50 of classifier training.
- `equilateral_vs_euler.gif` — 8-second morph from Euler configuration to equilateral configuration, illustrating section difference.
- `new_orbit_<N>.gif` — one comet-tail loop per surviving HP-verified orbit; these become the "new orbit" assets for the paper and the deck.

## Acceptance criteria

1. Basin map: 256×256 complete, ≥ 95% of points labelled (non-degenerate integration).
2. Classifier: validation accuracy ≥ 95% on 4-class task.
3. LD field resolves visible "ridges" distinct from class-boundary artifacts (visual inspection).
4. ≥ 10 candidates survive Newton refinement at residual < 10⁻⁹.
5. ≥ 1 candidate HP-verified at < 10⁻³⁰ closure residual.
6. All HP-verified candidates classified (stable / hyperbolic / mixed) via monodromy.
7. Every HP-verified candidate has an explicit NOVEL / REDISCOVERY / UNCERTAIN verdict backed by syzygy-word comparison to Hristov 2024 + 2025.
8. `section_text.tex` reports the full pipeline outcome with honest treatment of negative results ("0 novel orbits found" is a valid scientific result and should be reported cleanly if that's what happens).

## Risks / open questions

- **Negative result is possible.** The equilateral section may be mostly chaotic with no long-period periodic orbits in the explored bound-energy region; this is fine but must be reported honestly.
- **Basin may have very few periodic points** compared to Euler, because the Poincaré section is fundamentally different (S₃ symmetry is generic — the original figure-8 sits on it — but the section we define might miss the long-period figure-8 family if it's not at simple (u_x, u_y)).
- **The figure-8 orbit lives on this section**. Expected sanity: if our (u_x, u_y) grid covers E ≈ 0.876 (figure-8 energy) we *should* rediscover the figure-8. If not, that's a pipeline bug, not a scientific result.
- **Classifier bootstrap**: if S3BodyNet weights don't initialize, a fresh MLP will still get ≥ 95% on 4 visually-distinct escape classes + periodic region; this is the fallback.
- **De-duplication by S₃ symmetry**: the cyclic 2π/3 rotation relabelling needs careful bookkeeping — use canonical form (smallest (u_x, u_y) under the cyclic action) before de-duplicating.

## Dependencies

- Reuses the `S3BodyNet` / MLP architecture and Yoshida-6 integrator code from `experiments/orbit_discovery/` (Phase A).
- Reuses heyoka HP + Newton refiner from `experiments/orbit_verification/06_high_precision/`.
- Reuses variational-monodromy from `experiments/orbit_verification/11_monodromy/`.
- Reuses syzygy-word computation from `experiments/orbit_verification/09_syzygy_words/`.
- Independent of Threads 1 and 2.

## Estimated wall time

~8 hours on A100: ~1 h basin, ~30 min classifier, ~15 min LD, ~10 min candidates, ~30 min Newton, ~1 h HP verify (20 candidates), ~30 min monodromy, ~2 h topology + matching, ~1.5 h figures + LaTeX.

Can run in parallel with Threads 1 and 2 if we spin up a second process.
