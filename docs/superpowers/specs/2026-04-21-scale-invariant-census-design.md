# Scale-invariant census across Hristov 2024 topologies

**Date:** 2026-04-21
**Thread:** 1 of 3 in the Ravishankar follow-up paper (one combined paper, see co-specs for Threads 2 and 3)
**Target audience:** v2 / follow-up paper Section 2 ("Census")

---

## Context

The conference paper (`paper2/conference_paper.tex`, submission-ready) reports four verified periodic orbits A, B, C, D on the Euler velocity section and establishes that A is not a topological match anywhere in the 24,582-entry Hristov 2024 catalog (`experiments/orbit_verification/14_topology_sweep_hp_full/`). The Ravishankar note (`experiments/orbit_verification/15_scaling_and_invariants/NOTE_FOR_RAVISHANKAR.md`) develops the scaling symmetry (α, α^{3/2}) for the equal-mass 1/r problem, proves L = 0 identically on the Euler section, and reports 50-digit values of E and T⋆ = T·|E|^{3/2} for all four orbits.

This thread positions our four orbits against the entire Hristov 2024 topological landscape in the scale-invariant plane (E, T⋆) — the natural invariants of the L = 0 sector — colored by syzygy-word class. It also produces the α-family visualization Ravishankar asked for.

## Goal (one sentence)

Compute E and T⋆ for all 24,582 Hristov 2024 entries, overlay our four orbits in (E, T⋆), color by syzygy-word length, and publish a density analysis along the T⋆ axis per topological class.

## Scope

**In:**
- Compute E for each Hristov 2024 entry analytically from the entry's IC.
- Compute T⋆ = T·|E|^{3/2} per entry using the catalog-provided T.
- Re-use the existing syzygy-word assignment (word length) from `14_topology_sweep_hp_full/topology_sweep_hp_summary.json` and `length_histogram.json`.
- Produce four artefacts: scatter plot, per-class T⋆ density histogram, α-family schematic, numerical summary tables.
- Overlay Hristov 2025 stable-orbit entries (971) as a distinct marker set for reference.
- Write a self-contained LaTeX section fragment for the follow-up paper.

**Out:**
- No re-integration of Hristov entries (we trust their catalog T and IC).
- No family assignment work (free-group word classification beyond length is out of scope for this thread; length is sufficient for coloring).
- No Li-Liao comparison here (different section; handled in the conference paper already).
- No computation of new orbits.

## Data inputs

All paths relative to repo root.

| Artefact | Path | Status |
|---|---|---|
| Hristov 2024 solutions (ICs + T) | `experiments/orbit_verification/00_catalogs/hristov_2024_sol_80.txt` | present |
| Hristov 2024 eigenvalues (for sanity checks) | `experiments/orbit_verification/00_catalogs/hristov_2024_eigen.txt` | present |
| Hristov 2025 stable ICs | `experiments/orbit_verification/00_catalogs/hristov_2025_stable.txt` | present |
| Syzygy-word lengths per entry | `experiments/orbit_verification/14_topology_sweep_hp_full/length_histogram.json` + `topology_sweep_hp_summary.json` | present |
| 50-digit E, T⋆ for A, B, C, D | `experiments/orbit_verification/15_scaling_and_invariants/invariants.json` | present |

**Convention note.** Hristov 2024 uses the free-fall section; our orbits use the Euler velocity section. The scaling invariants (E, T⋆) are section-agnostic — both catalogs and our orbits can be placed on the same (E, T⋆) plane, since α-scaling is a symmetry of the global EoM and the definition of E and T⋆ does not depend on which section the IC sits on.

## Methodology

### Step 1 — Parse and compute E per Hristov 2024 entry

For each of the 24,582 entries:

1. Load (r_i, v_i) from `hristov_2024_sol_80.txt` (mass convention m_i = 1, G = 1 — same as our orbits).
2. Compute E = Σ_i (1/2)|v_i|² − Σ_{i<j} 1/|r_i − r_j|. Verify E < 0 (bound orbit).
3. Record T⋆ = T · |E|^{3/2} using the catalog-provided T.
4. Cross-check: test that rescaling all positions by α and velocities by α^{-1/2} leaves T⋆ invariant (spot-check on 100 random entries).

Output: `hristov_2024_invariants.json` — list of dicts `{entry_id, E, T⋆, word_length}` for all 24,582 entries.

Compute budget: ~10 seconds on CPU (analytic E, no integration).

### Step 2 — Scatter plot

Produce `figures/census_scatter.pdf` (paper) and `_blog.png` (web, for deck).

- x-axis: log10(T⋆), range to be data-driven
- y-axis: E (negative; flip if clearer)
- Hristov 2024 points: small dots, colored by word-length class (continuous colormap or discrete bins — prefer discrete bins at length 20, 30, 40, 50, 60, 70, 80+ for legibility)
- Hristov 2025 stable points: open circle markers, distinct edge color
- Our A, B, C, D: large distinct markers with labels; A and B in the upper-left (E ≈ −1.5), C nearby, D outlier at E ≈ −0.94
- Legend: the 7 length bins + "stable catalog" + "this work"

### Step 3 — Per-class density along T⋆

For each discrete word-length bin, plot the 1-D density f(T⋆) (kernel density or normalised histogram).

`figures/census_density.pdf` — overlaid KDEs, one per length-bin. Annotate T⋆(A) = 36.94, T⋆(B) = 64.65, T⋆(C) = 64.66, T⋆(D) = 73.81.

Reported numbers:
- Density at T⋆(A) per class (with A's word length highlighted) — substantiates the novelty claim *inside* its own topology class.
- Same for B, C, D.

### Step 4 — α-family schematic for A, B, C, D

Ravishankar's specific ask: visualise the 1-parameter scaling family per orbit.

`figures/scaling_families.pdf` — a panel of 4 (2×2), one per orbit. Each panel shows:
- The orbit's trajectory in configuration space at α = 1 (original).
- Two additional scaled copies at α = 0.5 and α = 2 (same topology, different linear size).
- Annotate (E(α), T(α), T⋆) — T⋆ is α-independent.

This is purely analytic (rescaling), no re-integration.

### Step 5 — Numerical tables

`RESULT.md` in thread directory — produced automatically by the script:

- Count of Hristov 2024 entries per word-length bin (sanity, should agree with `length_histogram.json`).
- Our 4 orbits' (E, T⋆, word_length) with 10-digit precision.
- Percentile of T⋆(A) within its word-length class, same for B, C, D.
- Density at T⋆(A) in the full catalog vs. in A's word-length class.
- The B–C T⋆ coincidence (relative Δ = 1.05×10⁻⁴): report whether any Hristov 2024 entries fall in that T⋆ interval at similar E.

## Outputs (artefacts)

```
experiments/ravishankar_followup/01_census/
├── run_census.py                 # produces everything
├── hristov_2024_invariants.json  # (E, T⋆, word_length) per entry
├── figures/
│   ├── census_scatter.pdf
│   ├── census_scatter_blog.png
│   ├── census_density.pdf
│   ├── census_density_blog.png
│   ├── scaling_families.pdf
│   └── scaling_families_blog.png
├── RESULT.md                     # numerical summary tables
└── section_text.tex              # LaTeX fragment for follow-up paper
```

## Acceptance criteria

1. `run_census.py` completes in under 60 s on CPU, no errors.
2. Sanity: the word-length distribution from our computed invariants matches `14_topology_sweep_hp_full/length_histogram.json` exactly in counts per bin.
3. Sanity: the four overlaid markers (A, B, C, D) land at the 50-digit (E, T⋆) values from `invariants.json` within plotting precision.
4. The scatter plot renders cleanly at a2paper 10pt without overlapping legend text.
5. α-family panel shows identical topology across scales per orbit (visual inspection).
6. `RESULT.md` reports: total entries = 24,582; per-class T⋆ percentiles for A, B, C, D; B/C T⋆-coincidence interval population.
7. `section_text.tex` compiles standalone when `\input{}`ed into the existing paper template.

## Risks / open questions

- **E computation convention mismatch**: Hristov 2024 uses rationalized units that might differ from our m=G=1 convention. Spot-check with 5 known entries from the catalog cross-referenced against published E values (if provided) to catch a constant-factor error before running over 24k.
- **Negative-E outliers**: if any entry computes with E ≥ 0, it's a bad IC (escape at t=0). Drop with a warning; expected zero but handle anyway.
- **Word-length coloring density**: if too many discrete bins, the legend becomes the figure. Collapse to 5 bins if needed (20-39, 40-49, 50-59, 60-69, 70+).
- **Hristov 2025 overlay cleanliness**: the stable-catalog plot symbol should not compete with our 4 orbits visually; use small open circles, not filled markers.

## Dependencies

- **None** from Thread 2 or Thread 3.
- **Consumed by**: the follow-up paper's Section 2 ("Census"), and optionally the conference paper's revision cycle if a reviewer asks for the scale-invariant positioning (we can lift this figure back into the conference paper if needed).

## Estimated wall time

~2 hours for Claude to execute end-to-end (parse, compute, plot, tabulate, write LaTeX fragment), assuming no catalog convention surprises.
