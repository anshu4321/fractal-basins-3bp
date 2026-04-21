# Thread 3 — reconnaissance notes

## Reusable modules located on pod

### Basin/classifier (from experiments/orbit_discovery/)

- `velocity_space_search.py` — full end-to-end pipeline: 2D basin map generation,
  classifier training, label-disagreement (LD) entropy field, candidate selection,
  Gauss-Newton refinement on (v1, v2, T). Primary reuse target for the equilateral
  section analogue.
- `extract_basin_map.py` / `extract_basin_map_cpu.py` — GPU/CPU basin map extraction
  via Yoshida-6 + vmap over a 2D grid; provides the loop structure and basin-label
  encoding to adapt for the equilateral IC section.
- `train_model.py` — trains `BasinMLP` (from `mega3bp.ml`) on NPZ basin data;
  provides the training loop, optax optimiser, equinox checkpoint save/load.
- `baselines.py` — reference baselines for classifier comparison and label-disagreement
  scoring; contains `label_disagreement` utility used by velocity-space search.
- `pipeline.py` — unified `verify_candidate()` / `search_and_verify()` wrappers
  using `mega3bp.refine`, `mega3bp.diagnostics`, `mega3bp.shape_sphere`, and
  `mega3bp.dynamics`; reusable for the Newton refinement step.
- `run_group4.py`, `run_group5.py`, `run_single_seed_method.py` — orchestration
  scripts with grid-search + basin-disagreement patterns; reference for batch runs.
- `tier1_orchestrator.py` / `tier1_aggregate.py` — two-stage orchestrate-then-aggregate
  pattern; useful for parallelising the equilateral grid scan.
- `visualize_basin.py` / `visualize.py` — basin-map and shape-sphere plotting helpers.

Key mega3bp package modules used by all of the above:
- `mega3bp.integrators.yoshida6_step` — 6th-order Yoshida stepper (float64, JAX vmap-safe)
- `mega3bp.ml.BasinMLP`, `mega3bp.ml.N_CLASSES`, `mega3bp.ml.loss_fn`, `mega3bp.ml.predict`
- `mega3bp.refine.refine_from_shape_point` — Gauss-Newton Newton refinement
- `mega3bp.shape_sphere.shape_to_config`, `mega3bp.shape_sphere.pairwise_distances`
- `mega3bp.dynamics.total_energy`
- `mega3bp.diagnostics` — conservation, min-distance, multi-period stability checks

### HP verification

Path: `experiments/orbit_verification/06_high_precision/run_hp_verification.py`
Schema: mpmath-based Taylor integrator (30-decimal precision), Newton-Raphson
refinement, residual threshold 1e-15 (strong) / 1e-20 (publication-grade). Takes
orbit ICs in collinear shape-sphere convention; needs IC convention adapter for
equilateral section inputs.

### Monodromy

Path: `experiments/orbit_verification/11_monodromy/compute_monodromy.py`
Reusable function: heyoka variational ODE system, 12-dim Hamiltonian 3BP lifted
to 156-dim (12 state + 144 sensitivities), reads HP results JSON for ICs. Hardcodes
path to `06_high_precision/hp_heyoka_newton_results.json`; needs path parameter
injection for equilateral results. Eigenvalue classification: |lambda|_max > 1+1e-4
=> hyperbolic, else elliptic.

### Syzygy words

Path: `experiments/orbit_verification/09_syzygy_words/`
Files: `compute_words.py`, `RESULT.md`, `run.log`, `status.json`, `verdict.json`, `words.json`
`compute_words.py` — Yoshida-6 integration + signed-area zero-crossing detection,
body-between labelling, cyclic/time-reversal equivalence; relabels internal body
indices (0,1,2) to Hristov convention (1,3,2). Reusable as-is; needs equilateral
ICs passed in.

## What we'll build fresh vs. reuse

| Component | Reuse | Build fresh | Notes |
|---|---|---|---|
| Section definition | — | build | equilateral IC geometry: bodies at vertices of equilateral triangle, velocities on 2D section |
| EOM | reuse `mega3bp.integrators.yoshida6_step` | — | already float64, vmap-safe |
| Basin map | adapt `extract_basin_map.py` + `velocity_space_search.py` | — | swap collinear IC encoder for equilateral encoder; 4 basin classes unchanged |
| Classifier | adapt `train_model.py` (`BasinMLP` from `mega3bp.ml`) | — | 7-feature MLP (PyTorch); note: current impl uses equinox/JAX, not PyTorch — verify feature count |
| LD field | adapt `velocity_space_search.py` entropy section | — | entropy of classifier softmax already implemented |
| Candidate selection | adapt `velocity_space_search.py` local-max + dedup | — | need S3 symmetry deduplication for equilateral section (collinear version uses simpler sym) |
| Newton refinement | adapt `pipeline.py` + `mega3bp.refine.refine_from_shape_point` | — | IC convention adapter needed |
| HP Taylor verify | reuse `run_hp_verification.py` | — | different IC convention; wrap with adapter |
| Monodromy | reuse `compute_monodromy.py` | — | path injection needed; IC convention adapter |
| Syzygy word match | reuse `compute_words.py` | — | pass equilateral ICs directly |

## NOTES / potential gaps

- `compute_monodromy.py` hardcodes the HP results JSON path; will need a path
  argument or a small wrapper to point it at equilateral HP output.
- Current `BasinMLP` in `mega3bp.ml` uses equinox (JAX), not PyTorch — the plan
  references a 7-feature PyTorch MLP. Verify feature engineering requirements
  before choosing equinox vs. PyTorch implementation.
- No `11_compute_fairness/` directory found on pod (listed in git status as `??`
  locally but absent remotely) — not needed for this thread.
- `ravishankar_followup/_01_census/` and `_02_quantization/` already exist;
  `_03_equilateral/` is the new scaffold.
