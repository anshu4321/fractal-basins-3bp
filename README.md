# Fractal Basins of the Three-Body Problem

**A benchmark, an active-learning finding, and an open scaling-law question.**

This repository contains the code, experiments, and paper for a study of machine learning classification on the fractal escape basins of the equal-mass planar three-body problem. Targeting the NeurIPS ML4PS 2026 workshop.

**Author:** Aishwarya Das, [Dirac Labs Inc.](https://diraclabs.com)

---

## The Problem

Drop three equal masses onto a plane with some initial configuration. Wait. Eventually one escapes, leaving a binary behind. *Which one escapes?*

The answer depends on the initial conditions with fractal sensitivity. The basin boundaries, the set of initial conditions where a tiny perturbation changes the outcome, are fractal curves that satisfy the Wada property: every boundary point simultaneously borders all three escape basins.

<p align="center">
  <img src="figures/basin_mollweide.png" width="700"/>
</p>
<p align="center"><i>Escape basins on the Montgomery shape sphere (Mollweide projection, ~10<sup>6</sup> initial conditions). Four classes: bound (blue), body 1 escapes (orange), body 2 escapes (green), body 3 escapes (red). The 120-degree S<sub>3</sub> symmetry is visible.</i></p>

---

## The Shape Sphere

The configuration space of three equal-mass bodies, modulo translation, rotation, and scaling, is the Montgomery shape sphere S<sup>2</sup>. Binary-collision singularities sit 120 degrees apart on the equator; the Lagrange equilateral configurations occupy the poles.

<p align="center">
  <img src="figures/shape_sphere_3d.png" width="350"/>
  <img src="figures/shape_sphere_configs.png" width="450"/>
</p>
<p align="center"><i>Left: The shape sphere with collision points (C<sub>12</sub>, C<sub>13</sub>, C<sub>23</sub>) and Lagrange points (L<sub>+</sub>, L<sub>-</sub>). Right: Physical triangle configurations at sampled shape-sphere coordinates.</i></p>

---

## Key Results

### 1. Fractal Basin Benchmark

- **2D at-rest dataset:** 953,143 clean labeled trajectories on S<sup>2</sup> (zero initial momenta)
- **6D phase-space dataset:** 1,010,567 clean trajectories (non-zero Jacobi momenta)
- **Uncertainty exponents:** alpha<sub>2D</sub> = 0.259 +/- 0.019, alpha<sub>6D</sub> = 0.145 +/- 0.004
- **Wada property:** Confirmed via merging test. Basin entropy S<sub>b</sub> = 0.222
- **Integrator:** Yoshida 6th-order symplectic, energy conservation < 10<sup>-12</sup>

<p align="center">
  <img src="figures/fractal_mollweide.png" width="500"/>
</p>
<p align="center"><i>Basin boundary points on the shape sphere. All three escape-basin boundaries overlap everywhere (Wada property). Box-counting dimension D = 1.59.</i></p>

<p align="center">
  <img src="figures/02_uncertainty_fit.png" width="600"/>
</p>
<p align="center"><i>MGOY uncertainty exponent measurement. Left: 2D (alpha = 0.259, D<sub>u</sub> = 1.74). Right: 6D (alpha = 0.145, D<sub>u</sub> = 5.86). Both R<sup>2</sup> > 0.99.</i></p>

### 2. Active Learning Fails in 6D

Boundary-targeted active learning improves macro-F1 by **+3.7%** in 2D but **degrades** it by -0.7% in 6D. The reason: in 6D, the uncertainty dimension D<sub>u</sub> = 5.86 means the fractal boundary nearly fills the entire phase space. "Targeted" sampling degenerates into noisy uniform sampling.

<p align="center">
  <img src="paper/figures/F2_al_2d_6d.png" width="600"/>
</p>
<p align="center"><i>Active learning vs uniform sampling. In 2D, AL wins. In 6D, uniform wins at every budget.</i></p>

### 3. The GOY Scaling-Law Question

The 1983 Grebogi-Ott-Yorke bound predicts classifier error scales as N<sup>-alpha/d</sup>. In 2D: predicted slope = -0.130.

Under **architecture-specific training**, measured slopes match the prediction to 6-9%:
- MLP: -0.131 +/- 0.005
- S3BodyNet: -0.141 +/- 0.009

Under a **pre-registered matched-compute protocol**, slopes are flat:
- MLP: -0.002 +/- 0.003
- S3BodyNet: +0.014 +/- 0.002

<p align="center">
  <img src="paper/figures/F3_scaling_law.png" width="650"/>
</p>
<p align="center"><i>Left: Architecture-specific training matches the GOY band. Right: Matched-compute protocol produces flat slopes.</i></p>

Training diagnostics reveal the cause: the fixed gradient-update budget gives small-N cells ~6,000 epochs (overfitting) and large-N cells ~200 epochs (undertraining). This differential convergence flattens the measured slope.

<p align="center">
  <img src="paper/figures/F4_diagnostic_curves.png" width="650"/>
</p>
<p align="center"><i>Training diagnostics for four cells. Blue: training loss. Orange: validation F1. Small-N overfits, large-N undertrained.</i></p>

**Whether classifier scaling matches the GOY bound under properly matched training remains an open question.**

---

## Repository Structure

```
mega3bp/                        # Core Python package
  dynamics.py                   # Newtonian gravitational dynamics
  integrators.py                # Yoshida 6th-order symplectic integrator
  shape_sphere.py               # Montgomery shape sphere transforms
  escape.py                     # Standish escape criterion
  batch.py                      # Batched GPU integration (JAX)
  ml.py                         # BasinMLP, loss functions, evaluation
  equivariant.py                # S3BodyNet, S3ReynoldsNet, group actions
  orbits.py                     # Periodic orbit search

experiments/
  01_dataset_regen/             # Dataset generation
  02_uncertainty_exponent/      # MGOY alpha measurement
  03_scaling_law/               # Phase A scaling law (MLP, 2D + 6D)
  04_wada_basin_entropy/        # Wada merging test, basin entropy
  05_ablation/                  # Four-way ablation at N=300k
  06_active_learning/           # AL vs uniform, 2D and 6D
  12_augmentation_diagnosis/    # S3 augmentation bug investigation
  13_scaling_2d_s3bodynet/      # Phase C: 2-architecture scaling law
  14_6d_multifractal/           # 6D multi-fractal analysis
  15_ablation_n1m/              # Ablation at N=1M
  16_convergence_check/         # Training convergence validation
  17_scaling_2d_matched_protocol/ # Phase D: matched-protocol attempts
  21_matched_protocol_final/    # Phase E: pre-registered matched test
    lr_tuning.json              #   Peak LR selection
    results.json                #   All 40 runs
    fit.json                    #   Slope fits + pass criteria
    VERDICT.md                  #   PASS/FAIL verdict
    PHASE_E_DIAGNOSTIC.md       #   Root cause analysis
    SUMMARY.md                  #   Complete Phase E summary

results/                        # Result artifacts (JSON)
figures/                        # All generated figures
paper/                          # LaTeX paper
  main.tex                      #   Complete manuscript
  refs.bib                      #   Bibliography (20 citations)
  main.pdf                      #   Compiled PDF
  figures/                      #   Paper figures

BRANCH_DECISION.md              # Phase E outcome + paper branch
PHASE_E_START.md                # Phase E clock
GATE_DECISION_v2.md             # Phase C gate decision
PHASE_C_REPORT.md               # Comprehensive Phase C review
```

## Tech Stack

- **JAX + Equinox** for GPU-accelerated integration and neural network training
- **Optax** for optimizer implementations (AdamW, cosine schedules)
- **NumPy/SciPy** for data processing and statistical analysis
- **Matplotlib** for all figures
- **Yoshida-6 symplectic integrator** for trajectory computation
- **H100 GPU** via RunPod for experiment execution

## Architectures

**BasinMLP:** 6-layer feed-forward network (384 hidden, GELU activation). Takes 3D shape-sphere coordinates as input. Simple, fast, strong baseline.

**S3BodyNet:** Permutation-equivariant architecture with shared per-body feature extraction. Computes 8 physically motivated invariants per body (distances, kinetic energies, angular momentum, radial velocities) plus 4 global invariants. Weight sharing across bodies enforces S<sub>3</sub> equivariance by construction. Inspired by DeepSets.

## Reproducing

```bash
# Install dependencies
pip install jax[cuda12] equinox optax numpy scipy matplotlib

# Generate the 2D dataset (~2 min on GPU)
python experiments/01_dataset_regen/run.py

# Run the scaling law experiment
python experiments/13_scaling_2d_s3bodynet/run.py

# Run the Phase E matched-compute test
python experiments/21_matched_protocol_final/lr_tuning.py
python experiments/21_matched_protocol_final/run_sweep.py
python experiments/21_matched_protocol_final/fit_and_verdict.py

# Compile the paper
cd paper && ~/bin/tectonic main.tex
```

## Citation

If you use this benchmark or code, please cite:

```bibtex
@inproceedings{das2026fractalbasins,
  title     = {A Benchmark and an Open Problem: Fractal-Basin Classification
               for the Planar Three-Body Problem},
  author    = {Das, Aishwarya},
  booktitle = {NeurIPS ML4PS Workshop},
  year      = {2026},
}
```

## License

MIT
