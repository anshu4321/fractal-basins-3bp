# Learning the Basin Structure of the Planar Three-Body Problem

## Paper outline (draft)

### Abstract (~150 words)
We construct the first ML-based basin partition of the equal-mass, zero-angular-momentum planar three-body problem on Montgomery's shape sphere. Using a Yoshida 6th-order symplectic integrator validated to |ΔE/E| < 10⁻¹³ on the Chenciner-Montgomery figure-eight orbit, we generate 10⁶ labeled trajectories on an H100 GPU at 467 M traj-steps/sec. The basin boundaries are fractal with box-counting dimension D = 1.592 ± 0.003, confirmed to respect the S₃ permutation symmetry at 0.1% precision. A baseline MLP with Fourier features plateaus at 91.3% classification accuracy; SIREN (sinusoidal activations) achieves the same — we show this plateau is an information bottleneck set by the training sample density, not the network architecture. Boundary-adaptive sampling (concentrating new samples near the D ≈ 1.6 fractal boundary) breaks the plateau, pushing accuracy to 93.8% and doubling per-class escape F1. Our results demonstrate that active learning on fractal basin boundaries is more cost-effective than uniform sampling for chaotic-scattering classification.

### 1. Introduction (~1 page)
- The three-body problem and its basin structure
- Prior work: Agekian-Anosova maps, Šuvakov-Dmitrašinović orbits, Breen et al. 2020 (trajectory prediction, not basin partition)
- Montgomery's shape sphere as the natural parameter space
- Our contribution: first basin MAP (not just trajectory prediction), fractal dimension measurement, information-bottleneck finding, active learning remedy

### 2. Setup (~1.5 pages)
#### 2.1 The planar equal-mass 3BP
- Hamiltonian, natural units (G=m=1), COM frame
- Zero angular momentum constraint

#### 2.2 Montgomery shape sphere
- Mass-weighted Jacobi coordinates
- Hopf map to S²
- Special points: three binary collisions (equator, 120° apart), two Lagrange equilateral configs (poles)

#### 2.3 Numerical integration
- Yoshida 6th-order symplectic composition of Störmer-Verlet
- Figure-eight validation: |ΔE/E| < 1.12 × 10⁻¹³ over 100 periods
- Escape criterion: Standish-style (pair distance + binary identification + radial velocity)
- Close-encounter flagging

### 3. Dataset and basin map (~1 page)
- 1,048,576 ICs sampled uniformly on S² with natural (Fubini-Study) measure
- Zero initial momentum (at rest)
- Integration to t_max = 500, h = 0.01, 50,000 steps per trajectory
- H100 GPU, JAX + diffrax, fp64, 467 M traj-steps/sec, 112 seconds total
- Outcome distribution: 93.5% bound, 2.0% each escape class (S₃ to 0.03%)
- 0.45% ambiguous (well below KS-regularization trigger)
- Basin map visualization: Mollweide projection, 3D scatter, stereographic views

### 4. Fractal dimension of basin boundaries (~1 page)
- Boundary detection via k=12 nearest-neighbor label disagreement
- 25.1% of sample points are boundary points
- Box-counting on 3D voxel grid, ε from 0.005 to 0.5
- D_all = 1.592, D_body_i ≈ 1.435 (spread 0.003 — S₃ symmetry)
- D_all > D_individual: triple-junction densification
- Comparison with chaotic-scattering literature (Bleher-Ott-Grebogi)

### 5. Baseline classifiers and the information plateau (~1.5 pages)
#### 5.1 MLP with Fourier features
- Architecture: 3 → Fourier(k=8, 48-dim) → 384×6 GELU → 5 outputs
- Class-weighted cross-entropy (sqrt inverse frequency)
- Best-checkpoint selection: acc 0.913, macro F1 0.415
- Error map: misclassifications concentrate on basin boundaries (Fig. X)

#### 5.2 SIREN baseline
- sin activations, ω₀ = 30, same capacity
- acc 0.906, macro F1 0.419 — ties MLP within noise
- Unweighted SIREN converges to trivial always-bound predictor (0.940, F1=0.24)

#### 5.3 The plateau is fundamental
- Both architectures sit on the same (accuracy, macro-F1) tradeoff frontier
- Nearest-neighbor spacing at 1M points: √(4π/10⁶) ≈ 0.0035 rad
- Basin boundaries finer than this are not in the training set
- D = 1.59 boundary set → ~25% of test points are within one NN spacing of a boundary → ~9% error floor

### 6. Breaking the plateau with adaptive sampling (~1 page)
- Detect 262k boundary points from the 1M dataset
- Generate 524k perturbations (σ = 0.005 rad, tangent-plane Gaussian)
- Integrate, merge → 1.57M enriched dataset
- Retrain same MLP on enriched data, evaluate on ORIGINAL held-out test set
- Results: acc 0.938 (+0.025), macro F1 0.585 (+0.170), escape F1 doubled
- [Learning curve figure: accuracy and F1 vs N for uniform vs adaptive]
- Active learning is more cost-effective than uniform scaling for fractal basins

### 7. Discussion (~0.5 page)
- Why R² on escape time fails: shape-sphere coordinates don't encode chaotic dynamics
- Implications for periodic orbit search (future work)
- Scaling to non-zero momentum (6D parameter space)
- Connection to spectral bias literature

### 8. Conclusion (~0.3 page)

### References
- Montgomery 1998
- Chenciner & Montgomery 2000
- Šuvakov & Dmitrašinović 2013
- Breen et al. 2020
- Sitzmann et al. 2020 (SIREN)
- Bleher, Ott, Grebogi (fractal basin boundaries)
- Agekian & Anosova (classical maps)
- Liao et al. (periodic orbit catalogs)

### Figures
1. Figure-eight energy conservation (already have)
2. Mollweide basin map (already have)
3. Fractal scaling plot (already have)
4. Boundary points on shape sphere (already have)
5. MLP error map showing boundary-localized failures (already have)
6. SIREN vs MLP tradeoff scatter (already have)
7. Learning curve: uniform vs adaptive (generating now)
8. Phase 3 enriched MLP errors (already have)
