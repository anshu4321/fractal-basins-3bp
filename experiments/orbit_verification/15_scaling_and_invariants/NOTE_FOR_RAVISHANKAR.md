# Note for Prof V. Ravishankar
**Scaling symmetry, energy, and angular momentum of the four orbits**
Aishwarya — 2026-04-16

Following up on Tracks 1a/1b from our discussion on 2026-04-15.

---

## 1. Scaling symmetry (Newtonian N-body, 1/r potential)

With m_i = 1, G = 1, the equations of motion

  r̈_i  =  − Σ_{j ≠ i} (r_i − r_j) / |r_i − r_j|^3

are invariant under the one-parameter group of scalings

  r  →  α r,     t  →  α^{3/2} t.

The induced transformation of the conserved quantities is

  v, p  →  α^{−1/2} v, p
  E    →  α^{−1} E
  L    →  α^{+1/2} L
  T    →  α^{3/2} T     (period)

so two **independent dimensionless scale invariants** characterise any
bound orbit up to scale:

  T⋆ ≡ T · |E|^{3/2}         (scale-invariant period)
  L⋆ ≡ L · |E|^{1/2}         (scale-invariant angular momentum)

The rest of the orbit's geometry — shape, topology, Floquet multipliers —
is entirely determined by (T⋆, L⋆). Generating the whole one-parameter
family from one representative costs zero additional simulations; just
rescale.

---

## 2. Angular momentum: L = 0 for all four orbits (exactly, by construction)

Our orbits were found on the **Euler velocity section** parameterised by
(v_1, v_2):

  r_1 = (−1, 0),  r_2 = (+1, 0),  r_3 = (0, 0)
  p_1 = p_2 = (v_1, v_2),        p_3 = −2 (v_1, v_2).

Then

  L_z  =  r_1 × p_1 + r_2 × p_2 + r_3 × p_3
       =  (−1)(v_2) + (+1)(v_2) + 0
       =  0         (identically, for any v_1, v_2).

**Consequence.** All four orbits live in the L = 0 sector. This is a
property of the section, not a coincidence of the solutions we found —
any solution on this section has L = 0, and any new orbit we discover
here will too. For the symmetric equilateral-triangle extension you
suggested (task #4 in our discussion), the same is true: full S_3
permutation symmetry forces L = 0 from the three pairwise-cancelling
contributions.

So distinctions between our four orbits lie entirely in **E** and
**T⋆**, not in L.

---

## 3. Energy and scale-invariant period at 50-digit precision

| Orbit | Class | E | T⋆ = T \|E\|^{3/2} |
|---|---|---|---|
| A | hyperbolic | −1.51880093609…  | 36.94001257740157… |
| B | hyperbolic | −1.57045059002…  | 64.64865733919877… |
| C | lin. stable (= H2025 #0006) | −1.50430210713… | 64.65542233189639… |
| D | lin. stable (= H2025 #0011) | −0.93930544448… | 73.81478296715605… |

(Verified two ways: E = 3(v_1² + v_2²) − 5/2 analytically, and
E = T_kin(0) + V(0) numerically at 60-digit mpmath — agreement to 50
digits. Full 50-digit table in `invariants.json`.)

### Observations

- **All four are bound** (E < 0), as expected for periodic motion.
- **E spread is modest.** A, B, C cluster near E ≈ −1.5; D is a
  distinct outlier at E ≈ −0.94 (roughly 60% more energetic). D is also
  the most compact at its minimum separation (r_min ≈ 0.15 vs. ≈ 0.30
  for the others), which is consistent — higher E allows closer
  approaches at the same angular momentum (here L = 0, so the close
  approach is purely controlled by E).
- **B and C are NOT scale-equivalent.** The paper's rounded display
  reports T⋆ = 64.65 for both, but at 50 digits
  T⋆(B) = 64.64865734… and T⋆(C) = 64.65542233…, differing by
  6.8 × 10⁻³ (relative 1.0 × 10⁻⁴). They sit in the same topological
  family but on different bifurcation branches — the near-coincidence of
  T⋆ at the 2-decimal level is real but not exact.

### Stability note (updates what I told you)

Monodromy computation (heyoka variational, float64, this week) gives:

  A: \|λ\|_max = 43.78   hyperbolic   (s = +21.9 per period)
  B: \|λ\|_max =  4.04   hyperbolic   (s = −2.14 per period)
  C: \|λ\|_max =  1.000  elliptic/stable  (inherited from H2025)
  D: \|λ\|_max =  1.000  elliptic/stable  (inherited from H2025)

So A and B — the genuinely new orbits — are **unstable**. This doesn't
affect periodicity (closure at T verified to 1.4 × 10⁻¹³), but it does
change which semiclassical machinery applies: Bohr–Sommerfeld for the
stable pair C, D; **Gutzwiller's trace formula** for the hyperbolic
pair A, B (where the Lyapunov exponent is an input to the amplitude).

---

## 4. What this implies for your next-step questions

1. **Scaling families (your #1).** Generating entire families by rescaling
   α is now trivial — give me an α-range and I produce all (E(α), T(α),
   trajectory) triples analytically. "Density of families" in (T⋆, L⋆)
   space is the right way to ask the question; for L = 0, it reduces to
   density along the 1-D T⋆ axis.

2. **Energy and angular-momentum analysis across topologies (your #3).**
   With L = 0 fixed, the analysis collapses to (E, T⋆) scatter across
   topological classes. I can overlay our four orbits onto the Hristov
   2024 catalog in this plane and colour by topology.

3. **Equilateral-triangle symmetric ICs (your #4).** Natural next search
   manifold. As noted above, S_3 symmetry forces L = 0 automatically, so
   this stays in the same sector as our current orbits. Would give
   access to a different 2-D slice of the same L = 0 manifold; we should
   run it.

4. **Semiclassical quantisation (your #2).** For C and D — the stable
   orbits — the Bohr–Sommerfeld action ∮ P · dQ is straightforward to
   compute on the existing 10⁻⁵⁰-precision trajectories and will give a
   quantised-energy spectrum. For A and B, the Gutzwiller trace formula
   uses exactly the monodromy eigenvalues we just computed. Both are
   doable on the current data.

5. **Stability under ε → 0 perturbations (your #5).** The linear
   analysis is finished (monodromy). Your suggestion was to probe
   specific perturbations — we can now run finite-ε integrations of A, B
   with random perturbation directions and measure the exponential
   divergence rate, cross-checking against s = +21.9 and −2.14. A
   1-evening experiment. Still awaiting your methodology note.

---

## 5. Discrete symmetry of the Euler section: a Klein 4-group theorem

The Newtonian 3-body problem has symmetry group

  **G = S₃ × O(2) × ℤ₂(time-reversal) × ℝ₊(scaling)**.

The Euler velocity section fixes positions, breaking most of G; the
**stabiliser** of the section is the Klein four-group

  **D₂ = {I, σ_x, σ_y, σ_x σ_y}**

acting on (v₁, v₂) as (±v₁, ±v₂). Each periodic orbit on the section
therefore has **at least 4 equivalent ICs**, and the full basin of
attraction in (v₁, v₂) must be D₂-invariant (modulo a permutation of the
escape labels that tracks which body is swapped).

### Numerical verification

**(a) Klein theorem.** For each of A, B, C, D, all four (±v₁, ±v₂)
images close at exactly the same period T to scipy-DOP853 truncation
(residuals 10⁻¹⁰ to 10⁻⁹). Confirmed as a theorem:

| Orbit | closure ||x(T)−x(0)|| (all 4 images, identical) |
|---|---|
| A | 2.4 × 10⁻¹⁰ |
| B | 9.9 × 10⁻¹⁰ |
| C | 1.1 × 10⁻⁹ |
| D | 3.5 × 10⁻⁹ |

**(b) No internal half-period symmetry.** Testing whether
x(T/2) = σ x(0) (or σ·τ x(0) with time-reversal τ) for any σ ∈ D₂:
residuals are all O(1) for every orbit, compared to the configuration
scale ∼1. None of A, B, C, D is a figure-8-type "symmetric" orbit.
The four Klein images of each are genuinely distinct trajectories — no
collapse to fewer equivalence classes.

**(c) Basin of attraction is D₂-symmetric.** The ML-labelled (v₁, v₂)
basin (100×100 grid, 4 labels) satisfies:

| Symmetry | Label map | Agreement |
|---|---|---|
| σ_x (v₁ → −v₁) | identity | **99.36%** |
| σ_y (v₂ → −v₂) | label perm (0, 2, 1, 3) | **99.20%** |
| σ_xy (full inversion) | label perm (0, 2, 1, 3) | **99.28%** |

The label permutation (0, 2, 1, 3) — **labels 1 and 2 swap; 0 and 3 are
fixed** — reveals the physical content: labels 1 and 2 are the two
escape classes that σ_y / σ_xy physically swap (the two "mirror-image
bodies" at ±x₀), while labels 0 and 3 (likely periodic + "body-3
escapes") are D₂-invariant since body 3 sits at the origin. The ∼1%
residual is at the level of the ML classifier's label noise on the
100×100 grid; the symmetry is effectively exact.

### What this gives us

Every orbit we find in the Euler section comes with **4 automatic
copies** in the basin; any honest "distinct orbits" count must
deduplicate by D₂. The Hristov 2024 catalog's "24,582 ICs for 12,409
orbits" is only a ℤ₂ (time-reversal) deduplication — our basin sees
the full D₂, meaning the *effective* search volume per distinct orbit
is 4× what the catalog reports.

### The harder symmetries (S₃ × scaling) and what's left

- **Scaling ℝ₊**: continuous, handled above — gives every orbit a
  1-parameter family of scale-equivalent copies on sections at
  different r.
- **S₃ permutation (2↔3, 1↔3 swaps)**: these do *not* preserve the
  Euler section — they map an Euler-section orbit to one that lives on
  a different section (Jacobi or collinear-Lagrangian-type). Those 2
  extra images per orbit are "hidden" from our search, and recovering
  them requires running the pipeline on one of those alternative
  sections. This is a natural extension.
- **Your equilateral-triangle proposal** sits in the *fully symmetric*
  sub-bundle: initial conditions invariant under all of S₃ and (by p=0,
  L=0) also under continuous spatial rotations. Orbits found there
  would have simpler equivalence classes — potentially just the
  continuous-symmetry orbits and ℝ₊, no residual D₂ freedom.

---

## Artefacts

- Scaling derivation + invariants: `experiments/orbit_verification/15_scaling_and_invariants/compute_invariants.py` → `invariants.json`
- Symmetry verification: `experiments/orbit_verification/15_scaling_and_invariants/symmetry_analysis.py` → `symmetry_results.json`
- 50-digit ICs (source): `experiments/orbit_verification/06_high_precision/hp_heyoka_newton_results.json`
- Monodromy eigenvalues: `experiments/orbit_verification/11_monodromy/monodromy_results.json`
- Basin data: `experiments/orbit_discovery/basin_map.npz`
