# Semiclassical quantization of A, B, C, D

**Date:** 2026-04-21
**Thread:** 2 of 3 in the Ravishankar follow-up paper
**Target audience:** v2 / follow-up paper Section 3 ("Semiclassical quantization")

---

## Context

All four verified periodic orbits have 50-digit high-precision trajectories and monodromy data (for A, B; monodromy for C, D was inherited from Hristov 2025 catalog membership and has not been independently computed in our pipeline yet). The stability classification is:

- **A**: hyperbolic, |λ|_max = 43.779, stability index s = +21.90
- **B**: hyperbolic, |λ|_max = 4.040, stability index s = −2.14
- **C**: linearly stable (elliptic), inherited from Hristov 2025 #0006
- **D**: linearly stable (elliptic), inherited from Hristov 2025 #0011

This thread applies semiclassical quantization to produce quantitative spectra (C, D via Bohr–Sommerfeld) and trace-formula components (A, B via Gutzwiller). This differentiates the follow-up paper from a pure-numerical-discovery result: periodic orbits are the *input data* for semiclassical quantum mechanics in chaotic and near-integrable systems, and both quantization regimes are relevant to different subsets of our orbit set.

Ravishankar's note #2 of 2026-04-15 requested this thread explicitly.

## Goal (one sentence)

Produce Bohr–Sommerfeld energy spectra {E_{n,m₁,m₂}} for the stable orbits C and D, and Gutzwiller trace-formula contributions {S_p, T_p, |det(M_p − I)|^{1/2}, ν_p} for the hyperbolic orbits A and B, on the 50-digit trajectories.

## Scope

**In:**
- Independent monodromy computation for C, D (heyoka variational, float64 — same pipeline as 11_monodromy; inherited-classification is not enough for quantitative Floquet frequencies).
- Action integral S_p = ∮ p·dq on each of the four orbits, computed two ways (trapezoidal on HP trajectory + Gauss–Legendre with heyoka dense output) and cross-checked.
- Maslov index ν_p per orbit via Conley–Zehnder count from the stability eigenvalue flow around the orbit.
- Transverse Floquet frequencies ω_⊥^{(k)} = arg(λ_k)/T for C, D from the non-trivial elliptic eigenvalue pairs.
- Bohr–Sommerfeld condition applied: S(E) = 2π(n + ν_p/4 + 1/2)ℏ for the longitudinal quantum number n; transverse zero-point + oscillator levels for C, D.
- Gutzwiller single-primitive-orbit trace contribution for A, B, plotted as cos(r S_p/ℏ − r ν_p π/2) / |det(M_p^r − I)|^{1/2} envelopes for r = 1, 2, 3.
- LaTeX section fragment.

**Out:**
- Full Schrödinger numerical spectrum comparison (would require a separate PDE solve; out of scope for tonight).
- Zeta-function / cycle-expansion resummations (would need many more orbits than 4).
- Gutzwiller trace summed over a family of orbits (we have 4 orbits; a cycle expansion needs hundreds).

## Data inputs

| Artefact | Path | Status |
|---|---|---|
| 50-digit ICs + T for A, B, C, D | `experiments/orbit_verification/06_high_precision/hp_heyoka_newton_results.json` | present |
| Scale invariants (E, T⋆, L) | `experiments/orbit_verification/15_scaling_and_invariants/invariants.json` | present |
| Monodromy A, B + full eigenvalue lists | `experiments/orbit_verification/11_monodromy/monodromy_results.json` | present |
| Monodromy for C, D | — | **to compute in this thread** |
| Heyoka variational pipeline | `experiments/orbit_verification/11_monodromy/compute_monodromy.py` | present, reusable |

## Methodology

### Step 1 — Monodromy for C, D

Re-run `compute_monodromy.py` with the 50-digit ICs of C and D from `06_high_precision/hp_heyoka_newton_results.json`. Expected: all non-trivial |λ_k| = 1 within 10⁻⁶ (elliptic), det(M) = 1 to 10⁻¹⁰. Record the arguments arg(λ_k) for transverse-frequency computation.

Output: `monodromy_CD.json` — same schema as existing `monodromy_results.json`.

### Step 2 — Action integral S_p

For each orbit:

S_p = ∮ Σ_i p_i · dq_i = ∫_0^T Σ_i m_i |v_i(t)|² dt = ∫_0^T 2T_kin(t) dt

(m_i = 1, so Σ p·dq = 2·∫T_kin dt.)

Computed two ways:

(a) **Trapezoidal on HP trajectory**: read out the 10⁵-sample HP trajectory from heyoka, compute 2T_kin per sample, trapezoidal-integrate over one period.

(b) **Gauss–Legendre with dense output**: use heyoka's continuous dense output to evaluate T_kin at 64 Gauss–Legendre nodes per period, integrate at high order.

Report both. Expected agreement to ≥ 12 significant digits (the 10⁵-sample trapezoidal is the limiting factor; GL-64 on dense output is near machine-precision-limited).

Tie-in with energy: for a periodic orbit E = T_kin(t) + V(t) is *not* a constant identically — it *is*, actually, since energy is conserved — so ∫ T_kin dt = E·T − ∫V dt, which is a consistency check on the integral.

### Step 3 — Maslov index ν_p

Compute Conley–Zehnder index from the symplectic flow of the linearised dynamics along the orbit. Implementation: parameterise the path of symplectic matrices Φ(t) = M(t, 0) as t: 0 → T where M is the variational-equation solution already produced in Step 1. Count signed crossings of the Krein-signature-weighted eigenvalues on the unit circle.

Standard reference: Sugita, "Geometrical properties of Maslov indices in the semiclassical trace formula" (1992); Muratore-Ginanneschi, Phys. Rep. 366 (2002) §5.

For a 12-dimensional symplectic flow (3 bodies × 2D × position+momentum − constraints) the effective Maslov index lives in the 4-D physical symplectic block after projecting out the trivial pairs (energy, momentum, angular momentum, center of mass). We have 10 trivial eigenvalues and 2 non-trivial per orbit — the Maslov lives in the 2×2 non-trivial block.

For hyperbolic orbits (A, B): ν_p = 0 mod 4 if the hyperbolic eigenvalue is positive real, ν_p = 1 mod 4 if negative real (loxodromic). Check sign of λ_max for A, B to pin down mod-4 class.

For elliptic orbits (C, D): ν_p depends on the argument of the elliptic eigenvalue; compute via Gel'fand–Lidskii winding number.

Output: `maslov_ABCD.json` with ν_p and its derivation notes per orbit.

### Step 4 — Bohr–Sommerfeld spectrum for C, D

On an elliptic periodic orbit in a 2-DOF-reduced problem (after energy and momentum reduction, the 12D phase space reduces to a 2D transverse slice around the orbit), the semiclassical spectrum near the orbit is:

E_{n,m} ≈ E_0(I_0) + (ℏ ω_⊥) (m + 1/2)

where I_0 is the longitudinal action fixed by the Bohr–Sommerfeld condition

S_p(E) = 2π (n + ν_p/4) ℏ  → determines E_n

and ω_⊥ = arg(λ_transverse)/T is the transverse libration frequency.

For our 3-body problem at L = 0, after reducing by energy and by the 2D translation + 1D rotation (3 continuous symmetries → 3 pairs of trivial ±1 eigenvalues) plus time-translation (1 pair), we have 12 − 8 = 4 non-trivial dimensions → 2 transverse frequency pairs for elliptic orbits.

Both pairs contribute:

E_{n, m₁, m₂} = E_n^{long} + ℏ ω_⊥^{(1)} (m₁ + 1/2) + ℏ ω_⊥^{(2)} (m₂ + 1/2)

Procedure:

1. Compute dS/dE via finite-difference on action, using nearby Newton-refined orbits at E + δE and E − δE along the scaling direction (S scales as S → α^{1/2} S, so dS/dE = -(3/2) S / E — analytic!). Verify numerically.
2. BS condition: S(E_n) = 2π(n + ν_p/4)ℏ.
3. Since S ∝ |E|^{-1/2} along the scaling direction, BS is explicitly solvable for E_n as function of n (power law).
4. Tabulate E_{n, m₁, m₂} for n = 0, 1, ..., 9 and m₁, m₂ = 0, 1, ..., 4. ℏ = 1 (report spectrum in natural units; physical conversion is trivial).
5. Report: quantized energies, transverse frequencies, energy-level spacing.

Output: `bs_spectrum_C.json`, `bs_spectrum_D.json` — tables of E_{n,m₁,m₂} with metadata.

### Step 5 — Gutzwiller contributions for A, B

Compute per orbit:

- S_p (Step 2)
- T_p (known, 50-digit)
- |det(M_p − I)|^{1/2} from eigenvalues in `monodromy_results.json`. For a 12D symplectic map with 10 trivial ±1 eigenvalues, det(M − I) is dominated by the two non-trivial ones: det(M − I) ≈ (λ − 1)(1/λ − 1) = −(λ − 1)²/λ for a hyperbolic pair.
- ν_p (Step 3)

Gutzwiller contribution to oscillatory density of states per primitive orbit:

ρ_p^{osc}(E) = (T_p / πℏ) Σ_{r=1}^{∞} cos(r S_p(E)/ℏ − r ν_p π/2) / |det(M_p^r − I)|^{1/2}

|det(M_p^r − I)|^{1/2} ≈ |λ|^{r/2} · (1 − O(|λ|^{-r})) for large r (pure hyperbolic). Sum truncates at r ≤ 5 for convergence of the displayed series (terms decay as |λ|^{-r/2}: for A, |λ|=43.78 → r=1 dominates ~100×; for B, |λ|=4.04 → r=3 still 1%).

Produce:

- `gutzwiller_A.json`, `gutzwiller_B.json` — {S_p, T_p, ν_p, |det(M-I)|, Lyapunov-per-period log|λ_max|/T_p, truncated-series amplitudes}.
- `figures/gutzwiller_AB.pdf` — plot the single-primitive-orbit oscillation ρ_p^{osc}(E) vs. E, sweeping E over a scaling range [0.5 E_0, 2 E_0]. This uses S_p(E) = S_p(E_0) · |E/E_0|^{-1/2} from scaling.

### Step 6 — LaTeX section fragment

`section_text.tex` — subsections: Action integrals, Maslov indices, Bohr-Sommerfeld spectra (C, D), Gutzwiller contributions (A, B), Discussion.

Tables: action values per orbit; Maslov indices; first 10 BS levels for C and D; Gutzwiller components for A, B.

## Outputs (artefacts)

```
experiments/ravishankar_followup/02_quantization/
├── run_monodromy_CD.py           # Step 1
├── run_action.py                 # Step 2
├── run_maslov.py                 # Step 3
├── run_bs.py                     # Step 4
├── run_gutzwiller.py             # Step 5
├── monodromy_CD.json
├── action_ABCD.json
├── maslov_ABCD.json
├── bs_spectrum_C.json
├── bs_spectrum_D.json
├── gutzwiller_A.json
├── gutzwiller_B.json
├── figures/
│   ├── gutzwiller_AB.pdf
│   ├── gutzwiller_AB_blog.png
│   ├── bs_spectrum_CD.pdf
│   └── bs_spectrum_CD_blog.png
├── RESULT.md                     # cross-orbit summary table
└── section_text.tex
```

## Acceptance criteria

1. Monodromy for C and D: non-trivial |λ_k| = 1 to 10⁻⁴ (elliptic confirmed), det(M) = 1 to 10⁻⁹.
2. Action S_p for each orbit: two computation paths agree to ≥ 12 digits.
3. Maslov index: derivation fully recorded in `maslov_ABCD.json` with rationale (not just a number).
4. BS spectra: lowest 10 longitudinal states for C and D printed with > 10-digit precision; per-state error bound from finite-action precision stated.
5. Gutzwiller contributions for A, B: agree with single-term closed-form |det(M−I)|^{-1/2} amplitude to 6 digits using the Step 1 eigenvalues.
6. Scaling consistency: S(α E) = α^{1/2} S(E) numerically within 10⁻¹² (checks both the action integration and the scaling-invariant theory).
7. Gutzwiller oscillation figure visibly resolves the O(|λ|^{-1/2}) amplitude difference between A (|λ|=43.78) and B (|λ|=4.04).
8. `section_text.tex` compiles standalone.

## Risks / open questions

- **Maslov index is the hard piece**. It is tractable but error-prone. Sanity check: for the figure-8 orbit (known ν = 2) we can re-run the same code on its published IC as a calibration. If no calibration, state the index mod 4 explicitly and discuss the uncertainty.
- **C, D elliptic degeneracy**: if the two transverse-pair arguments happen to be rationally commensurate with each other (resonance), the product BS quantization undergoes resonance-crossing; report the frequency ratio and flag if |ratio − p/q| < 10⁻³ for small p, q.
- **ℏ convention**: report spectra in the natural units where our G = m = 1. Physical conversion to a specific atomic/molecular analog system is a separate discussion paragraph — included in `section_text.tex` but not numerically.
- **The "inherited" C, D classification was monodromy-uncalculated**: if the Step 1 independent computation disagrees (e.g., gives a weakly hyperbolic or mixed classification), the entire BS machinery for that orbit becomes inappropriate — document and pivot that orbit to Gutzwiller treatment. Monitor closely.

## Dependencies

- Requires high-precision ICs (present).
- Requires monodromy (have for A, B; computing here for C, D).
- No dependency on Threads 1 or 3.
- **Consumed by**: follow-up paper Section 3.

## Estimated wall time

~4 hours for Claude to execute: ~30 min monodromy C, D; ~30 min action integrals; ~90 min Maslov (the involved piece); ~30 min BS and Gutzwiller closed-forms; ~30 min figures; ~30 min LaTeX fragment. Order-of-magnitude, not a hard budget.
