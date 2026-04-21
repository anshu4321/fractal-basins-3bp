# Quantization — RESULT

Wall time: 0.0 s

## Summary table

| Orbit | Class | E | T | S_p | ν mod 2 | Extra |
|---|---|---|---|---|---|---|
| A | hyperbolic | -1.518801 | 19.7354 | 59.9483 | 0 | |λ|_max = 43.7791 |
| B | hyperbolic | -1.570451 | 32.8491 | 103.1757 | 1 | |λ|_max = 4.0403 |
| C | linearly stable | -1.504302 | 35.0431 | 105.4308 | 1 | ω_⊥ = [0.04879805243975609, 0.028426661435506054] |
| D | linearly stable | -0.939305 | 81.0836 | 152.3246 | 0 | ω_⊥ = [0.014179548814103979, 0.0022438888519390084] |


## Notes

- **Action integrals** computed two ways (trapezoidal @ 50k steps + composite
  Gauss-Legendre 32×100), agree to ≥ 12 digits for all 4 orbits.
- **Monodromy** for C, D recomputed via `heyoka.var_ode_sys` (same pipeline as
  A, B previously); det(M) = 1 ± 10⁻¹¹ for both, all 12 |λ| = 1 within 10⁻³
  (elliptic).
- **Maslov index** computed only **mod 2** from terminal monodromy spectrum.
  The full Conley-Zehnder winding-number computation over the continuous
  Floquet path is deferred to a follow-up pass; all BS / Gutzwiller results
  inherit the ±k/2 integer-shift uncertainty.
- **BS spectra** for C, D: 160 levels
  for C, 144 levels for D
  (n=0..9, m1=0..3, m2=0..3 with ν_mod_2 filtering).
- **Gutzwiller** components for A, B: primitive r=1 amplitude ratio
  A/B ≈ 0.39 (B's single-repetition contribution dominates because |λ_B|=4.04
  is closer to unit-circle than |λ_A|=43.78).

## Artefacts

- `action_ABCD.json` / `maslov_ABCD.json` / `monodromy_CD.json`
- `bs_spectrum_C.json` / `bs_spectrum_D.json`
- `gutzwiller_A.json` / `gutzwiller_B.json`
- `figures/action_vs_energy_{paper.pdf,blog.png}`
- `figures/bs_spectrum_cd_{paper.pdf,blog.png}`
- `figures/gutzwiller_ab_{paper.pdf,blog.png}`
- `figures/monodromy_circle_{paper.pdf,blog.png}`
- `animations/gutzwiller_sweep.{gif,mp4}`
- `animations/orbit_trail_{A,B,C,D}.{gif,mp4}`
- `animations/monodromy_dance_AB.{gif,mp4}`

## TODO for follow-up

1. **Full Maslov (Conley-Zehnder) index** via continuous-path winding of the
   variational eigenvalues over [0, T]. This removes the ±k/2 ambiguity in
   BS energies and the ±π/2 phase ambiguity in Gutzwiller amplitudes.
2. **Numerical Schrödinger spectrum** cross-check for C, D (would require a
   2-D PDE solve in transverse action-angle coordinates near the orbit).
3. **Multi-orbit Gutzwiller sum** — our current A+B Gutzwiller is just two
   primitive contributions; a proper trace-formula spectrum needs a cycle
   expansion over ~10-100 orbits.
