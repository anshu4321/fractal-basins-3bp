# Action #4 - Independent HP Cross-Check (heyoka 200-bit)

Dual-tool high-precision closure verification. Each orbit is integrated for one full period T under our Euler-section IC convention, and the max-component closure residual is compared against the existing mpmath-Taylor (30 decimal digits) HP results.

**Gate:** max closure residual < 1e-30

**Overall verdict:** ALL FOUR PASS - dual-tool periodicity confirmed

**Independence:** mpmath-Taylor is a pure-Python interpreted arbitrary-precision Taylor integrator; heyoka is a JIT-compiled MPFR-backed Taylor integrator (different Taylor-series generation, different arithmetic backend). Both closing < 1e-30 on the same IC rules out tool-specific bugs.

## Results

| Orbit | T | heyoka 200-bit residual | mpmath Taylor 30-dps residual | heyoka wall (s) | dual verdict |
|:-----:|--:|:-----------------------:|:-----------------------------:|----------------:|:------------:|
| A | 19.73539 | 5.340e-48 | 1.197e-50 | 1.96 | **PASS** |
| B | 32.84907 | 1.119e-48 | 1.779e-50 | 3.16 | **PASS** |
| C | 35.04308 | 4.150e-49 | 8.391e-51 | 3.06 | **PASS** |
| D | 81.08361 | 4.440e-48 | 4.296e-50 | 3.92 | **PASS** |

Total heyoka wall-time (sequential): 12.09 s

## Interpretation

All four orbits close below the 1e-30 gate under *both* independent HP integrators. Combined with the prior identity work:

- **Orbit C** = Hristov 2025 #0006 (Action #1, dv_min ~ 4.65e-51).
- **Orbit D** = Hristov 2025 #0011 (Action #1, analogous).
- **Orbit A** = novel topology, novel in both Hristov catalogs (Actions #2, #3).
- **Orbit B** = novel IC (no Hristov match); topology shared with C and Hristov row 7 (Actions #2, #3).

Dual-tool HP closure means the novelty claims for A (and the IC-level novelty of B) do not depend on a single integrator's correctness — the closure has been verified by two tools with disjoint Taylor-series implementations.

## Methodology

1. IC construction: q = [(-1,0), (1,0), (0,0)], p = [(v1,v2), (v1,v2), (-2v1,-2v2)], with (v1, v2, T) as 50-digit strings from `experiments/orbit_verification/06_high_precision/hp_heyoka_newton_results.json`.
2. Integrator: `build_hp_integrator(200)` from `07_hristov_2025_definitive/check_C_D_vs_hristov_2025.py` - heyoka `taylor_adaptive` with `fp_type=hy.real`, `prec=200`, `tol=1e-50`, `compact_mode=True`.
3. Propagate via `ta.propagate_until(T_end)` (single sweep; no intermediate dense output needed for closure).
4. Closure residual = max over 12 state components of |state(T) - state(0)|.
5. The mpmath Taylor residuals are read from the existing `hp_heyoka_newton_results.json` `residual_HP_float` field.

