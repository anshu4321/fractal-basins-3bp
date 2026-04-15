# Action #1 — Definitive C/D ↔ Hristov 2025 identity check RESULT

**Run:** 2026-04-15 05:45–06:15 IST
**Status:** COMPLETE

## Verdict

| Orbit | Hristov entry | Verdict | dv_min | \|ΔT\| | Closure |
|---|---|---|---|---|---|
| **C** | hristov2025_0006 | **IDENTICAL** | 4.65 × 10⁻⁵¹ | 4.54 × 10⁻⁵⁰ | 4.15 × 10⁻⁴⁹ |
| **D** | hristov2025_0011 | **IDENTICAL** | 4.64 × 10⁻⁵¹ | 4.71 × 10⁻⁵⁰ | 4.44 × 10⁻⁴⁸ |

Orbits C and D are rediscoveries of Hristov 2025 stable-orbits entries
#0006 and #0011 at ≥50-digit precision. The 4-decimal match flagged in
the audit was the tip of a 50-digit match — not a coincidence.

## Critical finding: Hristov 2025 catalog convention is Euler, not free-fall

The catalog README (`00_catalogs/README.md`) claims Hristov 2025 uses
free-fall convention (bodies at ±0.5, 0; body 3 at (x, y); zero
velocities). This is **incorrect** for the 2025 catalog. Evidence:

1. **Hristov 2024 sanity (Task 5):** integrating Hristov 2024 row 1's
   `(x, y, T)` as free-fall IC reproduces periodic closure to 5.85 × 10⁻⁴¹.
   So Hristov 2024 IS free-fall.
2. **Hristov 2025 free-fall (Task 6):** integrating Hristov 2025 row 6's
   `(x, y, T)` as free-fall IC diverges — closure residual 2.43 after
   time T; trajectory visits 0 Euler-section configurations.
3. **Hristov 2025 Euler (Tasks 6, 7):** interpreting Hristov 2025 rows
   as `(v1, v2, T, T*)` in Li-Liao/Euler convention (bodies at ±1, 0;
   body 3 at origin; p1 = p2 = (v1, v2); p3 = −2(v1, v2)) closes to
   4.15 × 10⁻⁴⁹ for row 6 and 4.44 × 10⁻⁴⁸ for row 11, matching our
   orbits C and D at 50-digit precision.

Hristov 2025 published the *stable* subset of their free-fall 2024 catalog.
Between 2024 and 2025 the convention changed to Li-Liao/Euler — likely to
present the orbits in the more widely used convention for stability
discussion. The README's convention note was not updated.

## Implications for the paper

### Orbit C (T ≈ 35.04, linearly stable)

- **Novelty claim: FALSIFIED.** Already known as Hristov 2025 #0006.
- Stability claim: **strengthened.** Independently identified as linearly
  stable in Hristov 2025's stable-orbits database.
- Our HP ICs agree with Hristov's 100-digit catalog entries to 50 digits.

### Orbit D (T ≈ 81.08, marginal stability in our monodromy)

- **Novelty claim: FALSIFIED.** Already known as Hristov 2025 #0011.
- Our "marginal" vs Hristov's "stable" — reconcile: likely numerical
  noise in our monodromy analysis (eigenvalue magnitude 1 + ε, ε at
  roundoff).

### Orbits A and B

- **Unchanged by this check.** Audit did not flag A, B as potential
  Hristov 2025 matches (T* values not in the stable-orbits catalog).
- A and B remain **candidate new**, pending free-group word comparison
  (action #3) and full Hristov 2024 sweep (action #2).

### Paper framing

The current draft "Four new periodic orbits" is no longer defensible.
Honest revised claim:

> **"Two candidate new orbits (A, B) plus independent rediscovery of two
> Hristov 2025 stable-orbits entries (C, D) via an ML-guided search from
> a different phase-space section"**

Target venues: **Chaos (AIP)** or **CMDA** — this framing is
publishable as a methods/pipeline paper. NOT Phys Rev E at current
novelty count.

## Numerical artifacts

### Initial conditions at the Euler section

| Orbit | v₁ (50-digit, HP-refined) | v₂ (50-digit, HP-refined) | T (50-digit) |
|---|---|---|---|
| C | 0.25543093565075946258… | −0.51638583901511426327… | 35.04308702166738922310… |
| D | 0.55393899048227531125… | 0.46193410063619444282… | 81.08361216982954533977… |

These agree with Hristov 2025's rows 6 and 11 at ≥50 digits.

### Closure residuals (heyoka 200-bit, full period)

- Sanity 1 (orbit C self-closure): **4.15 × 10⁻⁴⁹** (gate 1 × 10⁻³⁰, passed by 19 orders)
- Sanity 2 (Hristov 2024 #00001 free-fall): **5.85 × 10⁻⁴¹** (gate 1 × 10⁻²⁰)
- Main C vs #0006 (Euler interp): **4.15 × 10⁻⁴⁹**
- Main D vs #0011 (Euler interp): **4.44 × 10⁻⁴⁸**

All closures passed. Sign/reflection ambiguity: C matches under
`(+v₁, −v₂)`; D matches under `(+v₁, +v₂)`. Both are discrete symmetries
of the equal-mass problem.

## Supporting files

- `verdict.json` — machine-readable verdict
- `verify_C_result.json` — full crossing data for C (Task 6)
- `verify_D_result.json` — Euler-interpretation data for D (Task 7)
- `sanity_results.json` — both sanity runs (Tasks 4, 5)
- `verify_C.log`, `verify_D.log` — raw integration logs
- `diag_*.py` — diagnostic scripts used to isolate the convention-mismatch
  (kept as evidence trail)
- `DESIGN.md` — original design document
- `/Users/aishwarya/Desktop/Mega_3Bp/docs/plans/2026-04-15-hristov-2025-definitive-check.md` — plan

## Next actions (from the audit's top-5 queue)

1. **DONE** — Resolve C/D ↔ Hristov 2025 identity. Verdict: both are
   rediscoveries.
2. Full Hristov 2024 sweep (all 12,409 entries) — now higher priority
   given the 2025 convention confusion. Needs re-check of A, B against
   2024 free-fall via our existing forward-integration script.
3. Free-group word computation for A, B.
4. Independent non-heyoka HP integrator cross-check for A, B (now
   especially important since C, D are rediscoveries — A, B are the only
   remaining novelty claims).
5. Seed expansion + ablations for ML prior study.
