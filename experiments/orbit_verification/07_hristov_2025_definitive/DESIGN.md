# Action #1 — Definitive resolution of C/D ↔ Hristov 2025

**Status:** design, approved 2026-04-15
**Goal:** answer one binary question — are our orbits C and D the same
periodic orbits as Hristov 2025 entries #0006 and #0011? Yes or no, with
quantitative evidence.

---

## 1. Motivation

Red-team audit (`audit/novelty_audit.md`) found a 4-decimal numerical match
between:
- our orbit C's Euler-section IC `(v1, v2, T) = (0.2554, -0.5164, 35.043)`
  and Hristov 2025 #0006's free-fall IC `(x3, y3, T) = (0.2554, 0.5164,
  35.043)`, and
- our orbit D's `(0.5539, 0.4619, 81.084)` and Hristov 2025 #0011's
  `(0.5539, 0.4619, 81.084)`.

Two conflicting signals:
- **Coincidence is implausible** (three numbers agree across independently
  produced catalogs at 4 decimals).
- **Brake-points check** (`01_brake_points/RESULT.md`) found KE_min over one
  full period was 0.77 (C) and 0.33 (D), never approaching zero. Hristov
  2025 orbits start at free-fall (KE = 0) by construction, so if C, D are
  Hristov orbits they should have KE = 0 at t = 0 and t = T. They do not.

Both cannot be true. This check resolves the contradiction.

## 2. Method

For each Hristov 2025 entry E ∈ {#0006, #0011}:

1. **Load IC at 100-digit precision** from
   `00_catalogs/hristov_2025_stable.txt` (row index E).
2. **Set up free-fall state** in E's convention: `q1 = (-0.5, 0)`,
   `q2 = (0.5, 0)`, `q3 = (x, y)`, `p = 0`. Total mass = 3, G = 1.
3. **Integrate forward** one full period T using heyoka with 200-bit
   floats (≈ 60 digits), adaptive step, absolute tolerance 1e-50.
4. **Periodicity sanity check:** verify `|q(T) − q(0)| + |p(T) − p(0)|
   < 1e-40`. If this fails, the Hristov catalog entry is not a valid
   periodic IC at this precision and we abort with an error.
5. **Energy conservation sanity check:** verify `|E(t) − E(0)| / |E(0)| <
   1e-40` across the integration.
6. **Sample the trajectory** densely (dense output at 100,000 steps of
   uniform fraction of T).
7. **Detect Euler-section crossings.** An Euler configuration in our
   canonical form is a collinear isosceles triangle with one body at the
   midpoint of the other two. Test:
   - Compute `det([q2 − q1, q3 − q1])` (signed area). At a collinear
     instant this crosses zero.
   - At each zero crossing, identify the body at the midpoint:
     the one for which `|q_i − ½(q_j + q_k)| / d_jk < 1e-6`.
   - Both conditions satisfied simultaneously defines a candidate Euler
     crossing.
8. **Canonical transform** at each crossing:
   - Let `body M` = midpoint body, `body 1/2` = outer pair.
   - Translate so body M is at origin.
   - Rotate so body 1 is at `(−d, 0)`, body 2 at `(+d, 0)` where
     `d = |q_1 − q_M| = |q_2 − q_M|` (equal by construction).
   - Scale by `1/d`, so bodies 1, 2 are at `(∓1, 0)`; body M at origin.
   - Momenta scale as `p → p · √d` (length scale `d → 1` means time
     scales as `d^(3/2)`, so velocity scales as `d^(−1/2)`, momentum
     same).
   - Read canonical `(v1, v2) = p_body1` after transformation.
9. **Compare to our orbit** at the matched crossing:
   - `dv = sqrt((v1_ours − v1_match)^2 + (v2_ours − v2_match)^2)`.
   - Consider all 4 discrete sign/reflection symmetries; take minimum dv.
   - Also report the time fraction `t/T` at which the best match occurs.

## 3. Verdict criteria

- `dv < 1e-10`: **identical orbits** (HP-level match). Declare
  rediscovery. Title change mandatory.
- `1e-10 < dv < 1e-4`: **likely identical** but precision drift. Needs
  refinement pass (refine the canonical transform).
- `1e-4 < dv < 0.01`: **AMBIGUOUS**. Could be different numerical
  realizations of the same topological orbit or genuinely different
  orbits. Escalate to free-group word comparison.
- `dv > 0.01` AND no antiparallel collinear+midpoint crossings exist:
  **orbits are distinct**. Audit's 4-decimal coincidence on (x, y) ↔
  (v1, v2) was fortuitous. Four-new-orbits framing restored.
- Integration fails periodic-closure sanity check: Hristov catalog's T is
  wrong or the entry is mis-indexed. Investigate.

## 4. Validation (run before the main check)

Three sanity runs to catch setup bugs:

1. **Known free-fall orbit round-trip.** Take Hristov 2024 entry #00001
   (Broucke A1 figure-eight-like), integrate in our setup, verify
   periodic closure. This validates our Hristov-convention integrator.
2. **Our orbit C round-trip in Euler convention.** Using our known C IC,
   integrate over T with heyoka at the same precision, verify closure
   < 1e-40. This validates our Euler-convention integrator and the
   precision target.
3. **Scale-transform round-trip.** Take our orbit C's IC, transform to
   Hristov-scale (binary = 1, recompute q3, rescale momenta), integrate,
   verify closure. This validates the canonical transform itself.

If any sanity run fails, the main check is invalid; fix the setup
before proceeding.

## 5. Compute plan

- **Platform:** GCP VM `34.173.13.145` (A100, 12 CPU). heyoka 7.10.1
  already installed via miniconda.
- **Per-entry wallclock:** ~15 min (60-digit, 100k dense steps, T=80
  worst case). Total for 2 entries + 3 sanity runs: ~2 h.
- **Artifacts:** for each entry, save
  - trajectory sample in `.npz` (float64 is fine for post-processing)
  - high-precision crossing states in `.json` (mpmath strings)
  - plot of |det| over time, log-scale
  - summary row in `RESULT.md`.

## 6. Scope (explicit)

**In scope:**
- Resolve #0006, #0011 identity vs. our C, D.
- Produce machine-readable verdict JSON + human-readable RESULT.md.

**Out of scope for this script:**
- Broader Hristov 2024 sweep (that is action #2).
- Free-group word computation (action #3).
- Independent non-heyoka cross-check (action #4).
- Checking whether #0006's or #0011's *other* Euler crossings match
  orbits A or B (natural extension, defer to follow-on if main check
  succeeds).

## 7. Deliverables

1. `check_C_D_vs_hristov_2025.py` — the integrator + match script.
2. `RESULT.md` — executive summary, verdict per orbit, tables of
   crossings, E conservation, closure residuals.
3. `verdict.json` — machine-readable verdict (`identical`/`distinct`/
   `ambiguous`/`setup_error`) per orbit, with supporting dv and phase.
4. `figures/` — optional `|det|(t)` and `KE(t)` plots.

## 8. Risk table

| Risk | Probability | Mitigation |
|---|---|---|
| GCP VM down | medium | Check first; boot if needed. Fall back to local mpmath Taylor if VM cannot be recovered. |
| heyoka 100-digit precision mismatch with our 50-digit HP ICs | low | heyoka 60-digit exceeds our 50-digit. Compare at 15 decimals (our baseline float64). |
| Euler crossing detection misses a subtle crossing | medium | Run at 100k dense samples; also scan for sub-threshold local minima of \|det\|. |
| Our brake-points check is wrong, and they DO have brake points (invalidating the prior reasoning) | low-medium | The main check is diagnostic regardless; a positive match would trigger re-running brake-points as a follow-up. |
| Canonical transform has a sign / parity bug | medium | Sanity run #3 (scale round-trip) catches it. |
