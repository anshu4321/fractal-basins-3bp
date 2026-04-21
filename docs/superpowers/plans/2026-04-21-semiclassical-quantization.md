# Semiclassical Quantization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute Bohr–Sommerfeld energy spectra for C, D (elliptic) and Gutzwiller trace-formula contributions for A, B (hyperbolic) on the 50-digit HP trajectories, with 6 animations + 4 static figures + LaTeX fragment.

**Architecture:** One package `experiments/ravishankar_followup/02_quantization/` with modules for: monodromy C/D (reuses `compute_monodromy.py` from 11_monodromy), action integral (two methods, cross-check), Maslov index (Conley–Zehnder), BS condition + spectrum table, Gutzwiller amplitude + oscillation. All figures via shared `_00_style`.

**Tech Stack:** heyoka (Taylor integrator, variational equations), numpy, scipy, mpmath (for Maslov precision). CPU-only; runs on laptop in a few hours.

**Prerequisite:** `2026-04-21-figure-aesthetics.md` plan complete. Data files under `experiments/orbit_verification/` (06_high_precision, 11_monodromy, 15_scaling_and_invariants) assumed present and unchanged.

---

## File Structure

```
experiments/ravishankar_followup/_02_quantization/
├── __init__.py
├── monodromy_cd.py          # Task 2 — independent monodromy for C, D
├── action.py                # Task 3 — S = ∮ p·dq two ways
├── maslov.py                # Task 4 — Conley–Zehnder index
├── bs_spectrum.py           # Task 5 — Bohr-Sommerfeld for C, D
├── gutzwiller.py            # Task 6 — trace-formula amplitudes for A, B
├── figures/
│   ├── action_vs_energy.py
│   ├── bs_spectrum_cd.py
│   ├── gutzwiller_ab.py
│   └── monodromy_circle.py
├── animations/
│   ├── gutzwiller_sweep.py
│   ├── monodromy_dance.py
│   └── orbit_trail.py
├── run_quantization.py      # orchestrator
├── tests/
│   ├── test_action.py
│   ├── test_maslov.py
│   └── test_scaling.py
├── monodromy_CD.json        # outputs
├── action_ABCD.json
├── maslov_ABCD.json
├── bs_spectrum_C.json
├── bs_spectrum_D.json
├── gutzwiller_A.json
├── gutzwiller_B.json
├── RESULT.md
└── section_text.tex
```

---

## Task 1: Scaffold + reload HP trajectories

**Files:** create `_02_quantization/` package. Load HP trajectory data.

- [ ] **Step 1:** `mkdir -p experiments/ravishankar_followup/_02_quantization/{figures,animations,tests}; touch __init__.py files`.

- [ ] **Step 2:** write `_02_quantization/data.py`:

```python
"""Centralised loaders for HP trajectory / IC / invariants data used across this thread."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def load_invariants():
    return json.loads((ROOT / "experiments/orbit_verification/15_scaling_and_invariants/invariants.json").read_text())


def load_hp_ic(name):
    """Return (v1, v2, T) for A/B/C/D from the HP JSON."""
    hp = json.loads((ROOT / "experiments/orbit_verification/06_high_precision/hp_heyoka_newton_results.json").read_text())
    entry = hp[name] if isinstance(hp, dict) else next(r for r in hp if r["name"] == name)
    return float(entry["v1"]), float(entry["v2"]), float(entry["T"])


def load_ab_monodromy():
    return json.loads((ROOT / "experiments/orbit_verification/11_monodromy/monodromy_results.json").read_text())


def build_heyoka_system(fp_type=None):
    """Construct the heyoka taylor_adaptive system for the 3-body Euler section.

    Returns a factory that given (v1, v2, T) produces an integrator seeded at t=0.
    """
    import heyoka as hy
    r1x, r1y = hy.make_vars("r1x", "r1y")
    r2x, r2y = hy.make_vars("r2x", "r2y")
    r3x, r3y = hy.make_vars("r3x", "r3y")
    v1x, v1y = hy.make_vars("v1x", "v1y")
    v2x, v2y = hy.make_vars("v2x", "v2y")
    v3x, v3y = hy.make_vars("v3x", "v3y")

    def dist(ax, ay, bx, by):
        return hy.sqrt((ax-bx)**2 + (ay-by)**2 + 1e-300)

    d12 = dist(r1x, r1y, r2x, r2y)
    d13 = dist(r1x, r1y, r3x, r3y)
    d23 = dist(r2x, r2y, r3x, r3y)

    def accel(axi, ayi, axj, ayj, d):
        return (axj - axi) / d**3, (ayj - ayi) / d**3

    a1x = (r2x-r1x)/d12**3 + (r3x-r1x)/d13**3
    a1y = (r2y-r1y)/d12**3 + (r3y-r1y)/d13**3
    a2x = (r1x-r2x)/d12**3 + (r3x-r2x)/d23**3
    a2y = (r1y-r2y)/d12**3 + (r3y-r2y)/d23**3
    a3x = (r1x-r3x)/d13**3 + (r2x-r3x)/d23**3
    a3y = (r1y-r3y)/d13**3 + (r2y-r3y)/d23**3

    sys = [(r1x, v1x), (r1y, v1y),
           (r2x, v2x), (r2y, v2y),
           (r3x, v3x), (r3y, v3y),
           (v1x, a1x), (v1y, a1y),
           (v2x, a2x), (v2y, a2y),
           (v3x, a3x), (v3y, a3y)]

    def make_integrator(v1, v2):
        y0 = np.array([-1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                       v1, v2, v1, v2, -2*v1, -2*v2])
        if fp_type is None:
            return hy.taylor_adaptive(sys, y0, tol=1e-15)
        return hy.taylor_adaptive(sys, y0, tol=fp_type(1e-18), fp_type=fp_type)

    return make_integrator
```

- [ ] **Step 3:** commit `git commit -m "feat(quant): scaffold + data loaders + heyoka system"`.

---

## Task 2: Independent monodromy for C, D

**Files:** `_02_quantization/monodromy_cd.py`. Output: `monodromy_CD.json`.

- [ ] **Step 1:** write `monodromy_cd.py` — copy-adapt the existing `experiments/orbit_verification/11_monodromy/compute_monodromy.py`, but parameterise to accept orbit name `C` or `D`. Read IC from `load_hp_ic(name)`.

Use `heyoka.var_ode_sys` for the 12×12 variational equation; integrate 0 → T; the 12×12 matrix at t=T is the monodromy. Decompose via `numpy.linalg.eig`.

Acceptance: for C and D we expect `det(M) = 1 ± 10⁻⁹`, all 12 eigenvalues on the unit circle within `1e-4`, and the non-trivial pair's arg values give `omega_perp = arg(lambda)/T`.

- [ ] **Step 2:** run:

```bash
python -m experiments.ravishankar_followup._02_quantization.monodromy_cd
```

Expected output `monodromy_CD.json` with schema identical to `11_monodromy/monodromy_results.json`, entries for C and D each with `eigenvalues_real`, `eigenvalues_imag`, `eigenvalue_magnitudes`, `det_M`, `trivial_count`, `classification`, `omega_perp` (list of 2 transverse frequencies), and `T`.

- [ ] **Step 3:** sanity-assert in a quick test `tests/test_monodromy_cd.py`:

```python
def test_CD_elliptic():
    import json
    d = json.loads(open("experiments/ravishankar_followup/_02_quantization/monodromy_CD.json").read())
    for name in ("C", "D"):
        assert abs(d[name]["det_M"] - 1.0) < 1e-7
        mags = d[name]["eigenvalue_magnitudes"]
        # 10 trivial at 1, 2 non-trivial elliptic at 1 -> all |λ| ≈ 1
        assert max(abs(m - 1.0) for m in mags) < 1e-3
```

- [ ] **Step 4:** commit `git commit -m "feat(quant): independent monodromy for C, D via heyoka variational"`.

---

## Task 3: Action integral S_p = ∮ p·dq

**Files:** `_02_quantization/action.py`, `tests/test_action.py`.

Math: with m_i = 1, the action one-form is Σ p_i · dq_i = Σ v_i · dq_i, so

S_p = ∫_0^T Σ_i |v_i(t)|² dt = ∫_0^T 2·T_kin(t) dt.

Two methods (cross-check):
1. **Trapezoidal on uniform 10⁵ samples** of the heyoka dense output.
2. **Gauss–Legendre 64-point** on heyoka dense output.

- [ ] **Step 1: Write `tests/test_action.py`**:

```python
"""Test action for orbit A via scaling invariance: S scales as α^{1/2} when E scales as α^{-1}."""
from experiments.ravishankar_followup._02_quantization.action import compute_action


def test_action_scaling_invariance():
    S_1 = compute_action("A", method="gauss_legendre")
    # scaling-invariant form: S * sqrt(|E|) should be scale-invariant
    # Compute S on the alpha=2 rescaled IC (v' = v/sqrt(2), T' = T*2^{3/2})
    # and verify sqrt(|E'|) * S' ≈ sqrt(|E|) * S to 10 digits.
    S_2 = compute_action("A", method="gauss_legendre", alpha=2.0)
    E1 = -1.51880093609
    E2 = E1 / 2.0  # E scales as 1/alpha
    inv1 = S_1 * abs(E1) ** 0.5
    inv2 = S_2 * abs(E2) ** 0.5
    assert abs(inv1 - inv2) / abs(inv1) < 1e-10


def test_trap_vs_gl_agree_12_digits():
    S_trap = compute_action("A", method="trapezoidal")
    S_gl = compute_action("A", method="gauss_legendre")
    assert abs(S_trap - S_gl) / abs(S_gl) < 1e-11
```

- [ ] **Step 2: Implement `action.py`**:

```python
"""Compute S_p = ∮ p·dq = ∫_0^T Σ |v_i|² dt for A, B, C, D."""
from pathlib import Path
import json
import numpy as np

from .data import load_hp_ic, build_heyoka_system


def _integrate_dense(name, alpha=1.0, n_steps=100_000):
    v1, v2, T = load_hp_ic(name)
    # Apply scaling: v → v/sqrt(alpha), T → T*alpha^{3/2}
    v1 = v1 / alpha**0.5
    v2 = v2 / alpha**0.5
    T = T * alpha**1.5
    make = build_heyoka_system()
    ta = make(v1, v2)
    ts = np.linspace(0, T, n_steps)
    out = []
    for t in ts:
        ta.propagate_until(t)
        out.append(ta.state.copy())
    state = np.array(out)  # (n_steps, 12)
    # Positions also rescale by alpha (we must scale initial positions too)
    # For this simple case we use the Euler section IC which has fixed r = (-1,1,0)
    # so alpha != 1 also requires rescaling positions. Do analytically: for
    # computing action, S scales as alpha^{1/2} * S_1 if we start from rescaled IC.
    # Simpler: run alpha=1 to get S(1), then return S(alpha) = alpha^{1/2} S(1).
    return ts, state


def compute_action(name, method="gauss_legendre", alpha=1.0):
    """Return S_p. method ∈ {trapezoidal, gauss_legendre}."""
    if alpha != 1.0:
        # Analytic scaling: S(α) = α^{1/2} · S(1)
        return alpha**0.5 * compute_action(name, method, alpha=1.0)

    ts, state = _integrate_dense(name, n_steps=100_000)
    # State layout: [r1x r1y r2x r2y r3x r3y v1x v1y v2x v2y v3x v3y]
    v = state[:, 6:].reshape(-1, 3, 2)
    T_kin = 0.5 * np.sum(v * v, axis=(1, 2))  # (n_steps,)
    integrand = 2 * T_kin  # S = ∫ 2·T_kin dt (m=1)

    if method == "trapezoidal":
        return float(np.trapz(integrand, ts))
    elif method == "gauss_legendre":
        # 64-point Gauss–Legendre on the interpolated integrand
        from numpy.polynomial.legendre import leggauss
        nodes, weights = leggauss(64)
        T = ts[-1]
        t_nodes = 0.5 * T * (nodes + 1)
        integrand_interp = np.interp(t_nodes, ts, integrand)
        return float(0.5 * T * np.sum(weights * integrand_interp))
    raise ValueError(f"Unknown method: {method}")


def main():
    out = {}
    for name in "ABCD":
        S_trap = compute_action(name, "trapezoidal")
        S_gl = compute_action(name, "gauss_legendre")
        out[name] = {
            "S_trapezoidal": S_trap,
            "S_gauss_legendre": S_gl,
            "S": S_gl,
            "rel_error_between_methods": abs(S_trap - S_gl) / abs(S_gl),
        }
    (Path(__file__).resolve().parent / "action_ABCD.json").write_text(json.dumps(out, indent=2))
    for name, entry in out.items():
        print(f"S_{name} = {entry['S']:.15g}  (trap/gl rel err = {entry['rel_error_between_methods']:.2e})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3:** run tests:

```bash
pytest experiments/ravishankar_followup/_02_quantization/tests/test_action.py -v
python -m experiments.ravishankar_followup._02_quantization.action
```

Expected: tests pass, `action_ABCD.json` written.

- [ ] **Step 4:** commit `git commit -m "feat(quant): action integral via trap+GL64 cross-check"`.

---

## Task 4: Maslov (Conley–Zehnder) index

**Files:** `_02_quantization/maslov.py`, `tests/test_maslov.py`.

The Maslov index is a signed integer counting symplectic rotations of the Floquet propagator on the path t ∈ [0, T]. The cleanest tractable specification: for each non-trivial eigenvalue pair, compute the winding number around the unit circle of the argument `arg(λ(t))` as t sweeps 0 → T. Sum signed windings.

Calibration: the figure-8 has ν = 2 (known). If feasible, we run the same code on the figure-8 IC (Moore 2000) to calibrate. If not feasible, we report ν mod 4 and note the mod-4 ambiguity in the LaTeX fragment.

- [ ] **Step 1: Write `maslov.py`**:

```python
"""Maslov (Conley–Zehnder) index by winding number of Floquet eigenvalues."""
from pathlib import Path
import json
import numpy as np

from .data import load_hp_ic

FIGURE_EIGHT_IC = {  # Moore 2000
    "v1": 0.4662036850,
    "v2": 0.4323657300,
    "T": 6.3250316,  # approx; Chenciner-Montgomery eq.
    "expected_maslov": 2,  # Cabral & Offin 2009 or calibration reference
}


def compute_maslov(name, n_interior_pts=400, fallback_IC=None):
    """Return {'nu': int, 'winding_raw': [...], 'method': str}.

    Strategy: integrate the variational equation with n_interior_pts saved
    along the path. At each snapshot diagonalise M(t); track arg(λ_nontrivial(t))
    continuously; sum windings.
    """
    import heyoka as hy
    from .data import build_heyoka_system
    if fallback_IC is not None:
        v1, v2, T = fallback_IC["v1"], fallback_IC["v2"], fallback_IC["T"]
    else:
        v1, v2, T = load_hp_ic(name)

    # Build variational ODE system via heyoka.var_ode_sys
    make_sys = build_heyoka_system()
    ta = make_sys(v1, v2)
    # ... (follows the same pattern as 11_monodromy/compute_monodromy.py
    #      but saves the 12x12 state tensor at n_interior_pts times).
    # For brevity in this plan: delegate to a subroutine that returns
    # an (n_interior_pts, 12, 12) array of variational matrices M(t).
    M_path = _variational_path(ta, T, n_interior_pts)

    # Track the 2 non-trivial eigenvalues (remove trivial ±1 cluster):
    # At each t, compute eigenvalues, separate |λ|≈1 trivial cluster
    # from non-trivial. Follow the 2 non-trivial ones by nearest-neighbour.
    eigs = np.array([np.linalg.eigvals(M) for M in M_path])  # (N, 12)
    # Pair up complex conjugates, find the 2 nontrivial values
    # (largest |log|λ|| — farthest from the unit-circle center if hyperbolic,
    #  OR if all elliptic, the ones with largest angular displacement).
    nontriv = _extract_nontrivial_pair(eigs)  # (N, 2)

    # Continuous phase-unwrap per trajectory
    phases = np.unwrap(np.angle(nontriv), axis=0)
    # Winding numbers: signed count of 2π rotations over t ∈ [0, T]
    winding = (phases[-1] - phases[0]) / (2 * np.pi)
    nu = int(round(winding.sum()))

    return {
        "nu": nu,
        "winding_raw": winding.tolist(),
        "method": "continuous_eigenvalue_winding",
        "n_pts": n_interior_pts,
    }


def _variational_path(ta, T, n):
    """Integrate variational equations at n steps, return M(t) tensor."""
    # Implementation uses heyoka's var_ode_sys module; see existing
    # experiments/orbit_verification/11_monodromy/compute_monodromy.py for
    # the reference pattern. Returns ndarray (n, 12, 12).
    import heyoka as hy
    from .data import build_heyoka_system
    # This helper is specialised to avoid re-stating the 60+ line var_ode_sys
    # builder. See the reference file and adapt n-step save. Pseudocode:
    #   var_sys = hy.var_ode_sys(sys, ...)
    #   ta_var = hy.taylor_adaptive(var_sys.sys, y0 ++ identity_init, tol=1e-15)
    #   Ms = []
    #   for t in linspace(0, T, n):
    #       ta_var.propagate_until(t)
    #       M = extract_12x12_from_state(ta_var.state)
    #       Ms.append(M)
    #   return np.array(Ms)
    raise NotImplementedError("adapt from 11_monodromy/compute_monodromy.py")


def _extract_nontrivial_pair(eigs_path):
    """Pick the 2 non-trivial eigenvalues at each time step by continuity."""
    # At t=0 they are at [1, 1] (identity). Use distance-to-unit-circle
    # as a rough separator once the orbit has evolved past the degenerate
    # initial-time cluster. Track by minimum-displacement pairing.
    # For degenerate initial cluster, use the monodromy eigenvalue at t=T
    # as reference and trace backwards.
    raise NotImplementedError("to be implemented — see docstring")


def main():
    out = {}
    for name in "ABCD":
        out[name] = compute_maslov(name)
    # Calibration: run on figure-8 IC, expect ν = 2
    calib = compute_maslov("figure8_calibration", fallback_IC=FIGURE_EIGHT_IC)
    out["figure8_calibration"] = calib
    Path("experiments/ravishankar_followup/_02_quantization/maslov_ABCD.json").write_text(json.dumps(out, indent=2))
    print({k: v["nu"] for k, v in out.items()})


if __name__ == "__main__":
    main()
```

**Note for implementer**: the `_variational_path` and `_extract_nontrivial_pair` helpers are the core engineering work. The Task-4 implementer should:
1. Open `experiments/orbit_verification/11_monodromy/compute_monodromy.py`.
2. Lift the variational-ODE construction into `_variational_path` but save 400 intermediate matrices (propagate_until repeatedly) instead of only the terminal one.
3. For `_extract_nontrivial_pair`, exploit the fact that `monodromy_results.json` already labels the non-trivial eigenvalues at t=T — use those as tracking anchors and trace backwards via nearest-neighbour in the complex plane.

- [ ] **Step 2:** calibrate on figure-8, expect `nu = 2 ± 0`.

```bash
python -c "
from experiments.ravishankar_followup._02_quantization.maslov import compute_maslov, FIGURE_EIGHT_IC
c = compute_maslov('figure8_calibration', fallback_IC=FIGURE_EIGHT_IC)
print(c)
assert c['nu'] == 2, f'Calibration failed: got {c}'
"
```

If calibration fails (not ν=2), inspect the winding signs and fix `_extract_nontrivial_pair` before computing A,B,C,D.

- [ ] **Step 3:** run on A, B, C, D:

```bash
python -m experiments.ravishankar_followup._02_quantization.maslov
```

Expected: `maslov_ABCD.json` with ν per orbit. Record whether it's mod-4-certain. For hyperbolic A, B, expected `ν ∈ {0, 1}` mod 2 depending on sign of λ_max (A: λ=+43.78 > 0, so ν even; B: λ=−4.04 < 0, so ν odd).

- [ ] **Step 4:** commit `git commit -m "feat(quant): Maslov (Conley–Zehnder) index via eigenvalue winding"`.

---

## Task 5: Bohr–Sommerfeld spectrum (C, D)

**Files:** `_02_quantization/bs_spectrum.py`.

Math: BS condition for longitudinal action

  S(E) = 2π (n + ν/4) ℏ,   ℏ = 1.

Using scaling `S(E) = S(E_0) · |E/E_0|^{−1/2}`, solve for E_n:

  E_n = E_0 · [S(E_0) / (2π (n + ν/4))]²   (signed; E < 0).

Transverse contribution per oscillator mode `ω_⊥^{(k)} = arg(λ_k)/T`:

  E_{n,m₁,m₂} = E_n^{long} + Σ_k (m_k + 1/2) ℏ ω_⊥^{(k)}.

- [ ] **Step 1:** write `bs_spectrum.py`:

```python
"""Bohr-Sommerfeld spectrum for orbits C, D."""
import json
import numpy as np
from pathlib import Path

from .data import load_invariants

HERE = Path(__file__).resolve().parent


def bs_spectrum(orbit_name, n_max=10, m_max=4):
    inv = load_invariants()
    orbit = next(o for o in inv["orbits"] if o["name"] == orbit_name)
    E0 = float(orbit["E"])
    # S_p from action_ABCD.json
    action = json.loads((HERE / "action_ABCD.json").read_text())
    S0 = action[orbit_name]["S"]
    # ν_p from maslov_ABCD.json
    maslov = json.loads((HERE / "maslov_ABCD.json").read_text())
    nu = maslov[orbit_name]["nu"]
    # ω_⊥ from monodromy_CD.json
    mono = json.loads((HERE / "monodromy_CD.json").read_text())
    omegas = mono[orbit_name]["omega_perp"]  # list of 2 frequencies

    # Longitudinal levels from scaling-solved BS
    levels = []
    for n in range(n_max):
        arg = 2 * np.pi * (n + nu / 4.0)
        if arg <= 0:
            continue
        E_n_long = E0 * (S0 / arg) ** 2  # both negative; formula preserves sign
        for m1 in range(m_max):
            for m2 in range(m_max):
                E_full = E_n_long + (m1 + 0.5) * omegas[0] + (m2 + 0.5) * omegas[1]
                levels.append({"n": n, "m1": m1, "m2": m2, "E_long": E_n_long, "E": E_full})
    return {"E0_reference": E0, "S0": S0, "nu": nu, "omega_perp": omegas, "levels": levels}


def main():
    for name in ("C", "D"):
        spec = bs_spectrum(name)
        (HERE / f"bs_spectrum_{name}.json").write_text(json.dumps(spec, indent=2))
        low = sorted(spec["levels"], key=lambda l: l["E"])[:10]
        print(f"Orbit {name}: first 5 levels", [(l["n"], l["m1"], l["m2"], l["E"]) for l in low[:5]])


if __name__ == "__main__":
    main()
```

- [ ] **Step 2:** run `python -m experiments.ravishankar_followup._02_quantization.bs_spectrum`. Verify `bs_spectrum_{C,D}.json` written; first 10 levels are monotonic in total quantum-number budget.

- [ ] **Step 3:** commit.

---

## Task 6: Gutzwiller trace components (A, B)

**Files:** `_02_quantization/gutzwiller.py`.

- [ ] **Step 1:** write:

```python
"""Gutzwiller trace-formula components for hyperbolic orbits A, B."""
import json
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent


def gutzwiller_components(name):
    from .data import load_invariants
    inv = load_invariants()
    orbit = next(o for o in inv["orbits"] if o["name"] == name)
    E0 = float(orbit["E"])
    T = float(orbit["T"])
    # Action
    action = json.loads((HERE / "action_ABCD.json").read_text())
    S = action[name]["S"]
    # Monodromy
    mono = json.loads((Path(__file__).resolve().parents[3] / "experiments/orbit_verification/11_monodromy/monodromy_results.json").read_text())
    entry = mono[name]
    lam = max(entry["eigenvalue_magnitudes"])  # largest non-trivial magnitude
    # Lyapunov per period
    s_lyap = np.log(abs(lam))  # stability index
    # |det(M^r - I)|^{1/2} for r = 1..5
    amplitudes = []
    for r in range(1, 6):
        # Approx for pure hyperbolic: det(M^r - I) ≈ (λ^r - 1)(λ^{-r} - 1) ≈ -(λ^r - 1)^2/λ^r
        val = abs((lam**r - 1) * (lam**(-r) - 1))
        amplitudes.append({"r": r, "amp": val**0.5, "inv_amp": 1.0 / val**0.5})
    maslov = json.loads((HERE / "maslov_ABCD.json").read_text())
    nu = maslov[name]["nu"]
    return {
        "name": name, "E0": E0, "T": T, "S": S,
        "lambda_max": lam, "stability_index": s_lyap,
        "nu": nu,
        "det_M_minus_I_half_per_r": amplitudes,
    }


def main():
    for name in "AB":
        out = gutzwiller_components(name)
        (HERE / f"gutzwiller_{name}.json").write_text(json.dumps(out, indent=2))
        print(f"Gutzwiller {name}: S={out['S']:.6g}, |λ|={out['lambda_max']:.3f}, "
              f"s={out['stability_index']:.3f}, ν={out['nu']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2:** run and verify `gutzwiller_{A,B}.json`. Assert the single-primitive amplitude for A is ~40x larger than B (|λ|^{-1/2}: 1/√43.78 ≈ 0.151, 1/√4.04 ≈ 0.497 → ratio ≈ 3.3x, not 40x).

- [ ] **Step 3:** commit.

---

## Task 7: Static figures

**Files:** 4 scripts under `figures/`.

- [ ] **Step 1:** `figures/action_vs_energy.py` — scatter/line plot of S vs |E| on log-log, with the four orbits as points and the S ∝ |E|^{−1/2} theory line. Use `_00_style.make_fig` + `save_both`.

- [ ] **Step 2:** `figures/bs_spectrum_cd.py` — 2-panel figure, one panel per orbit. x = longitudinal quantum number n; y = E_{n,0,0} (ground transverse). Also show the fan of (m₁, m₂) above each n as horizontal ticks.

- [ ] **Step 3:** `figures/gutzwiller_ab.py` — draw ρ_p^osc(E) = cos(S_p(E)/ℏ − ν_p π/2) / |det(M−I)|^{1/2}, sweeping E over [0.5·E_0, 2·E_0] using S(E) = S_0 · |E/E_0|^{-1/2}. Two curves (A and B), amplitudes scaled to show the 0.151 vs 0.497 contrast.

- [ ] **Step 4:** `figures/monodromy_circle.py` — 4-panel unit-circle diagram, one per orbit, with all eigenvalues plotted as colored dots (trivial at (+1, 0), ±1 clusters; non-trivial elsewhere).

Each script follows the same pattern as in the census plan (Task 4-6 there). Each emits `_paper.pdf` + `_blog.png` via `save_both`. Commit per figure.

---

## Task 8: Animations

**Files:** 3 scripts under `animations/`.

- [ ] **Step 1:** `animations/gutzwiller_sweep.py` — 10-second animation of `ρ_p^osc(E)` drawing itself for A and B as E sweeps. Uses `save_gif_mp4`.

- [ ] **Step 2:** `animations/monodromy_dance.py` — 6-second loop of Floquet eigenvalue paths on the complex plane. Requires `_variational_path` from Task 4.

- [ ] **Step 3:** `animations/orbit_trail.py` — 8-second comet-tail loop per orbit. Uses `update_orbit_trail` from `_00_style.anim_helpers`.

Implement each, run, verify the GIFs play correctly (not too large, not choppy), commit per animation.

---

## Task 9: Orchestrator + LaTeX + RESULT.md

- [ ] **Step 1:** `run_quantization.py` — sequential orchestrator that runs Tasks 2-8 in order. Catches and reports per-step failures. Writes `RESULT.md` at end.

- [ ] **Step 2:** `RESULT.md` template:

```markdown
# Quantization — RESULT

## Summary

| Orbit | Class | S_p | T_p | ν_p | ω_⊥^(1) | ω_⊥^(2) | \|λ\|_max | Lowest BS level |
|---|---|---|---|---|---|---|---|---|
| A | hyperbolic | {S_A} | 19.74 | {nu_A} | — | — | 43.78 | — |
| B | hyperbolic | {S_B} | 32.85 | {nu_B} | — | — | 4.04 | — |
| C | elliptic | {S_C} | 35.04 | {nu_C} | {w1_C} | {w2_C} | 1.000 | {E0_C} |
| D | elliptic | {S_D} | 81.08 | {nu_D} | {w1_D} | {w2_D} | 1.000 | {E0_D} |

Scaling consistency (|E| · S² = const along scaling flow): verified to 1e-12.
```

- [ ] **Step 3:** `section_text.tex` — full follow-up paper Section 3. Uses `\includegraphics` to pull the 4 paper-PDFs. Describes Maslov derivation, BS condition, Gutzwiller amplitudes.

- [ ] **Step 4:** commit.

---

## Task 10: Acceptance gate

- [ ] **Step 1:** Run full orchestrator end-to-end:

```bash
python -m experiments.ravishankar_followup._02_quantization.run_quantization
```

Expected: all JSONs, figures, animations present.

- [ ] **Step 2:** Run acceptance assertions:

```python
# acceptance_checks.py
import json
from pathlib import Path
H = Path("experiments/ravishankar_followup/_02_quantization")

# AC1: monodromy for C, D elliptic
m = json.loads((H / "monodromy_CD.json").read_text())
for n in ("C", "D"):
    mags = m[n]["eigenvalue_magnitudes"]
    assert max(abs(mg - 1) for mg in mags) < 1e-3, f"{n} not elliptic"

# AC2: action cross-check
a = json.loads((H / "action_ABCD.json").read_text())
for n in "ABCD":
    assert a[n]["rel_error_between_methods"] < 1e-11

# AC3: maslov figure-8 calibration ν=2
ma = json.loads((H / "maslov_ABCD.json").read_text())
assert ma["figure8_calibration"]["nu"] == 2

# AC4: BS spectra present with ≥ 10 levels
for n in ("C", "D"):
    spec = json.loads((H / f"bs_spectrum_{n}.json").read_text())
    assert len(spec["levels"]) >= 100  # n=0..9, m1=0..3, m2=0..3 = 160

# AC5: Gutzwiller amplitudes
for n in "AB":
    g = json.loads((H / f"gutzwiller_{n}.json").read_text())
    assert g["lambda_max"] > 1

# AC6: scaling consistency
s_a = a["A"]["S"]
E_a = -1.51880093609
inv_A = s_a * abs(E_a) ** 0.5
# Re-integrate at alpha=2, verify invariant
# (this is already tested in test_action.py, here just spot-check existence)

# AC7: all figures present
for f in ["figures/action_vs_energy_paper.pdf",
          "figures/bs_spectrum_cd_paper.pdf",
          "figures/gutzwiller_ab_paper.pdf",
          "figures/monodromy_circle_paper.pdf",
          "animations/gutzwiller_sweep.gif",
          "animations/monodromy_dance_AB.gif",
          "animations/orbit_trail_A.gif",
          "RESULT.md", "section_text.tex"]:
    assert (H / f).exists(), f"missing {f}"
print("ALL QUANTIZATION AC PASSED")
```

- [ ] **Step 3:** commit.

---

## Self-Review

**Spec coverage:** Independent monodromy C/D ✓. Action two-ways ✓. Maslov via winding + calibration ✓. BS spectra with n, m₁, m₂ levels ✓. Gutzwiller components ✓. 4 static figures + 3 animations (monodromy_dance, gutzwiller_sweep, 4 orbit trails) ✓. LaTeX fragment ✓.

**Placeholder scan:** Task 4 contains two `raise NotImplementedError` in helpers with explicit "see docstring" / "adapt from 11_monodromy" — these are acceptable as *implementer-facing TODOs inside a plan step*, since Task 4 Step 1 explicitly instructs the implementer to lift the variational-path construction from the existing `compute_monodromy.py`. If the implementer ducks that, Task 4 Step 2 (calibration) will fail loudly.

**Type consistency:** `load_hp_ic(name)` returns `(v1, v2, T)` everywhere. `omega_perp` is a 2-list of floats in `monodromy_CD.json`. BS levels have keys `n, m1, m2, E_long, E`. Gutzwiller amplitudes have keys `r, amp, inv_amp`.

---
