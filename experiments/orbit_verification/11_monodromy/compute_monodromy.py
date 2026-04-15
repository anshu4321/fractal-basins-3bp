"""
Monodromy computation for orbits A and B at float64 precision via
heyoka variational ODEs (order 1).

Strategy: build the 12-dim Hamiltonian 3BP as heyoka symbolic equations,
lift to a variational system in the 12 state variables, integrate the
156-dim combined ODE (12 state + 144 sensitivities) over one period T,
and read off the 12x12 monodromy matrix M = dstate(T)/dstate(0).

Eigenvalues of M classify linear stability:
  - |lambda|_max (non-trivial) > 1 + 1e-4  --> hyperbolic (unstable)
  - all non-trivial |lambda| = 1 within tol --> elliptic (linearly stable)
  - otherwise                               --> marginal

float64 eigenvalue precision ~ 1e-6 is sufficient for a hyperbolic vs
elliptic call; marginal cases would need a 200-bit rerun.
"""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import heyoka as hy

HP_RESULTS = pathlib.Path(
    "/home/aishwarya/3bp/experiments/orbit_verification/06_high_precision/"
    "hp_heyoka_newton_results.json"
)
OUT_DIR = pathlib.Path(
    "/home/aishwarya/3bp/experiments/orbit_verification/11_monodromy"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_var_system() -> hy.var_ode_sys:
    q1x, q1y = hy.make_vars("q1x", "q1y")
    q2x, q2y = hy.make_vars("q2x", "q2y")
    q3x, q3y = hy.make_vars("q3x", "q3y")
    p1x, p1y = hy.make_vars("p1x", "p1y")
    p2x, p2y = hy.make_vars("p2x", "p2y")
    p3x, p3y = hy.make_vars("p3x", "p3y")

    r12 = hy.sqrt((q1x - q2x) ** 2 + (q1y - q2y) ** 2)
    r13 = hy.sqrt((q1x - q3x) ** 2 + (q1y - q3y) ** 2)
    r23 = hy.sqrt((q2x - q3x) ** 2 + (q2y - q3y) ** 2)

    f12x = (q2x - q1x) / r12**3
    f12y = (q2y - q1y) / r12**3
    f13x = (q3x - q1x) / r13**3
    f13y = (q3y - q1y) / r13**3
    f23x = (q3x - q2x) / r23**3
    f23y = (q3y - q2y) / r23**3

    sys_eqs = [
        (q1x, p1x), (q1y, p1y),
        (q2x, p2x), (q2y, p2y),
        (q3x, p3x), (q3y, p3y),
        (p1x,  f12x + f13x), (p1y,  f12y + f13y),
        (p2x, -f12x + f23x), (p2y, -f12y + f23y),
        (p3x, -f13x - f23x), (p3y, -f13y - f23y),
    ]
    vsys = hy.var_ode_sys(sys_eqs, args=hy.var_args.vars, order=1)
    return vsys


def initial_state(v1: float, v2: float) -> np.ndarray:
    return np.array([
        -1.0, 0.0,
        +1.0, 0.0,
         0.0, 0.0,
          v1,  v2,
          v1,  v2,
        -2 * v1, -2 * v2,
    ], dtype=np.float64)


def classify(eigs: np.ndarray, trivial_tol: float = 1e-4):
    mags = np.abs(eigs)
    near_one = np.abs(mags - 1.0) < trivial_tol
    trivial_count = int(near_one.sum())
    nontrivial_mags = mags[~near_one]
    lambda_max_nt = float(nontrivial_mags.max()) if nontrivial_mags.size else 1.0
    if lambda_max_nt > 1.0 + trivial_tol:
        cls = "hyperbolic"
    elif trivial_count == len(eigs):
        cls = "linearly stable"
    else:
        cls = "marginal"
    return cls, trivial_count, lambda_max_nt


def reciprocal_pair_max_err(eigs: np.ndarray) -> float:
    remaining = list(range(len(eigs)))
    worst = 0.0
    while remaining:
        i = remaining.pop(0)
        lam = eigs[i]
        best_j, best_e = -1, 1e18
        for j in remaining:
            e = abs(lam * eigs[j] - 1.0)
            if e < best_e:
                best_e, best_j = float(e), j
        if best_j >= 0:
            worst = max(worst, best_e)
            remaining.remove(best_j)
    return worst


def stability_index(eigs: np.ndarray, trivial_tol: float = 1e-4) -> float:
    near_one = np.abs(np.abs(eigs) - 1.0) < trivial_tol
    nontrivial = eigs[~near_one]
    if nontrivial.size == 0:
        return 0.0
    s = 0.0
    used = np.zeros(len(nontrivial), dtype=bool)
    for i, lam in enumerate(nontrivial):
        if used[i]:
            continue
        best_j, best_e = -1, 1e18
        for j in range(i + 1, len(nontrivial)):
            if used[j]:
                continue
            e = abs(lam * nontrivial[j] - 1.0)
            if e < best_e:
                best_e, best_j = e, j
        if best_j >= 0:
            s += 0.5 * float((lam + nontrivial[best_j]).real)
            used[i] = True
            used[best_j] = True
    return float(s)


def compute_orbit(name: str, v1: float, v2: float, T: float, vsys) -> dict:
    state0 = initial_state(v1, v2)
    identity = np.eye(12, dtype=np.float64).flatten()
    init = np.concatenate([state0, identity])
    assert init.size == 156

    t0 = time.time()
    ta = hy.taylor_adaptive(vsys, init.tolist(), compact_mode=True, tol=1e-15)
    ta.propagate_until(T)
    wall = time.time() - t0

    final = np.array(ta.state)
    closure_err = float(np.linalg.norm(final[:12] - state0))
    M = final[12:156].reshape(12, 12)
    det_M = float(np.linalg.det(M))
    eigs = np.linalg.eigvals(M)
    order = np.argsort(-np.abs(eigs))
    eigs = eigs[order]
    mags = np.abs(eigs)

    cls, trivial, lam_max_nt = classify(eigs)
    rec_err = reciprocal_pair_max_err(eigs)
    s_idx = stability_index(eigs)

    return {
        "name": name,
        "v1": v1,
        "v2": v2,
        "T": T,
        "closure_err_at_T": closure_err,
        "det_M": det_M,
        "reciprocal_check_max_error": rec_err,
        "trivial_count": trivial,
        "lambda_max_nontrivial": lam_max_nt,
        "stability_index": s_idx,
        "classification": cls,
        "precision": "float64",
        "eigenvalues_real": [float(e.real) for e in eigs],
        "eigenvalues_imag": [float(e.imag) for e in eigs],
        "eigenvalue_magnitudes": [float(m) for m in mags],
        "wall_time_s": wall,
    }


def main() -> None:
    vsys = build_var_system()
    hp = json.loads(HP_RESULTS.read_text())
    results: dict = {}
    for rec in hp:
        name = rec["name"]
        if name not in ("A", "B"):
            continue
        v1 = float(rec["v1_HP"])
        v2 = float(rec["v2_HP"])
        T = float(rec["T_HP"])
        print(f"[{name}] v1={v1:.15e} v2={v2:.15e} T={T:.15e}", flush=True)
        r = compute_orbit(name, v1, v2, T, vsys)
        print(
            f"[{name}] class={r['classification']:<18} "
            f"|lambda|_max_nt={r['lambda_max_nontrivial']:.6e} "
            f"trivial={r['trivial_count']:2d} "
            f"det(M)={r['det_M']:+.6f} "
            f"rec_err={r['reciprocal_check_max_error']:.2e} "
            f"closure={r['closure_err_at_T']:.2e} "
            f"wall={r['wall_time_s']:.1f}s",
            flush=True,
        )
        results[name] = r

    results["C"] = {
        "classification": "linearly stable",
        "source": "Hristov 2025 #0006",
        "note": "inherited from stability-filtered catalog membership",
    }
    results["D"] = {
        "classification": "linearly stable",
        "source": "Hristov 2025 #0011",
        "note": "inherited from stability-filtered catalog membership",
    }

    out = OUT_DIR / "monodromy_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
