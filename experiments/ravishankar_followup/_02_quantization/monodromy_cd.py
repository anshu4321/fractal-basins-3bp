"""Compute monodromy for orbits C and D via heyoka variational equations.

Schema matches experiments/orbit_verification/11_monodromy/monodromy_results.json
so downstream scripts can load both sources uniformly.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from experiments.ravishankar_followup._02_quantization.data import load_hp_ic

HERE = Path(__file__).resolve().parent


def build_var_system():
    import heyoka as hy

    q1x, q1y = hy.make_vars("q1x", "q1y")
    q2x, q2y = hy.make_vars("q2x", "q2y")
    q3x, q3y = hy.make_vars("q3x", "q3y")
    p1x, p1y = hy.make_vars("p1x", "p1y")
    p2x, p2y = hy.make_vars("p2x", "p2y")
    p3x, p3y = hy.make_vars("p3x", "p3y")

    r12 = hy.sqrt((q1x - q2x) ** 2 + (q1y - q2y) ** 2)
    r13 = hy.sqrt((q1x - q3x) ** 2 + (q1y - q3y) ** 2)
    r23 = hy.sqrt((q2x - q3x) ** 2 + (q2y - q3y) ** 2)

    f12x = (q2x - q1x) / r12 ** 3
    f12y = (q2y - q1y) / r12 ** 3
    f13x = (q3x - q1x) / r13 ** 3
    f13y = (q3y - q1y) / r13 ** 3
    f23x = (q3x - q2x) / r23 ** 3
    f23y = (q3y - q2y) / r23 ** 3

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


def compute_omega_perp(eigs: np.ndarray, T: float, trivial_tol: float = 1e-4) -> list:
    """Extract transverse frequencies arg(lambda_nontrivial) / T for the non-trivial pairs."""
    mags = np.abs(eigs)
    near_one = np.abs(mags - 1.0) < trivial_tol
    # Among unit-circle eigenvalues, find those with non-zero imaginary part
    # (i.e. not +1 or -1 exactly — those are the trivial Goldstone modes)
    omega_perp = []
    seen_args: list[float] = []
    # Sort by descending |Im(eig)| to hit the non-trivial pair first
    im_part = np.abs(eigs.imag)
    sorted_idx = np.argsort(-im_part)
    for i in sorted_idx:
        if not near_one[i]:
            continue  # skip non-unit eigenvalues
        arg = float(np.angle(eigs[i]))
        if abs(arg) < 1e-8:
            continue  # skip real +1
        # Deduplicate conjugate pairs: keep positive-arg representative
        if any(abs(abs(arg) - sa) < 1e-6 for sa in seen_args):
            continue
        seen_args.append(abs(arg))
        omega_perp.append(abs(arg) / T)
        if len(omega_perp) == 2:
            break
    return omega_perp


def compute_monodromy_one(name: str, vsys) -> dict:
    """Compute monodromy for a single orbit. Returns dict matching 11_monodromy schema."""
    v1, v2, T = load_hp_ic(name)
    state0 = np.array([
        -1.0, 0.0,
        +1.0, 0.0,
         0.0, 0.0,
         v1,  v2,
         v1,  v2,
        -2 * v1, -2 * v2,
    ], dtype=np.float64)
    identity = np.eye(12, dtype=np.float64).flatten()
    init = np.concatenate([state0, identity])
    assert init.size == 156

    import heyoka as hy
    t0 = time.time()
    ta = hy.taylor_adaptive(vsys, init.tolist(), compact_mode=True, tol=1e-15)
    ta.propagate_until(T)
    wall = time.time() - t0

    final = np.array(ta.state)
    closure_err = float(np.linalg.norm(final[:12] - state0))
    # heyoka flattens the variational matrix row-by-row: M[i,j] = d_state_i(T)/d_ic_j
    # final[12:156] is the 144-element flattened 12x12 monodromy matrix
    M = final[12:156].reshape(12, 12)
    det_M = float(np.linalg.det(M))
    eigs = np.linalg.eigvals(M)
    # Sort by descending magnitude for reproducibility
    order = np.argsort(-np.abs(eigs))
    eigs = eigs[order]
    mags = np.abs(eigs)

    cls, trivial, lam_max_nt = classify(eigs)
    rec_err = reciprocal_pair_max_err(eigs)
    s_idx = stability_index(eigs)
    omega_perp = compute_omega_perp(eigs, T)

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
        "omega_perp": omega_perp,
        "wall_time_s": wall,
    }


def main() -> None:
    vsys = build_var_system()
    out: dict = {}
    for name in ("C", "D"):
        print(f"Computing monodromy for {name}...", flush=True)
        r = compute_monodromy_one(name, vsys)
        out[name] = r
        print(
            f"  [{name}] class={r['classification']:<18} "
            f"|lambda|_max_nt={r['lambda_max_nontrivial']:.6e} "
            f"trivial={r['trivial_count']:2d} "
            f"det(M)={r['det_M']:+.6f} "
            f"rec_err={r['reciprocal_check_max_error']:.2e} "
            f"closure={r['closure_err_at_T']:.2e} "
            f"omega_perp={r['omega_perp']} "
            f"wall={r['wall_time_s']:.1f}s",
            flush=True,
        )
    out_path = HERE / "monodromy_CD.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
