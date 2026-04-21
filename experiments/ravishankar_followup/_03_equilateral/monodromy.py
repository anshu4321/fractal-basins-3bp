"""Monodromy for HP-verified equilateral-section candidates via heyoka variational.

Adapted from experiments/ravishankar_followup/_02_quantization/monodromy_cd.py;
main difference: ICs built from the equilateral-triangle S3-symmetric section
(section_ic) instead of the Euler collinear section. Output schema matches
monodromy_CD.json so downstream figures can load both sources uniformly.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from experiments.ravishankar_followup._03_equilateral.section import section_ic

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


def classify(eigs, trivial_tol=1e-4):
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


def reciprocal_pair_max_err(eigs):
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


def stability_index(eigs, trivial_tol=1e-4):
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


def compute_omega_perp(eigs, T, trivial_tol=1e-4):
    mags = np.abs(eigs)
    near_one = np.abs(mags - 1.0) < trivial_tol
    omega_perp = []
    seen_args = []
    im_part = np.abs(eigs.imag)
    sorted_idx = np.argsort(-im_part)
    for i in sorted_idx:
        if not near_one[i]:
            continue
        arg = float(np.angle(eigs[i]))
        if abs(arg) < 1e-8:
            continue
        if any(abs(abs(arg) - sa) < 1e-6 for sa in seen_args):
            continue
        seen_args.append(abs(arg))
        omega_perp.append(abs(arg) / T)
        if len(omega_perp) == 2:
            break
    return omega_perp


def compute_monodromy_one(name, u_x, u_y, T, vsys):
    r, v = section_ic((u_x, u_y))
    state0 = np.concatenate([r.ravel(), v.ravel()]).astype(np.float64)
    assert state0.size == 12
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
    M = final[12:156].reshape(12, 12)
    det_M = float(np.linalg.det(M))
    eigs = np.linalg.eigvals(M)
    order = np.argsort(-np.abs(eigs))
    eigs = eigs[order]
    mags = np.abs(eigs)

    cls, trivial, lam_max_nt = classify(eigs)
    rec_err = reciprocal_pair_max_err(eigs)
    s_idx = stability_index(eigs)
    omega_perp = compute_omega_perp(eigs, T)

    return {
        "name": name,
        "u_x": float(u_x),
        "u_y": float(u_y),
        "T": float(T),
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


def main():
    cands = json.loads((HERE / "hp_verified_candidates.json").read_text())
    vsys = build_var_system()
    out = {}
    for c in cands:
        name = c.get("name") or ("EQ" + str(c.get("cluster_id", "?")))
        u = c.get("u") or [c.get("u_x"), c.get("u_y")]
        T = c["T"]
        ux, uy = float(u[0]), float(u[1])
        print("Computing monodromy for {} (u=({:.6f},{:.6f}), T={:.6f})...".format(name, ux, uy, T), flush=True)
        r = compute_monodromy_one(name, ux, uy, T, vsys)
        out[name] = r
        print("  [{}] class={:<18} |lambda|_max_nt={:.6e} trivial={:2d} det(M)={:+.6f} rec_err={:.2e} closure={:.2e} omega_perp={} wall={:.1f}s".format(
            name, r["classification"], r["lambda_max_nontrivial"], r["trivial_count"],
            r["det_M"], r["reciprocal_check_max_error"], r["closure_err_at_T"],
            r["omega_perp"], r["wall_time_s"]), flush=True)
    out_path = HERE / "monodromy_equilateral.json"
    out_path.write_text(json.dumps(out, indent=2))
    print("wrote {} ({} entries)".format(out_path, len(out)), flush=True)


if __name__ == "__main__":
    main()
