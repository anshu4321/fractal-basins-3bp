"""High-precision Taylor verification of equilateral-section candidates.

Adapts the pattern from experiments/orbit_verification/06_high_precision/
but with the S3-symmetric IC builder from _03_equilateral.section.
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def _build_heyoka_ld_system():
    """Build a heyoka taylor_adaptive_longdouble for the 3-body system."""
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
    return hy, sys


def _section_ic_ld(u_x, u_y):
    """Equilateral IC in longdouble."""
    R_TRI = np.longdouble(1.0) / np.sqrt(np.longdouble(3.0))
    thetas = [np.longdouble(0),
              np.longdouble(2) * np.pi / 3,
              np.longdouble(4) * np.pi / 3]
    r = np.array([
        [R_TRI * np.cos(t), R_TRI * np.sin(t)] for t in thetas
    ], dtype=np.longdouble)
    u_x = np.longdouble(u_x); u_y = np.longdouble(u_y)
    v = np.array([
        [np.cos(t)*u_x - np.sin(t)*u_y, np.sin(t)*u_x + np.cos(t)*u_y]
        for t in thetas
    ], dtype=np.longdouble)
    return r, v


def verify_one(u_x, u_y, T, max_iter=20, tol=1e-30):
    """HP Taylor + Newton polish at longdouble."""
    import heyoka as hy
    hy_mod, sys = _build_heyoka_ld_system()

    u = np.array([np.longdouble(u_x), np.longdouble(u_y)])
    T_cur = np.longdouble(T)

    def integrate(u_vec, T_val):
        r, v = _section_ic_ld(u_vec[0], u_vec[1])
        y0 = np.concatenate([r.ravel(), v.ravel()]).astype(np.longdouble)
        ta = hy_mod.taylor_adaptive(sys, y0, tol=np.longdouble(1e-18),
                                     fp_type=np.longdouble)
        ta.propagate_until(T_val)
        return y0, np.asarray(ta.state)

    # Newton polish at longdouble
    for it in range(max_iter):
        y0, yT = integrate(u, T_cur)
        residual = yT - y0
        res_norm = float(np.linalg.norm(residual.astype(float)))
        if res_norm < tol:
            return {"u": [float(u[0]), float(u[1])],
                    "T": float(T_cur),
                    "residual": res_norm,
                    "iterations": it, "hp_converged": True}
        # Jacobian wrt (u_x, u_y, T) at longdouble
        eps_u = np.longdouble(1e-8); eps_T = np.longdouble(1e-6)
        J = np.zeros((12, 3), dtype=np.longdouble)
        for k, (du0, du1, dT) in enumerate([
            (eps_u, np.longdouble(0), np.longdouble(0)),
            (np.longdouble(0), eps_u, np.longdouble(0)),
            (np.longdouble(0), np.longdouble(0), eps_T),
        ]):
            u_p = u + np.array([du0, du1])
            T_p = T_cur + dT
            y0_p, yT_p = integrate(u_p, T_p)
            J[:, k] = ((yT_p - y0_p) - residual) / (eps_u if k < 2 else eps_T)
        try:
            delta, *_ = np.linalg.lstsq(J.astype(np.float64),
                                         -residual.astype(np.float64), rcond=None)
        except np.linalg.LinAlgError:
            break
        u[0] += np.longdouble(delta[0])
        u[1] += np.longdouble(delta[1])
        T_cur += np.longdouble(delta[2])
    return {"u": [float(u[0]), float(u[1])],
            "T": float(T_cur),
            "residual": res_norm,
            "iterations": max_iter, "hp_converged": res_norm < tol}


def run_all():
    cands = json.loads((HERE / "refined_candidates.json").read_text())
    hp_results = []
    for i, c in enumerate(cands):
        print(f"HP verifying #{i+1}: u=({c['u'][0]:.4f}, {c['u'][1]:.4f}), T={c['T']:.4f} ...")
        try:
            r = verify_one(c["u"][0], c["u"][1], c["T"])
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        entry = {**c, **r, "name": f"EQ{i+1}"}
        status = "OK  " if r["hp_converged"] else "partial"
        print(f"  -> {status} res={r['residual']:.2e} T={r['T']:.6f}")
        if r["residual"] < 1e-11:
            hp_results.append(entry)

    (HERE / "hp_verified_candidates.json").write_text(json.dumps(hp_results, indent=2))
    print(f"\n{len(hp_results)}/{len(cands)} HP-verified at residual < 1e-11")
    return hp_results


if __name__ == "__main__":
    run_all()
