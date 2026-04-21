"""Section 2.6 fast: Single-orbit HP runner. Used by parallel launcher.

Usage: python3 run_hp_one_fast.py NAME v1 v2 T n_steps order dps max_iter
"""
from __future__ import annotations
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path("/Users/aishwarya/Desktop/Mega_3Bp")
sys.path.insert(0, str(PROJECT_ROOT))

import mpmath as mp
from mega3bp.taylor_hp import integrate

OUT = Path(__file__).parent


def _F_residual(v1_s, v2_s, T_s, n_steps, order, dps):
    mp.mp.dps = dps
    v1 = mp.mpf(v1_s); v2 = mp.mpf(v2_s); T = mp.mpf(T_s)
    q0 = [[mp.mpf(-1), mp.mpf(0)], [mp.mpf(1), mp.mpf(0)], [mp.mpf(0), mp.mpf(0)]]
    p0 = [[v1, v2], [v1, v2], [-2 * v1, -2 * v2]]
    qf, pf = integrate(q0, p0, T, n_steps=n_steps, order=order, dps=dps)
    F = []
    for i in range(3):
        for d in range(2):
            F.append(qf[i][d] - q0[i][d])
    for i in range(3):
        for d in range(2):
            F.append(pf[i][d] - p0[i][d])
    return [mp.nstr(x, dps + 5) for x in F]


def F_local(v1, v2, T, n_steps, order, dps):
    F_strs = _F_residual(str(v1), str(v2), str(T), n_steps, order, dps)
    F = [mp.mpf(s) for s in F_strs]
    norm = mp.sqrt(sum(x * x for x in F))
    return F, norm


def main():
    name = sys.argv[1]
    v1 = float(sys.argv[2]); v2 = float(sys.argv[3]); T = float(sys.argv[4])
    n_steps = int(sys.argv[5])
    order = int(sys.argv[6])
    dps = int(sys.argv[7])
    max_iter = int(sys.argv[8])

    mp.mp.dps = dps
    v1m = mp.mpf(str(v1)); v2m = mp.mpf(str(v2)); T0 = mp.mpf(str(T))
    best_res = mp.mpf("inf"); best_params = (v1m, v2m, T0); trace = []

    print(f"[{name}] start dps={dps} order={order} n_steps={n_steps} max_iter={max_iter}",
          flush=True)
    t_total = time.perf_counter()

    with ProcessPoolExecutor(max_workers=3) as pool:
        for it in range(max_iter):
            t_iter = time.perf_counter()
            F, normF = F_local(v1m, v2m, T0, n_steps, order, dps)
            print(f"[{name}] iter {it}: ||F|| = {mp.nstr(normF, 10)} "
                  f"({time.perf_counter() - t_iter:.0f}s)", flush=True)
            trace.append(mp.nstr(normF, 10))
            if normF < best_res:
                best_res = normF; best_params = (v1m, v2m, T0)
            if normF < mp.mpf(f"1e-{dps - 3}"):
                print(f"[{name}] converged", flush=True)
                break

            eps_fd = mp.mpf("1e-7")
            v1_s, v2_s, T_s = str(v1m), str(v2m), str(T0)
            args_list = [
                (str(v1m + eps_fd), v2_s, T_s, n_steps, order, dps),
                (v1_s, str(v2m + eps_fd), T_s, n_steps, order, dps),
                (v1_s, v2_s, str(T0 + eps_fd), n_steps, order, dps),
            ]
            futures = [pool.submit(_F_residual, *a) for a in args_list]
            F_pluses = [[mp.mpf(s) for s in f.result()] for f in futures]
            J_cols = []
            for c in range(3):
                col = [(fp - f) / eps_fd for fp, f in zip(F_pluses[c], F)]
                J_cols.append(col)
            J = mp.matrix([[J_cols[c][r] for c in range(3)] for r in range(12)])
            F_vec = mp.matrix(F)
            damping = mp.matrix([[mp.mpf("1e-8") if i == j else mp.mpf(0)
                                   for j in range(3)] for i in range(3)])
            dp = mp.lu_solve(J.T * J + damping, J.T * F_vec)
            v1m -= dp[0]; v2m -= dp[1]; T0 -= dp[2]

    elapsed = time.perf_counter() - t_total
    r = {
        "name": name,
        "v1_HP": str(best_params[0]),
        "v2_HP": str(best_params[1]),
        "T_HP": str(best_params[2]),
        "residual_HP": mp.nstr(best_res, 20),
        "residual_HP_float": float(best_res),
        "verified_below_1e_15": bool(best_res < mp.mpf("1e-15")),
        "verified_below_1e_12": bool(best_res < mp.mpf("1e-12")),
        "verified_below_1e_10": bool(best_res < mp.mpf("1e-10")),
        "trace": trace,
        "wall_time_s": elapsed,
        "params": {"n_steps": n_steps, "order": order, "dps": dps, "max_iter": max_iter},
        "original": {"v1": v1, "v2": v2, "T": T},
    }
    out = OUT / f"hp_{name}.json"
    out.write_text(json.dumps(r, indent=2, default=str))
    print(f"[{name}] DONE residual={r['residual_HP']} time={elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
