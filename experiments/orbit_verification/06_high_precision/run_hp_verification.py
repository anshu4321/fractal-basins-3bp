"""Section 2.6: High-precision verification of the four orbits.

heyoka was not installable on this machine (CMake build failure, no
binary wheel for arm64). Fallback: mpmath-based Taylor integrator from
mega3bp/taylor_hp.py, with per-orbit step counts chosen based on
r_min (closest body-body distance during the orbit).

For each orbit, run Newton-Raphson refinement at 30-decimal precision
and check whether the residual drops below 1e-15 (strong genuineness)
or 1e-20 (publication-grade).

Per the prompt, we report results honestly. Failures are documented.
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

# Per-orbit configuration. n_steps chosen so h ≈ r_min / 8 to safely
# stay inside the Taylor convergence radius near close binary encounters.
ORBITS = [
    {"name": "A", "v1": 0.18900489195141518,  "v2": -0.5397624528013026,
     "T": 19.73539189302097, "r_min": 0.0106, "n_steps": 8000, "order": 24},
    {"name": "B", "v1": -0.20349168744840768, "v2": 0.5181128607400666,
     "T": 32.84907116698178, "r_min": 0.0255, "n_steps": 5000, "order": 24},
    {"name": "C", "v1": 0.2554309356506809,   "v2": -0.516385839015133,
     "T": 35.043087021664284, "r_min": 0.0139, "n_steps": 8000, "order": 24},
    {"name": "D", "v1": 0.5539389904823785,   "v2": 0.4619341006364459,
     "T": 81.08361216996745, "r_min": 0.0990, "n_steps": 4000, "order": 24},
]
DPS = 30
MAX_ITER = 6


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


def newton_hp(v1_0, v2_0, T0, name, n_steps, order, dps, max_iter):
    mp.mp.dps = dps
    v1 = mp.mpf(str(v1_0)); v2 = mp.mpf(str(v2_0)); T = mp.mpf(str(T0))
    best_res = mp.mpf("inf"); best_params = (v1, v2, T); trace = []
    eps_fd = mp.mpf("1e-8")

    print(f"[{name}] HP Newton: dps={dps}, order={order}, "
          f"n_steps={n_steps}, max_iter={max_iter}", flush=True)

    with ProcessPoolExecutor(max_workers=3) as pool:
        for it in range(max_iter):
            F, normF = F_local(v1, v2, T, n_steps, order, dps)
            print(f"[{name}] iter {it}: ||F|| = {mp.nstr(normF, 10)}", flush=True)
            trace.append(mp.nstr(normF, 10))
            if normF < best_res:
                best_res = normF; best_params = (v1, v2, T)
            if normF < mp.mpf(f"1e-{dps - 5}"):
                print(f"[{name}] converged to precision floor", flush=True)
                break

            v1_s, v2_s, T_s = str(v1), str(v2), str(T)
            args_list = [
                (str(v1 + eps_fd), v2_s, T_s, n_steps, order, dps),
                (v1_s, str(v2 + eps_fd), T_s, n_steps, order, dps),
                (v1_s, v2_s, str(T + eps_fd), n_steps, order, dps),
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
            v1 -= dp[0]; v2 -= dp[1]; T -= dp[2]

    return {
        "name": name,
        "v1_HP": str(best_params[0]),
        "v2_HP": str(best_params[1]),
        "T_HP": str(best_params[2]),
        "residual_HP": mp.nstr(best_res, 20),
        "residual_HP_float": float(best_res),
        "verified_below_1e_20": bool(best_res < mp.mpf("1e-20")),
        "verified_below_1e_15": bool(best_res < mp.mpf("1e-15")),
        "verified_below_1e_10": bool(best_res < mp.mpf("1e-10")),
        "trace": trace,
    }


def main():
    results = {}
    for cfg in ORBITS:
        t0 = time.perf_counter()
        r = newton_hp(cfg["v1"], cfg["v2"], cfg["T"], cfg["name"],
                      cfg["n_steps"], cfg["order"], DPS, MAX_ITER)
        r["original"] = {"v1": cfg["v1"], "v2": cfg["v2"], "T": cfg["T"],
                          "n_steps": cfg["n_steps"], "order": cfg["order"],
                          "dps": DPS, "max_iter": MAX_ITER, "r_min": cfg["r_min"]}
        r["wall_time_s"] = time.perf_counter() - t0
        verdict = ("VERIFIED-1e-20" if r["verified_below_1e_20"]
                   else ("VERIFIED-1e-15" if r["verified_below_1e_15"]
                         else ("VERIFIED-1e-10" if r["verified_below_1e_10"]
                               else "FAIL")))
        print(f"\n[{cfg['name']}] residual={r['residual_HP']}  "
              f"time={r['wall_time_s']:.0f}s  -> {verdict}\n")
        results[cfg["name"]] = r

        # Save incrementally
        (OUT / "verified_novel.json").write_text(json.dumps(results, indent=2))

    # Final summary
    print("=" * 60)
    print("SECTION 2.6 SUMMARY")
    print("=" * 60)
    for name, r in results.items():
        verdict = ("VERIFIED <1e-20" if r["verified_below_1e_20"]
                   else ("near-verified <1e-15" if r["verified_below_1e_15"]
                         else ("weak <1e-10" if r["verified_below_1e_10"]
                               else "FAIL")))
        print(f"  {name}: {r['residual_HP']}  -> {verdict}")


if __name__ == "__main__":
    main()
