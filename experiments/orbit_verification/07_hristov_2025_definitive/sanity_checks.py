"""Sanity checks for the Hristov 2025 definitive identity check.

All must pass < 1e-30 absolute (target usually ~1e-40 or better) before trusting the main check.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import heyoka as hy
import mpmath as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_C_D_vs_hristov_2025 import build_hp_integrator, get_our_orbits

mp.mp.dps = 120


def _load_state(orbit_name: str):
    """Return 12-element list of heyoka real at 200-bit for orbit's Euler IC."""
    our_orbits = get_our_orbits()
    v1 = our_orbits[orbit_name]["v1"]
    v2 = our_orbits[orbit_name]["v2"]
    q = [(mp.mpf(-1), mp.mpf(0)),
         (mp.mpf(1),  mp.mpf(0)),
         (mp.mpf(0),  mp.mpf(0))]
    v = [(v1,      v2),
         (v1,      v2),
         (-2 * v1, -2 * v2)]
    ic = []
    for qi in q:
        ic += [hy.real(mp.nstr(qi[0], 50), prec=200),
               hy.real(mp.nstr(qi[1], 50), prec=200)]
    for vi in v:
        ic += [hy.real(mp.nstr(vi[0], 50), prec=200),
               hy.real(mp.nstr(vi[1], 50), prec=200)]
    return ic


def sanity_orbit_C_euler_roundtrip():
    print("=== Sanity 1: orbit C Euler-convention roundtrip ===")
    t0 = time.perf_counter()
    ta = build_hp_integrator(200)
    print(f"   integrator built in {time.perf_counter() - t0:.1f}s")

    ic = _load_state("C")
    ta.state[:] = ic
    ta.time = hy.real("0.0", prec=200)

    T_our = get_our_orbits()["C"]["T"]
    T_end = hy.real(mp.nstr(T_our, 50), prec=200)

    t0 = time.perf_counter()
    res = ta.propagate_until(T_end)
    wall = time.perf_counter() - t0
    print(f"   propagated T={float(T_our):.6f} in {wall:.0f}s")

    # Closure residual
    residuals = []
    for i in range(12):
        d = ta.state[i] - ic[i]
        residuals.append(abs(float(d)))
    residual = max(residuals)
    print(f"   max closure residual: {residual:.3e}")
    if residual > 1e-30:
        raise RuntimeError(f"Sanity 1 FAILED: residual {residual} > 1e-30")
    print(f"   PASS (< 1e-30).")
    return {"wall_s": wall, "residual": residual, "pass": True}


def sanity_hristov_00001_roundtrip():
    print("=== Sanity 2: Hristov 2024 #00001 free-fall roundtrip ===")
    # Hristov 2024 row 1 = Broucke-style figure-8-ish free-fall IC
    cat = Path(__file__).resolve().parents[1] / "00_catalogs/hristov_2024_sol_80.txt"
    line1 = cat.read_text().splitlines()[0].split()
    x3, y3, T = mp.mpf(line1[0]), mp.mpf(line1[1]), mp.mpf(line1[2])
    print(f"   IC: x3={mp.nstr(x3, 20)}, y3={mp.nstr(y3, 20)}, T={mp.nstr(T, 20)}")

    t0 = time.perf_counter()
    ta = build_hp_integrator(200)
    print(f"   integrator built in {time.perf_counter() - t0:.1f}s")

    q = [(mp.mpf("-0.5"), mp.mpf("0")),
         (mp.mpf("0.5"),  mp.mpf("0")),
         (x3, y3)]
    v = [(mp.mpf("0"), mp.mpf("0"))] * 3
    ic = []
    for qi in q:
        ic += [hy.real(mp.nstr(qi[0], 50), prec=200),
               hy.real(mp.nstr(qi[1], 50), prec=200)]
    for vi in v:
        ic += [hy.real(mp.nstr(vi[0], 50), prec=200),
               hy.real(mp.nstr(vi[1], 50), prec=200)]

    ta.state[:] = ic
    ta.time = hy.real("0.0", prec=200)
    T_end = hy.real(mp.nstr(T, 50), prec=200)

    t0 = time.perf_counter()
    ta.propagate_until(T_end)
    wall = time.perf_counter() - t0
    print(f"   propagated T={float(T):.6f} in {wall:.0f}s")

    residuals = [abs(float(ta.state[i] - ic[i])) for i in range(12)]
    residual = max(residuals)
    print(f"   max closure residual: {residual:.3e}")
    if residual > 1e-20:
        raise RuntimeError(f"Sanity 2 FAILED: residual {residual} > 1e-20")
    print(f"   PASS (< 1e-20).")
    return {"wall_s": wall, "residual": residual, "pass": True}


if __name__ == "__main__":
    results = {}
    for name, fn in [("sanity_1_orbit_C", sanity_orbit_C_euler_roundtrip),
                     ("sanity_2_hristov_00001", sanity_hristov_00001_roundtrip)]:
        try:
            results[name] = fn()
        except Exception as e:
            results[name] = {"pass": False, "error": str(e)}
    Path(__file__).parent.joinpath("sanity_results.json").write_text(
        json.dumps(results, default=str, indent=2))
    # Re-raise if any failed, after writing partial results
    failures = [k for k, v in results.items() if not v.get("pass")]
    if failures:
        raise SystemExit(f"Sanity failures: {failures}")
