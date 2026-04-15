"""Test hypothesis: Hristov 2025 catalog rows are (v1, v2, T) in our
Li-Liao Euler convention, NOT (x3, y3, T) in free-fall convention.

Take row 6 (#0006). Use (x, y) as (v1, v2) in our Euler setup. Integrate
for T. Closure should be ~1e-40 if hypothesis holds.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import heyoka as hy
import mpmath as mp

from check_C_D_vs_hristov_2025 import build_hp_integrator, load_hristov_entry


def test_euler_ic(v1_str, v2_str, T, label):
    ta = build_hp_integrator(200)
    v1 = mp.mpf(v1_str)
    v2 = mp.mpf(v2_str)

    q = [(mp.mpf(-1), mp.mpf(0)),
         (mp.mpf(1), mp.mpf(0)),
         (mp.mpf(0), mp.mpf(0))]
    v = [(v1, v2),
         (v1, v2),
         (-2*v1, -2*v2)]
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
    res = max(abs(float(ta.state[i] - ic[i])) for i in range(12))
    print(f"   {label}: wall={wall:.1f}s, closure={res:.3e}")
    return res


def main():
    # Test A: Hristov 2025 #0006 (x, y) as (v1, v2), T
    entry = load_hristov_entry(6)
    print("Test A: Interpret Hristov 2025 #0006 (x, y, T) as (v1, v2, T) in Euler conv")
    print(f"   x={mp.nstr(entry['x3'], 20)}, y={mp.nstr(entry['y3'], 20)}")
    test_euler_ic(str(entry["x3"]), str(entry["y3"]), entry["T"], "   (+v1, +v2)")
    test_euler_ic(str(entry["x3"]), "-" + str(entry["y3"]), entry["T"], "   (+v1, -v2)")
    test_euler_ic("-" + str(entry["x3"]), str(entry["y3"]), entry["T"], "   (-v1, +v2)")

    # Also test T_star
    print("Test B: Use T_star instead of T")
    test_euler_ic(str(entry["x3"]), "-" + str(entry["y3"]), entry["T_star"], "   T_star with (+v1, -v2)")


if __name__ == "__main__":
    main()
