"""Diagnostic: compare propagate_until vs propagate_grid for Hristov 2025 #0006 IC.

We suspect propagate_grid may be losing precision at 200-bit. Compare the
closure residual both ways.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import heyoka as hy
import mpmath as mp
import numpy as np

from check_C_D_vs_hristov_2025 import build_hp_integrator, load_hristov_entry


def build_ic(entry, prec=200):
    q = [(mp.mpf("-0.5"), mp.mpf("0")),
         (mp.mpf("0.5"),  mp.mpf("0")),
         (entry["x3"], entry["y3"])]
    v = [(mp.mpf("0"), mp.mpf("0"))] * 3
    ic = []
    for qi in q:
        ic += [hy.real(mp.nstr(qi[0], 50), prec=prec),
               hy.real(mp.nstr(qi[1], 50), prec=prec)]
    for vi in v:
        ic += [hy.real(mp.nstr(vi[0], 50), prec=prec),
               hy.real(mp.nstr(vi[1], 50), prec=prec)]
    return ic


def main():
    entry = load_hristov_entry(6)
    T = entry["T"]
    print(f"Hristov 2025 #0006: T = {mp.nstr(T, 30)}")

    # Test 1: propagate_until
    print("\n=== Test 1: propagate_until ===")
    ta1 = build_hp_integrator(200)
    ic1 = build_ic(entry)
    ta1.state[:] = ic1
    ta1.time = hy.real("0.0", prec=200)
    T_end = hy.real(mp.nstr(T, 50), prec=200)
    t0 = time.perf_counter()
    ta1.propagate_until(T_end)
    wall1 = time.perf_counter() - t0
    res1 = max(abs(float(ta1.state[i] - ic1[i])) for i in range(12))
    print(f"   wall: {wall1:.1f}s, closure: {res1:.3e}")

    # Test 2: propagate_grid, 50000 points
    print("\n=== Test 2: propagate_grid (50000 pts) ===")
    ta2 = build_hp_integrator(200)
    ic2 = build_ic(entry)
    ta2.state[:] = ic2
    ta2.time = hy.real("0.0", prec=200)
    n_grid = 50000
    t_grid_mp = [T * mp.mpf(i) / mp.mpf(n_grid - 1) for i in range(n_grid)]
    t_grid_real = [hy.real(mp.nstr(t, 50), prec=200) for t in t_grid_mp]
    t0 = time.perf_counter()
    pg_out = ta2.propagate_grid(t_grid_real)
    wall2 = time.perf_counter() - t0
    res2 = max(abs(float(ta2.state[i] - ic2[i])) for i in range(12))
    print(f"   wall: {wall2:.1f}s, closure (ta.state vs ic): {res2:.3e}")
    print(f"   ta.time: {ta2.time}")

    # Check last row of pg_out
    if isinstance(pg_out, tuple):
        results = pg_out[-1]
        print(f"   pg_out is tuple, len={len(pg_out)}, first elements: {pg_out[:-1]}")
    else:
        results = pg_out
    # Compare final grid row to IC
    final_row_diff = max(abs(float(results[-1][i]) - float(ic2[i])) for i in range(12))
    print(f"   final_grid_row - IC: {final_row_diff:.3e}")

    # Compare first grid row (at t=0) to IC — should be exactly IC
    first_row_diff = max(abs(float(results[0][i]) - float(ic2[i])) for i in range(12))
    print(f"   first_grid_row - IC: {first_row_diff:.3e}")

    # Look at position magnitudes on the grid to see if the integration blew up
    pos_norms = []
    for k in [0, n_grid // 4, n_grid // 2, 3 * n_grid // 4, n_grid - 1]:
        row = [float(results[k][i]) for i in range(12)]
        pos_max = max(abs(row[i]) for i in range(6))
        print(f"   row {k}: t={float(t_grid_mp[k]):.3f}, max|q|={pos_max:.3e}")


if __name__ == "__main__":
    main()
