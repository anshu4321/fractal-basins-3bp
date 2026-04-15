"""Diagnostic: Try integrating Hristov 2025 #0006 for T/2, T, T_star/2, T_star
to see which gives closure.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import heyoka as hy
import mpmath as mp

from check_C_D_vs_hristov_2025 import build_hp_integrator, load_hristov_entry


def test_t(ta, ic, label, T_test):
    ta.state[:] = ic
    ta.time = hy.real("0.0", prec=200)
    T_end = hy.real(mp.nstr(T_test, 50), prec=200)
    t0 = time.perf_counter()
    ta.propagate_until(T_end)
    wall = time.perf_counter() - t0
    res = max(abs(float(ta.state[i] - ic[i])) for i in range(12))
    # Also "halfway closure": after T, is the config a reflection of IC?
    # Check: if all 12 coords match IC up to sign, or match (q, -v, -v, q), etc.
    print(f"   {label}: T={float(T_test):.6f}, wall={wall:.1f}s, closure={res:.3e}")
    return res


def main():
    entry = load_hristov_entry(6)
    T = entry["T"]
    Tstar = entry["T_star"]

    q = [(mp.mpf("-0.5"), mp.mpf("0")),
         (mp.mpf("0.5"),  mp.mpf("0")),
         (entry["x3"], entry["y3"])]
    v = [(mp.mpf("0"), mp.mpf("0"))] * 3
    ic = []
    for qi in q:
        ic += [hy.real(mp.nstr(qi[0], 50), prec=200),
               hy.real(mp.nstr(qi[1], 50), prec=200)]
    for vi in v:
        ic += [hy.real(mp.nstr(vi[0], 50), prec=200),
               hy.real(mp.nstr(vi[1], 50), prec=200)]

    ta = build_hp_integrator(200)

    print(f"Hristov 2025 #0006:")
    print(f"   T      = {mp.nstr(T, 30)}")
    print(f"   T_star = {mp.nstr(Tstar, 30)}")

    print("\nTesting various integration times:")
    test_t(ta, ic, "T      (catalog)", T)
    test_t(ta, ic, "T/2    (half)   ", T / mp.mpf(2))
    test_t(ta, ic, "T_star          ", Tstar)
    test_t(ta, ic, "T_star/2        ", Tstar / mp.mpf(2))
    test_t(ta, ic, "2*T             ", T * mp.mpf(2))


if __name__ == "__main__":
    main()
