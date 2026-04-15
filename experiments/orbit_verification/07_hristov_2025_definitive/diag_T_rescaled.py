"""Check if Hristov 2025 entries need energy-rescaling to close.

Since catalog T doesn't close the orbit in m=1, G=1 raw units, test if
T_raw = T_catalog / |E|^(3/2) or T_raw = T_catalog * |E|^(3/2) closes.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import heyoka as hy
import mpmath as mp

from check_C_D_vs_hristov_2025 import build_hp_integrator, load_hristov_entry


def compute_E(entry):
    x3, y3 = entry["x3"], entry["y3"]
    # q1=(-0.5,0), q2=(+0.5,0), q3=(x3,y3); v=0 for all; m=1 each
    PE = mp.mpf(0)
    q = [(-mp.mpf("0.5"), mp.mpf(0)), (mp.mpf("0.5"), mp.mpf(0)), (x3, y3)]
    for i in range(3):
        for j in range(i+1, 3):
            dx = q[i][0] - q[j][0]
            dy = q[i][1] - q[j][1]
            r = mp.sqrt(dx*dx + dy*dy)
            PE -= 1 / r
    return PE  # KE is zero at free-fall IC


def main():
    entry = load_hristov_entry(6)
    T = entry["T"]
    Tstar = entry["T_star"]
    E = compute_E(entry)
    print(f"Hristov 2025 #0006:")
    print(f"   E (raw, m=1, G=1) = {mp.nstr(E, 30)}")
    print(f"   |E|^(3/2)         = {mp.nstr(mp.fabs(E)**mp.mpf(1.5), 30)}")
    print(f"   T      (cat)      = {mp.nstr(T, 30)}")
    print(f"   T_star (cat)      = {mp.nstr(Tstar, 30)}")
    print(f"   T / |E|^(3/2)     = {mp.nstr(T / mp.fabs(E)**mp.mpf(1.5), 30)}")
    print(f"   T * |E|^(3/2)     = {mp.nstr(T * mp.fabs(E)**mp.mpf(1.5), 30)}")
    print(f"   T_star / |E|^(3/2)= {mp.nstr(Tstar / mp.fabs(E)**mp.mpf(1.5), 30)}")
    print(f"   T_star * |E|^(3/2)= {mp.nstr(Tstar * mp.fabs(E)**mp.mpf(1.5), 30)}")
    print()

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

    def test(label, Tval):
        ta.state[:] = ic
        ta.time = hy.real("0.0", prec=200)
        T_end = hy.real(mp.nstr(Tval, 50), prec=200)
        t0 = time.perf_counter()
        ta.propagate_until(T_end)
        wall = time.perf_counter() - t0
        res = max(abs(float(ta.state[i] - ic[i])) for i in range(12))
        print(f"   {label}: T={float(Tval):.6f}, wall={wall:.1f}s, closure={res:.3e}")
        return res

    print("Testing integration times:")
    test("T/|E|^(3/2)", T / mp.fabs(E)**mp.mpf(1.5))
    test("T*|E|^(3/2)", T * mp.fabs(E)**mp.mpf(1.5))
    test("T_star/|E|^(3/2)", Tstar / mp.fabs(E)**mp.mpf(1.5))

    # Another possibility: Hristov uses G*M = 1 with M=3, so G=1/3
    # Then their time units are scaled as sqrt(M) = sqrt(3)
    # Or: maybe Hristov uses reduced mass units.
    # Test simple ratios
    test("T/sqrt(2)", T / mp.sqrt(mp.mpf(2)))
    test("T*sqrt(2)", T * mp.sqrt(mp.mpf(2)))
    test("T/sqrt(3)", T / mp.sqrt(mp.mpf(3)))
    test("T*sqrt(3)", T * mp.sqrt(mp.mpf(3)))


if __name__ == "__main__":
    main()
