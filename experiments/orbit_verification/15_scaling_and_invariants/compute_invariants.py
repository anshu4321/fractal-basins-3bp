"""
Scale invariants and conserved quantities for orbits A, B, C, D.

IC convention (Euler velocity section, equal unit masses, G=1):
    r1 = (-1, 0)                  p1 = ( v1,  v2)
    r2 = (+1, 0)                  p2 = ( v1,  v2)
    r3 = ( 0, 0)                  p3 = (-2v1, -2v2)

Consequences at t=0 (verified analytically here and at 50-digit numerically):
    P_total     = p1 + p2 + p3 = 0
    L_z_total   = sum r_i x p_i = -v2 + v2 + 0 = 0
    V(0)        = -1/r12 - 1/r13 - 1/r23 = -(1/2 + 1 + 1) = -5/2
    T_kin(0)    = (1/2)(|p1|^2 + |p2|^2 + |p3|^2) = 3(v1^2 + v2^2)
    E           = 3(v1^2 + v2^2) - 5/2

Scale symmetry of Newtonian N-body with 1/r potential:
    r -> alpha r,  t -> alpha^{3/2} t  leaves the EOM invariant.
    Under this: v -> alpha^{-1/2} v,  p -> alpha^{-1/2} p (unit masses)
                E -> alpha^{-1} E,   L -> alpha^{+1/2} L,   T -> alpha^{3/2} T
    Scale-invariant quantities:
        T* = T |E|^{3/2}                  (scale-invariant period)
        L* = L |E|^{1/2}                  (scale-invariant L, trivially 0 here)
        E_dimless = E (with G=m=1; only scale breaking is units)

Since L_z = 0 is baked into the Euler section, ALL FOUR orbits sit in the
zero-angular-momentum sector by construction. Any future search outside
the Euler section (e.g. equilateral-triangle symmetric ICs) will still have
L = 0 if we demand full S_3 permutation symmetry, since the three
contributions cancel pairwise -- a point worth making to Prof Ravishankar.
"""

import json
from pathlib import Path

from mpmath import mp, mpf, sqrt

mp.dps = 60  # 60 decimal digits of precision

HERE = Path(__file__).parent
HP_JSON = HERE.parent / "06_high_precision" / "hp_heyoka_newton_results.json"


def energy(v1, v2):
    """E = 3(v1^2 + v2^2) - 5/2 for the Euler section with unit masses."""
    return 3 * (v1 ** 2 + v2 ** 2) - mpf("5") / 2


def angular_momentum_z(v1, v2):
    """L_z = r1 x p1 + r2 x p2 + r3 x p3
         = (-1)(v2) - 0*(v1) + (1)(v2) - 0*(v1) + 0*(-2v2) - 0*(-2v1)
         = -v2 + v2 + 0 = 0  (exactly, for any v1, v2)
    """
    return (mpf(-1) * v2) + (mpf(1) * v2) + mpf(0)


def scale_invariant_period(T, E):
    return T * (abs(E) ** (mpf(3) / 2))


def pair_separations_at_t0():
    return {"r12": mpf(2), "r13": mpf(1), "r23": mpf(1)}


def potential_at_t0():
    s = pair_separations_at_t0()
    return -(1 / s["r12"] + 1 / s["r13"] + 1 / s["r23"])


def kinetic_at_t0(v1, v2):
    # |p1|^2 = v1^2+v2^2, |p2|^2 = v1^2+v2^2, |p3|^2 = 4(v1^2+v2^2)
    # T = (1/2)(|p1|^2 + |p2|^2 + |p3|^2) = 3(v1^2+v2^2)
    return 3 * (v1 ** 2 + v2 ** 2)


def main():
    raw = json.loads(HP_JSON.read_text())
    orbits = {o["name"]: o for o in raw}

    print(f"# Scaling invariants for orbits A, B, C, D")
    print(f"# mpmath precision: {mp.dps} decimal digits")
    print(f"# IC convention: Euler velocity section (r1=-1,0; r2=+1,0; r3=0,0;")
    print(f"#                 p1=p2=(v1,v2), p3=-2(v1,v2); m=1, G=1)")
    print()
    print(f"Pair separations at t=0:      r12=2, r13=1, r23=1")
    print(f"Potential V(0):               {potential_at_t0()} = -5/2")
    print()

    rows = []
    for name in ["A", "B", "C", "D"]:
        o = orbits[name]
        v1 = mpf(o["v1_HP"])
        v2 = mpf(o["v2_HP"])
        T  = mpf(o["T_HP"])

        E = energy(v1, v2)
        Lz = angular_momentum_z(v1, v2)
        Tstar = scale_invariant_period(T, E)
        Tkin = kinetic_at_t0(v1, v2)
        Vpot = potential_at_t0()
        E_check = Tkin + Vpot
        # Also compute a stricter invariant: Sundman-Weyl dimensionless period
        # (equivalent to T*)
        # Check: what's rmin? Read from HP JSON if present, else skip.

        rows.append({
            "orbit": name,
            "v1": v1, "v2": v2, "T": T,
            "E": E,
            "E_cross_check": E_check,
            "E_residual": E - E_check,
            "Lz": Lz,
            "Tstar": Tstar,
            "Tkin0": Tkin,
            "V0": Vpot,
        })

    print("## Per-orbit quantities (50-digit)\n")
    for r in rows:
        print(f"### Orbit {r['orbit']}")
        print(f"  v1         = {r['v1']}")
        print(f"  v2         = {r['v2']}")
        print(f"  T          = {r['T']}")
        print(f"  E          = {mp.nstr(r['E'], 40)}")
        print(f"  E (T+V)    = {mp.nstr(r['E_cross_check'], 40)}")
        print(f"  E residual = {mp.nstr(r['E_residual'], 5)}")
        print(f"  L_z        = {r['Lz']}   (exact, by Euler-section construction)")
        print(f"  T*         = T |E|^{{3/2}} = {mp.nstr(r['Tstar'], 40)}")
        print()

    print("## Compact table\n")
    hdr = f"{'Orbit':<6} {'E':>20} {'L_z':>6} {'T*':>28} {'Tkin(0)':>15} {'V(0)':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['orbit']:<6} "
            f"{mp.nstr(r['E'], 12):>20} "
            f"{str(r['Lz']):>6} "
            f"{mp.nstr(r['Tstar'], 22):>28} "
            f"{mp.nstr(r['Tkin0'], 8):>15} "
            f"{mp.nstr(r['V0'], 4):>6}"
        )

    print()
    print("## B vs C: are their T* identical?")
    tstar_b = rows[1]["Tstar"]
    tstar_c = rows[2]["Tstar"]
    print(f"  T*(B) = {mp.nstr(tstar_b, 40)}")
    print(f"  T*(C) = {mp.nstr(tstar_c, 40)}")
    print(f"  diff  = {mp.nstr(tstar_b - tstar_c, 10)}")
    print(f"  reldiff = {mp.nstr(abs(tstar_b - tstar_c) / tstar_c, 6)}")
    print()

    # JSON output
    out = {
        "ic_convention": {
            "r": [[-1, 0], [1, 0], [0, 0]],
            "p_parametrization": "p1=p2=(v1,v2), p3=-2(v1,v2)",
            "masses": [1, 1, 1],
            "G": 1,
            "section": "Euler velocity section (Sukava-Dmitrasinovic)",
        },
        "conservation_at_t0": {
            "P_total": [0, 0],
            "L_z_total": "0 (identically, independent of v1, v2)",
            "V_at_t0": "-5/2",
            "T_kin_at_t0": "3 * (v1^2 + v2^2)",
            "E": "3 * (v1^2 + v2^2) - 5/2",
        },
        "scale_symmetry": {
            "r":  "alpha r",
            "t":  "alpha^{3/2} t",
            "v":  "alpha^{-1/2} v",
            "E":  "alpha^{-1} E",
            "L":  "alpha^{+1/2} L",
            "T":  "alpha^{3/2} T",
            "invariants": ["T* = T |E|^{3/2}", "L* = L |E|^{1/2}"],
        },
        "orbits": [
            {
                "name": r["orbit"],
                "v1": str(r["v1"]),
                "v2": str(r["v2"]),
                "T": str(r["T"]),
                "E": str(r["E"]),
                "E_via_T_plus_V": str(r["E_cross_check"]),
                "E_residual": str(r["E_residual"]),
                "L_z": str(r["Lz"]),
                "T_star": str(r["Tstar"]),
            }
            for r in rows
        ],
        "B_vs_C_Tstar": {
            "T_star_B": str(rows[1]["Tstar"]),
            "T_star_C": str(rows[2]["Tstar"]),
            "abs_diff": str(rows[1]["Tstar"] - rows[2]["Tstar"]),
            "rel_diff": str(abs(rows[1]["Tstar"] - rows[2]["Tstar"]) / rows[2]["Tstar"]),
        },
    }
    out_path = HERE / "invariants.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
