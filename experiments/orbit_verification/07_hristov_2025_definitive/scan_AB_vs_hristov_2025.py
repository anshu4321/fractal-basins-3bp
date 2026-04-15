"""Scan all 971 Hristov 2025 stable-orbits rows for matches to orbits A, B.

Context: Tasks 6, 7 established that the Hristov 2025 catalog uses
Euler/Li-Liao convention (v1, v2, T, T*), not free-fall as the README
claims. Orbits C and D are identical to rows 6 and 11 at 50 digits.

This script runs the same comparison for orbits A and B against all
971 rows, under all 4 sign-reflection symmetries (v1, v2) -> (s1*v1, s2*v2).

A hit means:
  dv_min = min over (s1, s2) of sqrt((s1*h_v1 - a_v1)^2 + (s2*h_v2 - a_v2)^2)
           < 1e-4 (plausible match)
  AND |T_hristov - T_a| / T_a < 1e-6 (same period at identification precision)

Stricter "identical" gate: dv_min < 1e-10 AND |dT|/T < 1e-10.
"""
from __future__ import annotations

import json
from pathlib import Path
import mpmath as mp

mp.mp.dps = 60

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CAT = PROJECT_ROOT / "experiments/orbit_verification/00_catalogs/hristov_2025_stable.txt"
HP = PROJECT_ROOT / "experiments/orbit_verification/06_high_precision/hp_heyoka_newton_results.json"

# Load HP-refined ICs for A, B
hp_data = json.loads(HP.read_text())
orbits = {}
for entry in hp_data:
    name = entry.get("name")
    if name in ("A", "B"):
        orbits[name] = {
            "v1": mp.mpf(entry["v1_HP"]),
            "v2": mp.mpf(entry["v2_HP"]),
            "T":  mp.mpf(entry["T_HP"]),
        }

# Load all 971 Hristov 2025 rows
rows = []
for i, line in enumerate(CAT.read_text().splitlines(), 1):
    parts = line.split()
    if len(parts) < 4:
        continue
    rows.append({
        "id": f"hristov2025_{i:04d}",
        "row": i,
        "v1": mp.mpf(parts[0]),
        "v2": mp.mpf(parts[1]),
        "T":  mp.mpf(parts[2]),
        "Ts": mp.mpf(parts[3]),
    })

print(f"Loaded {len(rows)} Hristov 2025 rows.\n")

# Scan
for name in ("A", "B"):
    a = orbits[name]
    print(f"=== Orbit {name} (v1={mp.nstr(a['v1'],8)}, v2={mp.nstr(a['v2'],8)}, T={mp.nstr(a['T'],10)}) ===")

    best_dv = mp.mpf("inf")
    best_row = None
    hits = []  # rows with dv_min < 1e-4 AND |dT|/T < 1e-6

    for r in rows:
        # 4 sign symmetries
        dv_min = mp.mpf("inf")
        for s1 in (mp.mpf(1), mp.mpf(-1)):
            for s2 in (mp.mpf(1), mp.mpf(-1)):
                dv = mp.sqrt((s1 * r["v1"] - a["v1"])**2
                             + (s2 * r["v2"] - a["v2"])**2)
                if dv < dv_min:
                    dv_min = dv
        dT = abs(r["T"] - a["T"]) / a["T"]

        if dv_min < best_dv:
            best_dv = dv_min
            best_row = (r, dv_min, dT)

        if dv_min < mp.mpf("1e-4") and dT < mp.mpf("1e-6"):
            hits.append((r, dv_min, dT))

    # Report
    if hits:
        print(f"   {len(hits)} strict hit(s) (dv<1e-4 AND |dT|/T<1e-6):")
        for r, dv, dT in sorted(hits, key=lambda x: x[1])[:5]:
            verdict = "IDENTICAL" if dv < mp.mpf("1e-10") else "LIKELY"
            print(f"      {r['id']} (row {r['row']}): dv_min = {mp.nstr(dv, 5)}, "
                  f"|dT|/T = {mp.nstr(dT, 5)}  -> {verdict}")
    else:
        print(f"   NO strict hits. Best near-match:")
        r, dv, dT = best_row
        print(f"      {r['id']} (row {r['row']}): dv_min = {mp.nstr(dv, 5)}, "
              f"|dT|/T = {mp.nstr(dT, 5)}")
        # Also report nearest in T alone, in case signs are off
        def dT_rel(r): return abs(r["T"] - a["T"]) / a["T"]
        nearest_T = min(rows, key=dT_rel)
        print(f"   Nearest in T alone: {nearest_T['id']} "
              f"T={mp.nstr(nearest_T['T'], 10)}, |dT|/T={mp.nstr(dT_rel(nearest_T), 5)}")
    print()

print(f"DONE. Scanned {len(rows)} rows against orbits A, B.")
