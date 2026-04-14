"""Action #1 — Definitive identity check: orbits C, D ↔ Hristov 2025 #0006, #0011.

See DESIGN.md in this directory.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import mpmath as mp
import numpy as np

mp.mp.dps = 120  # 120 digits header for IO; heyoka integration uses 60

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
CAT_FILE = PROJECT_ROOT / "experiments/orbit_verification/00_catalogs/hristov_2025_stable.txt"

# Our orbit HP ICs from 06_high_precision/hp_heyoka_newton_results.json.
# We'll load exact strings from JSON in a later task; for now use frozen
# double-precision values for quick sanity checks.
OUR_ORBITS = {
    "C": {"v1": mp.mpf("0.2554309356506809"),
          "v2": mp.mpf("-0.516385839015133"),
          "T":  mp.mpf("35.043087021664284")},
    "D": {"v1": mp.mpf("0.5539389904823785"),
          "v2": mp.mpf("0.4619341006364459"),
          "T":  mp.mpf("81.08361216996745")},
}

HRISTOV_TARGETS = {
    "C": 6,   # Hristov 2025 row index (1-based) for #0006
    "D": 11,  # #0011
}


def load_hristov_entry(row_index_1based: int) -> dict:
    """Load Hristov 2025 entry at 1-based row index, preserving ~100-digit precision."""
    lines = CAT_FILE.read_text().splitlines()
    if row_index_1based < 1 or row_index_1based > len(lines):
        raise IndexError(f"Row {row_index_1based} out of range (have {len(lines)} lines).")
    parts = lines[row_index_1based - 1].split()
    if len(parts) < 4:
        raise ValueError(f"Row {row_index_1based}: expected 4 fields, got {len(parts)}.")
    x, y, T, T_star = [mp.mpf(p) for p in parts[:4]]
    return {
        "row": row_index_1based,
        "id": f"hristov2025_{row_index_1based:04d}",
        "x3": x, "y3": y, "T": T, "T_star": T_star,
    }


def main():
    for orbit_name, row in HRISTOV_TARGETS.items():
        entry = load_hristov_entry(row)
        print(f"[{orbit_name}] {entry['id']}:")
        print(f"   x3     = {mp.nstr(entry['x3'], 25)}")
        print(f"   y3     = {mp.nstr(entry['y3'], 25)}")
        print(f"   T      = {mp.nstr(entry['T'], 25)}")
        print(f"   T_star = {mp.nstr(entry['T_star'], 25)}")
        print(f"   our T  = {mp.nstr(OUR_ORBITS[orbit_name]['T'], 25)}")
        dT = abs(entry["T"] - OUR_ORBITS[orbit_name]["T"])
        print(f"   |ΔT|   = {mp.nstr(dT, 6)}")
        print()


if __name__ == "__main__":
    main()
