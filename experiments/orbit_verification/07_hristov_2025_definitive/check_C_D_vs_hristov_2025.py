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


def canonical_euler_transform(q: np.ndarray, p: np.ndarray,
                              tol_midpoint: float = 1e-6) -> tuple:
    """Transform a 3-body Euler-section state to the canonical (v1, v2).

    Expects q shape (3, 2), p shape (3, 2), unit masses. Identifies the
    body at the midpoint of the other two, translates+rotates+scales so
    outer bodies are at (+-1, 0) and midpoint body at origin, and returns
    the canonical v1 = p[outer1, 0], v2 = p[outer1, 1] after transform.

    Returns (v1, v2, aux_dict). Raises ValueError if no body is at the
    midpoint within tol_midpoint.
    """
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)

    # Find the midpoint body
    midpoint_body = -1
    best_dist = np.inf
    for m in range(3):
        others = [i for i in range(3) if i != m]
        mid = 0.5 * (q[others[0]] + q[others[1]])
        d_others = np.linalg.norm(q[others[0]] - q[others[1]])
        if d_others < 1e-12:
            continue
        dist_to_mid = np.linalg.norm(q[m] - mid) / d_others
        if dist_to_mid < best_dist:
            best_dist = dist_to_mid
            midpoint_body = m
    if midpoint_body < 0 or best_dist > tol_midpoint:
        raise ValueError(
            f"No body at the midpoint within {tol_midpoint}. "
            f"Best: body {midpoint_body}, scaled dist = {best_dist}")

    # Translate to put midpoint body at origin
    origin = q[midpoint_body].copy()
    q_shift = q - origin  # p is unchanged under translation

    # Identify outer bodies in a deterministic order:
    # "outer1" = the one with the smaller x after rotation, = body at (-1, 0).
    others = [i for i in range(3) if i != midpoint_body]
    # Half-binary distance
    d_half = 0.5 * np.linalg.norm(q_shift[others[0]] - q_shift[others[1]])

    # Rotation: align the outer bodies with the x-axis.
    # The axis vector points from outer0 to outer1.
    axis = q_shift[others[1]] - q_shift[others[0]]
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        raise ValueError("Outer bodies coincide.")
    cos_th = axis[0] / axis_norm
    sin_th = axis[1] / axis_norm
    # Rotation by -theta (to align axis with +x)
    R = np.array([[cos_th, sin_th], [-sin_th, cos_th]])

    q_rot = q_shift @ R.T
    p_rot = p @ R.T

    # After rotation: outer0 should be at (-d_half, 0), outer1 at (+d_half, 0).
    # Scale by 1/d_half so outer bodies are at (+-1, 0).
    q_canonical = q_rot / d_half
    # Velocity scaling under r -> r/d_half, t -> t/d_half^{3/2}:
    # v = dr/dt scales as (1/d_half) / (1/d_half^{1.5}) = sqrt(d_half)
    # So v_canonical = v_rot * sqrt(d_half). With unit mass, p == v.
    p_canonical = p_rot * np.sqrt(d_half)

    # Determine which outer is "body 1" (at -1) — the one with x = -1.
    # After rotation: outer0 had position with dot(axis, outer0) < 0 relative
    # to midpoint, so outer0 is at -d_half after rotation. Confirm:
    x_outer0 = q_canonical[others[0], 0]
    if x_outer0 > 0:
        # outer1 is at -1; swap.
        outer_minus_idx = others[1]
    else:
        outer_minus_idx = others[0]

    v1 = float(p_canonical[outer_minus_idx, 0])
    v2 = float(p_canonical[outer_minus_idx, 1])

    return v1, v2, {
        "midpoint_body": int(midpoint_body),
        "outer_minus_idx": int(outer_minus_idx),
        "half_binary_d": float(d_half),
        "scale_factor_to_canonical": float(1.0 / d_half),
        "midpoint_residual": float(best_dist),
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
