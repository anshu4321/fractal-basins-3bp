"""Action #1 — Definitive identity check: orbits C, D ↔ Hristov 2025 #0006, #0011.

See DESIGN.md in this directory.
"""
from __future__ import annotations

import functools
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

@functools.lru_cache(maxsize=1)
def _load_our_orbits():
    """Load HP-refined IC strings for each orbit present at 50-digit precision.

    Reads 06_high_precision/hp_heyoka_newton_results.json (list of entries
    with keys {name, v1_HP, v2_HP, T_HP, ...}) and returns a dict keyed by
    orbit name. Any orbit missing the required HP string fields is skipped
    so future tasks can add orbits (A, B, ...) without breaking callers.
    """
    src = PROJECT_ROOT / "experiments/orbit_verification/06_high_precision/hp_heyoka_newton_results.json"
    data = json.loads(src.read_text())
    out = {}
    for r in data:
        name = r.get("name")
        if name is None:
            continue
        try:
            out[name] = {
                "v1": mp.mpf(r["v1_HP"]),
                "v2": mp.mpf(r["v2_HP"]),
                "T":  mp.mpf(r["T_HP"]),
            }
        except KeyError:
            # Orbit entry present but HP strings missing — skip gracefully.
            continue
    if not out:
        raise RuntimeError(f"No HP orbits loaded from {src}; run 06_high_precision first")
    return out


def get_our_orbits():
    """Accessor for HP-refined orbit ICs (lazy-loaded, cached)."""
    return _load_our_orbits()

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


def build_hp_integrator(dps_bits: int = 200):
    """Build a heyoka.taylor_adaptive in mpfr (real) mode for the planar
    equal-mass 3-body problem with unit masses and G=1.

    State layout (12 vars):
      q0x, q0y, q1x, q1y, q2x, q2y, v0x, v0y, v1x, v1y, v2x, v2y
    """
    import heyoka as hy
    (q0x, q0y, q1x, q1y, q2x, q2y,
     v0x, v0y, v1x, v1y, v2x, v2y) = hy.make_vars(
        "q0x", "q0y", "q1x", "q1y", "q2x", "q2y",
        "v0x", "v0y", "v1x", "v1y", "v2x", "v2y",
    )

    def r3(ax, ay, bx, by):
        dx = bx - ax
        dy = by - ay
        r2 = dx * dx + dy * dy
        return r2 ** 1.5

    r01 = r3(q0x, q0y, q1x, q1y)
    r02 = r3(q0x, q0y, q2x, q2y)
    r12 = r3(q1x, q1y, q2x, q2y)

    # a_i = sum_j (q_j - q_i) / r_ij^3
    a0x = (q1x - q0x) / r01 + (q2x - q0x) / r02
    a0y = (q1y - q0y) / r01 + (q2y - q0y) / r02
    a1x = (q0x - q1x) / r01 + (q2x - q1x) / r12
    a1y = (q0y - q1y) / r01 + (q2y - q1y) / r12
    a2x = (q0x - q2x) / r02 + (q1x - q2x) / r12
    a2y = (q0y - q2y) / r02 + (q1y - q2y) / r12

    rhs = [
        (q0x, v0x), (q0y, v0y),
        (q1x, v1x), (q1y, v1y),
        (q2x, v2x), (q2y, v2y),
        (v0x, a0x), (v0y, a0y),
        (v1x, a1x), (v1y, a1y),
        (v2x, a2x), (v2y, a2y),
    ]

    # Dummy IC (zeros) for compile; will be overwritten per-run.
    ic = [hy.real("0.0", prec=dps_bits)] * 12

    ta = hy.taylor_adaptive(
        rhs,
        ic,
        fp_type=hy.real,
        prec=dps_bits,
        tol=hy.real("1e-50", prec=dps_bits),
        compact_mode=True,
    )
    return ta


def _smoketest():
    """Build the integrator and run a 1-second check it doesn't crash."""
    import heyoka as hy
    print("Building HP integrator (200-bit)...")
    t0 = time.perf_counter()
    ta = build_hp_integrator(dps_bits=200)
    print(f"   built in {time.perf_counter() - t0:.1f}s")

    # Free-fall IC for Hristov 2025 #0006
    entry = load_hristov_entry(6)
    ic_q = [(-mp.mpf("0.5"), mp.mpf("0")),
            (mp.mpf("0.5"), mp.mpf("0")),
            (entry["x3"], entry["y3"])]
    ic_v = [(mp.mpf("0"), mp.mpf("0"))] * 3
    ic = []
    for q in ic_q:
        ic += [hy.real(mp.nstr(q[0], 50), prec=200),
               hy.real(mp.nstr(q[1], 50), prec=200)]
    for v in ic_v:
        ic += [hy.real(mp.nstr(v[0], 50), prec=200),
               hy.real(mp.nstr(v[1], 50), prec=200)]
    ta.state[:] = ic
    ta.time = hy.real("0.0", prec=200)
    t_end = hy.real("0.01", prec=200)
    t0 = time.perf_counter()
    res = ta.propagate_until(t_end)
    print(f"   stepped to t=0.01 in {time.perf_counter() - t0:.2f}s, status={res}")
    print(f"   final state[0]: {ta.state[0]}")


def main():
    our_orbits = get_our_orbits()
    for orbit_name, row in HRISTOV_TARGETS.items():
        entry = load_hristov_entry(row)
        print(f"[{orbit_name}] {entry['id']}:")
        print(f"   x3     = {mp.nstr(entry['x3'], 25)}")
        print(f"   y3     = {mp.nstr(entry['y3'], 25)}")
        print(f"   T      = {mp.nstr(entry['T'], 25)}")
        print(f"   T_star = {mp.nstr(entry['T_star'], 25)}")
        print(f"   our T  = {mp.nstr(our_orbits[orbit_name]['T'], 25)}")
        dT = abs(entry["T"] - our_orbits[orbit_name]["T"])
        print(f"   |ΔT|   = {mp.nstr(dT, 6)}")
        print()


if __name__ == "__main__":
    if "--smoketest" in sys.argv:
        _smoketest()
    else:
        main()
