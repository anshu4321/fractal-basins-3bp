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
        # r2 * hy.sqrt(r2) is faster than r2 ** 1.5 at 200-bit (avoids exp/log).
        return r2 * hy.sqrt(r2)

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


def find_euler_crossings_from_grid(q_grid: np.ndarray, p_grid: np.ndarray,
                                   t_grid: np.ndarray,
                                   tol_midpoint: float = 1e-4):
    """Scan dense trajectory samples for Euler-section crossings.

    q_grid, p_grid: shape (N, 3, 2) float64.
    Returns list of dicts: each describes a crossing (index, t, score, ...).
    """
    N = q_grid.shape[0]
    crossings = []
    # Area of triangle (signed). Zero when collinear.
    def signed_area(Q):
        a = Q[1] - Q[0]
        b = Q[2] - Q[0]
        return a[0] * b[1] - a[1] * b[0]

    areas = np.array([signed_area(q_grid[k]) for k in range(N)])

    # Locate zero-crossings of `areas` (sign changes).
    sign_changes = []
    for k in range(N - 1):
        if areas[k] == 0 or (areas[k] > 0 and areas[k + 1] < 0) or (areas[k] < 0 and areas[k + 1] > 0):
            # Refine: linear interpolation in t
            a0, a1 = areas[k], areas[k + 1]
            if a1 == a0:
                alpha = 0.0
            else:
                alpha = a0 / (a0 - a1)
            alpha = max(0.0, min(1.0, alpha))
            t_cross = t_grid[k] + alpha * (t_grid[k + 1] - t_grid[k])
            q_cross = q_grid[k] + alpha * (q_grid[k + 1] - q_grid[k])
            p_cross = p_grid[k] + alpha * (p_grid[k + 1] - p_grid[k])
            sign_changes.append((k, alpha, t_cross, q_cross, p_cross))

    for k, alpha, t, q, p in sign_changes:
        # Check for midpoint body
        midpoint_body = -1
        best_mid_dist = np.inf
        for m in range(3):
            others = [i for i in range(3) if i != m]
            mid = 0.5 * (q[others[0]] + q[others[1]])
            d_others = np.linalg.norm(q[others[0]] - q[others[1]])
            if d_others < 1e-12:
                continue
            d_to_mid = np.linalg.norm(q[m] - mid) / d_others
            if d_to_mid < best_mid_dist:
                best_mid_dist = d_to_mid
                midpoint_body = m
        is_euler = best_mid_dist < tol_midpoint
        crossings.append({
            "step_index": k, "alpha": float(alpha), "t": float(t),
            "area_residual_at_k": float(areas[k]),
            "midpoint_body": int(midpoint_body),
            "midpoint_scaled_dist": float(best_mid_dist),
            "is_euler": bool(is_euler),
            "q": q.tolist(), "p": p.tolist(),
        })
    return crossings


def compare_to_our_orbit(v1_match: float, v2_match: float,
                         v1_ours: float, v2_ours: float) -> dict:
    """Compute min dv under the 4 sign-reflection symmetries."""
    candidates = []
    for s1, s2 in [(1, 1), (-1, -1), (1, -1), (-1, 1)]:
        dv = ((s1 * v1_match - v1_ours)**2
              + (s2 * v2_match - v2_ours)**2)**0.5
        candidates.append((dv, s1, s2))
    candidates.sort()
    return {"dv_min": candidates[0][0],
            "sign_s1": candidates[0][1],
            "sign_s2": candidates[0][2],
            "all_candidates": [(c[0], c[1], c[2]) for c in candidates]}


def verify_hristov_entry(orbit_name: str, row: int,
                         n_grid: int = 50000) -> dict:
    """Main verification for one Hristov 2025 entry vs our orbit."""
    import heyoka as hy
    entry = load_hristov_entry(row)
    print(f"\n=== Main: orbit {orbit_name} vs {entry['id']} ===")

    ta = build_hp_integrator(200)

    # Build IC as heyoka real values
    q = [(mp.mpf("-0.5"), mp.mpf("0")),
         (mp.mpf("0.5"),  mp.mpf("0")),
         (entry["x3"], entry["y3"])]
    v = [(mp.mpf("0"), mp.mpf("0"))] * 3
    ic_real = []
    for qi in q:
        ic_real += [hy.real(mp.nstr(qi[0], 50), prec=200),
                    hy.real(mp.nstr(qi[1], 50), prec=200)]
    for vi in v:
        ic_real += [hy.real(mp.nstr(vi[0], 50), prec=200),
                    hy.real(mp.nstr(vi[1], 50), prec=200)]
    ta.state[:] = ic_real
    ta.time = hy.real("0.0", prec=200)

    T = entry["T"]
    T_end = hy.real(mp.nstr(T, 50), prec=200)

    # Dense output at n_grid uniformly spaced times.
    t_grid_mp = [T * mp.mpf(i) / mp.mpf(n_grid - 1) for i in range(n_grid)]
    t_grid_real = [hy.real(mp.nstr(t, 50), prec=200) for t in t_grid_mp]

    t0 = time.perf_counter()
    pg_out = ta.propagate_grid(t_grid_real)
    wall = time.perf_counter() - t0
    print(f"   propagated {n_grid}-grid to T={float(T):.6f} in {wall:.0f}s")

    # heyoka's propagate_grid API has varied across versions. It may return:
    #   - a bare numpy array of shape (n_grid, 12), OR
    #   - a tuple (status, min_h, max_h, nsteps, output) or similar.
    # Unwrap defensively.
    if isinstance(pg_out, tuple):
        # Output array is the last element in all variants we've seen.
        results = pg_out[-1]
    else:
        results = pg_out

    # results shape: (n_grid, 12). Convert to float64 for post-processing.
    state_grid = np.asarray([[float(results[i][j]) for j in range(12)]
                             for i in range(n_grid)])
    q_grid = state_grid[:, :6].reshape(n_grid, 3, 2)
    p_grid = state_grid[:, 6:].reshape(n_grid, 3, 2)
    t_grid = np.asarray([float(t) for t in t_grid_mp])

    # Closure check
    residuals = [abs(float(ta.state[i] - ic_real[i])) for i in range(12)]
    closure = max(residuals)
    print(f"   closure residual: {closure:.3e}")
    if closure > 1e-20:
        print(f"   WARN: closure > 1e-20; check precision.")

    # Find all Euler crossings
    crossings = find_euler_crossings_from_grid(q_grid, p_grid, t_grid,
                                               tol_midpoint=1e-4)
    euler_crossings = [c for c in crossings if c["is_euler"]]
    print(f"   found {len(crossings)} collinear events, "
          f"{len(euler_crossings)} with Euler-midpoint config")

    # Apply canonical transform at each Euler crossing
    matches = []
    our_orbits = get_our_orbits()
    v1_ours = float(our_orbits[orbit_name]["v1"])
    v2_ours = float(our_orbits[orbit_name]["v2"])
    for c in euler_crossings:
        q = np.asarray(c["q"])
        p = np.asarray(c["p"])
        try:
            v1_match, v2_match, aux = canonical_euler_transform(q, p)
        except ValueError as e:
            c["transform_error"] = str(e)
            continue
        cmp = compare_to_our_orbit(v1_match, v2_match, v1_ours, v2_ours)
        c["canonical_v1"] = v1_match
        c["canonical_v2"] = v2_match
        c["aux"] = aux
        c["cmp"] = cmp
        matches.append({"t_over_T": c["t"] / float(T),
                        "dv_min": cmp["dv_min"],
                        "half_binary_d": aux["half_binary_d"]})

    # Summary
    if matches:
        matches.sort(key=lambda m: m["dv_min"])
        best = matches[0]
        verdict = (
            "identical" if best["dv_min"] < 1e-10
            else ("likely" if best["dv_min"] < 1e-4
                  else ("ambiguous" if best["dv_min"] < 1e-2
                        else "distinct_with_section_crossing")))
        print(f"   BEST match: dv_min={best['dv_min']:.3e} at t/T={best['t_over_T']:.4f}")
    else:
        verdict = "distinct_no_euler_crossing"
        best = None
        print(f"   NO Euler crossings found.")

    print(f"   VERDICT: {verdict}")

    return {
        "orbit": orbit_name,
        "hristov_id": entry["id"],
        "T_catalog": mp.nstr(T, 20),
        "closure_residual": closure,
        "wall_time_s": wall,
        "n_collinear_events": len(crossings),
        "n_euler_crossings": len(euler_crossings),
        "best_match": best,
        "all_matches": matches,
        "verdict": verdict,
    }


def verify_hristov_entry_as_euler(orbit_name: str, row: int) -> dict:
    """Alternative verification: treat Hristov 2025 (x, y) as (v1, v2) in our
    Li-Liao Euler convention.

    The README claims Hristov 2025 uses the free-fall convention. But as of
    2026-04-15 we have strong numerical evidence that the 2025 catalog's
    columns in ics_971_100.txt are actually (v1, v2, T, T_star) in our
    Li-Liao/Euler convention. Sign of y may correspond to a t -> -t
    symmetry.  This function probes that hypothesis directly: it sets up
    the IC in our Euler convention using Hristov (x, y) as (v1, v2),
    integrates for T, and reports closure + dv to our reference orbit.
    """
    import heyoka as hy
    entry = load_hristov_entry(row)
    print(f"\n=== Euler-conv hypothesis: orbit {orbit_name} vs {entry['id']} ===")

    ta = build_hp_integrator(200)
    T = entry["T"]
    v1_h = entry["x3"]
    v2_h = entry["y3"]

    out = {"orbit": orbit_name, "hristov_id": entry["id"],
           "hypothesis": "Hristov_catalog_(x,y)_equals_(v1,v2)_in_our_Euler_convention",
           "sign_tests": []}

    # Try four sign combinations of (v1, v2), since the orbit has discrete
    # symmetries; report all closures.
    for s1, s2 in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        v1 = s1 * v1_h
        v2 = s2 * v2_h
        q = [(mp.mpf(-1), mp.mpf(0)),
             (mp.mpf(1), mp.mpf(0)),
             (mp.mpf(0), mp.mpf(0))]
        v = [(v1, v2), (v1, v2), (-2*v1, -2*v2)]
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
        out["sign_tests"].append({
            "sign_v1": s1, "sign_v2": s2,
            "closure_residual": res, "wall_s": wall,
        })
        print(f"   ({s1:+d}v1, {s2:+d}v2): closure={res:.3e}, wall={wall:.1f}s")

    # Compare directly in HP
    our_orbits = get_our_orbits()
    v1_ours = our_orbits[orbit_name]["v1"]
    v2_ours = our_orbits[orbit_name]["v2"]
    T_ours = our_orbits[orbit_name]["T"]
    dv_candidates = []
    for s1, s2 in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        dv = mp.sqrt((s1 * v1_h - v1_ours)**2 + (s2 * v2_h - v2_ours)**2)
        dv_candidates.append((float(dv), s1, s2, mp.nstr(dv, 20)))
    dv_candidates.sort()
    dT = abs(T - T_ours)
    best = dv_candidates[0]
    out["dv_min_vs_our_orbit"] = best[0]
    out["dv_min_sign"] = (best[1], best[2])
    out["dv_min_HP_string"] = best[3]
    out["dT_abs"] = float(dT)
    out["dT_abs_HP_string"] = mp.nstr(dT, 20)
    out["all_dv_candidates"] = [[c[0], c[1], c[2], c[3]] for c in dv_candidates]

    print(f"   HP dv_min (best sign {best[1]},{best[2]}) = {best[3]}")
    print(f"   HP |dT|                                 = {mp.nstr(dT, 20)}")

    # Verdict
    min_closure = min(t["closure_residual"] for t in out["sign_tests"])
    if min_closure < 1e-40 and best[0] < 1e-40:
        out["verdict"] = "identical_under_euler_interpretation"
    elif min_closure < 1e-30 and best[0] < 1e-30:
        out["verdict"] = "likely_identical_under_euler_interpretation"
    else:
        out["verdict"] = "not_identical_under_euler_interpretation"
    print(f"   VERDICT (Euler hypothesis): {out['verdict']}")
    return out


def main():
    results_all = {}
    for orbit_name, row in HRISTOV_TARGETS.items():
        if orbit_name != "C":  # D is task 7, do C here
            continue

        # Primary: run the original plan (integrate as free-fall IC,
        # find Euler crossings). If Hristov really is free-fall this is
        # the right test.
        r_ff = verify_hristov_entry(orbit_name, row, n_grid=50000)

        # Secondary: test the hypothesis that Hristov 2025's catalog
        # columns are (v1, v2, T, T*) in Li-Liao/Euler convention (not
        # free-fall). If true, closure residual will be HP-clean and
        # dv_min will be ~1e-49.
        r_eu = verify_hristov_entry_as_euler(orbit_name, row)

        results_all[orbit_name] = {
            "free_fall_interpretation": r_ff,
            "euler_interpretation": r_eu,
        }

    (EXP_DIR / "verify_C_result.json").write_text(
        json.dumps(results_all, indent=2, default=str))
    print(f"\nSaved: {EXP_DIR / 'verify_C_result.json'}")


if __name__ == "__main__":
    if "--smoketest" in sys.argv:
        _smoketest()
    else:
        main()
