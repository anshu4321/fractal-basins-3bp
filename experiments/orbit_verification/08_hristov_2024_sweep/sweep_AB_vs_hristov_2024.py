"""Action #2: full Hristov 2024 free-fall catalog sweep vs our orbits A, B.

For each of 24,582 Hristov 2024 free-fall initial conditions (x, y, T, T*),
integrate forward one period in Hristov units using a pure-numpy 6th-order
Yoshida symplectic integrator. At every collinear-with-midpoint event
(Euler-section crossing) in the trajectory, apply the canonical transform
(translate midpoint to origin, rotate outer pair onto the x-axis, scale so
outer bodies land at (+-1, 0)) and compare the recovered (v1, v2, T_can) to
our HP-refined orbits A and B.

Match criteria (relative to each orbit):
  dv_min = min over 4 sign symmetries (s1, s2) in {+-1} of
             sqrt((s1 v1_can - v1_our)^2 + (s2 v2_can - v2_our)^2)
  dT_rel = |T_can - T_our| / T_our

  STRONG: dv_min < 1e-4 AND dT_rel < 1e-3
  WEAK:   dv_min < 1e-2 AND dT_rel < 1e-2

Parallelism: 12 workers via multiprocessing.Pool. Each worker receives a
chunk of catalog rows and returns their per-entry best-match records.

Outputs (in this script's directory):
  - sweep_results.json: list of dicts, one per entry, with the best dv / dT
    versus each of A and B, plus the crossing phase t/T and half-binary
    distance at the crossing.
  - RESULT.md: executive summary tables for strong and weak hits.

Runtime target on the GCP 12-CPU VM: ~5 minutes total.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp_proc
import os
import time
from pathlib import Path

import numpy as np

# Make numpy stay single-threaded in workers (prevent BLAS from oversubscribing).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# -----------------------------------------------------------------------------
# Yoshida-6 symplectic integrator (pure numpy, matches mega3bp/integrators.py).
# -----------------------------------------------------------------------------
_W1 = -1.17767998417887
_W2 = 0.235573213359357
_W3 = 0.784513610477560
_W0 = 1.0 - 2.0 * (_W1 + _W2 + _W3)
YOSHIDA6_WEIGHTS = np.array([_W3, _W2, _W1, _W0, _W1, _W2, _W3], dtype=np.float64)


def _force(q: np.ndarray) -> np.ndarray:
    """F_i = sum_{j != i} (q_j - q_i) / |q_j - q_i|^3 for unit masses, G=1."""
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    d12 = q2 - q1
    d13 = q3 - q1
    d23 = q3 - q2
    r12_sq = d12[0] * d12[0] + d12[1] * d12[1]
    r13_sq = d13[0] * d13[0] + d13[1] * d13[1]
    r23_sq = d23[0] * d23[0] + d23[1] * d23[1]
    # Inverse r^3. Guard against zeros (will let the caller detect blowup).
    inv12_3 = r12_sq ** -1.5
    inv13_3 = r13_sq ** -1.5
    inv23_3 = r23_sq ** -1.5
    F = np.empty_like(q)
    F[0] = d12 * inv12_3 + d13 * inv13_3
    F[1] = -d12 * inv12_3 + d23 * inv23_3
    F[2] = -d13 * inv13_3 - d23 * inv23_3
    return F


def _verlet_substep(q: np.ndarray, p: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Drift-Kick-Drift Stormer-Verlet sub-step (unit mass, p == v)."""
    q_half = q + 0.5 * dt * p
    p_new = p + dt * _force(q_half)
    q_new = q_half + 0.5 * dt * p_new
    return q_new, p_new


def yoshida6_step(q: np.ndarray, p: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    for w in YOSHIDA6_WEIGHTS:
        q, p = _verlet_substep(q, p, w * h)
    return q, p


# Fast scalar version of the whole step: unpacks q, p into 12 scalars and
# runs all 7 Verlet sub-steps with scalar math. Avoids numpy array allocation
# and indexing overhead in the inner loop — ~10x speedup on pure Python.
def _yoshida6_step_scalar(
    q0x: float, q0y: float, q1x: float, q1y: float, q2x: float, q2y: float,
    p0x: float, p0y: float, p1x: float, p1y: float, p2x: float, p2y: float,
    h: float,
) -> tuple[float, ...]:
    """Scalar Yoshida-6 step; returns 12 new coordinates."""
    for w in (_W3, _W2, _W1, _W0, _W1, _W2, _W3):
        dt = w * h
        half_dt = 0.5 * dt
        # Drift: q += 0.5 dt * p
        q0x += half_dt * p0x
        q0y += half_dt * p0y
        q1x += half_dt * p1x
        q1y += half_dt * p1y
        q2x += half_dt * p2x
        q2y += half_dt * p2y

        # Kick: p += dt * F(q_half), unit mass
        d01x = q1x - q0x; d01y = q1y - q0y
        d02x = q2x - q0x; d02y = q2y - q0y
        d12x = q2x - q1x; d12y = q2y - q1y
        r01_sq = d01x * d01x + d01y * d01y
        r02_sq = d02x * d02x + d02y * d02y
        r12_sq = d12x * d12x + d12y * d12y
        inv01 = r01_sq ** -1.5
        inv02 = r02_sq ** -1.5
        inv12 = r12_sq ** -1.5
        f0x = d01x * inv01 + d02x * inv02
        f0y = d01y * inv01 + d02y * inv02
        f1x = -d01x * inv01 + d12x * inv12
        f1y = -d01y * inv01 + d12y * inv12
        f2x = -d02x * inv02 - d12x * inv12
        f2y = -d02y * inv02 - d12y * inv12
        p0x += dt * f0x; p0y += dt * f0y
        p1x += dt * f1x; p1y += dt * f1y
        p2x += dt * f2x; p2y += dt * f2y

        # Drift: q += 0.5 dt * p (new p)
        q0x += half_dt * p0x
        q0y += half_dt * p0y
        q1x += half_dt * p1x
        q1y += half_dt * p1y
        q2x += half_dt * p2x
        q2y += half_dt * p2y
    return (q0x, q0y, q1x, q1y, q2x, q2y,
            p0x, p0y, p1x, p1y, p2x, p2y)


# -----------------------------------------------------------------------------
# Hristov 2024 free-fall integration + Euler-section crossing detection.
# -----------------------------------------------------------------------------

def integrate_and_find_euler_events(x3_init: float,
                                     y3_init: float,
                                     T_H: float,
                                     n_steps: int,
                                     tol_midpoint: float) -> list[dict]:
    """Integrate the free-fall IC forward one period T_H at n_steps resolution
    and return every collinear-midpoint event as a dict with step k, t, q, p.

    Convention: body 1 at (-0.5, 0), body 2 at (0.5, 0), body 3 at (x_init, y_init),
    zero momenta, unit mass, G = 1.

    Uses running "last 3 triangles" buffer + signed-area-sign-change detection so
    memory is O(1) per step, not O(n_steps).

    Raises RuntimeError if integration diverges (NaN/Inf encountered) — caller
    is expected to catch and record.
    """
    # Scalar state: (q0x, q0y, q1x, q1y, q2x, q2y, p0x, p0y, p1x, p1y, p2x, p2y)
    q0x, q0y = -0.5, 0.0
    q1x, q1y = 0.5, 0.0
    q2x, q2y = float(x3_init), float(y3_init)
    p0x = p0y = p1x = p1y = p2x = p2y = 0.0
    h = T_H / n_steps

    events: list[dict] = []

    # Keep previous state so we can linearly interpolate across sign changes.
    # Signed triangle area = (q1-q0) x (q2-q0).
    a0 = q1x - q0x; a1 = q1y - q0y
    b0 = q2x - q0x; b1 = q2y - q0y
    area_prev = a0 * b1 - a1 * b0

    q_prev = (q0x, q0y, q1x, q1y, q2x, q2y)
    p_prev = (p0x, p0y, p1x, p1y, p2x, p2y)

    step_scalar = _yoshida6_step_scalar
    import math as _math
    for k in range(1, n_steps + 1):
        out = step_scalar(
            q0x, q0y, q1x, q1y, q2x, q2y,
            p0x, p0y, p1x, p1y, p2x, p2y, h)
        q0x, q0y, q1x, q1y, q2x, q2y, p0x, p0y, p1x, p1y, p2x, p2y = out

        # Cheap divergence check every 1000 steps (full NaN scan is expensive).
        if k % 1000 == 0:
            if not (_math.isfinite(q0x) and _math.isfinite(q2x) and _math.isfinite(p2x)):
                raise RuntimeError("non-finite state at step %d" % k)

        # Signed triangle area.
        a0 = q1x - q0x; a1 = q1y - q0y
        b0 = q2x - q0x; b1 = q2y - q0y
        area = a0 * b1 - a1 * b0

        # Detect zero-crossing of the signed area between step k-1 and step k.
        if (area_prev == 0.0) or (area_prev > 0.0 and area < 0.0) or (area_prev < 0.0 and area > 0.0):
            denom = area_prev - area
            alpha = 0.5 if denom == 0.0 else area_prev / denom
            if alpha < 0.0:
                alpha = 0.0
            elif alpha > 1.0:
                alpha = 1.0
            t_cross = (k - 1 + alpha) * h
            q0x_c = q_prev[0] + alpha * (q0x - q_prev[0])
            q0y_c = q_prev[1] + alpha * (q0y - q_prev[1])
            q1x_c = q_prev[2] + alpha * (q1x - q_prev[2])
            q1y_c = q_prev[3] + alpha * (q1y - q_prev[3])
            q2x_c = q_prev[4] + alpha * (q2x - q_prev[4])
            q2y_c = q_prev[5] + alpha * (q2y - q_prev[5])
            p0x_c = p_prev[0] + alpha * (p0x - p_prev[0])
            p0y_c = p_prev[1] + alpha * (p0y - p_prev[1])
            p1x_c = p_prev[2] + alpha * (p1x - p_prev[2])
            p1y_c = p_prev[3] + alpha * (p1y - p_prev[3])
            p2x_c = p_prev[4] + alpha * (p2x - p_prev[4])
            p2y_c = p_prev[5] + alpha * (p2y - p_prev[5])

            # Is this an Euler midpoint crossing? For each choice of midpoint
            # body m, compute scaled distance from m to midpoint of the other
            # two, and keep the minimum.
            best_mid_dist = float("inf")
            midpoint_body = -1
            # body 0 at midpoint of 1,2?
            mx = 0.5 * (q1x_c + q2x_c); my = 0.5 * (q1y_c + q2y_c)
            dx = q1x_c - q2x_c; dy = q1y_c - q2y_c
            d_out = (dx * dx + dy * dy) ** 0.5
            if d_out > 1e-12:
                ex = q0x_c - mx; ey = q0y_c - my
                dist = ((ex * ex + ey * ey) ** 0.5) / d_out
                if dist < best_mid_dist:
                    best_mid_dist = dist
                    midpoint_body = 0
            # body 1 at midpoint of 0,2?
            mx = 0.5 * (q0x_c + q2x_c); my = 0.5 * (q0y_c + q2y_c)
            dx = q0x_c - q2x_c; dy = q0y_c - q2y_c
            d_out = (dx * dx + dy * dy) ** 0.5
            if d_out > 1e-12:
                ex = q1x_c - mx; ey = q1y_c - my
                dist = ((ex * ex + ey * ey) ** 0.5) / d_out
                if dist < best_mid_dist:
                    best_mid_dist = dist
                    midpoint_body = 1
            # body 2 at midpoint of 0,1?
            mx = 0.5 * (q0x_c + q1x_c); my = 0.5 * (q0y_c + q1y_c)
            dx = q0x_c - q1x_c; dy = q0y_c - q1y_c
            d_out = (dx * dx + dy * dy) ** 0.5
            if d_out > 1e-12:
                ex = q2x_c - mx; ey = q2y_c - my
                dist = ((ex * ex + ey * ey) ** 0.5) / d_out
                if dist < best_mid_dist:
                    best_mid_dist = dist
                    midpoint_body = 2

            if best_mid_dist < tol_midpoint:
                q_arr = np.array([[q0x_c, q0y_c], [q1x_c, q1y_c], [q2x_c, q2y_c]])
                p_arr = np.array([[p0x_c, p0y_c], [p1x_c, p1y_c], [p2x_c, p2y_c]])
                events.append({
                    "step": k,
                    "alpha": float(alpha),
                    "t": float(t_cross),
                    "midpoint_body": int(midpoint_body),
                    "midpoint_residual": float(best_mid_dist),
                    "q": q_arr,
                    "p": p_arr,
                })
        q_prev = (q0x, q0y, q1x, q1y, q2x, q2y)
        p_prev = (p0x, p0y, p1x, p1y, p2x, p2y)
        area_prev = area

    return events


def canonical_euler_transform(q: np.ndarray, p: np.ndarray,
                              tol_midpoint: float = 1e-3) -> tuple[float, float, dict]:
    """Transform an Euler-section state (q, p) to canonical (v1, v2).

    Mirrors 07_hristov_2025_definitive/check_C_D_vs_hristov_2025.py:
    translate the midpoint body to the origin, rotate the outer-pair axis onto
    +x, rescale by 1/d_half so outer bodies land at (+-1, 0). Velocities
    transform as p_can = p_rot * sqrt(d_half) (t -> t / d_half^{1.5}, r -> r / d_half).

    Returns (v1_can, v2_can, aux). aux["half_binary_d"] is the d_half in input
    units (= Hristov units if called with a trajectory integrated in those units).
    """
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)

    midpoint_body = -1
    best_dist = np.inf
    for m in range(3):
        others = [i for i in range(3) if i != m]
        d_others = np.linalg.norm(q[others[0]] - q[others[1]])
        if d_others < 1e-12:
            continue
        mid = 0.5 * (q[others[0]] + q[others[1]])
        d_to_mid = np.linalg.norm(q[m] - mid) / d_others
        if d_to_mid < best_dist:
            best_dist = d_to_mid
            midpoint_body = m
    if midpoint_body < 0 or best_dist > tol_midpoint:
        raise ValueError(
            "No midpoint body within tol_midpoint=%g (best=%g)" % (tol_midpoint, best_dist))

    origin = q[midpoint_body].copy()
    q_shift = q - origin

    others = [i for i in range(3) if i != midpoint_body]
    d_half = 0.5 * np.linalg.norm(q_shift[others[0]] - q_shift[others[1]])

    axis = q_shift[others[1]] - q_shift[others[0]]
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        raise ValueError("Outer bodies coincide.")
    cos_th = axis[0] / axis_norm
    sin_th = axis[1] / axis_norm
    R = np.array([[cos_th, sin_th], [-sin_th, cos_th]])

    q_rot = q_shift @ R.T
    p_rot = p @ R.T
    q_canonical = q_rot / d_half
    p_canonical = p_rot * np.sqrt(d_half)

    # outer_minus_idx = the one that ends up at x = -1 after rotation.
    x_outer0 = q_canonical[others[0], 0]
    outer_minus_idx = others[1] if x_outer0 > 0 else others[0]

    v1 = float(p_canonical[outer_minus_idx, 0])
    v2 = float(p_canonical[outer_minus_idx, 1])

    return v1, v2, {
        "midpoint_body": int(midpoint_body),
        "outer_minus_idx": int(outer_minus_idx),
        "half_binary_d": float(d_half),
        "midpoint_residual": float(best_dist),
    }


def _dv_min_4_signs(v1_can: float, v2_can: float,
                    v1_our: float, v2_our: float) -> tuple[float, int, int]:
    """Min dv over (v1, v2) -> (+-v1, +-v2)."""
    best = (float("inf"), 0, 0)
    for s1 in (1, -1):
        for s2 in (1, -1):
            dv = ((s1 * v1_can - v1_our) ** 2 + (s2 * v2_can - v2_our) ** 2) ** 0.5
            if dv < best[0]:
                best = (dv, s1, s2)
    return best


# -----------------------------------------------------------------------------
# Worker + Pool driver.
# -----------------------------------------------------------------------------

OUR_ORBITS: dict[str, tuple[float, float, float]] = {}  # set by main()


def _process_chunk(args: tuple) -> list[dict]:
    """Process one chunk of catalog entries. Returns list of per-entry dicts."""
    chunk_rows, n_steps, tol_midpoint, our_orbits = args
    results = []
    for row_idx, x3, y3, T_H in chunk_rows:
        rec: dict = {
            "row": row_idx,
            "x3": x3, "y3": y3, "T_H": T_H,
            "best": {name: {"dv_min": None, "dT_rel": None,
                            "s1": 0, "s2": 0,
                            "T_can": None, "d_half": None,
                            "t_over_T": None, "n_events": 0}
                     for name in our_orbits},
            "n_events": 0,
            "diverged": False,
            "error": None,
        }
        try:
            events = integrate_and_find_euler_events(
                x3, y3, T_H, n_steps=n_steps, tol_midpoint=tol_midpoint)
        except RuntimeError as e:
            rec["diverged"] = True
            rec["error"] = str(e)
            results.append(rec)
            continue
        except Exception as e:  # pragma: no cover
            rec["diverged"] = True
            rec["error"] = "exception:%s" % type(e).__name__
            results.append(rec)
            continue

        rec["n_events"] = len(events)
        for name, (v1_our, v2_our, T_our) in our_orbits.items():
            rec["best"][name]["n_events"] = len(events)

        for ev in events:
            try:
                v1c, v2c, aux = canonical_euler_transform(ev["q"], ev["p"],
                                                          tol_midpoint=tol_midpoint)
            except ValueError:
                continue
            d_half = aux["half_binary_d"]
            if d_half <= 0.0:
                continue
            T_can = T_H * (1.0 / d_half) ** 1.5

            for name, (v1_our, v2_our, T_our) in our_orbits.items():
                dv, s1, s2 = _dv_min_4_signs(v1c, v2c, v1_our, v2_our)
                dT_rel = abs(T_can - T_our) / T_our
                cur = rec["best"][name]
                if cur["dv_min"] is None or dv < cur["dv_min"]:
                    cur["dv_min"] = dv
                    cur["dT_rel"] = dT_rel
                    cur["s1"] = int(s1)
                    cur["s2"] = int(s2)
                    cur["T_can"] = T_can
                    cur["d_half"] = d_half
                    cur["t_over_T"] = ev["t"] / T_H
                    cur["v1_can"] = v1c
                    cur["v2_can"] = v2c
        results.append(rec)
    return results


def load_catalog(catalog_path: Path) -> list[tuple[int, float, float, float]]:
    """Parse hristov_2024_sol_80.txt into (row, x, y, T) float64 tuples."""
    import mpmath as mp

    mp.mp.dps = 40  # Keep parsing cheap; we only need ~double precision downstream.
    rows = []
    with catalog_path.open() as f:
        for i, line in enumerate(f, 1):
            parts = line.split()
            if len(parts) < 4:
                continue
            x = float(mp.mpf(parts[0]))
            y = float(mp.mpf(parts[1]))
            T = float(mp.mpf(parts[2]))
            rows.append((i, x, y, T))
    return rows


def load_our_orbits(hp_path: Path) -> dict[str, tuple[float, float, float]]:
    data = json.loads(hp_path.read_text())
    out: dict[str, tuple[float, float, float]] = {}
    for entry in data:
        name = entry.get("name")
        if name in ("A", "B"):
            out[name] = (
                float(entry["v1_HP"]),
                float(entry["v2_HP"]),
                float(entry["T_HP"]),
            )
    missing = {"A", "B"} - set(out)
    if missing:
        raise RuntimeError("Missing HP orbits %s in %s" % (missing, hp_path))
    return out


def chunked(xs: list, n: int):
    k, r = divmod(len(xs), n)
    i = 0
    for c in range(n):
        size = k + (1 if c < r else 0)
        yield xs[i:i + size]
        i += size


def write_result_md(sweep_results: list[dict], our_orbits: dict,
                    meta: dict, path: Path) -> None:
    strong_thresh_dv = 1e-4
    strong_thresh_dT = 1e-3
    weak_thresh_dv = 1e-2
    weak_thresh_dT = 1e-2

    def classify(rec, name):
        b = rec["best"].get(name, {})
        dv = b.get("dv_min")
        dT = b.get("dT_rel")
        if dv is None or dT is None:
            return None
        if dv < strong_thresh_dv and dT < strong_thresh_dT:
            return "STRONG"
        if dv < weak_thresh_dv and dT < weak_thresh_dT:
            return "WEAK"
        return None

    hits = {name: {"strong": [], "weak": []} for name in our_orbits}
    for rec in sweep_results:
        for name in our_orbits:
            c = classify(rec, name)
            if c == "STRONG":
                hits[name]["strong"].append(rec)
            elif c == "WEAK":
                hits[name]["weak"].append(rec)

    for name in our_orbits:
        for tier in ("strong", "weak"):
            hits[name][tier].sort(
                key=lambda r: r["best"][name]["dv_min"])

    n_total = len(sweep_results)
    n_diverged = sum(1 for r in sweep_results if r["diverged"])
    n_with_events = sum(1 for r in sweep_results if r["n_events"] > 0)

    lines = []
    lines.append("# Action #2: Hristov 2024 full-sweep vs orbits A, B\n")
    lines.append("Generated by sweep_AB_vs_hristov_2024.py.\n")
    lines.append("## Sweep parameters\n")
    lines.append(f"- Catalog entries processed: {n_total}")
    lines.append(f"- Workers: {meta['n_workers']}")
    lines.append(f"- Yoshida-6 steps per period: {meta['n_steps']}")
    lines.append(f"- Midpoint tolerance: {meta['tol_midpoint']}")
    lines.append(f"- Wall time: {meta['wall_time_s']:.1f} s "
                 f"({meta['wall_time_s']/60:.2f} min)")
    lines.append(f"- Entries with at least one Euler crossing: {n_with_events}")
    lines.append(f"- Diverged integrations: {n_diverged}\n")
    lines.append("## Match criteria\n")
    lines.append(f"- STRONG: dv_min < {strong_thresh_dv:g} AND |dT|/T < {strong_thresh_dT:g}")
    lines.append(f"- WEAK:   dv_min < {weak_thresh_dv:g} AND |dT|/T < {weak_thresh_dT:g}\n")

    for name in our_orbits:
        v1, v2, T = our_orbits[name]
        lines.append(f"## Orbit {name} (v1={v1:+.6f}, v2={v2:+.6f}, T={T:.6f})\n")
        for tier, label in (("strong", "STRONG"), ("weak", "WEAK")):
            rows = hits[name][tier]
            if tier == "weak":
                # Remove rows already in strong tier to avoid double-listing.
                strong_ids = {r["row"] for r in hits[name]["strong"]}
                rows = [r for r in rows if r["row"] not in strong_ids]
            lines.append(f"### {label} matches ({len(rows)})\n")
            if not rows:
                lines.append("_none._\n")
                continue
            lines.append("| row | dv_min | dT_rel | t/T | d_half | T_can | (s1, s2) |")
            lines.append("|---:|---:|---:|---:|---:|---:|:---:|")
            for r in rows[:25]:
                b = r["best"][name]
                lines.append(
                    f"| {r['row']} | {b['dv_min']:.3e} | {b['dT_rel']:.3e} | "
                    f"{b['t_over_T']:.4f} | {b['d_half']:.4f} | {b['T_can']:.4f} | "
                    f"({b['s1']:+d},{b['s2']:+d}) |")
            if len(rows) > 25:
                lines.append(f"\n_... {len(rows) - 25} more not shown._\n")
            lines.append("")

    lines.append("## Summary\n")
    for name in our_orbits:
        lines.append(f"- Orbit {name}: {len(hits[name]['strong'])} STRONG, "
                     f"{len([r for r in hits[name]['weak'] if r['row'] not in {x['row'] for x in hits[name]['strong']}])} WEAK.")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", type=Path,
                    default=Path(__file__).resolve().parents[1]
                    / "00_catalogs/hristov_2024_sol_80.txt")
    ap.add_argument("--hp-json", type=Path,
                    default=Path(__file__).resolve().parents[1]
                    / "06_high_precision/hp_heyoka_newton_results.json")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent)
    ap.add_argument("--n-steps", type=int, default=10000)
    ap.add_argument("--tol-midpoint", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0,
                    help="Limit to first N rows (debug).")
    args = ap.parse_args()

    print(f"Catalog : {args.catalog}")
    print(f"HP JSON : {args.hp_json}")
    print(f"Out dir : {args.out_dir}")
    print(f"n_steps={args.n_steps}, tol_midpoint={args.tol_midpoint}, "
          f"workers={args.workers}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    our_orbits = load_our_orbits(args.hp_json)
    for name, (v1, v2, T) in our_orbits.items():
        print(f"  Orbit {name}: v1={v1:+.12f}, v2={v2:+.12f}, T={T:.12f}")

    rows = load_catalog(args.catalog)
    print(f"Loaded {len(rows)} catalog rows.")
    if args.limit > 0:
        rows = rows[:args.limit]
        print(f"  (limited to first {len(rows)})")

    # Build many small chunks so imap_unordered returns results frequently and
    # we can print rolling progress. Target ~150 rows/chunk => ~164 chunks for
    # 24,582 rows × 12 workers. Each worker picks up a new chunk when it
    # finishes one.
    chunk_size = max(50, min(200, len(rows) // (args.workers * 12) or 100))
    chunks = []
    i = 0
    while i < len(rows):
        chunks.append(rows[i:i + chunk_size])
        i += chunk_size
    task_inputs = [(ch, args.n_steps, args.tol_midpoint, our_orbits)
                   for ch in chunks]
    print(f"[SWEEP] START: {len(rows)} entries, {args.workers} workers, "
          f"batch size {chunk_size}", flush=True)

    t0 = time.perf_counter()
    results: list[dict] = []

    # Track running strong/weak hit counts for progress display.
    counts = {"strong_A": 0, "strong_B": 0, "weak_A": 0, "weak_B": 0}

    def _update_counts(batch: list[dict]) -> None:
        for r in batch:
            for name in ("A", "B"):
                b = r["best"].get(name, {})
                dv = b.get("dv_min"); dT = b.get("dT_rel")
                if dv is None or dT is None:
                    continue
                if dv < 1e-4 and dT < 1e-3:
                    counts["strong_" + name] += 1
                elif dv < 1e-2 and dT < 1e-2:
                    counts["weak_" + name] += 1

    def _log_progress(n_done: int, n_total: int, force: bool = False) -> None:
        elapsed = time.perf_counter() - t0
        rate = n_done / elapsed if elapsed > 0 else 0.0
        eta = (n_total - n_done) / rate if rate > 0 else float("inf")
        print(
            f"[SWEEP] {n_done}/{n_total} ({100.0*n_done/n_total:.1f}%) - "
            f"elapsed {elapsed:.1f}s - ETA {eta:.1f}s - "
            f"strong: A={counts['strong_A']} B={counts['strong_B']} "
            f"weak: A={counts['weak_A']} B={counts['weak_B']}",
            flush=True,
        )

    # Print progress line every ~PROGRESS_EVERY completed entries.
    PROGRESS_EVERY = 100

    if args.workers <= 1:
        next_threshold = PROGRESS_EVERY
        for task in task_inputs:
            batch = _process_chunk(task)
            results.extend(batch)
            _update_counts(batch)
            if len(results) >= next_threshold:
                _log_progress(len(results), len(rows))
                next_threshold = ((len(results) // PROGRESS_EVERY) + 1) * PROGRESS_EVERY
    else:
        next_threshold = PROGRESS_EVERY
        with mp_proc.Pool(processes=args.workers) as pool:
            for out in pool.imap_unordered(_process_chunk, task_inputs):
                results.extend(out)
                _update_counts(out)
                if len(results) >= next_threshold:
                    _log_progress(len(results), len(rows))
                    next_threshold = ((len(results) // PROGRESS_EVERY) + 1) * PROGRESS_EVERY

    wall = time.perf_counter() - t0
    # Final progress + DONE line.
    _log_progress(len(results), len(rows), force=True)
    print(
        f"[SWEEP] DONE: {len(results)}/{len(rows)} in {wall:.1f}s "
        f"({len(results)/wall:.1f} entries/s)",
        flush=True,
    )

    results.sort(key=lambda r: r["row"])

    # Build strong/weak buckets in the user-requested output schema.
    strong_matches: dict[str, list[dict]] = {"A": [], "B": []}
    weak_matches: dict[str, list[dict]] = {"A": [], "B": []}
    for r in results:
        for name in ("A", "B"):
            b = r["best"].get(name, {})
            dv = b.get("dv_min"); dT = b.get("dT_rel")
            if dv is None or dT is None:
                continue
            entry = {
                "row": r["row"],
                "T_hristov": r["T_H"],
                "x3_init": r["x3"],
                "y3_init": r["y3"],
                "dv_min": dv,
                "dT_rel": dT,
                "t_over_T": b.get("t_over_T"),
                "d_half": b.get("d_half"),
                "T_can": b.get("T_can"),
                "s1": b.get("s1"), "s2": b.get("s2"),
                "v1_can": b.get("v1_can"), "v2_can": b.get("v2_can"),
                "n_events": b.get("n_events"),
            }
            if dv < 1e-4 and dT < 1e-3:
                strong_matches[name].append(entry)
            elif dv < 1e-2 and dT < 1e-2:
                weak_matches[name].append(entry)
    for bucket in (strong_matches, weak_matches):
        for name in bucket:
            bucket[name].sort(key=lambda e: e["dv_min"])

    n_diverged = sum(1 for r in results if r["diverged"])
    meta = {
        "catalog": str(args.catalog),
        "hp_json": str(args.hp_json),
        "n_steps": args.n_steps,
        "tol_midpoint": args.tol_midpoint,
        "n_workers": args.workers,
        "n_rows": len(rows),
        "wall_time_s": wall,
    }
    out_json = args.out_dir / "sweep_results.json"
    out_json.write_text(json.dumps(
        {
            "n_entries": len(results),
            "n_diverged": n_diverged,
            "elapsed_s": wall,
            "meta": meta,
            "our_orbits": {n: {"v1": v[0], "v2": v[1], "T": v[2]}
                           for n, v in our_orbits.items()},
            "strong_matches": strong_matches,
            "weak_matches": weak_matches,
            "results": results,
        },
        indent=1, default=str))
    print(f"Wrote {out_json}.")

    out_md = args.out_dir / "RESULT.md"
    write_result_md(results, our_orbits, meta, out_md)
    print(f"Wrote {out_md}.")

    # Short text summary.
    for name in our_orbits:
        dvs = [r["best"][name]["dv_min"] for r in results
               if r["best"][name]["dv_min"] is not None]
        strong = [r for r in results
                  if r["best"][name]["dv_min"] is not None
                  and r["best"][name]["dv_min"] < 1e-4
                  and r["best"][name]["dT_rel"] < 1e-3]
        weak = [r for r in results
                if r["best"][name]["dv_min"] is not None
                and r["best"][name]["dv_min"] < 1e-2
                and r["best"][name]["dT_rel"] < 1e-2
                and r not in strong]
        dv_min_overall = min(dvs) if dvs else float("nan")
        print(f"  Orbit {name}: strong={len(strong)}, weak={len(weak)}, "
              f"min dv seen = {dv_min_overall:.3e}")


if __name__ == "__main__":
    main()
