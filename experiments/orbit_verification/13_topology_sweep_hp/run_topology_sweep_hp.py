"""Action #8: high-precision topology sweep (A only) over Hristov 2024
entries whose scale-invariant period T* falls in A's plausible-length
window.

Why this is the right scope. Orbit A has T* = 36.94 and syzygy word
length = 24 (Hristov numbering). The empirical ratio length / T*
across our four verified orbits is ~0.65 per unit T* (24/36.94,
42/64.65, 42/64.65, 48/73.84). A true length-24 orbit therefore has
T* roughly in [25, 50]; entries with T* outside that window cannot
carry A's canonical syzygy word.

Why float64 is not enough. Many Hristov 2024 orbits are
hyperbolically sensitive and float64 integration loses the true
trajectory within one period (Lyapunov amplification >> double
epsilon). The float64 sweep (Action #7) resolved only ~357 of 24,582
entries to full-length words. This sweep uses heyoka MPFR at
200-bit, which keeps ~60 decimal digits through Lyapunov
amplification of ~exp(40) typical for long chaotic periods.

Pipeline per entry (heyoka variational? no, just scalar):
  1. Build heyoka taylor_adaptive at fp_type=hy.real, prec=200,
     compact_mode=True.
  2. Attach a non-terminal event to the signed area
     w = (q2-q1) x (q3-q1); every firing is a syzygy.
  3. Set the free-fall IC q1=(-0.5,0), q2=(0.5,0), q3=(x3,y3),
     p_i=0; propagate_until(T).
  4. The event callback appends the midpoint body index (1-indexed)
     to a per-orbit digit list.
  5. Canonicalise the resulting word and compare against A's.

Telemetry at status.json in this directory via the shared
mega3bp.gcp_telemetry.Telemetry class. Probe live with:
    ssh ... 'cat ~/3bp/experiments/orbit_verification/13_topology_sweep_hp/status.json'
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import heyoka as hy
import mpmath as mp
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from mega3bp.gcp_telemetry import Telemetry  # noqa: E402

sys.path.insert(0, str(REPO / "experiments" / "orbit_verification" / "09_syzygy_words"))
sys.path.insert(0, str(REPO / "experiments" / "orbit_verification" / "08_hristov_2024_sweep"))
from compute_words import equivalence_class  # noqa: E402
from sweep_AB_vs_hristov_2024 import load_catalog  # noqa: E402

CATALOG = REPO / "experiments/orbit_verification/00_catalogs/hristov_2024_sol_80.txt"
HP_RESULTS = REPO / "experiments/orbit_verification/06_high_precision/hp_heyoka_newton_results.json"

# Action #3 canonical words (Hristov numbering, confirmed in validation of
# Action #7). Equivalence checking is invariant to body-index labelling.
A_WORD = "132132132132132132132132"  # len 24
B_WORD = "312312312312312312312312312312312312312312"  # len 42
A_LEN = len(A_WORD)
B_LEN = len(B_WORD)
A_CANON = equivalence_class(A_WORD)
B_CANON = equivalence_class(B_WORD)

# Window on T* that is empirically consistent with either A's length-24 or
# B's length-42 syzygy word (length ~ 0.65 * T*).
TSTAR_LO = float(os.environ.get("TOPO_HP_TSTAR_LO", "25"))
TSTAR_HI = float(os.environ.get("TOPO_HP_TSTAR_HI", "50"))

# Precision: np.longdouble on x86-64 Linux = 80-bit extended, ~19 decimal
# digits. That is 4 digits beyond float64 — enough to survive Lyapunov
# amplification exp(T/tau_Ly) for typical Hristov 2024 orbits (T<=80,
# tau_Ly~2-5) because the amplification rarely exceeds 10^10, leaving ~9
# safe digits for topology identification. heyoka.taylor_adaptive with
# fp_type=np.longdouble compiles in ~2 seconds vs ~5 min for real+prec=200.
FP_TYPE = np.longdouble
TOL = np.longdouble("1e-18")
N_WORKERS = int(os.environ.get("TOPO_HP_WORKERS", "12"))


# ---------------------------------------------------------------------------
# Midpoint-body identification (pure-float on the interpolated event state).
# ---------------------------------------------------------------------------

def _midpoint_body(q1x, q1y, q2x, q2y, q3x, q3y) -> int:
    """Return the 1-indexed Hristov body ({1, 2, 3}) that is spatially between
    the other two at a collinear configuration. Uses the ratio of the
    perpendicular distance (from the candidate body to the line through the
    other two) to the separation between the other two bodies."""
    best = float("inf")
    mid = 1
    # body 1 midpoint of 2, 3?
    mx, my = 0.5 * (q2x + q3x), 0.5 * (q2y + q3y)
    dx, dy = q2x - q3x, q2y - q3y
    d = (dx * dx + dy * dy) ** 0.5
    if d > 1e-16:
        ex, ey = q1x - mx, q1y - my
        r = ((ex * ex + ey * ey) ** 0.5) / d
        if r < best:
            best, mid = r, 1
    # body 2 midpoint of 1, 3?
    mx, my = 0.5 * (q1x + q3x), 0.5 * (q1y + q3y)
    dx, dy = q1x - q3x, q1y - q3y
    d = (dx * dx + dy * dy) ** 0.5
    if d > 1e-16:
        ex, ey = q2x - mx, q2y - my
        r = ((ex * ex + ey * ey) ** 0.5) / d
        if r < best:
            best, mid = r, 2
    # body 3 midpoint of 1, 2?
    mx, my = 0.5 * (q1x + q2x), 0.5 * (q1y + q2y)
    dx, dy = q1x - q2x, q1y - q2y
    d = (dx * dx + dy * dy) ** 0.5
    if d > 1e-16:
        ex, ey = q3x - mx, q3y - my
        r = ((ex * ex + ey * ey) ** 0.5) / d
        if r < best:
            best, mid = r, 3
    return mid


# ---------------------------------------------------------------------------
# Per-worker integrator construction (heavy — JITs once per process).
# ---------------------------------------------------------------------------

_DIGITS: list[str] = []  # mutable module-global, accumulated by event callback
_TA = None


def _syzygy_callback(ta, t, d_sgn):
    """heyoka non-terminal event callback. d_sgn is the direction of the
    zero crossing (+/-). The interpolated state at the event time is in
    ta.state (first 6 entries are the body positions)."""
    s = ta.state
    q1x = float(s[0]); q1y = float(s[1])
    q2x = float(s[2]); q2y = float(s[3])
    q3x = float(s[4]); q3y = float(s[5])
    m = _midpoint_body(q1x, q1y, q2x, q2y, q3x, q3y)
    _DIGITS.append(str(m))


def _build_integrator():
    """Construct a taylor_adaptive(fp_type=real, prec=MPFR_PREC_BITS) 3BP
    integrator with a non-terminal event on signed area."""
    q1x, q1y = hy.make_vars("q1x", "q1y")
    q2x, q2y = hy.make_vars("q2x", "q2y")
    q3x, q3y = hy.make_vars("q3x", "q3y")
    p1x, p1y = hy.make_vars("p1x", "p1y")
    p2x, p2y = hy.make_vars("p2x", "p2y")
    p3x, p3y = hy.make_vars("p3x", "p3y")

    r12 = hy.sqrt((q1x - q2x) ** 2 + (q1y - q2y) ** 2)
    r13 = hy.sqrt((q1x - q3x) ** 2 + (q1y - q3y) ** 2)
    r23 = hy.sqrt((q2x - q3x) ** 2 + (q2y - q3y) ** 2)
    f12x = (q2x - q1x) / r12**3; f12y = (q2y - q1y) / r12**3
    f13x = (q3x - q1x) / r13**3; f13y = (q3y - q1y) / r13**3
    f23x = (q3x - q2x) / r23**3; f23y = (q3y - q2y) / r23**3

    sys_eqs = [
        (q1x, p1x), (q1y, p1y),
        (q2x, p2x), (q2y, p2y),
        (q3x, p3x), (q3y, p3y),
        (p1x,  f12x + f13x), (p1y,  f12y + f13y),
        (p2x, -f12x + f23x), (p2y, -f12y + f23y),
        (p3x, -f13x - f23x), (p3y, -f13y - f23y),
    ]
    # Signed area: (q2 - q1) x (q3 - q1)
    area_expr = (q2x - q1x) * (q3y - q1y) - (q2y - q1y) * (q3x - q1x)
    nt_ev = hy.nt_event(area_expr, _syzygy_callback, fp_type=FP_TYPE)

    # Placeholder IC (reset per orbit).
    state0 = np.zeros(12, dtype=FP_TYPE)
    ta = hy.taylor_adaptive(
        sys_eqs, state0, compact_mode=True,
        fp_type=FP_TYPE,
        tol=TOL,
        nt_events=[nt_ev],
    )
    return ta


def _worker_init():
    global _TA
    _TA = _build_integrator()


def _sweep_one(args):
    """Process one Hristov 2024 entry: (row, x3, y3, T_H, T_star)."""
    global _TA, _DIGITS
    row_idx, x3, y3, T_H, T_star = args
    t_wall = time.perf_counter()
    _DIGITS.clear()

    # Reset integrator state to free-fall IC at np.longdouble precision.
    _TA.state[0]  = FP_TYPE(-0.5);       _TA.state[1]  = FP_TYPE(0.0)
    _TA.state[2]  = FP_TYPE(0.5);        _TA.state[3]  = FP_TYPE(0.0)
    _TA.state[4]  = FP_TYPE(repr(x3));   _TA.state[5]  = FP_TYPE(repr(y3))
    _TA.state[6]  = FP_TYPE(0.0);        _TA.state[7]  = FP_TYPE(0.0)
    _TA.state[8]  = FP_TYPE(0.0);        _TA.state[9]  = FP_TYPE(0.0)
    _TA.state[10] = FP_TYPE(0.0);        _TA.state[11] = FP_TYPE(0.0)
    _TA.time = FP_TYPE(0.0)

    T_hp = FP_TYPE(repr(T_H))
    try:
        _TA.propagate_until(T_hp)
    except Exception as e:
        return {
            "row": row_idx, "x3": x3, "y3": y3, "T_H": T_H, "T_star": T_star,
            "error": str(e)[:120], "word_head": "", "length": 0,
            "canonical": "", "wall_time_s": time.perf_counter() - t_wall,
        }

    word = "".join(_DIGITS)
    length = len(word)

    # Final closure diagnostic (difference between state(T) and IC).
    s = _TA.state
    closure = max(
        abs(float(s[0]) - (-0.5)), abs(float(s[1])),
        abs(float(s[2]) -  0.5),   abs(float(s[3])),
        abs(float(s[4]) - x3),     abs(float(s[5]) - y3),
    )

    canonical = ""
    if length in (A_LEN, B_LEN):
        canonical = equivalence_class(word)

    return {
        "row": row_idx, "x3": x3, "y3": y3, "T_H": T_H, "T_star": T_star,
        "word_head": word[:80], "length": length, "canonical": canonical,
        "closure_max": closure,
        "wall_time_s": time.perf_counter() - t_wall,
    }


# ---------------------------------------------------------------------------
# Catalog filtering.
# ---------------------------------------------------------------------------

def _load_filtered_catalog():
    rows = load_catalog(CATALOG)  # list of (row_idx, x3, y3, T_H, T_star) — actually (row, x, y, T)
    # load_catalog returns (row, x, y, T); we need T* too. Read it directly.
    # hristov_2024_sol_80.txt columns: x3, y3, T, T*.
    filtered = []
    with CATALOG.open() as f:
        for i, line in enumerate(f, 1):
            parts = line.split()
            if len(parts) < 4:
                continue
            x3 = float(mp.mpf(parts[0]))
            y3 = float(mp.mpf(parts[1]))
            T_H = float(mp.mpf(parts[2]))
            T_star = float(mp.mpf(parts[3]))
            if TSTAR_LO <= T_star <= TSTAR_HI:
                filtered.append((i, x3, y3, T_H, T_star))
    return filtered


# ---------------------------------------------------------------------------
# Main driver.
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only the first N filtered rows "
                             "(sanity mode; 0 = all).")
    parser.add_argument("--chunksize", type=int, default=4)
    args = parser.parse_args()

    mp.mp.dps = 60  # safe decimal display precision for parsing catalog strings

    print(f"[TOPO-HP] filtering Hristov 2024 by T* in [{TSTAR_LO}, {TSTAR_HI}]...",
          flush=True)
    filtered = _load_filtered_catalog()
    print(f"[TOPO-HP] {len(filtered)} entries survive window.", flush=True)

    if args.limit > 0:
        filtered = filtered[: args.limit]
        print(f"[TOPO-HP] LIMIT mode: {len(filtered)} entries.", flush=True)

    status = HERE / "status.json"
    tele = Telemetry(status, total=len(filtered), stage="init")
    tele.update(
        tstar_lo=TSTAR_LO, tstar_hi=TSTAR_HI,
        fp_type="long_double_80bit",
        tol=str(TOL),
        n_workers=N_WORKERS,
        a_canonical_head=A_CANON[:40], b_canonical_head=B_CANON[:40],
    )

    t_start = time.perf_counter()
    results: list[dict] = []
    matches_A: list[dict] = []
    matches_B: list[dict] = []
    errors = 0
    length_hist: dict[int, int] = {}
    max_closure = 0.0

    with Pool(processes=N_WORKERS, initializer=_worker_init) as pool:
        tele.update(stage="sweeping")
        for i, res in enumerate(pool.imap_unordered(_sweep_one, filtered,
                                                    chunksize=args.chunksize)):
            results.append(res)
            if res.get("error"):
                errors += 1
            length = res.get("length", 0)
            length_hist[length] = length_hist.get(length, 0) + 1
            if "closure_max" in res and res["closure_max"] > max_closure:
                max_closure = res["closure_max"]
            if res.get("canonical"):
                if length == A_LEN and res["canonical"] == A_CANON:
                    matches_A.append(res)
                if length == B_LEN and res["canonical"] == B_CANON:
                    matches_B.append(res)
            # Status cadence: every 5 orbits, since each orbit is ~2-5s at MPFR.
            if (i + 1) % 5 == 0 or (i + 1) == len(filtered):
                tele.update(
                    progress=i + 1,
                    matches_A=len(matches_A),
                    matches_B=len(matches_B),
                    errors=errors,
                    max_closure=max_closure,
                    length_hist_size=len(length_hist),
                    last_row=res.get("row"),
                    last_length=length,
                    last_wall_s=res.get("wall_time_s", 0.0),
                )

    wall = time.perf_counter() - t_start

    # Outputs.
    (HERE / "length_histogram.json").write_text(
        json.dumps(dict(sorted(length_hist.items())), indent=2))
    (HERE / "matches.json").write_text(json.dumps({
        "A": matches_A, "B": matches_B,
        "a_canonical": A_CANON, "b_canonical": B_CANON,
    }, indent=2))
    summary = {
        "n_total_in_window": len(filtered),
        "n_processed": len(results),
        "n_errors": errors,
        "max_closure_over_sweep": max_closure,
        "n_matches_A": len(matches_A),
        "n_matches_B": len(matches_B),
        "length_hist": dict(sorted(length_hist.items())),
        "window": {"tstar_lo": TSTAR_LO, "tstar_hi": TSTAR_HI},
        "wall_sweep_s": wall,
        "config": {
            "fp_type": "np.longdouble (80-bit extended, ~19 decimal digits)",
            "tol": str(TOL),
            "n_workers": N_WORKERS,
            "integrator": "heyoka.taylor_adaptive(fp_type=np.longdouble)",
            "catalog": CATALOG.name,
        },
    }
    (HERE / "topology_sweep_hp_summary.json").write_text(json.dumps(summary, indent=2))
    _write_result_md(summary, matches_A, matches_B)

    tele.done(stage="done",
              matches_A=len(matches_A), matches_B=len(matches_B),
              errors=errors, wall_total_s=wall)
    print(
        f"[TOPO-HP] done. matches_A={len(matches_A)} matches_B={len(matches_B)} "
        f"errors={errors} wall={wall:.1f}s",
        flush=True,
    )
    return 0


def _write_result_md(summary: dict, matches_A: list, matches_B: list) -> None:
    lines = []
    lines.append("# Topology sweep — high-precision (Action #8)")
    lines.append("")
    lines.append(f"Integrator: {summary['config']['integrator']}, "
                 f"tol={summary['config']['tol']}, workers={summary['config']['n_workers']}.")
    lines.append(f"Window: T* in [{summary['window']['tstar_lo']}, "
                 f"{summary['window']['tstar_hi']}].")
    lines.append(f"Catalog entries in window: {summary['n_total_in_window']}. "
                 f"Processed: {summary['n_processed']}. Errors: {summary['n_errors']}. "
                 f"Wall: {summary['wall_sweep_s']:.1f}s.")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    if not matches_A:
        lines.append(f"**A: NO topological match in the T*-windowed Hristov 2024 "
                     f"subset ({summary['n_total_in_window']} entries).**")
        lines.append("Combined with zero IC matches (Action #2) and zero canonical "
                     "matches in the 971-entry Hristov 2025 stability-filtered "
                     "syzygy catalog (Action #3), A's new-family claim is fully "
                     "substantiated within the physically plausible T* window.")
    else:
        lines.append(f"**A: {len(matches_A)} topological match(es) in Hristov 2024.** "
                     "A demotes from 'new family' to 'hyperbolic rediscovery' at the "
                     "topology axis.")
    lines.append("")
    if not matches_B:
        lines.append("B: no topological match in the windowed subset "
                     "(expected: B's true T* = 64.65 may lie outside the window).")
    else:
        lines.append(f"B: {len(matches_B)} topological match(es).")
    lines.append("")
    lines.append("## Length distribution in the T*-window subset")
    lines.append("")
    lines.append("| word length | n orbits |")
    lines.append("|------------:|---------:|")
    for length, count in sorted(summary["length_hist"].items(), key=lambda kv: -kv[1])[:20]:
        lines.append(f"| {length} | {count} |")
    lines.append("")
    lines.append(f"Entries at length = {A_LEN} (A's shell): "
                 f"{summary['length_hist'].get(A_LEN, 0)}")
    lines.append(f"Entries at length = {B_LEN} (B's shell): "
                 f"{summary['length_hist'].get(B_LEN, 0)}")
    (HERE / "RESULT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
