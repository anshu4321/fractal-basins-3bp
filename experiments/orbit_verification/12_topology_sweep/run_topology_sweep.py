"""Action #7: Topology sweep of orbits A and B against the full Hristov 2024
catalog (24,582 free-fall initial conditions, 12,409 distinct orbits).

Closes the gap flagged in the conference-paper Limitations (ii): the paper's
new-family claim for A rests on the 971 Hristov 2025 stability-filtered
syzygy labels. The full Hristov 2024 catalog does NOT publish syzygies, so
we must forward-integrate every entry and extract its syzygy word here.

Pipeline per catalog entry:
  1. Forward-integrate free-fall IC (x3, y3, zero momenta, bodies at
     (-0.5, 0), (+0.5, 0), (x3, y3)) for one published period T_H at
     Yoshida-6 float64, N_STEPS steps.
  2. At each step, compute signed area. On sign change, linearly interpolate
     to locate the collinear instant and identify the midpoint body (which
     of bodies 0, 1, 2 is spatially between the other two).
  3. Accumulate the sequence of midpoint body indices over the period.
  4. Canonicalize the word under cyclic rotation + S_3 body permutation +
     time reversal (reuse 09_syzygy_words/compute_words.equivalence_class).
  5. Compare against A's and B's precomputed canonical forms.

Telemetry: writes `status.json` in this directory at min_interval_s = 0.5.
Probe with:
    ssh ... 'cat ~/3bp/experiments/orbit_verification/12_topology_sweep/status.json'

Output:
    topology_sweep_results.json — full result set (24,582 rows summarised)
    matches.json                — just the rows that match A or B under
                                   equivalence (if any)
    length_histogram.json       — word-length distribution across the catalog
    RESULT.md                   — short verdict

Verification mode (run with --validate-only) re-runs the pipeline on our
own A, B, C, D orbits (via the EULER section, not free-fall) and checks the
canonical words match what Action #3 recorded. This validates the
extraction+canonicalisation before the sweep.
"""
from __future__ import annotations

import argparse
import json
import math as _math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from mega3bp.gcp_telemetry import Telemetry  # noqa: E402

# Import the canonicalisation + integrator machinery from prior actions.
sys.path.insert(0, str(REPO / "experiments" / "orbit_verification" / "09_syzygy_words"))
sys.path.insert(0, str(REPO / "experiments" / "orbit_verification" / "08_hristov_2024_sweep"))
from compute_words import equivalence_class  # noqa: E402
from sweep_AB_vs_hristov_2024 import _yoshida6_step_scalar, load_catalog  # noqa: E402

CATALOG = REPO / "experiments/orbit_verification/00_catalogs/hristov_2024_sol_80.txt"
HP_RESULTS = REPO / "experiments/orbit_verification/06_high_precision/hp_heyoka_newton_results.json"

# Action #3 canonical words (Hristov body numbering).
# These are precomputed; we do NOT reparse words.json because the exact
# canonical form is what we must match.
# A: "132132132132132132132132" (length 24)
# B: "312312312312312312312312312312312312312312" (length 42)
# C: "213213213213213213213213213213213213213213" (length 42)
# D: "213213213213213213213213213213213213213213213213" (length 48)
A_WORD_HRISTOV = "132132132132132132132132"
B_WORD_HRISTOV = "312312312312312312312312312312312312312312"
C_WORD_HRISTOV = "213213213213213213213213213213213213213213"
D_WORD_HRISTOV = "213213213213213213213213213213213213213213213213"

# The sweep's integrator uses internal body indices 0, 1, 2. We will build
# the word in internal indices 1..3 (1-indexed), then canonicalise. Because
# canonicalisation permutes all of {1, 2, 3}, the integrator's index labels
# are equivalent to Hristov's for the purpose of matching under S_3.
A_CANONICAL: str = ""  # filled in main()
B_CANONICAL: str = ""
C_CANONICAL: str = ""
D_CANONICAL: str = ""

N_STEPS = 50_000
N_WORKERS = int(os.environ.get("TOPO_WORKERS", "12"))


# ---------------------------------------------------------------------------
# Worker: integrate one free-fall IC and extract its syzygy word.
# ---------------------------------------------------------------------------

def _midpoint_body(q0x, q0y, q1x, q1y, q2x, q2y) -> int:
    """At a near-collinear configuration, return internal body index (0, 1,
    or 2) that lies between the other two. Uses ratio of perpendicular
    distance to pair separation (returns body with smallest ratio)."""
    best = float("inf")
    mid = 0
    # body 0 midpoint of 1, 2?
    mx, my = 0.5 * (q1x + q2x), 0.5 * (q1y + q2y)
    dx, dy = q1x - q2x, q1y - q2y
    d_out = (dx * dx + dy * dy) ** 0.5
    if d_out > 1e-12:
        ex, ey = q0x - mx, q0y - my
        dist = ((ex * ex + ey * ey) ** 0.5) / d_out
        if dist < best:
            best, mid = dist, 0
    # body 1 midpoint of 0, 2?
    mx, my = 0.5 * (q0x + q2x), 0.5 * (q0y + q2y)
    dx, dy = q0x - q2x, q0y - q2y
    d_out = (dx * dx + dy * dy) ** 0.5
    if d_out > 1e-12:
        ex, ey = q1x - mx, q1y - my
        dist = ((ex * ex + ey * ey) ** 0.5) / d_out
        if dist < best:
            best, mid = dist, 1
    # body 2 midpoint of 0, 1?
    mx, my = 0.5 * (q0x + q1x), 0.5 * (q0y + q1y)
    dx, dy = q0x - q1x, q0y - q1y
    d_out = (dx * dx + dy * dy) ** 0.5
    if d_out > 1e-12:
        ex, ey = q2x - mx, q2y - my
        dist = ((ex * ex + ey * ey) ** 0.5) / d_out
        if dist < best:
            best, mid = dist, 2
    return mid


def sweep_one(args):
    """Worker entrypoint. args = (row_idx, x3, y3, T_H, target_lengths).

    target_lengths is the set of word lengths for which we need the canonical
    form (e.g., {24, 42}). For other lengths we only return the length, no
    canonical (a big speedup)."""
    row_idx, x3, y3, T_H, target_lengths = args
    t0 = time.perf_counter()

    q0x, q0y = -0.5, 0.0
    q1x, q1y = 0.5, 0.0
    q2x, q2y = float(x3), float(y3)
    p0x = p0y = p1x = p1y = p2x = p2y = 0.0
    h = T_H / N_STEPS

    # initial signed area
    area_prev = (q1x - q0x) * (q2y - q0y) - (q1y - q0y) * (q2x - q0x)
    q0x_p, q0y_p = q0x, q0y
    q1x_p, q1y_p = q1x, q1y
    q2x_p, q2y_p = q2x, q2y

    digits: list[str] = []
    initial_is_syzygy = abs(area_prev) < 1e-10
    q_init = (q0x, q0y, q1x, q1y, q2x, q2y)
    first_sign = 0  # sign of the first non-zero area sample we see (+1/-1)
    last_sign = 0

    n_crossings = 0

    for k in range(1, N_STEPS + 1):
        out = _yoshida6_step_scalar(
            q0x, q0y, q1x, q1y, q2x, q2y,
            p0x, p0y, p1x, p1y, p2x, p2y, h)
        (q0x, q0y, q1x, q1y, q2x, q2y,
         p0x, p0y, p1x, p1y, p2x, p2y) = out

        if (k & 1023) == 0 and not _math.isfinite(q0x):
            return {
                "row": row_idx, "error": "divergence_at_step_%d" % k,
                "word": "", "length": 0, "canonical": "",
                "wall_time_s": time.perf_counter() - t0,
            }

        area = (q1x - q0x) * (q2y - q0y) - (q1y - q0y) * (q2x - q0x)
        if area > 1e-12:
            if first_sign == 0:
                first_sign = 1
            last_sign = 1
        elif area < -1e-12:
            if first_sign == 0:
                first_sign = -1
            last_sign = -1

        if (area_prev > 0.0 and area < 0.0) or (area_prev < 0.0 and area > 0.0):
            # sign change -> linear interpolation to locate the collinear instant
            denom = area_prev - area
            alpha = 0.5 if denom == 0.0 else area_prev / denom
            if alpha < 0.0:
                alpha = 0.0
            elif alpha > 1.0:
                alpha = 1.0
            q0x_c = q0x_p + alpha * (q0x - q0x_p)
            q0y_c = q0y_p + alpha * (q0y - q0y_p)
            q1x_c = q1x_p + alpha * (q1x - q1x_p)
            q1y_c = q1y_p + alpha * (q1y - q1y_p)
            q2x_c = q2x_p + alpha * (q2x - q2x_p)
            q2y_c = q2y_p + alpha * (q2y - q2y_p)
            m = _midpoint_body(q0x_c, q0y_c, q1x_c, q1y_c, q2x_c, q2y_c)
            digits.append(str(m + 1))  # 0,1,2 -> 1,2,3 (1-indexed)
            n_crossings += 1

        area_prev = area
        q0x_p, q0y_p = q0x, q0y
        q1x_p, q1y_p = q1x, q1y
        q2x_p, q2y_p = q2x, q2y

    # Wrap-around crossing: if the orbit begins at a collinear configuration
    # (|area(0)| ~ 0) and the first and last non-zero area samples have
    # opposite signs, the cyclic sequence has one sign change through t=0/T
    # that the linear loop missed. Register it at the initial config
    # (equivalent to the final by periodicity).
    if initial_is_syzygy and first_sign != 0 and last_sign != 0 and first_sign != last_sign:
        m_wrap = _midpoint_body(*q_init)
        digits.append(str(m_wrap + 1))

    closure_max = max(
        abs(q0x - (-0.5)), abs(q0y), abs(q1x - 0.5), abs(q1y),
        abs(q2x - float(x3)), abs(q2y - float(y3)),
    )
    word = "".join(digits)
    length = len(word)
    canonical = equivalence_class(word) if length in target_lengths and length > 0 else ""

    return {
        "row": row_idx,
        "x3": x3, "y3": y3, "T_H": T_H,
        "word_head": word[:60],
        "length": length,
        "canonical": canonical,
        "closure_max": closure_max,
        "wall_time_s": time.perf_counter() - t0,
    }


# ---------------------------------------------------------------------------
# Validation: re-run our own orbits through the same pipeline (Euler IC
# path, not free-fall) and confirm canonical forms match the committed ones.
# ---------------------------------------------------------------------------

def _integrate_euler_ic_and_extract_word(v1: float, v2: float, T: float, n_steps: int = N_STEPS):
    """Euler-section IC: q1=(-1,0), q2=(1,0), q3=(0,0), p1=p2=(v1,v2),
    p3=-2(v1,v2). Returns the raw syzygy word (digit string, 1-indexed)."""
    q0x, q0y = -1.0, 0.0
    q1x, q1y = 1.0, 0.0
    q2x, q2y = 0.0, 0.0
    p0x, p0y = v1, v2
    p1x, p1y = v1, v2
    p2x, p2y = -2.0 * v1, -2.0 * v2
    h = T / n_steps

    area_prev = (q1x - q0x) * (q2y - q0y) - (q1y - q0y) * (q2x - q0x)
    q0x_p, q0y_p = q0x, q0y
    q1x_p, q1y_p = q1x, q1y
    q2x_p, q2y_p = q2x, q2y
    digits: list[str] = []
    initial_is_syzygy = abs(area_prev) < 1e-10
    q_init = (q0x, q0y, q1x, q1y, q2x, q2y)
    first_sign = 0
    last_sign = 0

    for k in range(1, n_steps + 1):
        out = _yoshida6_step_scalar(
            q0x, q0y, q1x, q1y, q2x, q2y,
            p0x, p0y, p1x, p1y, p2x, p2y, h)
        (q0x, q0y, q1x, q1y, q2x, q2y,
         p0x, p0y, p1x, p1y, p2x, p2y) = out

        area = (q1x - q0x) * (q2y - q0y) - (q1y - q0y) * (q2x - q0x)
        if area > 1e-12:
            if first_sign == 0:
                first_sign = 1
            last_sign = 1
        elif area < -1e-12:
            if first_sign == 0:
                first_sign = -1
            last_sign = -1

        if (area_prev > 0.0 and area < 0.0) or (area_prev < 0.0 and area > 0.0):
            denom = area_prev - area
            alpha = 0.5 if denom == 0.0 else area_prev / denom
            if alpha < 0.0:
                alpha = 0.0
            elif alpha > 1.0:
                alpha = 1.0
            q0x_c = q0x_p + alpha * (q0x - q0x_p)
            q0y_c = q0y_p + alpha * (q0y - q0y_p)
            q1x_c = q1x_p + alpha * (q1x - q1x_p)
            q1y_c = q1y_p + alpha * (q1y - q1y_p)
            q2x_c = q2x_p + alpha * (q2x - q2x_p)
            q2y_c = q2y_p + alpha * (q2y - q2y_p)
            m = _midpoint_body(q0x_c, q0y_c, q1x_c, q1y_c, q2x_c, q2y_c)
            digits.append(str(m + 1))

        area_prev = area
        q0x_p, q0y_p = q0x, q0y
        q1x_p, q1y_p = q1x, q1y
        q2x_p, q2y_p = q2x, q2y

    if initial_is_syzygy and first_sign != 0 and last_sign != 0 and first_sign != last_sign:
        m_wrap = _midpoint_body(*q_init)
        digits.append(str(m_wrap + 1))

    return "".join(digits)


def validate_pipeline(tele: Telemetry) -> dict:
    """Re-extract A, B, C, D syzygy words from our HP Euler ICs and confirm
    their canonical forms match the Action #3 reference words."""
    tele.update(stage="validating_pipeline")
    hp = json.loads(HP_RESULTS.read_text())
    ics = {r["name"]: r for r in hp if r.get("name") in {"A", "B", "C", "D"}}
    report = {}
    for name, expected in (
        ("A", A_WORD_HRISTOV),
        ("B", B_WORD_HRISTOV),
        ("C", C_WORD_HRISTOV),
        ("D", D_WORD_HRISTOV),
    ):
        v1, v2, T = float(ics[name]["v1_HP"]), float(ics[name]["v2_HP"]), float(ics[name]["T_HP"])
        w = _integrate_euler_ic_and_extract_word(v1, v2, T)
        ok = equivalence_class(w) == equivalence_class(expected) and len(w) == len(expected)
        report[name] = {
            "expected_length": len(expected),
            "observed_length": len(w),
            "expected_canonical_head": equivalence_class(expected)[:30],
            "observed_canonical_head": equivalence_class(w)[:30] if w else "",
            "equivalent": ok,
        }
        print(
            f"[TOPO][validate] {name}: len_obs={len(w):3d} vs "
            f"len_exp={len(expected):3d}  equivalent={ok}",
            flush=True,
        )
    tele.update(stage="validation_complete", validation=report)
    return report


# ---------------------------------------------------------------------------
# Main driver.
# ---------------------------------------------------------------------------

def main() -> int:
    global A_CANONICAL, B_CANONICAL, C_CANONICAL, D_CANONICAL

    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true",
                        help="Run only the Euler-IC validation on A/B/C/D and exit.")
    parser.add_argument("--chunksize", type=int, default=32)
    args = parser.parse_args()

    A_CANONICAL = equivalence_class(A_WORD_HRISTOV)
    B_CANONICAL = equivalence_class(B_WORD_HRISTOV)
    C_CANONICAL = equivalence_class(C_WORD_HRISTOV)
    D_CANONICAL = equivalence_class(D_WORD_HRISTOV)

    status_path = HERE / "status.json"
    tele = Telemetry(status_path, total=1, stage="init")
    tele.update(
        a_canonical_head=A_CANONICAL[:40],
        b_canonical_head=B_CANONICAL[:40],
    )

    # ---- validation ----
    val = validate_pipeline(tele)
    if not all(v["equivalent"] for v in val.values()):
        tele.error("validation_failed", validation=val)
        (HERE / "validation.json").write_text(json.dumps(val, indent=2))
        print("[TOPO] validation failed; aborting sweep.", flush=True)
        return 2
    (HERE / "validation.json").write_text(json.dumps(val, indent=2))

    if args.validate_only:
        tele.done(stage="validation_only_done", validation=val)
        return 0

    # ---- full sweep ----
    print(f"[TOPO] loading catalog {CATALOG.name}...", flush=True)
    catalog = load_catalog(CATALOG)
    n_total = len(catalog)
    print(f"[TOPO] catalog loaded: {n_total} rows", flush=True)

    tele.total = n_total
    tele.update(progress=0, stage="sweeping", total=n_total,
                n_workers=N_WORKERS, n_steps=N_STEPS,
                validation={k: v["equivalent"] for k, v in val.items()})

    target_lengths = {len(A_WORD_HRISTOV), len(B_WORD_HRISTOV)}
    args_list = [(r[0], r[1], r[2], r[3], target_lengths) for r in catalog]

    t_sweep = time.perf_counter()
    results: list[dict] = []
    matches_A: list[dict] = []
    matches_B: list[dict] = []
    length_hist: dict[int, int] = {}
    errors = 0
    max_closure = 0.0

    with Pool(processes=N_WORKERS) as pool:
        for i, res in enumerate(pool.imap_unordered(sweep_one, args_list,
                                                    chunksize=args.chunksize)):
            results.append(res)
            if res.get("error"):
                errors += 1
            length = res.get("length", 0)
            length_hist[length] = length_hist.get(length, 0) + 1
            if "closure_max" in res:
                if res["closure_max"] > max_closure:
                    max_closure = res["closure_max"]
            if res.get("canonical"):
                if length == len(A_WORD_HRISTOV) and res["canonical"] == A_CANONICAL:
                    matches_A.append(res)
                if length == len(B_WORD_HRISTOV) and res["canonical"] == B_CANONICAL:
                    matches_B.append(res)
            if (i + 1) % 50 == 0 or (i + 1) == n_total:
                tele.update(
                    progress=i + 1,
                    stage="sweeping",
                    matches_A=len(matches_A),
                    matches_B=len(matches_B),
                    errors=errors,
                    max_closure=max_closure,
                    length_hist_size=len(length_hist),
                )

    wall = time.perf_counter() - t_sweep
    tele.update(stage="finalising", matches_A=len(matches_A),
                matches_B=len(matches_B), errors=errors,
                max_closure=max_closure, wall_sweep_s=wall)

    # persist outputs
    (HERE / "length_histogram.json").write_text(
        json.dumps(dict(sorted(length_hist.items())), indent=2))
    (HERE / "matches.json").write_text(json.dumps(
        {"A": matches_A, "B": matches_B, "a_canonical": A_CANONICAL,
         "b_canonical": B_CANONICAL},
        indent=2))
    # full results summary (drop the per-row canonical strings to save size)
    summary = {
        "n_total": n_total,
        "n_processed": len(results),
        "n_errors": errors,
        "max_closure_over_sweep": max_closure,
        "n_matches_A": len(matches_A),
        "n_matches_B": len(matches_B),
        "length_hist": dict(sorted(length_hist.items())),
        "target_lengths": sorted(target_lengths),
        "wall_sweep_s": wall,
        "config": {
            "n_steps": N_STEPS, "n_workers": N_WORKERS,
            "integrator": "yoshida6_float64",
            "catalog": str(CATALOG.name),
        },
        "validation": val,
    }
    (HERE / "topology_sweep_summary.json").write_text(json.dumps(summary, indent=2))

    _write_result_md(summary, matches_A, matches_B)

    tele.done(stage="done", matches_A=len(matches_A),
              matches_B=len(matches_B), errors=errors,
              wall_total_s=wall)
    print(
        f"[TOPO] done. matches_A={len(matches_A)}, matches_B={len(matches_B)}, "
        f"errors={errors}, wall={wall:.1f}s",
        flush=True,
    )
    return 0


def _write_result_md(summary: dict, matches_A: list, matches_B: list) -> None:
    lines = []
    lines.append("# Topology sweep (Action #7) — A, B vs full Hristov 2024 catalog")
    lines.append("")
    lines.append(f"Integrator: Yoshida-6 float64, N_STEPS={summary['config']['n_steps']} "
                 f"per published period, workers={summary['config']['n_workers']}.")
    lines.append(f"Wall: {summary['wall_sweep_s']:.1f}s on GCP; "
                 f"max closure across sweep: {summary['max_closure_over_sweep']:.3e}.")
    lines.append(f"Catalog: {summary['n_processed']} / {summary['n_total']} rows "
                 f"processed; errors: {summary['n_errors']}.")
    lines.append("")
    lines.append("## Validation gate")
    lines.append("")
    lines.append("| orbit | eq-class match |")
    lines.append("|-------|---------------:|")
    for name in ("A", "B", "C", "D"):
        ok = summary["validation"][name]["equivalent"]
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} |")
    lines.append("")
    lines.append("## Sweep verdict")
    lines.append("")
    if len(matches_A) == 0:
        lines.append(f"**A: no topological match in the full {summary['n_total']}-row "
                     "Hristov 2024 catalog.** A's new-family claim hardens.")
    else:
        lines.append(f"**A: {len(matches_A)} topological match(es) in Hristov 2024.** "
                     "A demotes from 'new family' to 'hyperbolic rediscovery' at the topology "
                     "axis. IC-level sweep (Action #2) already showed zero STRONG/WEAK matches.")
    lines.append("")
    if len(matches_B) == 0:
        lines.append("B: no topological match in Hristov 2024.")
    else:
        lines.append(f"B: {len(matches_B)} topological match(es) in Hristov 2024 "
                     "(expected — B lies in a known family shared with orbit C and "
                     "Hristov 2025 row 7).")
    lines.append("")
    lines.append("## Per-orbit details")
    lines.append("")
    for name, matches in (("A", matches_A), ("B", matches_B)):
        lines.append(f"### Orbit {name}")
        if not matches:
            lines.append(f"No rows in Hristov 2024 with equivalent canonical syzygy word.")
        else:
            lines.append(f"| row | x3 | y3 | T_H | word head |")
            lines.append(f"|----:|----|----|-----|-----------|")
            for m in matches[:25]:
                lines.append(
                    f"| {m['row']} | {m['x3']:.6f} | {m['y3']:.6f} | "
                    f"{m['T_H']:.4f} | `{m['word_head'][:30]}` |")
            if len(matches) > 25:
                lines.append(f"| ... | ... | ... | ... | {len(matches) - 25} more |")
        lines.append("")
    lines.append("## Length histogram (top 15 bins by count)")
    lines.append("")
    lines.append("| word length | n orbits |")
    lines.append("|------------:|---------:|")
    sorted_hist = sorted(summary["length_hist"].items(), key=lambda kv: -kv[1])
    for length, count in sorted_hist[:15]:
        lines.append(f"| {length} | {count} |")
    lines.append("")
    (HERE / "RESULT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
