"""Action #4 — Independent HP cross-check via heyoka 200-bit.

For each of our four novel/match orbits (A, B, C, D):
  1. Load HP-refined initial conditions (v1, v2, T) from
     06_high_precision/hp_heyoka_newton_results.json.
  2. Build the Euler-section IC in our convention:
         q = [(-1, 0), (1, 0), (0, 0)]
         p = [(v1, v2), (v1, v2), (-2 v1, -2 v2)]
  3. Integrate for one full period T using heyoka's mpfr Taylor integrator
     at 200-bit precision (roughly 60 decimal digits of working precision).
  4. Report max-component closure residual |state(T) - state(0)|_inf.
  5. Compare with the existing mpmath-Taylor HP residuals and check that
     residual < 1e-30 under BOTH tools (dual confirmation).

This is the gold-standard independence check: mpmath Taylor and heyoka mpfr
Taylor share almost nothing in their implementation (pure-Python interpreted
arbitrary-precision Taylor in one case, JIT-compiled MPFR Taylor in the
other), so agreement of both tools at < 1e-30 residual confirms the orbit is
genuinely periodic — not a product of bugs in a single integrator.

Sanity 1 from Action #1 already produced the heyoka-200bit residual for
orbit C (4.15e-49). This script extends the check to A, B, D so all four
orbits carry a dual-tool verification record.

Usage (on GCP):
    python hp_cross_check.py

Outputs written in this directory:
    verdict.json        machine-readable dual-tool summary
    RESULT.md           human-readable report with comparison table
    status.json         live telemetry (via mega3bp.gcp_telemetry.Telemetry)
    run.log             stdout/stderr (if launched via nohup)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import heyoka as hy
import mpmath as mp

mp.mp.dps = 120

EXP_DIR = Path(__file__).resolve().parent
# File lives at <root>/experiments/orbit_verification/10_hp_cross_check/hp_cross_check.py,
# so parents[3] is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
# Add project root so `from mega3bp...` works even if PYTHONPATH is unset.
sys.path.insert(0, str(PROJECT_ROOT))
# And the 07 dir so we can reuse build_hp_integrator / get_our_orbits.
sys.path.insert(0, str(PROJECT_ROOT / "experiments/orbit_verification/07_hristov_2025_definitive"))

from check_C_D_vs_hristov_2025 import build_hp_integrator, get_our_orbits  # noqa: E402
from mega3bp.gcp_telemetry import Telemetry  # noqa: E402


HP_RESULTS_FILE = (PROJECT_ROOT
                   / "experiments/orbit_verification/06_high_precision/hp_heyoka_newton_results.json")

ORBITS = ["A", "B", "C", "D"]
GATE = 1e-30


def _build_euler_ic(v1: mp.mpf, v2: mp.mpf):
    """Construct the 12-element heyoka real IC at 200-bit for our Euler section.

    Convention (from the orbit discovery pipeline):
        q = [(-1, 0), (1, 0), (0, 0)]
        p = [(v1, v2), (v1, v2), (-2 v1, -2 v2)]
    """
    q = [(mp.mpf(-1), mp.mpf(0)),
         (mp.mpf(1),  mp.mpf(0)),
         (mp.mpf(0),  mp.mpf(0))]
    v = [(v1,       v2),
         (v1,       v2),
         (-2 * v1, -2 * v2)]
    ic = []
    for qi in q:
        ic += [hy.real(mp.nstr(qi[0], 50), prec=200),
               hy.real(mp.nstr(qi[1], 50), prec=200)]
    for vi in v:
        ic += [hy.real(mp.nstr(vi[0], 50), prec=200),
               hy.real(mp.nstr(vi[1], 50), prec=200)]
    return ic


def _load_mpmath_taylor_residuals() -> dict[str, dict]:
    """Return {orbit: {residual_float, residual_str, pass}} from the existing
    30-dps mpmath Taylor HP verification."""
    data = json.loads(HP_RESULTS_FILE.read_text())
    out = {}
    for r in data:
        name = r.get("name")
        if name not in ORBITS:
            continue
        residual = float(r["residual_HP_float"])
        out[name] = {
            "residual": residual,
            "residual_HP_string": r["residual_HP"],
            "pass": bool(residual < GATE),
            "v1_HP": r["v1_HP"],
            "v2_HP": r["v2_HP"],
            "T_HP": r["T_HP"],
        }
    missing = [o for o in ORBITS if o not in out]
    if missing:
        raise RuntimeError(f"HP results missing orbits: {missing}")
    return out


def run_heyoka_hp_cross_check(orbit_name: str, ta, T_mp: mp.mpf,
                              v1: mp.mpf, v2: mp.mpf) -> dict:
    """Integrate a single orbit over one period at heyoka 200-bit."""
    ic = _build_euler_ic(v1, v2)
    ta.state[:] = ic
    ta.time = hy.real("0.0", prec=200)
    T_end = hy.real(mp.nstr(T_mp, 50), prec=200)

    t0 = time.perf_counter()
    ta.propagate_until(T_end)
    wall = time.perf_counter() - t0

    residuals = [abs(float(ta.state[i] - ic[i])) for i in range(12)]
    residual = max(residuals)
    per_component = [float(r) for r in residuals]

    passed = residual < GATE
    print(f"   orbit {orbit_name}: T={float(T_mp):.4f}, wall={wall:.2f}s, "
          f"residual={residual:.3e}  [{'PASS' if passed else 'FAIL'}]")
    return {
        "orbit": orbit_name,
        "residual": residual,
        "pass": passed,
        "wall_s": wall,
        "per_component_residual": per_component,
        "T": mp.nstr(T_mp, 20),
    }


def write_result_md(verdict: dict, out_path: Path) -> None:
    """Produce a terse human-readable dual-tool comparison report."""
    heyoka = verdict["heyoka_200bit"]
    mpm = verdict["mpmath_taylor_30dps"]
    dual = verdict["dual_confirmation"]
    gate = verdict["gate"]
    total_wall = sum(heyoka[o]["wall_s"] for o in ORBITS)
    all_pass = all(dual[o] == "PASS" for o in ORBITS)

    lines = []
    lines.append("# Action #4 - Independent HP Cross-Check (heyoka 200-bit)\n")
    lines.append(
        "Dual-tool high-precision closure verification. Each orbit is integrated "
        "for one full period T under our Euler-section IC convention, and the "
        "max-component closure residual is compared against the existing "
        "mpmath-Taylor (30 decimal digits) HP results.\n")
    lines.append(f"**Gate:** max closure residual < {gate:g}\n")
    lines.append(f"**Overall verdict:** "
                 f"{'ALL FOUR PASS - dual-tool periodicity confirmed' if all_pass else 'FAILURE on >=1 orbit'}\n")
    lines.append(
        "**Independence:** mpmath-Taylor is a pure-Python interpreted "
        "arbitrary-precision Taylor integrator; heyoka is a JIT-compiled "
        "MPFR-backed Taylor integrator (different Taylor-series generation, "
        "different arithmetic backend). Both closing < 1e-30 on the same IC "
        "rules out tool-specific bugs.\n")

    lines.append("## Results\n")
    lines.append("| Orbit | T | heyoka 200-bit residual | mpmath Taylor 30-dps residual | heyoka wall (s) | dual verdict |")
    lines.append("|:-----:|--:|:-----------------------:|:-----------------------------:|----------------:|:------------:|")
    for o in ORBITS:
        T_str = heyoka[o]["T"]
        lines.append(
            f"| {o} | {T_str[:8]} | {heyoka[o]['residual']:.3e} | "
            f"{mpm[o]['residual']:.3e} | {heyoka[o]['wall_s']:.2f} | "
            f"**{dual[o]}** |")
    lines.append(f"\nTotal heyoka wall-time (sequential): {total_wall:.2f} s\n")

    lines.append("## Interpretation\n")
    if all_pass:
        lines.append(
            "All four orbits close below the 1e-30 gate under *both* "
            "independent HP integrators. Combined with the prior identity work:\n\n"
            "- **Orbit C** = Hristov 2025 #0006 (Action #1, dv_min ~ 4.65e-51).\n"
            "- **Orbit D** = Hristov 2025 #0011 (Action #1, analogous).\n"
            "- **Orbit A** = novel topology, novel in both Hristov catalogs (Actions #2, #3).\n"
            "- **Orbit B** = novel IC (no Hristov match); topology shared with C and Hristov row 7 (Actions #2, #3).\n\n"
            "Dual-tool HP closure means the novelty claims for A (and the IC-level novelty of B) "
            "do not depend on a single integrator's correctness — the closure has been verified by two "
            "tools with disjoint Taylor-series implementations.\n")
    else:
        lines.append(
            "One or more orbits did NOT pass the 1e-30 gate under heyoka 200-bit. "
            "Investigate the failing row(s) before relying on dual confirmation for the paper.\n")

    lines.append("## Methodology\n")
    lines.append(
        "1. IC construction: q = [(-1,0), (1,0), (0,0)], "
        "p = [(v1,v2), (v1,v2), (-2v1,-2v2)], with (v1, v2, T) as 50-digit "
        f"strings from `{HP_RESULTS_FILE.relative_to(PROJECT_ROOT)}`.\n"
        "2. Integrator: `build_hp_integrator(200)` from "
        "`07_hristov_2025_definitive/check_C_D_vs_hristov_2025.py` - heyoka "
        "`taylor_adaptive` with `fp_type=hy.real`, `prec=200`, `tol=1e-50`, "
        "`compact_mode=True`.\n"
        "3. Propagate via `ta.propagate_until(T_end)` (single sweep; no "
        "intermediate dense output needed for closure).\n"
        "4. Closure residual = max over 12 state components of |state(T) - state(0)|.\n"
        "5. The mpmath Taylor residuals are read from the existing "
        "`hp_heyoka_newton_results.json` `residual_HP_float` field.\n")

    out_path.write_text("\n".join(lines) + "\n")


def main():
    t0_total = time.perf_counter()
    print(f"=== Action #4: HP cross-check (heyoka 200-bit) ===")
    print(f"Gate: residual < {GATE:g}")

    status = Telemetry(EXP_DIR / "status.json", total=len(ORBITS), stage="load_refs")
    our_orbits = get_our_orbits()
    mpmath_residuals = _load_mpmath_taylor_residuals()
    for o in ORBITS:
        if o not in our_orbits:
            raise RuntimeError(f"Orbit {o} missing from HP results - run 06_high_precision first.")
    print(f"Loaded HP ICs for: {sorted(our_orbits)}")

    status.update(stage="build_integrator")
    print("Building heyoka HP integrator at 200-bit precision...")
    t_build0 = time.perf_counter()
    ta = build_hp_integrator(200)
    print(f"   built in {time.perf_counter() - t_build0:.1f}s")

    heyoka_results = {}
    for i, orbit in enumerate(ORBITS):
        status.update(progress=i, stage=f"integrate_{orbit}", force=True)
        v1 = our_orbits[orbit]["v1"]
        v2 = our_orbits[orbit]["v2"]
        T = our_orbits[orbit]["T"]
        r = run_heyoka_hp_cross_check(orbit, ta, T, v1, v2)
        heyoka_results[orbit] = {
            "residual": r["residual"],
            "pass": r["pass"],
            "wall_s": r["wall_s"],
            "per_component_residual": r["per_component_residual"],
            "T": r["T"],
        }
        status.update(progress=i + 1, stage=f"done_{orbit}",
                      latest_residual=r["residual"], latest_pass=r["pass"],
                      force=True)

    # Dual-tool verdict
    dual = {}
    for orbit in ORBITS:
        h_pass = heyoka_results[orbit]["pass"]
        m_pass = mpmath_residuals[orbit]["pass"]
        dual[orbit] = "PASS" if (h_pass and m_pass) else (
            "HEYOKA_ONLY" if (h_pass and not m_pass) else (
                "MPMATH_ONLY" if (m_pass and not h_pass) else "FAIL"))

    mpmath_out = {orbit: {
        "residual": mpmath_residuals[orbit]["residual"],
        "pass": mpmath_residuals[orbit]["pass"],
        "residual_HP_string": mpmath_residuals[orbit]["residual_HP_string"],
    } for orbit in ORBITS}

    total_wall = time.perf_counter() - t0_total
    verdict = {
        "gate": GATE,
        "orbits": ORBITS,
        "heyoka_200bit": heyoka_results,
        "mpmath_taylor_30dps": mpmath_out,
        "dual_confirmation": dual,
        "total_wall_s": total_wall,
    }

    (EXP_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str))
    print(f"\nSaved: {EXP_DIR / 'verdict.json'}")

    write_result_md(verdict, EXP_DIR / "RESULT.md")
    print(f"Saved: {EXP_DIR / 'RESULT.md'}")

    all_pass = all(v == "PASS" for v in dual.values())
    print(f"\nDual confirmation: {dual}")
    print(f"Overall: {'ALL PASS' if all_pass else 'NEEDS ATTENTION'}")
    print(f"Total wall time: {total_wall:.2f}s")

    status.done(stage="complete", total_wall_s=total_wall,
                all_pass=all_pass, dual_confirmation=dual)

    if not all_pass:
        raise SystemExit(f"Dual-tool gate FAILED: {dual}")


if __name__ == "__main__":
    main()
