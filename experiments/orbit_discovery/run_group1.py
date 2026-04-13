"""Group 1: Orbit Verification (Tests 1.1-1.4).

Reads candidates from experiment 22 results, runs full Group 1 verification,
writes results to subdirectories 01-04 and GROUP1_VERDICT.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from pipeline import verify_candidate

OUT_DIR = Path(__file__).parent
MIN_SURVIVORS = 3  # PROMPT stop condition


def load_experiment22_candidates():
    """Load refined candidates from experiment 22."""
    results_path = PROJECT_ROOT / "experiments" / "22_periodic_orbit_search" / "results.json"
    with open(results_path) as f:
        data = json.load(f)
    candidates = []
    for r in data["refined"]:
        candidates.append({
            "n0": r["n0_refined"],
            "T": r["period"],
            "original_dist": r["return_dist_refined"],
            "classification": r["classification"],
        })
    return candidates


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def main():
    print("=" * 60)
    print("GROUP 1: ORBIT VERIFICATION")
    print("=" * 60)

    candidates = load_experiment22_candidates()
    print(f"Loaded {len(candidates)} candidates from experiment 22")

    all_results = []
    for i, cand in enumerate(candidates):
        print(f"\n{'='*40}")
        print(f"Candidate {i+1}/{len(candidates)}: "
              f"T={cand['T']:.3f}, orig_class={cand['classification']}")
        print(f"{'='*40}")

        result = verify_candidate(
            cand["n0"], cand["T"],
            refine=True, n_steps=10_000, verbose=True)
        result["candidate_idx"] = i
        result["original_classification"] = cand["classification"]
        all_results.append(result)

    # Write per-test outputs
    write_json(OUT_DIR / "01_phase_closure" / "verified.json",
               [r for r in all_results if r["test_1_1"]["pass"]])
    write_json(OUT_DIR / "02_conservation" / "conservation_report.json",
               [{"idx": r["candidate_idx"], **r.get("test_1_2", {})} for r in all_results
                if "test_1_2" in r])
    write_json(OUT_DIR / "03_min_distance" / "distance_report.json",
               [{"idx": r["candidate_idx"], **r.get("test_1_3", {})} for r in all_results
                if "test_1_3" in r])
    write_json(OUT_DIR / "04_multi_period" / "stability_trace.json",
               [{"idx": r["candidate_idx"], **r.get("test_1_4", {})} for r in all_results
                if "test_1_4" in r])

    # Group 1 gate
    survivors = [r for r in all_results if r.get("group1_pass", False)]
    write_json(OUT_DIR / "group1_verified.json", survivors)

    # Write verdict
    verdict_lines = [
        "# GROUP 1 VERDICT: Orbit Verification\n\n",
        f"## Summary\n\n",
        f"Tested {len(all_results)} candidates from experiment 22. "
        f"{len(survivors)} passed all Group 1 tests.\n\n",
        f"## Per-test results\n\n",
    ]
    for r in all_results:
        idx = r["candidate_idx"]
        t11 = r["test_1_1"]["classification"]
        t12 = r.get("test_1_2", {}).get("pass", "N/A")
        t13_col = r.get("test_1_3", {}).get("collisionless", "N/A")
        t14 = r.get("test_1_4", {}).get("pass", "N/A")
        gp = r.get("group1_pass", False)
        verdict_lines.append(
            f"- Candidate {idx}: closure={t11}, conservation={t12}, "
            f"collisionless={t13_col}, multi-period={t14}, "
            f"**GROUP1={'PASS' if gp else 'FAIL'}**\n")

    verdict_lines.append(f"\n## Gate decision\n\n")
    if len(survivors) >= MIN_SURVIVORS:
        verdict_lines.append(
            f"**PASS**: {len(survivors)} >= {MIN_SURVIVORS} survivors. "
            f"Proceed to Group 2.\n")
    else:
        verdict_lines.append(
            f"**FAIL**: {len(survivors)} < {MIN_SURVIVORS} survivors. "
            f"Per PROMPT: rerun classifier-prior search (Test 4.4 protocol) "
            f"to generate a new candidate pool, then restart Group 1.\n")

    (OUT_DIR / "GROUP1_VERDICT.md").write_text("".join(verdict_lines))
    print(f"\nGroup 1 complete: {len(survivors)}/{len(all_results)} passed")
    print(f"Verdict written to {OUT_DIR / 'GROUP1_VERDICT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
