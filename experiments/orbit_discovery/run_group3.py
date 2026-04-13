"""Group 3: Stability Analysis (Tests 3.1-3.3).

Computes monodromy matrix, extracts eigenvalues, classifies stability,
and cross-checks against catalog stability data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from mega3bp.monodromy import monodromy_matrix, analyze_monodromy

OUT_DIR = Path(__file__).parent
N_STEPS = 10_000


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def main():
    print("=" * 60)
    print("GROUP 3: STABILITY ANALYSIS")
    print("=" * 60)

    # Load Group 1 verified orbits
    verified_path = OUT_DIR / "group1_verified.json"
    if not verified_path.exists():
        print("ERROR: group1_verified.json not found. Run Group 1 first.")
        return 1

    with open(verified_path) as f:
        verified = json.load(f)
    print(f"Loaded {len(verified)} Group-1-verified orbits")

    # Test 3.1: Monodromy matrix computation
    eigenvalue_results = []
    stability_results = []

    for v in verified:
        idx = v["candidate_idx"]
        q0 = jnp.array(v["q0"], dtype=jnp.float64).reshape(3, 2)
        p0 = jnp.array(v["p0"], dtype=jnp.float64).reshape(3, 2)
        T = v["T_refined"]

        print(f"\nCandidate {idx}: T={T:.6f}")
        M = monodromy_matrix(q0, p0, T, n_steps=N_STEPS)
        M_np = np.array(M)

        analysis = analyze_monodromy(M_np)
        analysis["candidate_idx"] = idx
        eigenvalue_results.append(analysis)

        # Test 3.2: Stability classification
        stab = {
            "candidate_idx": idx,
            "stability_class": analysis["stability_class"],
            "stability_index": analysis["stability_index"],
            "n_trivial_eigenvalues": analysis["n_trivial"],
            "det_M": analysis["det_M"],
            "reciprocal_check": analysis["reciprocal_check"],
        }
        stability_results.append(stab)

        print(f"  Class: {analysis['stability_class']}, "
              f"Index: {analysis['stability_index']:.4f}, "
              f"Trivial eigs: {analysis['n_trivial']}, "
              f"det(M): {analysis['det_M']:.6f}")

        # Test 3.1 pass criterion: 4 trivial eigenvalues at +1 (tol 1e-6)
        if analysis["n_trivial"] < 4:
            print(f"  WARNING: Only {analysis['n_trivial']} trivial eigenvalues "
                  f"(expected 4). Check integration accuracy.")

    write_json(OUT_DIR / "09_monodromy" / "eigenvalues.json", eigenvalue_results)
    write_json(OUT_DIR / "10_stability" / "stability_table.json", stability_results)

    # Test 3.3: Cross-check with catalog
    matching_path = OUT_DIR / "07_matching" / "matching_table.json"
    cross_check = []
    if matching_path.exists():
        with open(matching_path) as f:
            matches = json.load(f)
        known_orbits = {m["candidate_idx"]: m for m in matches
                        if m["classification"] == "known"}

        for stab in stability_results:
            idx = stab["candidate_idx"]
            if idx in known_orbits:
                cross_check.append({
                    "candidate_idx": idx,
                    "our_class": stab["stability_class"],
                    "catalog_id": known_orbits[idx]["nearest_catalog_id"],
                    "catalog_class": "not_available",
                    "agreement": "unknown",
                })

    write_json(OUT_DIR / "11_stability_cross_check" / "report.json", cross_check)

    # Write cross-check report
    report = ["# Stability Cross-Check Report\n\n"]
    if cross_check:
        report.append("| Orbit | Our Class | Catalog ID | Catalog Class | Agreement |\n")
        report.append("|-------|-----------|------------|---------------|----------|\n")
        for cc in cross_check:
            report.append(f"| {cc['candidate_idx']} | {cc['our_class']} | "
                          f"{cc['catalog_id']} | {cc['catalog_class']} | "
                          f"{cc['agreement']} |\n")
    else:
        report.append("No known-orbit matches have catalog stability data for comparison.\n")
    (OUT_DIR / "11_stability_cross_check" / "report.md").write_text("".join(report))

    # Write Group 3 verdict
    n_stable = sum(1 for s in stability_results if s["stability_class"] == "stable")
    n_unstable = sum(1 for s in stability_results if s["stability_class"] == "unstable")
    verdict = [
        "# GROUP 3 VERDICT: Stability Analysis\n\n",
        f"## Summary\n\n",
        f"Computed monodromy matrices for {len(stability_results)} verified orbits.\n\n",
        f"- Stable (elliptic): {n_stable}\n",
        f"- Unstable (hyperbolic): {n_unstable}\n",
        f"- Parabolic: {len(stability_results) - n_stable - n_unstable}\n\n",
        f"## Cross-check\n\n",
        f"{len(cross_check)} orbits matched to catalog entries for stability comparison.\n",
    ]
    (OUT_DIR / "GROUP3_VERDICT.md").write_text("".join(verdict))

    print(f"\nGroup 3 complete. Stable: {n_stable}, Unstable: {n_unstable}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
