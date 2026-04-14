"""Stability analysis (Group 3) for verified velocity-space orbits.

Computes monodromy matrix, eigenvalue spectrum, stability classification
for each verified periodic orbit from velocity_space_results.json and
group4_baselines.json.
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
Q_COLLINEAR = jnp.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 0.0]], dtype=jnp.float64)


def make_ic(v1, v2):
    q0 = Q_COLLINEAR
    p0 = jnp.array([[v1, v2], [v1, v2], [-2*v1, -2*v2]], dtype=jnp.float64)
    return q0, p0


def collect_verified_orbits():
    """Gather all verified orbits from all experiment result files."""
    orbits = []
    seen = set()

    # From velocity_space_search.py
    vss_path = OUT_DIR / "velocity_space_results.json"
    if vss_path.exists():
        with open(vss_path) as f:
            d = json.load(f)
        for o in d.get("verified", []):
            key = (round(o["v1"], 6), round(o["v2"], 6), round(o["T"], 4))
            if key not in seen:
                seen.add(key)
                orbits.append({"source": "velocity_space_search",
                               "v1": o["v1"], "v2": o["v2"], "T": o["T"],
                               "residual": o["residual"]})

    # From group4_baselines.json
    g4_path = OUT_DIR / "group4_baselines.json"
    if g4_path.exists():
        with open(g4_path) as f:
            d = json.load(f)
        for method, runs in d.items():
            for run in runs:
                for o in run.get("verified_orbits", []):
                    key = (round(o["v1"], 6), round(o["v2"], 6), round(o["T"], 4))
                    if key not in seen:
                        seen.add(key)
                        orbits.append({"source": f"{method}_seed{run['seed']}",
                                       "v1": o["v1"], "v2": o["v2"], "T": o["T"],
                                       "residual": o["residual"]})
    return orbits


def main():
    print("=" * 60)
    print("GROUP 3: STABILITY ANALYSIS")
    print("=" * 60)

    orbits = collect_verified_orbits()
    print(f"Collected {len(orbits)} unique verified orbits")

    if not orbits:
        print("No verified orbits found. Run baselines first.")
        return 1

    results = []
    for i, orb in enumerate(orbits):
        print(f"\n--- Orbit {i+1}/{len(orbits)} from {orb['source']}: "
              f"v1={orb['v1']:.6f}, v2={orb['v2']:.6f}, T={orb['T']:.4f} ---")
        q0, p0 = make_ic(orb["v1"], orb["v2"])
        M = monodromy_matrix(q0, p0, orb["T"], n_steps=20_000)
        M_np = np.array(M)
        analysis = analyze_monodromy(M_np)
        result = {
            **orb,
            "stability_class": analysis["stability_class"],
            "stability_index": analysis["stability_index"],
            "n_trivial_eigenvalues": analysis["n_trivial"],
            "det_M": analysis["det_M"],
            "reciprocal_check": analysis["reciprocal_check"],
            "eigenvalues": [str(e) for e in analysis["eigenvalues"]],
        }
        results.append(result)
        print(f"  Class: {analysis['stability_class']}, "
              f"Index: {analysis['stability_index']:.4f}, "
              f"Trivial: {analysis['n_trivial']}, "
              f"det(M): {analysis['det_M']:.6f}")

    out = OUT_DIR / "stability_velocity.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out}")

    # Summary
    stable = sum(1 for r in results if r["stability_class"] == "stable")
    unstable = sum(1 for r in results if r["stability_class"] == "unstable")
    print(f"\nStable: {stable}, Unstable: {unstable}")


if __name__ == "__main__":
    sys.exit(main())
