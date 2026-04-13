"""Diagnose the 3.54e-2 residual floor.

Decomposes the 12D residual into position vs momentum components
to understand what's preventing closure. Also tests whether the
residual is an integration accuracy issue or a fundamental barrier.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from functools import partial

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from mega3bp.shape_sphere import shape_to_config, config_to_shape, pairwise_distances, moment_of_inertia
from mega3bp.integrators import yoshida6_step
from mega3bp.dynamics import total_energy, angular_momentum, kinetic_energy, potential_energy
from mega3bp.refine import _integrate_for_T, shooting_residual, REFINE_N_STEPS

OUT_DIR = Path(__file__).parent


def diagnose_candidate(n0, T, n_steps=10_000):
    """Full diagnostic of a single candidate orbit."""
    n0 = jnp.array(n0, dtype=jnp.float64)
    theta = jnp.float64(float(jnp.arccos(jnp.clip(n0[2], -1, 1))))
    phi = jnp.float64(float(jnp.arctan2(n0[1], n0[0])))
    T_jnp = jnp.float64(T)

    q0, p0, qf, pf = _integrate_for_T(theta, phi, T_jnp, n_steps)

    # Decompose residual
    dq = qf - q0  # position error (3, 2)
    dp = pf - p0  # momentum error (3, 2) = pf since p0=0

    dq_norm = float(jnp.linalg.norm(dq.ravel()))
    dp_norm = float(jnp.linalg.norm(dp.ravel()))
    total_norm = float(jnp.linalg.norm(jnp.concatenate([dq.ravel(), dp.ravel()])))

    # Shape sphere return
    n_final = config_to_shape(qf)
    shape_dist = float(jnp.arccos(jnp.clip(jnp.sum(n0 * n_final), -1.0, 1.0)))

    # Moment of inertia (scale) return
    I0 = float(moment_of_inertia(q0))
    If = float(moment_of_inertia(qf))
    dI_rel = abs(If - I0) / I0

    # Energy and angular momentum
    E0 = float(total_energy(q0, p0))
    Ef = float(total_energy(qf, pf))
    L0 = float(angular_momentum(q0, p0))
    Lf = float(angular_momentum(qf, pf))
    KE_f = float(kinetic_energy(pf))

    # Per-body momentum at T
    p_per_body = [float(jnp.linalg.norm(pf[i])) for i in range(3)]

    return {
        "total_residual": total_norm,
        "dq_norm": dq_norm,
        "dp_norm": dp_norm,
        "dq_fraction": dq_norm / max(total_norm, 1e-30),
        "dp_fraction": dp_norm / max(total_norm, 1e-30),
        "shape_sphere_dist": shape_dist,
        "moment_of_inertia_0": I0,
        "moment_of_inertia_T": If,
        "dI_relative": dI_rel,
        "E0": E0,
        "ET": Ef,
        "dE_rel": abs(Ef - E0) / abs(E0),
        "L0": L0,
        "LT": Lf,
        "KE_at_T": KE_f,
        "p_per_body": p_per_body,
        "T": T,
        "n_steps": n_steps,
    }


def test_convergence_with_n_steps(n0, T):
    """Test if residual improves with more integration steps."""
    results = []
    for n_steps in [1000, 2000, 5000, 10000, 20000, 50000]:
        d = diagnose_candidate(n0, T, n_steps=n_steps)
        results.append({
            "n_steps": n_steps,
            "h": T / n_steps,
            "total_residual": d["total_residual"],
            "dq_norm": d["dq_norm"],
            "dp_norm": d["dp_norm"],
        })
        print(f"  n_steps={n_steps:6d}, h={T/n_steps:.2e}: "
              f"||F||={d['total_residual']:.6e}, "
              f"||dq||={d['dq_norm']:.6e}, ||dp||={d['dp_norm']:.6e}")
    return results


def main():
    print("=" * 60)
    print("RESIDUAL FLOOR DIAGNOSTIC")
    print("=" * 60)

    # Load the best candidates from fresh search
    results_path = OUT_DIR / "fresh_search_results.json"
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
        candidates = data["all_results"]
    else:
        # Fallback: use experiment 22 candidates
        exp22_path = PROJECT_ROOT / "experiments" / "22_periodic_orbit_search" / "results.json"
        with open(exp22_path) as f:
            data = json.load(f)
        candidates = [{"n0_input": r["n0_refined"], "T_input": r["period"]}
                      for r in data["refined"]]

    # Pick a representative T=0.303 candidate (the ones that reach 3.54e-2)
    # and one from the original exp 22 (NEAR-PERIODIC)
    test_cases = []

    # From fresh search: find best candidate
    if "all_results" in data:
        best = min(data["all_results"],
                   key=lambda x: x.get("refinement", {}).get("closure_12d", 999))
        test_cases.append({
            "name": "best_fresh_search",
            "n0": best["n0_input"],
            "T": best.get("refinement", {}).get("T", best["T_input"]),
        })

    # Known T=0.303 candidate from exp 22
    exp22_path = PROJECT_ROOT / "experiments" / "22_periodic_orbit_search" / "results.json"
    with open(exp22_path) as f:
        exp22 = json.load(f)
    for r in exp22["refined"]:
        if r["classification"] == "NEAR-PERIODIC" and abs(r["period"] - 0.303) < 0.01:
            test_cases.append({
                "name": f"exp22_T{r['period']:.3f}",
                "n0": r["n0_refined"],
                "T": r["period"],
            })
            break

    for tc in test_cases:
        print(f"\n{'='*50}")
        print(f"Candidate: {tc['name']}")
        print(f"T = {tc['T']:.6f}")
        print(f"{'='*50}")

        # Full diagnostic
        d = diagnose_candidate(tc["n0"], tc["T"])
        print(f"\n  Total residual:    {d['total_residual']:.6e}")
        print(f"  Position error:    {d['dq_norm']:.6e} ({d['dq_fraction']*100:.1f}%)")
        print(f"  Momentum error:    {d['dp_norm']:.6e} ({d['dp_fraction']*100:.1f}%)")
        print(f"  Shape sphere dist: {d['shape_sphere_dist']:.6e}")
        print(f"  Inertia change:    {d['dI_relative']*100:.4f}%")
        print(f"  Energy drift:      {d['dE_rel']:.2e}")
        print(f"  L(0)={d['L0']:.6e}, L(T)={d['LT']:.6e}")
        print(f"  KE at T:           {d['KE_at_T']:.6e}")
        print(f"  |p| per body at T: {d['p_per_body']}")

        # Test if more steps help
        print(f"\n  Convergence test (varying n_steps):")
        conv = test_convergence_with_n_steps(tc["n0"], tc["T"])

    # Save diagnostics
    diag_path = OUT_DIR / "residual_diagnostic.json"
    with open(diag_path, "w") as f:
        json.dump({"test_cases": test_cases, "diagnostics": [
            diagnose_candidate(tc["n0"], tc["T"]) for tc in test_cases
        ]}, f, indent=2, default=str)
    print(f"\nDiagnostics saved to {diag_path}")


if __name__ == "__main__":
    main()
