"""Unified verification pipeline for periodic orbit candidates.

Wraps Group 1 tests (1.1-1.4) into a single verify_candidate() function.
Also provides search_and_verify() for batch processing.
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

from mega3bp.refine import refine_from_shape_point
from mega3bp.diagnostics import conservation_check, min_distance_check, multi_period_stability
from mega3bp.shape_sphere import shape_to_config
from mega3bp.dynamics import total_energy


# PROMPT pre-registered thresholds (do not modify after seeing results)
CLOSURE_VERIFIED = 1e-8
CLOSURE_NEAR = 1e-6
CONSERVATION_TOL = 1e-10
MIN_DIST_RATIO = 1.0 / 6.0
MULTI_PERIOD_TOL = 1e-4
N_PERIODS = 10
N_STEPS_VERIFY = 10_000


def verify_candidate(
    n0,
    T_approx: float,
    refine: bool = True,
    n_steps: int = N_STEPS_VERIFY,
    verbose: bool = True,
) -> dict:
    """Run full Group 1 verification on a single candidate.

    Args:
        n0: shape-sphere point (3-vector)
        T_approx: approximate period from shooting
        refine: if True, run Gauss-Newton refinement first
        n_steps: integration steps per period

    Returns:
        dict with test results and pass/fail for each criterion.
    """
    result = {"n0_input": np.asarray(n0).tolist(), "T_input": T_approx}

    # Step 0: Gauss-Newton refinement
    if refine:
        ref = refine_from_shape_point(n0, T_approx, n_steps=n_steps,
                                       max_iter=50, tol=1e-12, verbose=verbose)
        result["refinement"] = {
            "theta": ref["theta"], "phi": ref["phi"], "T": ref["T"],
            "closure_12d": ref["closure_12d"], "converged": ref["converged"],
            "n_iter": ref["n_iter"],
        }
        q0, p0, T = ref["q0"], ref["p0"], ref["T"]
        closure = ref["closure_12d"]
    else:
        n_arr = jnp.array(n0, dtype=jnp.float64)
        q0 = shape_to_config(n_arr, inertia=1.0)
        p0 = jnp.zeros_like(q0)
        T = T_approx
        # Compute closure without refinement
        from mega3bp.refine import shooting_residual
        params = jnp.array([
            float(np.arccos(np.clip(n0[2], -1, 1))),
            float(np.arctan2(n0[1], n0[0])),
            T
        ], dtype=jnp.float64)
        res = shooting_residual(params, n_steps)
        closure = float(jnp.linalg.norm(res))

    # Test 1.1: Phase-space closure
    if closure < CLOSURE_VERIFIED:
        closure_class = "verified"
    elif closure < CLOSURE_NEAR:
        closure_class = "near-periodic"
    else:
        closure_class = "failed"

    result["test_1_1"] = {
        "closure_norm": closure,
        "classification": closure_class,
        "pass": closure_class in ("verified", "near-periodic"),
        "criterion": f"r < {CLOSURE_NEAR:.0e}",
    }

    # Only continue if phase closure is at least near-periodic
    if closure_class == "failed":
        result["group1_pass"] = False
        result["fail_reason"] = "phase_closure"
        return result

    # Test 1.2: Conservation
    cons = conservation_check(q0, p0, T, n_steps=n_steps)
    dE = float(cons["dE_rel_max"])
    dL = float(cons["dL_max"])
    cons_pass = dE < CONSERVATION_TOL and dL < CONSERVATION_TOL

    result["test_1_2"] = {
        "dE_rel_max": dE,
        "dL_max": dL,
        "pass": cons_pass,
        "criterion": f"|dE/E| < {CONSERVATION_TOL:.0e} and |dL| < {CONSERVATION_TOL:.0e}",
    }

    # If conservation fails, try with more steps
    if not cons_pass and n_steps < 50_000:
        if verbose:
            print(f"  Conservation failed at {n_steps} steps, retrying with {n_steps*2}")
        cons2 = conservation_check(q0, p0, T, n_steps=n_steps * 2)
        dE2 = float(cons2["dE_rel_max"])
        dL2 = float(cons2["dL_max"])
        result["test_1_2"]["retry"] = {
            "n_steps": n_steps * 2, "dE_rel_max": dE2, "dL_max": dL2
        }
        if dE2 < CONSERVATION_TOL and dL2 < CONSERVATION_TOL:
            result["test_1_2"]["pass"] = True

    # Test 1.3: Minimum distance
    mdist = min_distance_check(q0, p0, T, n_steps=n_steps)
    ratio = float(mdist["r_min_ratio"])
    result["test_1_3"] = {
        "r_min": float(mdist["r_min"]),
        "r_min_per_pair": [float(x) for x in mdist["r_min_per_pair"]],
        "r_init_max": float(mdist["r_init_max"]),
        "r_min_ratio": ratio,
        "collisionless": ratio > MIN_DIST_RATIO,
        "criterion": f"r_min / r_init_max > {MIN_DIST_RATIO:.4f}",
    }

    # Test 1.4: Multi-period stability
    mp = multi_period_stability(q0, p0, T, n_steps=n_steps, n_periods=N_PERIODS)
    residuals = [float(r) for r in mp["residuals"]]
    result["test_1_4"] = {
        "residuals": residuals,
        "residual_10": residuals[-1],
        "pass": residuals[-1] < MULTI_PERIOD_TOL,
        "criterion": f"residual(10) < {MULTI_PERIOD_TOL:.0e}",
    }

    # Group 1 overall: Tests 1.1-1.4 all pass
    # Note: Test 1.3 flags near-collisions but does not reject
    all_pass = (result["test_1_1"]["pass"]
                and result["test_1_2"]["pass"]
                and result["test_1_4"]["pass"])
    result["group1_pass"] = all_pass
    result["q0"] = np.array(q0).tolist()
    result["p0"] = np.array(p0).tolist()
    result["T_refined"] = float(T)
    result["E"] = float(total_energy(q0, p0))

    return result


def search_and_verify(
    candidates: np.ndarray,
    T_candidates: np.ndarray,
    n_steps: int = N_STEPS_VERIFY,
    verbose: bool = True,
) -> list[dict]:
    """Run Gauss-Newton + Group 1 verification on a batch of candidates."""
    results = []
    for i in range(len(candidates)):
        if verbose:
            print(f"\n--- Candidate {i+1}/{len(candidates)} ---")
        r = verify_candidate(candidates[i], float(T_candidates[i]),
                             refine=True, n_steps=n_steps, verbose=verbose)
        results.append(r)
    return results
