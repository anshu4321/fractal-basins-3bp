"""Multi-period scan: check if the true period is a multiple of the detected shape return.

The shooting finds the FIRST time the shape sphere point returns close.
But the true periodic orbit period could be 2x, 3x, ... Nx of that.
At the true period, BOTH position AND momentum should return to zero.

Strategy:
1. For each near-return candidate, integrate for much longer (up to 100x the detected T)
2. At each shape-sphere near-return event, record the full state residual
3. Find the return time where ||dp|| is minimized (momentum closest to zero)
4. Try Gauss-Newton refinement at THAT period
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from functools import partial

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from mega3bp.shape_sphere import shape_to_config, config_to_shape, pairwise_distances
from mega3bp.integrators import yoshida6_step
from mega3bp.dynamics import total_energy
from mega3bp.refine import refine_from_shape_point
from pipeline import verify_candidate

OUT_DIR = Path(__file__).parent

H = 0.001          # finer timestep for longer integration
N_STEPS = 300_000  # integrate up to T_max = 300
SHAPE_RETURN_THRESHOLD = 0.05  # same as shooting
MIN_STEP = 300     # skip first 300 steps (~0.3 time units)


@partial(jax.jit, static_argnames=("n_steps",))
def scan_all_returns(n0, n_steps):
    """Integrate and record ALL shape-sphere near-returns with full state info.

    Returns arrays of shape (n_steps,) with:
    - shape_dist: geodesic distance to initial shape at each step
    - dp_norm: ||p(t)||_2 at each step (should be 0 for periodic free-fall)
    - dq_norm: ||q(t) - q(0)||_2 at each step
    - r_min: min pairwise distance at each step
    """
    h = jnp.float64(H)
    q0 = shape_to_config(n0, inertia=1.0)
    p0 = jnp.zeros_like(q0)
    state0 = jnp.concatenate([q0.ravel(), p0.ravel()])

    def step_fn(carry, step_idx):
        q, p = carry
        q, p = yoshida6_step(q, p, h)
        n = config_to_shape(q)
        shape_dist = jnp.arccos(jnp.clip(jnp.sum(n0 * n), -1.0, 1.0))
        dp_norm = jnp.linalg.norm(p.ravel())
        dq_norm = jnp.linalg.norm((q - q0).ravel())
        full_residual = jnp.linalg.norm(
            jnp.concatenate([q.ravel(), p.ravel()]) - state0)
        return (q, p), (shape_dist, dp_norm, dq_norm, full_residual)

    (_, _), (shape_dists, dp_norms, dq_norms, full_residuals) = jax.lax.scan(
        step_fn, (q0, p0), jnp.arange(n_steps, dtype=jnp.int64))

    return shape_dists, dp_norms, dq_norms, full_residuals


def find_best_returns(shape_dists, dp_norms, dq_norms, full_residuals,
                      shape_threshold=SHAPE_RETURN_THRESHOLD, min_step=MIN_STEP):
    """Find time steps where shape returns AND momentum is small.

    Returns list of (step, time, shape_dist, dp_norm, dq_norm, full_residual)
    sorted by full_residual (best first).
    """
    shape_dists = np.array(shape_dists)
    dp_norms = np.array(dp_norms)
    dq_norms = np.array(dq_norms)
    full_residuals = np.array(full_residuals)

    # Find local minima of shape distance that are below threshold
    returns = []
    for i in range(min_step, len(shape_dists) - 1):
        if (shape_dists[i] < shape_threshold and
            shape_dists[i] < shape_dists[i-1] and
            shape_dists[i] < shape_dists[i+1]):
            returns.append({
                "step": int(i),
                "time": float((i + 1) * H),
                "shape_dist": float(shape_dists[i]),
                "dp_norm": float(dp_norms[i]),
                "dq_norm": float(dq_norms[i]),
                "full_residual": float(full_residuals[i]),
            })

    # Sort by full residual (position + momentum combined)
    returns.sort(key=lambda x: x["full_residual"])
    return returns


def main():
    print("=" * 60)
    print("MULTI-PERIOD SCAN")
    print(f"h={H}, N_steps={N_STEPS}, T_max={H*N_STEPS:.0f}")
    print("=" * 60)

    # Load best candidates from fresh search (the T=0.303 ones that hit 3.54e-2)
    exp22_path = PROJECT_ROOT / "experiments" / "22_periodic_orbit_search" / "results.json"
    with open(exp22_path) as f:
        exp22 = json.load(f)

    # Pick candidates to scan: the NEAR-PERIODIC ones from exp 22
    test_cases = []
    for i, r in enumerate(exp22["refined"]):
        if r["classification"] in ("NEAR-PERIODIC", "PERIODIC"):
            test_cases.append({
                "name": f"exp22_{i}_T{r['period']:.3f}",
                "n0": r["n0_refined"],
                "T_original": r["period"],
                "idx": i,
            })

    if not test_cases:
        # Fallback: use the best from fresh search
        fresh_path = OUT_DIR / "fresh_search_results.json"
        if fresh_path.exists():
            with open(fresh_path) as f:
                fresh = json.load(f)
            # Pick first few candidates
            for i, r in enumerate(fresh["all_results"][:5]):
                test_cases.append({
                    "name": f"fresh_{i}",
                    "n0": r["n0_input"],
                    "T_original": r["T_input"],
                    "idx": i,
                })

    print(f"Scanning {len(test_cases)} candidates")

    all_scan_results = []
    best_candidates_for_refinement = []

    for tc in test_cases:
        print(f"\n{'='*50}")
        print(f"{tc['name']}: original T={tc['T_original']:.6f}")
        print(f"{'='*50}")

        n0 = jnp.array(tc["n0"], dtype=jnp.float64)
        t0 = time.perf_counter()
        shape_dists, dp_norms, dq_norms, full_residuals = scan_all_returns(n0, N_STEPS)
        dt = time.perf_counter() - t0
        print(f"Scan done in {dt:.1f}s")

        returns = find_best_returns(shape_dists, dp_norms, dq_norms, full_residuals)
        print(f"Found {len(returns)} shape-sphere returns")

        # Show top 10 by full residual
        print(f"\nTop 10 returns by full residual (pos+mom):")
        print(f"{'Step':>8} {'Time':>10} {'Shape':>10} {'||dp||':>12} {'||dq||':>12} {'||F||':>12}")
        for r in returns[:10]:
            print(f"{r['step']:8d} {r['time']:10.4f} {r['shape_dist']:10.6f} "
                  f"{r['dp_norm']:12.6e} {r['dq_norm']:12.6e} {r['full_residual']:12.6e}")

        # Show top 10 by momentum (dp_norm)
        returns_by_dp = sorted(returns, key=lambda x: x["dp_norm"])
        print(f"\nTop 10 returns by momentum (||dp|| closest to zero):")
        print(f"{'Step':>8} {'Time':>10} {'Shape':>10} {'||dp||':>12} {'||dq||':>12} {'||F||':>12}")
        for r in returns_by_dp[:10]:
            print(f"{r['step']:8d} {r['time']:10.4f} {r['shape_dist']:10.6f} "
                  f"{r['dp_norm']:12.6e} {r['dq_norm']:12.6e} {r['full_residual']:12.6e}")

        # Record the best for refinement
        if returns:
            best = returns[0]  # best by full residual
            best_dp = returns_by_dp[0]  # best by momentum
            tc["best_full"] = best
            tc["best_dp"] = best_dp
            all_scan_results.append(tc)

            # Queue for refinement if significantly better than 3.54e-2
            if best["full_residual"] < 3e-2:
                best_candidates_for_refinement.append({
                    "n0": tc["n0"],
                    "T": best["time"],
                    "source": tc["name"],
                    "expected_residual": best["full_residual"],
                })
            if best_dp["dp_norm"] < 3e-2 and best_dp != best:
                best_candidates_for_refinement.append({
                    "n0": tc["n0"],
                    "T": best_dp["time"],
                    "source": tc["name"] + "_dp",
                    "expected_residual": best_dp["full_residual"],
                })

    # Try Gauss-Newton refinement on the best multi-period candidates
    if best_candidates_for_refinement:
        print(f"\n{'='*60}")
        print(f"REFINING {len(best_candidates_for_refinement)} MULTI-PERIOD CANDIDATES")
        print(f"{'='*60}")

        verified = []
        for cand in best_candidates_for_refinement:
            print(f"\n--- {cand['source']}: T={cand['T']:.4f}, "
                  f"expected ||F||={cand['expected_residual']:.2e} ---")
            result = verify_candidate(cand["n0"], cand["T"],
                                      refine=True, n_steps=10_000, verbose=True)
            if result.get("group1_pass", False):
                verified.append(result)
                print(f"  >>> GROUP 1 PASS!")
            else:
                closure = result.get("test_1_1", {}).get("closure_norm", "?")
                print(f"  closure: {closure}")

        print(f"\nVerified: {len(verified)}/{len(best_candidates_for_refinement)}")
    else:
        print(f"\nNo candidates with full_residual < 3e-2 found at any return time.")
        print("The shape-sphere returns never coincide with momentum return.")

    # Save results
    save_data = {
        "parameters": {"h": H, "n_steps": N_STEPS, "T_max": H * N_STEPS},
        "scan_results": all_scan_results,
        "refinement_candidates": best_candidates_for_refinement,
    }
    save_path = OUT_DIR / "multi_period_scan.json"
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
