"""Full-state return scan: find the time where ||state(t) - state(0)|| is minimized.

Instead of looking for shape-sphere returns (position only, quotiented),
scan for the minimum of the FULL 12D phase-space residual. This directly
targets periodic orbits without the shape-sphere indirection.

Also: try a DENSE grid of starting points on the shape sphere,
not just the entropy-selected ones. Periodic orbits should show up
as points where the full-state residual has deep minima.
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
from mega3bp.refine import refine_from_shape_point
from pipeline import verify_candidate

OUT_DIR = Path(__file__).parent

H = 0.003
N_STEPS = 30000   # T_max = 90
EXCURSION_THRESHOLD = 2.0  # orbit must reach ||state-state0|| > this before we count returns


@partial(jax.jit, static_argnames=("n_steps",))
def shoot_fullstate_return(n0, n_steps):
    """Track the minimum full-state residual ||state(t) - state(0)|| over time.

    Key: only counts returns AFTER the orbit has excursed beyond EXCURSION_THRESHOLD.
    This ensures we skip trivial short-time near-returns where the orbit hasn't
    gone anywhere yet.

    Returns: (best_residual, best_step, best_dp, best_dq)
    """
    h = jnp.float64(H)
    q0 = shape_to_config(n0, inertia=1.0)
    p0 = jnp.zeros_like(q0)
    state0 = jnp.concatenate([q0.ravel(), p0.ravel()])

    def step_fn(carry, step_idx):
        q, p, best_res, best_step, best_dp, best_dq, has_excursed = carry
        q, p = yoshida6_step(q, p, h)

        state = jnp.concatenate([q.ravel(), p.ravel()])
        residual = jnp.linalg.norm(state - state0)
        dp = jnp.linalg.norm(p.ravel())
        dq = jnp.linalg.norm((q - q0).ravel())

        # Track whether orbit has ever gone far enough from start
        has_excursed = has_excursed | (residual > EXCURSION_THRESHOLD)

        # Only count as a valid return if:
        # 1. Orbit previously excursed beyond threshold (went somewhere)
        # 2. No close encounter (collision avoidance)
        r_min = jnp.min(pairwise_distances(q))
        valid = has_excursed & (r_min >= 0.01)
        is_better = valid & (residual < best_res)

        best_res = jnp.where(is_better, residual, best_res)
        best_step = jnp.where(is_better, step_idx, best_step)
        best_dp = jnp.where(is_better, dp, best_dp)
        best_dq = jnp.where(is_better, dq, best_dq)

        return (q, p, best_res, best_step, best_dp, best_dq, has_excursed), None

    init = (q0, p0, jnp.float64(999.0), jnp.int64(0),
            jnp.float64(999.0), jnp.float64(999.0), jnp.bool_(False))
    (_, _, best_res, best_step, best_dp, best_dq, _), _ = jax.lax.scan(
        step_fn, init, jnp.arange(n_steps, dtype=jnp.int64))

    return best_res, best_step, best_dp, best_dq


@partial(jax.jit, static_argnames=("n_steps",))
def batch_shoot_fullstate(n0_batch, n_steps):
    """Batch version of shoot_fullstate_return."""
    return jax.vmap(lambda n0: shoot_fullstate_return(n0, n_steps))(n0_batch)


def main():
    print("=" * 60)
    print("FULL-STATE RETURN SCAN")
    print(f"h={H}, N_steps={N_STEPS}, T_max={H*N_STEPS:.0f}")
    print("Looking for ||state(t) - state(0)|| minima")
    print("=" * 60)

    # Load dataset
    npz_path = PROJECT_ROOT / "results" / "01_dataset_regen" / "at_rest_with_diagnostics.npz"
    d = np.load(npz_path)
    mask = (d["r_min_ever"] >= 0.01) & (d["label"] != -1)
    shape_pts = d["shape_n"][mask].astype(np.float64)
    labels = d["label"][mask].astype(np.int32)
    print(f"Dataset: {len(labels)} clean samples")

    # Use entropy-selected candidates (top 5000 from trained model)
    import equinox as eqx
    from mega3bp.ml import BasinMLP, predict_proba

    model_path = OUT_DIR / "baseline_model.eqx"
    if model_path.exists():
        jax.config.update("jax_enable_x64", False)
        model = eqx.tree_deserialise_leaves(
            model_path, BasinMLP(in_dim=3, hidden=256, n_layers=5,
                                  key=jax.random.PRNGKey(0)))
        jax.config.update("jax_enable_x64", True)

        # Compute entropy
        all_ent = []
        for start in range(0, len(shape_pts), 8192):
            batch = jnp.array(shape_pts[start:start+8192], dtype=jnp.float32)
            probs = predict_proba(model, batch)
            ent = -jnp.sum(probs * jnp.log(probs + 1e-10), axis=-1)
            all_ent.append(np.asarray(ent))
        entropies = np.concatenate(all_ent)
        top_idx = np.argsort(-entropies)[:5000]
        candidates = shape_pts[top_idx].astype(np.float64)
        print(f"Selected 5000 candidates by entropy")
    else:
        # Fallback: k-NN disagreement
        from scipy.spatial import cKDTree
        tree = cKDTree(shape_pts[:50000])
        _, idx = tree.query(shape_pts[:50000], k=6)
        neighbor_labels = labels[idx[:, 1:]]
        own_labels = labels[:50000, None]
        disagree = np.mean(neighbor_labels != own_labels, axis=1)
        top_idx = np.argsort(-disagree)[:5000]
        candidates = shape_pts[top_idx].astype(np.float64)
        print(f"Selected 5000 candidates by k-NN disagreement")

    # Batch shoot with FULL-STATE return tracking
    print(f"\n=== SHOOTING {len(candidates)} candidates (full-state return) ===")
    BATCH_SIZE = 200
    all_res, all_steps, all_dp, all_dq = [], [], [], []
    t0 = time.perf_counter()

    for i in range(0, len(candidates), BATCH_SIZE):
        batch = jnp.array(candidates[i:i+BATCH_SIZE])
        res, steps, dp, dq = batch_shoot_fullstate(batch, N_STEPS)
        all_res.append(np.array(res))
        all_steps.append(np.array(steps))
        all_dp.append(np.array(dp))
        all_dq.append(np.array(dq))
        n_done = min(i + BATCH_SIZE, len(candidates))
        if n_done % 1000 == 0:
            best_so_far = min(np.min(r) for r in all_res)
            print(f"  {n_done}/{len(candidates)} [{time.perf_counter()-t0:.0f}s] "
                  f"best ||F||: {best_so_far:.6e}")

    residuals = np.concatenate(all_res)
    steps = np.concatenate(all_steps)
    dp_norms = np.concatenate(all_dp)
    dq_norms = np.concatenate(all_dq)

    dt_shoot = time.perf_counter() - t0
    print(f"\nShooting done: {dt_shoot:.0f}s")

    # Sort by full-state residual
    order = np.argsort(residuals)

    print(f"\nTop 20 by full-state residual:")
    print(f"{'Rank':>4} {'||F||':>12} {'||dp||':>12} {'||dq||':>12} {'T':>8} {'Step':>8}")
    for rank, idx in enumerate(order[:20]):
        T = (steps[idx] + 1) * H
        print(f"{rank+1:4d} {residuals[idx]:12.6e} {dp_norms[idx]:12.6e} "
              f"{dq_norms[idx]:12.6e} {T:8.3f} {steps[idx]:8d}")

    # Select candidates with full-state residual < 1.0 for refinement
    threshold = 1.0
    good = order[residuals[order] < threshold]
    print(f"\n{len(good)} candidates with ||F|| < {threshold}")

    # Refine the best ones
    N_REFINE = min(50, len(good))
    print(f"\n=== REFINING top {N_REFINE} candidates ===")

    verified = []
    for rank, idx in enumerate(good[:N_REFINE]):
        T = float((steps[idx] + 1) * H)
        n0 = candidates[idx]
        print(f"\n--- Rank {rank+1}: ||F||={residuals[idx]:.4e}, "
              f"||dp||={dp_norms[idx]:.4e}, T={T:.3f} ---")
        result = verify_candidate(n0.tolist(), T,
                                  refine=True, n_steps=10_000, verbose=True)
        if result.get("group1_pass", False):
            verified.append(result)
            print(f"  >>> GROUP 1 PASS!")
        else:
            closure = result.get("test_1_1", {}).get("closure_norm", "?")
            print(f"  Final closure: {closure}")

    print(f"\n{'='*60}")
    print(f"RESULT: {len(verified)}/{N_REFINE} verified periodic orbits")
    print(f"{'='*60}")

    # Save
    save_data = {
        "parameters": {"h": H, "n_steps": N_STEPS, "n_candidates": len(candidates)},
        "shooting_time_s": dt_shoot,
        "top_20": [{
            "rank": rank + 1,
            "n0": candidates[order[rank]].tolist(),
            "residual": float(residuals[order[rank]]),
            "dp_norm": float(dp_norms[order[rank]]),
            "dq_norm": float(dq_norms[order[rank]]),
            "T": float((steps[order[rank]] + 1) * H),
        } for rank in range(min(20, len(order)))],
        "n_verified": len(verified),
    }
    with open(OUT_DIR / "fullstate_scan_results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    if verified:
        with open(OUT_DIR / "group1_verified.json", "w") as f:
            json.dump(verified, f, indent=2, default=str)

    return 0


if __name__ == "__main__":
    sys.exit(main())
