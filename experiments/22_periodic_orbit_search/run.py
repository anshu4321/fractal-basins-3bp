"""Periodic orbit search via entropy-guided shooting on the shape sphere.

Vectorized: uses jax.vmap to shoot batches of candidates in parallel on GPU.
Tracks only the minimum return distance per trajectory (not the full trace)
to keep memory bounded.

Strategy:
  1. Find basin-boundary candidates (k-NN label disagreement)
  2. Batch-shoot candidates on GPU, find near-returns
  3. Re-shoot best candidates with full trace for refinement
  4. Refine with gradient-based optimization
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

H = 0.003
N_STEPS = 30000
R_MIN_THRESHOLD = 0.01
RETURN_THRESHOLD = 0.05
T_MIN_SKIP = 100
N_CANDIDATES = 5000
N_BOUNDARY = 50000
BATCH_SIZE = 200


def find_boundary_candidates(shape_pts, labels, n_boundary, n_candidates, k=5):
    from scipy.spatial import cKDTree
    n_boundary = min(n_boundary, len(shape_pts))
    tree = cKDTree(shape_pts[:n_boundary])
    _, idx = tree.query(shape_pts[:n_boundary], k=k+1)
    neighbor_labels = labels[idx[:, 1:]]
    own_labels = labels[:n_boundary, None]
    disagree_frac = np.mean(neighbor_labels != own_labels, axis=1)

    top = np.argsort(-disagree_frac)[:n_candidates]
    print(f"Found {np.sum(disagree_frac > 0.3)} high-disagreement points, "
          f"selected top {len(top)} candidates")
    return shape_pts[top], disagree_frac[top]


@partial(jax.jit, static_argnames=("n_steps",))
def batch_shoot_min_return(n0_batch, n_steps):
    """Shoot a batch of candidates, return (min_return_dist, min_step, r_min_ever) per candidate.

    Tracks only the running minimum, not the full trace. Memory: O(batch_size).
    """
    h = jnp.float64(H)

    def single_shoot(n0):
        q0 = shape_to_config(n0, inertia=1.0)
        p0 = jnp.zeros_like(q0)

        def step_fn(carry, step_idx):
            q, p, best_dist, best_step, r_min_ever = carry
            q, p = yoshida6_step(q, p, h)
            n = config_to_shape(q)
            dot = jnp.clip(jnp.sum(n0 * n), -1.0, 1.0)
            dist = jnp.arccos(dot)
            r_min = jnp.min(pairwise_distances(q))
            r_min_ever = jnp.minimum(r_min_ever, r_min)
            valid = (step_idx >= T_MIN_SKIP) & (r_min >= R_MIN_THRESHOLD)
            is_better = valid & (dist < best_dist)
            best_dist = jnp.where(is_better, dist, best_dist)
            best_step = jnp.where(is_better, step_idx, best_step)
            return (q, p, best_dist, best_step, r_min_ever), None

        init = (q0, p0, jnp.float64(999.0), jnp.int64(0), jnp.float64(999.0))
        (_, _, best_dist, best_step, r_min_ever), _ = jax.lax.scan(
            step_fn, init, jnp.arange(n_steps, dtype=jnp.int64))
        return best_dist, best_step, r_min_ever

    return jax.vmap(single_shoot)(n0_batch)


@partial(jax.jit, static_argnames=("n_steps",))
def shoot_full_trace(n0, n_steps):
    """Single trajectory with full shape trace for visualization."""
    q0 = shape_to_config(n0, inertia=1.0)
    p0 = jnp.zeros_like(q0)
    h = jnp.float64(H)

    def step_fn(carry, _):
        q, p = carry
        q, p = yoshida6_step(q, p, h)
        n = config_to_shape(q)
        return (q, p), (q, n)

    (_, _), (q_trace, n_trace) = jax.lax.scan(step_fn, (q0, p0), None, length=n_steps)
    return q_trace, n_trace


def refine_candidate(n0, best_step, n_refine=300, lr=1e-4):
    """Refine via gradient descent on return distance at best_step."""
    theta = float(np.arccos(np.clip(n0[2], -1, 1)))
    phi = float(np.arctan2(n0[1], n0[0]))

    @jax.jit
    def return_dist(theta, phi):
        n = jnp.array([jnp.sin(theta)*jnp.cos(phi),
                        jnp.sin(theta)*jnp.sin(phi),
                        jnp.cos(theta)])
        q0 = shape_to_config(n, inertia=1.0)
        p0 = jnp.zeros_like(q0)
        h = jnp.float64(H)
        def step_fn(carry, _):
            q, p = carry
            q, p = yoshida6_step(q, p, h)
            return (q, p), None
        (qf, _), _ = jax.lax.scan(step_fn, (q0, p0), None, length=best_step)
        nf = config_to_shape(qf)
        return jnp.arccos(jnp.clip(jnp.sum(n * nf), -1.0, 1.0))

    grad_fn = jax.jit(jax.grad(return_dist, argnums=(0, 1)))

    best_dist = float('inf')
    best_theta, best_phi = theta, phi

    for i in range(n_refine):
        d = float(return_dist(jnp.float64(theta), jnp.float64(phi)))
        if d < best_dist:
            best_dist = d
            best_theta, best_phi = theta, phi
        gt, gp = grad_fn(jnp.float64(theta), jnp.float64(phi))
        theta -= lr * float(gt)
        phi -= lr * float(gp)
        theta = max(0.01, min(np.pi - 0.01, theta))
        if i % 100 == 0:
            print(f"      iter {i}: dist={d:.6f}")

    n_ref = np.array([np.sin(best_theta)*np.cos(best_phi),
                       np.sin(best_theta)*np.sin(best_phi),
                       np.cos(best_theta)])
    return n_ref, best_dist


def main():
    out_dir = PROJECT_ROOT / "experiments" / "22_periodic_orbit_search"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"jax backend: {jax.default_backend()}")
    print(f"Integration: h={H}, n_steps={N_STEPS}, T_max={H*N_STEPS:.1f}")
    print(f"Batch size: {BATCH_SIZE}")

    npz_path = PROJECT_ROOT / "results" / "01_dataset_regen" / "at_rest_with_diagnostics.npz"
    d = np.load(npz_path)
    mask = (d["r_min_ever"] >= R_MIN_THRESHOLD) & (d["label"] != -1)
    shape_pts = d["shape_n"][mask].astype(np.float64)
    labels = d["label"][mask].astype(np.int32)
    print(f"Loaded {len(labels)} clean samples")

    # Phase 1
    print("\n=== Phase 1: Finding boundary candidates ===")
    candidates, entropies = find_boundary_candidates(
        shape_pts, labels, N_BOUNDARY, N_CANDIDATES)

    # Phase 2: batch shooting
    print(f"\n=== Phase 2: Batch shooting {len(candidates)} candidates ===")
    all_dists = []
    all_steps = []
    all_rmins = []
    t0 = time.perf_counter()

    for batch_start in range(0, len(candidates), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(candidates))
        batch = jnp.array(candidates[batch_start:batch_end])

        dists, steps, rmins = batch_shoot_min_return(batch, N_STEPS)
        all_dists.append(np.array(dists))
        all_steps.append(np.array(steps))
        all_rmins.append(np.array(rmins))

        dt = time.perf_counter() - t0
        n_done = batch_end
        near = sum(np.sum(d < RETURN_THRESHOLD) for d in all_dists)
        print(f"  {n_done}/{len(candidates)} [{dt:.1f}s] "
              f"near-returns so far: {near}")

    all_dists = np.concatenate(all_dists)
    all_steps = np.concatenate(all_steps)
    all_rmins = np.concatenate(all_rmins)

    dt = time.perf_counter() - t0
    near_mask = all_dists < RETURN_THRESHOLD
    print(f"\nShooting done in {dt:.1f}s")
    print(f"Near-returns (dist < {RETURN_THRESHOLD}): {np.sum(near_mask)}")

    # Sort by distance
    order = np.argsort(all_dists)
    results = []
    for i in order[:50]:
        if all_dists[i] < RETURN_THRESHOLD:
            results.append({
                "idx": int(i),
                "n0": candidates[i].tolist(),
                "min_dist": float(all_dists[i]),
                "min_step": int(all_steps[i]),
                "period": float((all_steps[i] + 1) * H),
                "r_min_ever": float(all_rmins[i]),
                "entropy": float(entropies[i]),
            })

    print(f"\nTop 10 near-returns:")
    for r in results[:10]:
        print(f"  dist={r['min_dist']:.5f}  T={r['period']:.3f}  "
              f"shape=({r['n0'][0]:.4f}, {r['n0'][1]:.4f}, {r['n0'][2]:.4f})")

    # Phase 3: Refine top candidates
    print(f"\n=== Phase 3: Refining top {min(10, len(results))} ===")
    refined = []
    for j, r in enumerate(results[:10]):
        print(f"\n  #{j+1}: dist={r['min_dist']:.5f}, T={r['period']:.3f}")

        n_ref, ref_dist = refine_candidate(
            np.array(r["n0"]), r["min_step"])

        q0 = shape_to_config(jnp.array(n_ref), inertia=1.0)
        p0 = jnp.zeros_like(q0)
        E = float(total_energy(q0, p0))
        r_min_init = float(jnp.min(pairwise_distances(q0)))

        cls = ("PERIODIC" if ref_dist < 0.001 else
               "NEAR-PERIODIC" if ref_dist < 0.01 else "CANDIDATE")

        refined.append({
            "n0_original": r["n0"],
            "n0_refined": n_ref.tolist(),
            "return_dist_original": r["min_dist"],
            "return_dist_refined": ref_dist,
            "period": r["period"],
            "energy": E,
            "r_min_initial": r_min_init,
            "entropy": r["entropy"],
            "classification": cls,
        })
        print(f"    -> {cls}: dist={ref_dist:.6f}, T={r['period']:.3f}, E={E:.4f}")

    output = {
        "parameters": {
            "h": H, "n_steps": N_STEPS, "T_max": H * N_STEPS,
            "return_threshold": RETURN_THRESHOLD,
            "n_candidates_searched": len(candidates),
            "n_near_returns": len(results),
            "n_refined": len(refined),
            "batch_size": BATCH_SIZE,
        },
        "near_returns": results,
        "refined": refined,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"Searched: {len(candidates)} candidates in {dt:.1f}s")
    print(f"Near-returns: {len(results)}")
    for r in refined:
        print(f"  {r['classification']}: dist={r['return_dist_refined']:.6f}, "
              f"T={r['period']:.3f}, E={r['energy']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
