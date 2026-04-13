"""Fresh classifier-prior search per PROMPT Group 1 gate fallback.

When Group 1 fails with <3 survivors: run classifier-prior search from scratch,
generate new candidate pool, then re-run Group 1 verification.

This combines the shooting pipeline from experiment 22 with the new
Gauss-Newton refiner and Group 1 verification.
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
from mega3bp.ml import BasinMLP, predict_proba
from pipeline import verify_candidate

OUT_DIR = Path(__file__).parent

# Search parameters
H = 0.003
N_STEPS_SHOOT = 30000
RETURN_THRESHOLD = 0.05
T_MIN_SKIP = 100
BATCH_SIZE = 200
N_CANDIDATES = 5000
N_REFINE_TOP = 100   # refine top 100 near-returns


@partial(jax.jit, static_argnames=("n_steps",))
def _shoot_batch(n0_batch, n_steps):
    """Shoot a batch, return (min_dist, min_step, r_min_ever) per candidate."""
    h = jnp.float64(H)
    def single(n0):
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
            valid = (step_idx >= T_MIN_SKIP) & (r_min >= 0.01)
            is_better = valid & (dist < best_dist)
            best_dist = jnp.where(is_better, dist, best_dist)
            best_step = jnp.where(is_better, step_idx, best_step)
            return (q, p, best_dist, best_step, r_min_ever), None
        init = (q0, p0, jnp.float64(999.0), jnp.int64(0), jnp.float64(999.0))
        (_, _, bd, bs, rm), _ = jax.lax.scan(
            step_fn, init, jnp.arange(n_steps, dtype=jnp.int64))
        return bd, bs, rm
    return jax.vmap(single)(n0_batch)


def main():
    print("=" * 60)
    print("FRESH CLASSIFIER-PRIOR SEARCH")
    print("(Group 1 gate fallback)")
    print("=" * 60)

    # 1. Load dataset
    npz_path = PROJECT_ROOT / "results" / "01_dataset_regen" / "at_rest_with_diagnostics.npz"
    d = np.load(npz_path)
    mask = (d["r_min_ever"] >= 0.01) & (d["label"] != -1)
    shape_pts = d["shape_n"][mask].astype(np.float64)
    labels = d["label"][mask].astype(np.int32)
    print(f"Dataset: {len(labels)} clean samples")

    # 2. Load trained classifier
    import equinox as eqx
    model_path = OUT_DIR / "baseline_model.eqx"
    if not model_path.exists():
        print(f"ERROR: No model at {model_path}. Run train_model.py first.")
        return 1

    # Need fp32 model for inference then cast inputs
    jax.config.update("jax_enable_x64", False)
    model = eqx.tree_deserialise_leaves(model_path,
                                         BasinMLP(in_dim=3, hidden=256, n_layers=5,
                                                  key=jax.random.PRNGKey(0)))
    jax.config.update("jax_enable_x64", True)
    print(f"Loaded model from {model_path}")

    # 3. Compute classifier entropy over dataset
    print("\nComputing prediction entropy...")
    all_ent = []
    bs = 8192
    for start in range(0, len(shape_pts), bs):
        batch = jnp.array(shape_pts[start:start + bs], dtype=jnp.float32)
        probs = predict_proba(model, batch)
        ent = -jnp.sum(probs * jnp.log(probs + 1e-10), axis=-1)
        all_ent.append(np.asarray(ent))
    entropies = np.concatenate(all_ent)
    print(f"Entropy range: [{entropies.min():.4f}, {entropies.max():.4f}]")
    print(f"High-entropy points (>0.5): {np.sum(entropies > 0.5)}")

    # 4. Select top-N candidates by entropy
    top_idx = np.argsort(-entropies)[:N_CANDIDATES]
    candidates = shape_pts[top_idx].astype(np.float64)
    cand_entropies = entropies[top_idx]
    print(f"\nSelected {len(candidates)} candidates by entropy")
    print(f"Entropy range of selected: [{cand_entropies.min():.4f}, {cand_entropies.max():.4f}]")

    # 5. Batch shoot candidates
    print(f"\n=== SHOOTING {len(candidates)} candidates ===")
    all_dists, all_steps, all_rmins = [], [], []
    t0 = time.perf_counter()

    for i in range(0, len(candidates), BATCH_SIZE):
        batch = jnp.array(candidates[i:i + BATCH_SIZE])
        d_batch, s_batch, r_batch = _shoot_batch(batch, N_STEPS_SHOOT)
        all_dists.append(np.array(d_batch))
        all_steps.append(np.array(s_batch))
        all_rmins.append(np.array(r_batch))

        n_done = min(i + BATCH_SIZE, len(candidates))
        near = sum(np.sum(dd < RETURN_THRESHOLD) for dd in all_dists)
        if n_done % 1000 == 0 or n_done == len(candidates):
            print(f"  {n_done}/{len(candidates)} [{time.perf_counter()-t0:.0f}s] "
                  f"near-returns: {near}")

    dists = np.concatenate(all_dists)
    steps = np.concatenate(all_steps)
    rmins = np.concatenate(all_rmins)

    dt_shoot = time.perf_counter() - t0
    n_near = int(np.sum(dists < RETURN_THRESHOLD))
    print(f"\nShooting done: {dt_shoot:.0f}s, {n_near} near-returns (< {RETURN_THRESHOLD})")

    # 6. Select top near-returns for refinement
    sorted_idx = np.argsort(dists)
    near_returns = []
    for idx in sorted_idx[:N_REFINE_TOP]:
        if dists[idx] < RETURN_THRESHOLD:
            near_returns.append({
                "n0": candidates[idx].tolist(),
                "T_approx": float((steps[idx] + 1) * H),
                "return_dist": float(dists[idx]),
                "entropy": float(cand_entropies[idx]) if idx < len(cand_entropies) else 0.0,
            })
    print(f"Selected {len(near_returns)} for Gauss-Newton refinement + verification")

    # 7. Refine and verify through Group 1
    print(f"\n=== REFINING + VERIFYING ===")
    all_results = []
    verified = []
    t1 = time.perf_counter()

    for j, nr in enumerate(near_returns):
        print(f"\n--- Near-return {j+1}/{len(near_returns)}: "
              f"dist={nr['return_dist']:.5f}, T={nr['T_approx']:.3f} ---")
        result = verify_candidate(nr["n0"], nr["T_approx"],
                                  refine=True, n_steps=10_000, verbose=True)
        result["candidate_idx"] = j
        result["search_entropy"] = nr["entropy"]
        result["search_return_dist"] = nr["return_dist"]
        all_results.append(result)

        if result.get("group1_pass", False):
            verified.append(result)
            print(f"  >>> GROUP 1 PASS! closure={result['test_1_1']['closure_norm']:.2e}")

        if (j + 1) % 10 == 0:
            print(f"\n  Progress: {j+1}/{len(near_returns)}, "
                  f"verified so far: {len(verified)}")

    dt_refine = time.perf_counter() - t1
    print(f"\nRefinement done: {dt_refine:.0f}s")
    print(f"Verified orbits: {len(verified)}/{len(near_returns)}")

    # 8. Save results
    output = {
        "parameters": {
            "n_candidates": N_CANDIDATES,
            "n_near_returns": n_near,
            "n_refined": len(near_returns),
            "n_verified": len(verified),
            "shoot_time_s": dt_shoot,
            "refine_time_s": dt_refine,
        },
        "all_results": all_results,
        "verified": verified,
    }

    results_path = OUT_DIR / "fresh_search_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Update group1_verified.json if we got survivors
    if verified:
        g1_path = OUT_DIR / "group1_verified.json"
        with open(g1_path, "w") as f:
            json.dump(verified, f, indent=2, default=str)
        print(f"Updated {g1_path} with {len(verified)} verified orbits")

    # Update verdict
    verdict = [
        "# GROUP 1 VERDICT: Orbit Verification (Fresh Search)\n\n",
        f"## Summary\n\n",
        f"Original 10 candidates: 0/10 passed.\n",
        f"Fresh classifier-prior search: {N_CANDIDATES} candidates, "
        f"{n_near} near-returns, {len(near_returns)} refined, "
        f"**{len(verified)} verified**.\n\n",
        f"## Gate decision\n\n",
    ]
    if len(verified) >= 3:
        verdict.append(f"**PASS**: {len(verified)} >= 3 survivors. Proceed to Group 2.\n")
    else:
        verdict.append(
            f"**FAIL**: {len(verified)} < 3 survivors even after fresh search.\n"
            f"Per PROMPT Section 7: consider high-precision integrator (heyoka/Taylor).\n")

    (OUT_DIR / "GROUP1_VERDICT.md").write_text("".join(verdict))
    print(f"\nFinal verdict: {'PASS' if len(verified) >= 3 else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
