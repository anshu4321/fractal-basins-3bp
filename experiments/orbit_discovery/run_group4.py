"""Group 4: Baseline Comparison (Tests 4.1-4.5).

Runs all 4 methods with identical budgets, verification, and 5 seeds.
Produces comparison table and headline plot.

Budget per method per seed: 5,000 integrations, top-100 Newton refinement,
Group 1 verification on all refined.
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
from pipeline import verify_candidate
from baselines import (uniform_random_candidates, physics_grid_candidates,
                       label_disagreement_candidates, classifier_entropy_candidates)

OUT_DIR = Path(__file__).parent
N_CANDIDATES = 5000
N_REFINE_TOP = 100
N_SEEDS = 5
H = 0.003
N_STEPS_SHOOT = 30000
RETURN_THRESHOLD = 0.05
BATCH_SIZE = 200


@partial(jax.jit, static_argnames=("n_steps",))
def _shoot_batch(n0_batch, n_steps):
    """Shoot a batch of candidates, return (min_dist, min_step, r_min_ever)."""
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
            valid = (step_idx >= 100) & (r_min >= 0.01)
            is_better = valid & (dist < best_dist)
            best_dist = jnp.where(is_better, dist, best_dist)
            best_step = jnp.where(is_better, step_idx, best_step)
            return (q, p, best_dist, best_step, r_min_ever), None
        init = (q0, p0, jnp.float64(999.0), jnp.int64(0), jnp.float64(999.0))
        (_, _, bd, bs, rm), _ = jax.lax.scan(
            step_fn, init, jnp.arange(n_steps, dtype=jnp.int64))
        return bd, bs, rm
    return jax.vmap(single)(n0_batch)


def batch_shoot(candidates):
    """Shoot all candidates in batches, return sorted near-returns."""
    all_dists, all_steps = [], []
    t0 = time.perf_counter()
    for i in range(0, len(candidates), BATCH_SIZE):
        batch = jnp.array(candidates[i:i+BATCH_SIZE])
        d, s, _ = _shoot_batch(batch, N_STEPS_SHOOT)
        all_dists.append(np.array(d))
        all_steps.append(np.array(s))
        n_done = min(i + BATCH_SIZE, len(candidates))
        if n_done % 1000 == 0:
            print(f"  {n_done}/{len(candidates)} [{time.perf_counter()-t0:.0f}s]")

    dists = np.concatenate(all_dists)
    steps = np.concatenate(all_steps)

    near_returns = []
    for idx in np.argsort(dists)[:N_REFINE_TOP]:
        if dists[idx] < RETURN_THRESHOLD:
            near_returns.append({
                "n0": candidates[idx].tolist(),
                "T_approx": float((steps[idx] + 1) * H),
                "return_dist": float(dists[idx]),
            })

    return near_returns


def run_method(name, candidates, seed):
    """Run one method: shoot candidates, refine top-K, verify all."""
    print(f"\n  Shooting {len(candidates)} candidates...")
    t0 = time.perf_counter()

    near_returns = batch_shoot(candidates)
    n_near = len(near_returns)
    print(f"  Near-returns: {n_near}")

    verified = []
    for j, nr in enumerate(near_returns):
        result = verify_candidate(nr["n0"], nr["T_approx"],
                                  refine=True, n_steps=10_000, verbose=False)
        if result.get("group1_pass", False):
            verified.append(result)
        if (j + 1) % 20 == 0:
            print(f"  Refined {j+1}/{n_near}, verified so far: {len(verified)}")

    wall_time = time.perf_counter() - t0
    print(f"  Verified: {len(verified)}, wall time: {wall_time:.0f}s")

    return {
        "method": name,
        "seed": seed,
        "n_candidates": len(candidates),
        "n_near_returns": n_near,
        "n_verified": len(verified),
        "hit_rate": len(verified) / max(len(candidates), 1),
        "wall_time_s": wall_time,
        "verified_orbits": verified,
    }


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def main():
    print("=" * 60)
    print("GROUP 4: BASELINE COMPARISON")
    print("=" * 60)

    # Load dataset for label-based methods
    npz_path = PROJECT_ROOT / "results" / "01_dataset_regen" / "at_rest_with_diagnostics.npz"
    d = np.load(npz_path)
    mask = (d["r_min_ever"] >= 0.01) & (d["label"] != -1)
    shape_pts = d["shape_n"][mask].astype(np.float64)
    labels = d["label"][mask].astype(np.int32)
    print(f"Dataset: {len(labels)} clean samples")

    # Load trained classifier for entropy method
    model = None
    model_paths = [
        PROJECT_ROOT / "data" / "baseline_model.eqx",
        PROJECT_ROOT / "results" / "baseline_model.eqx",
    ]
    for mp in model_paths:
        if mp.exists():
            import equinox as eqx
            from mega3bp.ml import BasinMLP
            model = eqx.tree_deserialise_leaves(mp, BasinMLP(key=jax.random.PRNGKey(0)))
            print(f"Loaded model from {mp}")
            break
    if model is None:
        print("WARNING: No trained model found. Test 4.4 (classifier entropy) will be skipped.")
        print("Searched:", [str(p) for p in model_paths])

    all_results = {}

    for seed in range(N_SEEDS):
        print(f"\n{'='*50} SEED {seed} {'='*50}")

        # Test 4.1: Uniform random
        print(f"\n--- Test 4.1: Uniform Random (seed={seed}) ---")
        cands = uniform_random_candidates(N_CANDIDATES, seed=seed)
        all_results.setdefault("uniform", []).append(
            run_method("uniform", cands, seed))

        # Test 4.2: Physics-informed grid (deterministic)
        print(f"\n--- Test 4.2: Physics Grid (seed={seed}) ---")
        cands = physics_grid_candidates(N_CANDIDATES)
        all_results.setdefault("physics_grid", []).append(
            run_method("physics_grid", cands, seed))

        # Test 4.3: Label disagreement
        print(f"\n--- Test 4.3: Label Disagreement (seed={seed}) ---")
        cands, _ = label_disagreement_candidates(shape_pts, labels, N_CANDIDATES)
        all_results.setdefault("label_disagreement", []).append(
            run_method("label_disagreement", cands, seed))

        # Test 4.4: Classifier entropy
        if model is not None:
            print(f"\n--- Test 4.4: Classifier Entropy (seed={seed}) ---")
            cands, _ = classifier_entropy_candidates(model, shape_pts, N_CANDIDATES)
            all_results.setdefault("classifier_entropy", []).append(
                run_method("classifier_entropy", cands, seed))

    # Write per-method results
    dir_map = {
        "uniform": "12_baseline_uniform",
        "physics_grid": "13_baseline_physics_grid",
        "label_disagreement": "14_baseline_label_only",
        "classifier_entropy": "15_method_classifier",
    }
    for method, results in all_results.items():
        write_json(OUT_DIR / dir_map[method] / "results.json", results)

    # Test 4.5: Comparison table
    comparison = {}
    for method, results in all_results.items():
        hit_rates = [r["hit_rate"] for r in results]
        n_verified = [r["n_verified"] for r in results]
        comparison[method] = {
            "hit_rate_mean": float(np.mean(hit_rates)),
            "hit_rate_std": float(np.std(hit_rates)),
            "hit_rate_2sigma": float(2 * np.std(hit_rates)),
            "n_verified_mean": float(np.mean(n_verified)),
            "n_verified_std": float(np.std(n_verified)),
            "wall_time_mean": float(np.mean([r["wall_time_s"] for r in results])),
        }

    write_json(OUT_DIR / "16_headline" / "comparison_table.json", comparison)

    # Determine claim level per PROMPT Test 4.5
    claim = "incomplete"
    if "classifier_entropy" in comparison and "uniform" in comparison:
        ce = comparison["classifier_entropy"]
        ur = comparison["uniform"]
        pg = comparison.get("physics_grid", {})

        ratio_vs_random = ce["hit_rate_mean"] / max(ur["hit_rate_mean"], 1e-10)
        ratio_vs_grid = ce["hit_rate_mean"] / max(pg.get("hit_rate_mean", 1e-10), 1e-10)

        ce_lo = ce["hit_rate_mean"] - ce["hit_rate_2sigma"]
        ur_hi = ur["hit_rate_mean"] + ur.get("hit_rate_2sigma", 0)
        pg_hi = pg.get("hit_rate_mean", 0) + pg.get("hit_rate_2sigma", 0)

        beats_random = ce_lo > ur_hi
        beats_grid = ce_lo > pg_hi

        if (ratio_vs_random >= 2 or ratio_vs_grid >= 2) and beats_random and beats_grid:
            claim = "strong"
        elif beats_random:
            claim = "weak"
        else:
            claim = "none"

    # Write verdict
    verdict = [
        "# GROUP 4 VERDICT: Baseline Comparison\n\n",
        f"## Summary\n\n",
        f"Ran {len(all_results)} methods x {N_SEEDS} seeds x {N_CANDIDATES} candidates.\n\n",
        "## Results\n\n",
        "| Method | Hit Rate (mean +/- 2sigma) | Verified (mean) | Wall Time (s) |\n",
        "|--------|---------------------------|-----------------|---------------|\n",
    ]
    for method, stats in comparison.items():
        verdict.append(
            f"| {method} | {stats['hit_rate_mean']:.4f} +/- {stats['hit_rate_2sigma']:.4f} | "
            f"{stats['n_verified_mean']:.1f} | {stats['wall_time_mean']:.0f} |\n")

    verdict.append(f"\n## Claim level: **{claim.upper()}**\n\n")
    claim_text = {
        "strong": "Classifier beats both random AND physics grid by >=2x outside error bars.",
        "weak": "Classifier beats random but ties/loses to physics grid. Modest contribution.",
        "none": "Classifier not significantly better than random. See PROMPT Section 7 fallback.",
        "incomplete": "Classifier not tested (model not available).",
    }
    verdict.append(claim_text.get(claim, "") + "\n")

    (OUT_DIR / "GROUP4_VERDICT.md").write_text("".join(verdict))
    print(f"\nGroup 4 complete. Claim level: {claim.upper()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
