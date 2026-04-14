"""Group 4 baselines in velocity space: the headline comparison.

Runs 4 candidate-selection methods × N_SEEDS seeds each through the
full shoot + LM-refine pipeline, logs hit rate / verified count / wall time.

Methods:
  1. Uniform random sampling of (v1, v2)
  2. Physics-informed grid (collinear + isosceles patterns in velocity space)
  3. Label disagreement on velocity-space basin labels (no classifier)
  4. Classifier entropy (our main method)

Requires: basin_map.npz (from extract_basin_map.py)
          velocity_classifier.eqx (saved by extract_basin_map.py)
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
from scipy.spatial import cKDTree

from mega3bp.integrators import yoshida6_step
from mega3bp.shape_sphere import pairwise_distances

OUT_DIR = Path(__file__).parent

# Same params as velocity_space_search.py
H = 0.002
N_STEPS = 50_000
BATCH_SIZE = 500
EXCURSION_THRESHOLD = 2.0
N_CANDIDATES = 5000
N_REFINE_TOP = 50
N_SEEDS = 3  # 3 seeds for reasonable error bars without killing compute budget
V_RANGE = 0.8

Q_COLLINEAR = jnp.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 0.0]], dtype=jnp.float64)


# ============================================================
# Candidate selection methods
# ============================================================

def uniform_candidates(n, seed):
    rng = np.random.default_rng(seed)
    v1 = rng.uniform(-V_RANGE, V_RANGE, n)
    v2 = rng.uniform(-V_RANGE, V_RANGE, n)
    return np.column_stack([v1, v2])


def grid_candidates(n):
    """Physics-informed grid: collinear (v2=0) + isosceles (v1=0) + diagonal axes."""
    # Concentrate on symmetry axes + uniform fill
    n_axis = n // 4
    n_fill = n - 3 * n_axis

    # Along v2 = 0 axis (collinear-like momenta)
    v1_a = np.linspace(-V_RANGE, V_RANGE, n_axis)
    v2_a = np.zeros(n_axis)
    # Along v1 = 0 axis (isosceles-like)
    v1_b = np.zeros(n_axis)
    v2_b = np.linspace(-V_RANGE, V_RANGE, n_axis)
    # Along v1 = v2 diagonal
    t = np.linspace(-V_RANGE, V_RANGE, n_axis)
    v1_c = t / np.sqrt(2)
    v2_c = t / np.sqrt(2)
    # Uniform fill
    rng = np.random.default_rng(0)
    v1_d = rng.uniform(-V_RANGE, V_RANGE, n_fill)
    v2_d = rng.uniform(-V_RANGE, V_RANGE, n_fill)

    v1 = np.concatenate([v1_a, v1_b, v1_c, v1_d])
    v2 = np.concatenate([v2_a, v2_b, v2_c, v2_d])
    return np.column_stack([v1, v2])


def label_disagreement_candidates(labels_grid, v1_grid, v2_grid, n, seed):
    """k-NN label disagreement on velocity-space basin labels (no classifier)."""
    V1, V2 = np.meshgrid(v1_grid, v2_grid)
    pts = np.column_stack([V1.ravel(), V2.ravel()])
    labels = labels_grid.ravel()

    # Filter ambiguous
    valid = labels >= 0
    pts_v = pts[valid]
    lab_v = labels[valid]

    # k-NN disagreement
    tree = cKDTree(pts_v)
    _, idx = tree.query(pts_v, k=6)
    neighbor_labels = lab_v[idx[:, 1:]]
    own_labels = lab_v[:, None]
    disagree = np.mean(neighbor_labels != own_labels, axis=1)

    # Add noise for seed variation, then pick top n
    rng = np.random.default_rng(seed)
    noisy = disagree + rng.normal(0, 0.01, size=len(disagree))
    top = np.argsort(-noisy)[:n]
    return pts_v[top]


def classifier_entropy_candidates(entropy_grid, v1_grid, v2_grid, n, seed):
    """Classifier entropy (our method). Same across seeds — deterministic."""
    V1, V2 = np.meshgrid(v1_grid, v2_grid)
    pts = np.column_stack([V1.ravel(), V2.ravel()])
    entropies = entropy_grid.ravel()

    # Add small noise for seed variation
    rng = np.random.default_rng(seed)
    noisy = entropies + rng.normal(0, 0.001, size=len(entropies))
    top = np.argsort(-noisy)[:n]
    return pts[top]


# ============================================================
# Shooting (full-state return)
# ============================================================

@partial(jax.jit, static_argnames=("n_steps",))
def shoot_one(v1, v2, n_steps):
    h = jnp.float64(H)
    q0 = Q_COLLINEAR
    p0 = jnp.array([[v1, v2], [v1, v2], [-2*v1, -2*v2]], dtype=jnp.float64)
    state0 = jnp.concatenate([q0.ravel(), p0.ravel()])

    def step_fn(carry, step_idx):
        q, p, best_res, best_step, has_excursed = carry
        q, p = yoshida6_step(q, p, h)
        state = jnp.concatenate([q.ravel(), p.ravel()])
        residual = jnp.linalg.norm(state - state0)
        has_excursed = has_excursed | (residual > EXCURSION_THRESHOLD)
        r_min = jnp.min(pairwise_distances(q))
        valid = has_excursed & (r_min >= 0.01)
        is_better = valid & (residual < best_res)
        best_res = jnp.where(is_better, residual, best_res)
        best_step = jnp.where(is_better, step_idx, best_step)
        return (q, p, best_res, best_step, has_excursed), None

    init = (q0, p0, jnp.float64(999.0), jnp.int64(0), jnp.bool_(False))
    (_, _, best_res, best_step, _), _ = jax.lax.scan(
        step_fn, init, jnp.arange(n_steps, dtype=jnp.int64))
    return best_res, best_step


@partial(jax.jit, static_argnames=("n_steps",))
def shoot_batch(v1_batch, v2_batch, n_steps):
    return jax.vmap(lambda v1, v2: shoot_one(v1, v2, n_steps))(v1_batch, v2_batch)


# ============================================================
# LM refinement (imported from velocity_space_search)
# ============================================================

@partial(jax.jit, static_argnames=("n_steps",))
def shooting_residual(params, n_steps):
    v1, v2, T = params[0], params[1], params[2]
    q0 = Q_COLLINEAR
    p0 = jnp.array([[v1, v2], [v1, v2], [-2*v1, -2*v2]], dtype=jnp.float64)
    h = T / n_steps
    def step_fn(carry, _):
        q, p = carry
        q, p = yoshida6_step(q, p, jnp.float64(h))
        return (q, p), None
    (qf, pf), _ = jax.lax.scan(step_fn, (q0, p0), None, length=n_steps)
    return jnp.concatenate([(qf - q0).ravel(), (pf - p0).ravel()])


@partial(jax.jit, static_argnames=("n_steps",))
def lm_step(params, lam, n_steps):
    def F(p):
        return shooting_residual(p, n_steps)
    residual = F(params)
    J = jax.jacfwd(F)(params)
    JtJ = J.T @ J
    JtF = J.T @ residual
    damping = lam * jnp.diag(jnp.diag(JtJ) + 1e-12)
    dp = jnp.linalg.solve(JtJ + damping, JtF)
    return params - dp, jnp.linalg.norm(residual)


def refine_lm(v1_0, v2_0, T0, n_steps=20_000, max_iter=40, tol=1e-10):
    params = jnp.array([v1_0, v2_0, T0], dtype=jnp.float64)
    lam = jnp.float64(1e-2)
    res0 = shooting_residual(params, n_steps)
    best_residual = float(jnp.linalg.norm(res0))
    best_params = params
    current_residual = best_residual

    stall_count = 0
    for i in range(max_iter):
        candidate, _ = lm_step(params, lam, n_steps)
        candidate = candidate.at[2].set(jnp.maximum(candidate[2], 0.01))
        new_res = shooting_residual(candidate, n_steps)
        new_residual = float(jnp.linalg.norm(new_res))
        if new_residual < current_residual:
            rel_improvement = (current_residual - new_residual) / max(current_residual, 1e-30)
            params = candidate
            current_residual = new_residual
            lam = jnp.maximum(lam / 3.0, jnp.float64(1e-12))
            if new_residual < best_residual:
                best_residual = new_residual
                best_params = params
            if rel_improvement > 1e-4:
                stall_count = 0
            else:
                stall_count += 1
        else:
            lam = jnp.minimum(lam * 10.0, jnp.float64(1e6))
            stall_count += 1
        if current_residual < tol:
            break
        if stall_count > 10:
            break
        if (i + 1) >= 15 and best_residual > 1e-2:
            break

    v1, v2, T = float(best_params[0]), float(best_params[1]), float(best_params[2])
    return {"v1": v1, "v2": v2, "T": T, "residual": best_residual, "converged": best_residual < tol}


# ============================================================
# Run one method × seed
# ============================================================

def run_one(method_name, candidates, seed):
    t0 = time.perf_counter()
    print(f"\n--- {method_name} seed={seed} ---")

    # Shoot
    all_res, all_steps = [], []
    for i in range(0, len(candidates), BATCH_SIZE):
        v1b = jnp.array(candidates[i:i+BATCH_SIZE, 0])
        v2b = jnp.array(candidates[i:i+BATCH_SIZE, 1])
        res, steps = shoot_batch(v1b, v2b, N_STEPS)
        all_res.append(np.array(res))
        all_steps.append(np.array(steps))
    residuals = np.concatenate(all_res)
    steps = np.concatenate(all_steps)
    t_shoot = time.perf_counter() - t0
    print(f"  Shooting: {t_shoot:.0f}s")

    # Top-K refine
    order = np.argsort(residuals)
    verified = []
    n_refine = min(N_REFINE_TOP, np.sum(residuals < 5.0))
    t_refine_start = time.perf_counter()
    for idx in order[:n_refine]:
        if residuals[idx] > 5.0:
            continue
        v1, v2 = float(candidates[idx, 0]), float(candidates[idx, 1])
        T = float((steps[idx] + 1) * H)
        result = refine_lm(v1, v2, T)
        if result["converged"]:
            verified.append(result)
    t_refine = time.perf_counter() - t_refine_start
    t_total = time.perf_counter() - t0
    print(f"  Refine: {t_refine:.0f}s, verified: {len(verified)}/{n_refine}")

    return {
        "method": method_name,
        "seed": seed,
        "n_candidates": len(candidates),
        "n_near_returns": int(np.sum(residuals < 0.5)),
        "n_refined": int(n_refine),
        "n_verified": len(verified),
        "hit_rate_shoot": float(np.mean(residuals < 0.5)),
        "hit_rate_verify": len(verified) / max(len(candidates), 1),
        "shoot_time_s": t_shoot,
        "refine_time_s": t_refine,
        "total_time_s": t_total,
        "best_residual": float(residuals.min()),
        "verified_orbits": verified,
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("GROUP 4 BASELINES IN VELOCITY SPACE")
    print(f"N_SEEDS={N_SEEDS}, N_CANDIDATES={N_CANDIDATES}, N_REFINE_TOP={N_REFINE_TOP}")
    print("=" * 60)

    # Load basin map + classifier data
    basin_path = OUT_DIR / "basin_map.npz"
    if not basin_path.exists():
        print(f"ERROR: {basin_path} not found. Run extract_basin_map.py first.")
        return 1

    d = np.load(basin_path)
    labels_grid = d["labels"]
    entropy_grid = d["entropy"]
    v1_grid = d["v1_grid"]
    v2_grid = d["v2_grid"]
    print(f"Basin map: {labels_grid.shape}, entropy range [{entropy_grid.min():.3f}, {entropy_grid.max():.3f}]")

    all_results = {}

    for seed in range(N_SEEDS):
        print(f"\n{'=' * 50}")
        print(f"SEED {seed}")
        print(f"{'=' * 50}")

        cands_uniform = uniform_candidates(N_CANDIDATES, seed=seed)
        all_results.setdefault("uniform", []).append(
            run_one("uniform", cands_uniform, seed))

        cands_grid = grid_candidates(N_CANDIDATES)
        all_results.setdefault("physics_grid", []).append(
            run_one("physics_grid", cands_grid, seed))

        cands_label = label_disagreement_candidates(
            labels_grid, v1_grid, v2_grid, N_CANDIDATES, seed=seed)
        all_results.setdefault("label_disagreement", []).append(
            run_one("label_disagreement", cands_label, seed))

        cands_entropy = classifier_entropy_candidates(
            entropy_grid, v1_grid, v2_grid, N_CANDIDATES, seed=seed)
        all_results.setdefault("classifier_entropy", []).append(
            run_one("classifier_entropy", cands_entropy, seed))

        # Incremental save
        with open(OUT_DIR / "group4_baselines.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Summary
    print("\n" + "=" * 60)
    print("GROUP 4 SUMMARY")
    print("=" * 60)
    print(f"{'Method':<22} {'Hit Rate (verify)':>20} {'Verified (mean)':>18} {'Wall (s)':>10}")
    print("-" * 75)
    for method, runs in all_results.items():
        hrs = [r["hit_rate_verify"] for r in runs]
        nvs = [r["n_verified"] for r in runs]
        wts = [r["total_time_s"] for r in runs]
        print(f"{method:<22} {np.mean(hrs):.5f} +/- {2*np.std(hrs):.5f}   "
              f"{np.mean(nvs):6.1f} +/- {2*np.std(nvs):4.1f}   {np.mean(wts):8.0f}")

    print(f"\nSaved to {OUT_DIR / 'group4_baselines.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
