"""Velocity-space periodic orbit search at the collinear configuration.

Key insight: ALL known Suvakov orbits start from the same shape sphere
point (collinear: bodies at -1, 0, +1). They differ only in initial
velocities (v1, v2) with body1=body2=(v1,v2), body3=(-2v1,-2v2).

This script:
1. Generates a 2D basin map in velocity space (v1, v2)
2. Trains a classifier on velocity-space basin labels
3. Uses classifier entropy to select orbit candidates
4. Shoots candidates with full phase-space return tracking
5. Refines with Gauss-Newton on (v1, v2, T)
6. Validates by checking closure + comparing to Suvakov catalog
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

from mega3bp.integrators import yoshida6_step
from mega3bp.dynamics import total_energy, angular_momentum
from mega3bp.shape_sphere import pairwise_distances

OUT_DIR = Path(__file__).parent

# Integration parameters
H = 0.002
N_STEPS = 50_000   # T_max = 100
BATCH_SIZE = 500
EXCURSION_THRESHOLD = 2.0

# Collinear configuration (fixed for all orbits)
Q_COLLINEAR = jnp.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 0.0]], dtype=jnp.float64)


def make_ic(v1, v2):
    """Create (q0, p0) from velocity parameters at collinear config."""
    q0 = Q_COLLINEAR
    p0 = jnp.array([[v1, v2], [v1, v2], [-2*v1, -2*v2]], dtype=jnp.float64)
    return q0, p0


# ============================================================
# STEP 1: Generate velocity-space basin dataset
# ============================================================

@partial(jax.jit, static_argnames=("n_steps",))
def classify_outcome(v1, v2, n_steps):
    """Integrate orbit from collinear config with velocity (v1,v2).

    Returns: (label, escape_time, r_min_ever)
    Label: 0=bound, 1/2/3=body escapes, -1=ambiguous
    """
    h = jnp.float64(H)
    q0, p0 = make_ic(v1, v2)
    sep_threshold = 50.0  # escape detection

    def step_fn(carry, step_idx):
        q, p, label, esc_step, r_min_ever, escaped = carry
        q, p = yoshida6_step(q, p, h)

        dists = pairwise_distances(q)
        r_min = jnp.min(dists)
        r_min_ever = jnp.minimum(r_min_ever, r_min)
        max_sep = jnp.max(dists)

        # Check escape: if max separation > threshold
        new_escaped = (~escaped) & (max_sep > sep_threshold)

        # Determine which body escaped (furthest from COM)
        com = jnp.mean(q, axis=0)
        body_dists = jnp.linalg.norm(q - com, axis=1)
        escaper = jnp.int32(jnp.argmax(body_dists) + 1)  # 1, 2, or 3

        label = jnp.where(new_escaped, escaper, label)
        esc_step = jnp.where(new_escaped, jnp.int32(step_idx), esc_step)
        escaped = escaped | new_escaped

        return (q, p, label, esc_step, r_min_ever, escaped), None

    init = (q0, p0, jnp.int32(0), jnp.int32(n_steps),
            jnp.float64(999.0), jnp.bool_(False))
    (_, _, label, esc_step, r_min_ever, _), _ = jax.lax.scan(
        step_fn, init, jnp.arange(n_steps, dtype=jnp.int32))

    return label, jnp.float64(esc_step) * h, r_min_ever


@partial(jax.jit, static_argnames=("n_steps",))
def batch_classify(v1_batch, v2_batch, n_steps):
    """Classify a batch of velocity ICs."""
    return jax.vmap(lambda v1, v2: classify_outcome(v1, v2, n_steps))(v1_batch, v2_batch)


# ============================================================
# STEP 4: Full phase-space return shooting
# ============================================================

@partial(jax.jit, static_argnames=("n_steps",))
def shoot_fullstate_velocity(v1, v2, n_steps):
    """Shoot from collinear config with (v1,v2), track full-state return.

    Only counts returns after excursion beyond EXCURSION_THRESHOLD.
    """
    h = jnp.float64(H)
    q0, p0 = make_ic(v1, v2)
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
def batch_shoot_velocity(v1_batch, v2_batch, n_steps):
    return jax.vmap(lambda v1, v2: shoot_fullstate_velocity(v1, v2, n_steps))(v1_batch, v2_batch)


# ============================================================
# STEP 5: Gauss-Newton refinement in velocity space
# ============================================================

@partial(jax.jit, static_argnames=("n_steps",))
def shooting_residual_velocity(params, n_steps):
    """F(v1, v2, T) = state(T) - state(0) for collinear IC with velocity (v1,v2)."""
    v1, v2, T = params[0], params[1], params[2]
    q0, p0 = make_ic(v1, v2)
    h = T / n_steps

    def step_fn(carry, _):
        q, p = carry
        q, p = yoshida6_step(q, p, jnp.float64(h))
        return (q, p), None

    (qf, pf), _ = jax.lax.scan(step_fn, (q0, p0), None, length=n_steps)
    dq = (qf - q0).ravel()
    dp = (pf - p0).ravel()
    return jnp.concatenate([dq, dp])


@partial(jax.jit, static_argnames=("n_steps",))
def lm_step_velocity(params, lam, n_steps):
    """One Levenberg-Marquardt step for velocity-space refinement."""
    def F(p):
        return shooting_residual_velocity(p, n_steps)

    residual = F(params)
    J = jax.jacfwd(F)(params)
    JtJ = J.T @ J
    JtF = J.T @ residual
    damping = lam * jnp.diag(jnp.diag(JtJ) + 1e-12)
    dp = jnp.linalg.solve(JtJ + damping, JtF)
    return params - dp, jnp.linalg.norm(residual), jnp.linalg.norm(dp)


def refine_velocity(v1_0, v2_0, T0, n_steps=20_000, max_iter=50, tol=1e-10, verbose=True):
    """Levenberg-Marquardt refinement in (v1, v2, T) space."""
    params = jnp.array([v1_0, v2_0, T0], dtype=jnp.float64)
    lam = jnp.float64(1e-2)

    res0 = shooting_residual_velocity(params, n_steps)
    best_residual = float(jnp.linalg.norm(res0))
    best_params = params
    current_residual = best_residual

    if verbose:
        print(f"  LM iter   0: ||F|| = {best_residual:.2e}, lam = {float(lam):.1e}")

    stall_count = 0
    for i in range(max_iter):
        candidate, _, _ = lm_step_velocity(params, lam, n_steps)
        candidate = candidate.at[2].set(jnp.maximum(candidate[2], 0.01))  # T > 0

        new_res = shooting_residual_velocity(candidate, n_steps)
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

        if verbose and (i + 1) % 5 == 0:
            print(f"  LM iter {i+1:3d}: ||F|| = {current_residual:.2e}, "
                  f"lam = {float(lam):.1e}, best = {best_residual:.2e}")

        if current_residual < tol:
            if verbose:
                print(f"  Converged at iter {i+1}: ||F|| = {current_residual:.2e}")
            break

        if stall_count > 10:
            if verbose:
                print(f"  Stalled at iter {i+1}: ||F|| = {current_residual:.2e}")
            break

        if (i + 1) >= 15 and best_residual > 1e-2:
            if verbose:
                print(f"  Hopeless at iter {i+1}: best = {best_residual:.2e}")
            break

    v1, v2, T = float(best_params[0]), float(best_params[1]), float(best_params[2])
    return {"v1": v1, "v2": v2, "T": T, "residual": best_residual,
            "converged": best_residual < tol}


# ============================================================
# Suvakov catalog for validation
# ============================================================

SUVAKOV = [
    ("Figure-eight", 0.3471168881, 0.5327249454, 6.3259139829),
    ("Butterfly I", 0.3068934205, 0.1255065670, 6.2346748391),
    ("Moth I", 0.4644451728, 0.3960600146, 14.8943051743),
    ("Goggles", 0.0833000718, 0.1278892555, 10.4648495256),
    ("Dragonfly", 0.0805842255, 0.5888360898, 21.2723373956),
    ("Yin-Yang Ib", 0.2826986823, 0.3272087861, 10.9633031497),
]


def main():
    print("=" * 70)
    print("VELOCITY-SPACE PERIODIC ORBIT SEARCH")
    print("Collinear config, scanning (v1, v2) velocity plane")
    print("=" * 70)

    # ---- STEP 1: Generate velocity-space basin dataset ----
    print("\n=== STEP 1: Generating velocity-space basin map ===")
    grid_size = 500  # 500x500 = 250K points
    v_range = 0.8
    v1_grid = np.linspace(-v_range, v_range, grid_size)
    v2_grid = np.linspace(-v_range, v_range, grid_size)
    V1, V2 = np.meshgrid(v1_grid, v2_grid)
    v1_flat = V1.ravel().astype(np.float64)
    v2_flat = V2.ravel().astype(np.float64)
    print(f"Grid: {grid_size}x{grid_size} = {len(v1_flat)} points, v in [-{v_range}, {v_range}]")

    labels_all = []
    t0 = time.perf_counter()
    for i in range(0, len(v1_flat), BATCH_SIZE):
        v1b = jnp.array(v1_flat[i:i+BATCH_SIZE])
        v2b = jnp.array(v2_flat[i:i+BATCH_SIZE])
        lab, _, _ = batch_classify(v1b, v2b, N_STEPS)
        labels_all.append(np.array(lab))
        n_done = min(i + BATCH_SIZE, len(v1_flat))
        if n_done % 50000 == 0:
            print(f"  {n_done}/{len(v1_flat)} [{time.perf_counter()-t0:.0f}s]")

    labels = np.concatenate(labels_all)
    dt_dataset = time.perf_counter() - t0
    print(f"Dataset generated in {dt_dataset:.0f}s")

    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Label {u}: {c} ({100*c/len(labels):.1f}%)")

    # ---- STEP 2: Train classifier on velocity space ----
    print("\n=== STEP 2: Training velocity-space classifier ===")

    # Filter out ambiguous labels
    valid = labels >= 0
    X = np.column_stack([v1_flat[valid], v2_flat[valid]]).astype(np.float32)
    y = labels[valid].astype(np.int32)

    jax.config.update("jax_enable_x64", False)  # fp32 for training
    import equinox as eqx
    import optax
    from mega3bp.ml import BasinMLP, loss_fn, predict_proba

    key = jax.random.PRNGKey(42)
    model = BasinMLP(in_dim=2, hidden=128, n_layers=4, key=key)

    # Class weights
    unique_y, counts_y = np.unique(y, return_counts=True)
    cw = np.ones(4, dtype=np.float32)
    for u, c in zip(unique_y, counts_y):
        if u < 4:
            cw[u] = len(y) / (4.0 * c)
    class_weights = jnp.array(cw)

    optimizer = optax.adamw(1e-3, weight_decay=1e-4)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    X_jnp = jnp.array(X)
    y_jnp = jnp.array(y)
    lt = jnp.zeros(len(y), dtype=jnp.float32)
    em = jnp.zeros(len(y), dtype=jnp.bool_)

    @eqx.filter_jit
    def train_step(model, opt_state, x, y_b, lt_b, em_b):
        grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)
        (loss, aux), grads = grad_fn(model, x, y_b, lt_b, em_b, class_weights, 0.0)
        updates, opt_state_new = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state_new, aux

    batch_size = 4096
    n_epochs = 20
    t0 = time.perf_counter()
    for epoch in range(n_epochs):
        perm = jax.random.permutation(jax.random.PRNGKey(epoch), len(y))
        for i in range(0, len(y), batch_size):
            idx = perm[i:i+batch_size]
            model, opt_state, _ = train_step(
                model, opt_state, X_jnp[idx], y_jnp[idx], lt[idx], em[idx])
    dt_train = time.perf_counter() - t0
    print(f"Training done in {dt_train:.0f}s")

    # ---- STEP 3: Compute entropy map + check Suvakov locations ----
    print("\n=== STEP 3: Entropy map + Suvakov validation ===")

    # Compute entropy for all grid points
    all_ent = []
    X_full = np.column_stack([v1_flat, v2_flat]).astype(np.float32)
    for i in range(0, len(X_full), 8192):
        batch = jnp.array(X_full[i:i+8192])
        probs = predict_proba(model, batch)
        ent = -jnp.sum(probs * jnp.log(probs + 1e-10), axis=-1)
        all_ent.append(np.asarray(ent))
    entropies = np.concatenate(all_ent)

    print(f"Entropy range: [{entropies.min():.4f}, {entropies.max():.4f}]")
    print(f"High-entropy (>0.5): {np.sum(entropies > 0.5)}")

    # Check entropy at Suvakov orbit locations
    print(f"\nEntropy at known Suvakov orbits:")
    print(f"{'Name':<20} {'v1':>10} {'v2':>10} {'Entropy':>10} {'Percentile':>10}")
    for name, sv1, sv2, sT in SUVAKOV:
        # Find nearest grid point
        dist = (v1_flat - sv1)**2 + (v2_flat - sv2)**2
        idx = np.argmin(dist)
        ent_at_orbit = entropies[idx]
        percentile = 100 * np.mean(entropies <= ent_at_orbit)
        print(f"  {name:<20} {sv1:10.6f} {sv2:10.6f} {ent_at_orbit:10.4f} {percentile:9.1f}%")

    jax.config.update("jax_enable_x64", True)  # back to fp64

    # ---- STEP 4: Select candidates by entropy + shoot ----
    print("\n=== STEP 4: Entropy-guided shooting ===")
    n_candidates = 5000
    top_idx = np.argsort(-entropies)[:n_candidates]
    cand_v1 = v1_flat[top_idx].astype(np.float64)
    cand_v2 = v2_flat[top_idx].astype(np.float64)
    print(f"Selected {n_candidates} candidates by entropy")

    all_res, all_steps = [], []
    t0 = time.perf_counter()
    for i in range(0, n_candidates, BATCH_SIZE):
        v1b = jnp.array(cand_v1[i:i+BATCH_SIZE])
        v2b = jnp.array(cand_v2[i:i+BATCH_SIZE])
        res, steps = batch_shoot_velocity(v1b, v2b, N_STEPS)
        all_res.append(np.array(res))
        all_steps.append(np.array(steps))
        n_done = min(i + BATCH_SIZE, n_candidates)
        if n_done % 1000 == 0:
            best = min(np.min(r) for r in all_res)
            print(f"  {n_done}/{n_candidates} [{time.perf_counter()-t0:.0f}s] best ||F||: {best:.4e}")

    residuals = np.concatenate(all_res)
    steps = np.concatenate(all_steps)
    dt_shoot = time.perf_counter() - t0
    print(f"Shooting done in {dt_shoot:.0f}s")

    # Sort by residual
    order = np.argsort(residuals)
    print(f"\nTop 20 by full-state residual:")
    print(f"{'Rank':>4} {'||F||':>12} {'v1':>10} {'v2':>10} {'T':>8}")
    for rank in range(min(20, len(order))):
        idx = order[rank]
        T = (steps[idx] + 1) * H
        if residuals[idx] < 900:
            print(f"{rank+1:4d} {residuals[idx]:12.4e} {cand_v1[idx]:10.6f} "
                  f"{cand_v2[idx]:10.6f} {T:8.3f}")

    # ---- STEP 5: Refine best candidates ----
    good = order[residuals[order] < 5.0]
    n_refine = min(50, len(good))
    print(f"\n=== STEP 5: Refining top {n_refine} candidates ===")

    verified = []
    for rank in range(n_refine):
        idx = good[rank]
        T = float((steps[idx] + 1) * H)
        v1, v2 = float(cand_v1[idx]), float(cand_v2[idx])

        print(f"\n--- Rank {rank+1}: ||F||={residuals[idx]:.4e}, "
              f"v1={v1:.6f}, v2={v2:.6f}, T={T:.3f} ---")

        result = refine_velocity(v1, v2, T, n_steps=20_000, max_iter=50,
                                 tol=1e-10, verbose=True)

        if result["converged"]:
            verified.append(result)
            print(f"  >>> CONVERGED! ||F|| = {result['residual']:.2e}")

            # Check against Suvakov catalog
            for name, sv1, sv2, sT in SUVAKOV:
                dv = np.sqrt((result["v1"] - sv1)**2 + (result["v2"] - sv2)**2)
                if dv < 0.01:
                    print(f"  >>> MATCHES {name}! dv={dv:.6e}")
        else:
            print(f"  Final: ||F|| = {result['residual']:.2e}")

    print(f"\n{'='*70}")
    print(f"RESULT: {len(verified)} verified periodic orbits")
    print(f"{'='*70}")

    for v in verified:
        print(f"  v1={v['v1']:.10f}, v2={v['v2']:.10f}, T={v['T']:.10f}, "
              f"||F||={v['residual']:.2e}")

    # Save everything
    save_data = {
        "grid": {"size": grid_size, "v_range": v_range},
        "dataset_time_s": dt_dataset,
        "train_time_s": dt_train,
        "shoot_time_s": dt_shoot,
        "n_verified": len(verified),
        "verified": verified,
        "top_20": [{
            "rank": rank + 1,
            "v1": float(cand_v1[order[rank]]),
            "v2": float(cand_v2[order[rank]]),
            "residual": float(residuals[order[rank]]),
            "T": float((steps[order[rank]] + 1) * H),
        } for rank in range(min(20, len(order))) if residuals[order[rank]] < 900],
    }
    with open(OUT_DIR / "velocity_space_results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\nResults saved to {OUT_DIR / 'velocity_space_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
