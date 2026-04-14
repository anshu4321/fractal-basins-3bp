"""Regenerate velocity-space basin map + entropy map and save to NPZ.

The velocity_space_search.py script computes the basin map and entropy
but doesn't save them -- only the top candidates. This script re-runs
just Steps 1-3 and saves the full grid for visualization.

Reuses the same parameters as velocity_space_search.py so the result
matches what was used for the actual search.
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
from mega3bp.shape_sphere import pairwise_distances

OUT_DIR = Path(__file__).parent

# Same parameters as velocity_space_search.py
H = 0.002
N_STEPS = 50_000
BATCH_SIZE = 500
GRID_SIZE = 500
V_RANGE = 0.8

Q_COLLINEAR = jnp.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 0.0]], dtype=jnp.float64)


@partial(jax.jit, static_argnames=("n_steps",))
def classify_outcome(v1, v2, n_steps):
    h = jnp.float64(H)
    q0 = Q_COLLINEAR
    p0 = jnp.array([[v1, v2], [v1, v2], [-2*v1, -2*v2]], dtype=jnp.float64)
    sep_threshold = 50.0

    def step_fn(carry, step_idx):
        q, p, label, esc_step, r_min_ever, escaped = carry
        q, p = yoshida6_step(q, p, h)
        dists = pairwise_distances(q)
        r_min = jnp.min(dists)
        r_min_ever = jnp.minimum(r_min_ever, r_min)
        max_sep = jnp.max(dists)
        new_escaped = (~escaped) & (max_sep > sep_threshold)
        com = jnp.mean(q, axis=0)
        body_dists = jnp.linalg.norm(q - com, axis=1)
        escaper = jnp.int32(jnp.argmax(body_dists) + 1)
        label = jnp.where(new_escaped, escaper, label)
        esc_step = jnp.where(new_escaped, jnp.int32(step_idx), esc_step)
        escaped = escaped | new_escaped
        return (q, p, label, esc_step, r_min_ever, escaped), None

    init = (q0, p0, jnp.int32(0), jnp.int32(n_steps),
            jnp.float64(999.0), jnp.bool_(False))
    (_, _, label, esc_step, r_min_ever, _), _ = jax.lax.scan(
        step_fn, init, jnp.arange(n_steps, dtype=jnp.int32))

    return label, esc_step, r_min_ever


@partial(jax.jit, static_argnames=("n_steps",))
def batch_classify(v1_batch, v2_batch, n_steps):
    return jax.vmap(lambda v1, v2: classify_outcome(v1, v2, n_steps))(v1_batch, v2_batch)


def main():
    print("=" * 60)
    print("EXTRACT BASIN MAP (velocity space)")
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE} = {GRID_SIZE**2} points")
    print(f"v in [-{V_RANGE}, {V_RANGE}], n_steps={N_STEPS}")
    print("=" * 60)

    # Build grid
    v1_grid = np.linspace(-V_RANGE, V_RANGE, GRID_SIZE)
    v2_grid = np.linspace(-V_RANGE, V_RANGE, GRID_SIZE)
    V1, V2 = np.meshgrid(v1_grid, v2_grid)
    v1_flat = V1.ravel().astype(np.float64)
    v2_flat = V2.ravel().astype(np.float64)
    N = len(v1_flat)

    # Classify
    print(f"\n=== Classifying {N} points ===")
    labels_all, esc_all, rmin_all = [], [], []
    t0 = time.perf_counter()
    for i in range(0, N, BATCH_SIZE):
        v1b = jnp.array(v1_flat[i:i+BATCH_SIZE])
        v2b = jnp.array(v2_flat[i:i+BATCH_SIZE])
        lab, esc, rm = batch_classify(v1b, v2b, N_STEPS)
        labels_all.append(np.array(lab))
        esc_all.append(np.array(esc))
        rmin_all.append(np.array(rm))
        n_done = min(i + BATCH_SIZE, N)
        if n_done % 50000 == 0 or n_done == N:
            print(f"  {n_done}/{N} [{time.perf_counter()-t0:.0f}s]")

    labels = np.concatenate(labels_all)
    esc_step = np.concatenate(esc_all)
    r_min = np.concatenate(rmin_all)

    dt = time.perf_counter() - t0
    print(f"Classification done: {dt:.0f}s")

    # Statistics
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nLabel distribution:")
    for u, c in zip(unique, counts):
        print(f"  Label {u}: {c} ({100*c/N:.1f}%)")

    # Train classifier for entropy
    print(f"\n=== Training classifier for entropy ===")
    valid = labels >= 0
    X = np.column_stack([v1_flat[valid], v2_flat[valid]]).astype(np.float32)
    y = labels[valid].astype(np.int32)

    jax.config.update("jax_enable_x64", False)
    import equinox as eqx
    import optax
    from mega3bp.ml import BasinMLP, loss_fn, predict_proba

    key = jax.random.PRNGKey(42)
    model = BasinMLP(in_dim=2, hidden=128, n_layers=4, key=key)

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

    t_train = time.perf_counter()
    batch_size = 4096
    for epoch in range(20):
        perm = jax.random.permutation(jax.random.PRNGKey(epoch), len(y))
        for i in range(0, len(y), batch_size):
            idx = perm[i:i+batch_size]
            model, opt_state, _ = train_step(
                model, opt_state, X_jnp[idx], y_jnp[idx], lt[idx], em[idx])
    print(f"Training: {time.perf_counter() - t_train:.0f}s")

    # Compute entropy over full grid
    print(f"\n=== Computing entropy over grid ===")
    all_ent = []
    X_full = np.column_stack([v1_flat, v2_flat]).astype(np.float32)
    for i in range(0, N, 8192):
        batch = jnp.array(X_full[i:i+8192])
        probs = predict_proba(model, batch)
        ent = -jnp.sum(probs * jnp.log(probs + 1e-10), axis=-1)
        all_ent.append(np.asarray(ent))
    entropies = np.concatenate(all_ent)

    print(f"Entropy range: [{entropies.min():.4f}, {entropies.max():.4f}]")

    jax.config.update("jax_enable_x64", True)

    # Save everything
    out_path = OUT_DIR / "basin_map.npz"
    np.savez_compressed(
        out_path,
        v1_grid=v1_grid,
        v2_grid=v2_grid,
        labels=labels.reshape(GRID_SIZE, GRID_SIZE),
        entropy=entropies.reshape(GRID_SIZE, GRID_SIZE),
        esc_step=esc_step.reshape(GRID_SIZE, GRID_SIZE),
        r_min=r_min.reshape(GRID_SIZE, GRID_SIZE),
    )
    print(f"\nSaved basin map to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Also save classifier
    ckpt_path = OUT_DIR / "velocity_classifier.eqx"
    eqx.tree_serialise_leaves(ckpt_path, model)
    print(f"Saved classifier to {ckpt_path}")

    print(f"\nTotal time: {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
