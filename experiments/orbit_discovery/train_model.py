"""Train a BasinMLP classifier on the at-rest dataset (NPZ format).

Minimal training script for orbit discovery pipeline.
Reads at_rest_with_diagnostics.npz, trains BasinMLP, saves checkpoint.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
jax.config.update("jax_enable_x64", False)  # fp32 for training speed

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optax

from mega3bp.ml import N_CLASSES, BasinMLP, loss_fn, predict

OUT_DIR = Path(__file__).parent


def main():
    print("=" * 60)
    print("TRAINING BASIN CLASSIFIER")
    print("=" * 60)
    print(f"JAX devices: {jax.devices()}")

    # Load dataset
    npz_path = PROJECT_ROOT / "results" / "01_dataset_regen" / "at_rest_with_diagnostics.npz"
    d = np.load(npz_path)
    mask = (d["r_min_ever"] >= 0.01) & (d["label"] != -1)
    shape_pts = d["shape_n"][mask].astype(np.float32)
    labels = d["label"][mask].astype(np.int32)
    print(f"Dataset: {len(labels)} clean samples")

    # Escape time for regression (use 0 for bound, actual for escape)
    escape_time = d["escape_time"][mask]
    log_t = np.where(np.isfinite(escape_time) & (escape_time > 0),
                     np.log(escape_time), 0.0).astype(np.float32)
    escape_mask_np = np.isfinite(escape_time) & (escape_time > 0)

    # Train/val/test split (80/10/10)
    rng = np.random.default_rng(42)
    N = len(labels)
    perm = rng.permutation(N)
    n_train = int(0.8 * N)
    n_val = int(0.1 * N)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    X_train = jnp.array(shape_pts[train_idx])
    y_train = jnp.array(labels[train_idx])
    lt_train = jnp.array(log_t[train_idx])
    em_train = jnp.array(escape_mask_np[train_idx])

    X_val = jnp.array(shape_pts[val_idx])
    y_val = jnp.array(labels[val_idx])
    X_test = jnp.array(shape_pts[test_idx])
    y_test = np.array(labels[test_idx])

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Class weights (inverse frequency)
    unique, counts = np.unique(labels[train_idx], return_counts=True)
    cw = np.ones(N_CLASSES, dtype=np.float32)
    for u, c in zip(unique, counts):
        cw[u] = len(train_idx) / (N_CLASSES * c)
    class_weights = jnp.array(cw)
    print(f"Class weights: {cw}")

    # Model
    key = jax.random.PRNGKey(42)
    model = BasinMLP(in_dim=3, hidden=256, n_layers=5, key=key)
    print(f"Model params: {sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))}")

    # Optimizer: AdamW with cosine schedule
    n_epochs = 30
    batch_size = 4096
    steps_per_epoch = max(1, n_train // batch_size)
    total_steps = n_epochs * steps_per_epoch
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-4, peak_value=1e-3, warmup_steps=steps_per_epoch,
        decay_steps=total_steps, end_value=1e-5)
    optimizer = optax.adamw(schedule, weight_decay=1e-4)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, opt_state, x, y, lt, em):
        grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)
        (loss, aux), grads = grad_fn(model, x, y, lt, em, class_weights, 0.3)
        updates, opt_state_new = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state_new, aux

    # Training loop
    best_val_acc = 0.0
    best_model = model
    t0 = time.perf_counter()

    for epoch in range(n_epochs):
        perm_e = jax.random.permutation(jax.random.PRNGKey(epoch), n_train)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm_e[i:i + batch_size]
            model, opt_state, aux = train_step(
                model, opt_state,
                X_train[idx], y_train[idx], lt_train[idx], em_train[idx])
            epoch_loss += float(aux["total_loss"])
            n_batches += 1

        # Validation
        val_preds, _ = predict(model, X_val)
        val_acc = float(jnp.mean(val_preds == y_val))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model

        if (epoch + 1) % 5 == 0:
            dt = time.perf_counter() - t0
            print(f"  Epoch {epoch+1:3d}/{n_epochs}: "
                  f"loss={epoch_loss/n_batches:.4f}, "
                  f"val_acc={val_acc:.4f}, "
                  f"best={best_val_acc:.4f}, "
                  f"[{dt:.0f}s]")

    # Test evaluation
    test_preds, _ = predict(best_model, X_test)
    test_preds_np = np.array(test_preds)
    test_acc = float(np.mean(test_preds_np == y_test))
    print(f"\nTest accuracy: {test_acc:.4f}")

    # Per-class F1
    for c in range(N_CLASSES):
        tp = np.sum((test_preds_np == c) & (y_test == c))
        fp = np.sum((test_preds_np == c) & (y_test != c))
        fn = np.sum((test_preds_np != c) & (y_test == c))
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-10)
        print(f"  Class {c}: P={prec:.3f} R={rec:.3f} F1={f1:.3f} (n={np.sum(y_test==c)})")

    # Save checkpoint
    ckpt_path = OUT_DIR / "baseline_model.eqx"
    eqx.tree_serialise_leaves(ckpt_path, best_model)
    print(f"\nModel saved to {ckpt_path}")

    dt = time.perf_counter() - t0
    print(f"Total training time: {dt:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
