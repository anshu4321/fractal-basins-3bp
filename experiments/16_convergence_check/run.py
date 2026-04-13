"""Experiment 16: Training convergence sanity check (PROMPT_v2 Section 3.5).

Train MLP-raw at N=300k for 10x the Phase A epoch count (600 epochs).
Compare final F1 to Phase A number. If F1 increases by >0.01, Phase A
runs were undertrained.

Pass criterion: F1 change <= 0.01.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
jax.config.update("jax_enable_x64", False)

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optax

from mega3bp.ml import N_CLASSES, BasinMLP, loss_fn, predict

N_TRAIN = 300_000
N_TEST = 100_000
SEED = 42
PHASE_A_EPOCHS = 60
EXTENDED_EPOCHS = 600
BATCH = 4096
LR = 3e-4
R_MIN_THRESHOLD = 0.01
EVAL_EVERY = 10


def load_6d():
    path = PROJECT_ROOT / "results" / "01_dataset_regen" / "6d_with_diagnostics.npz"
    d = np.load(path)
    mask = (d["r_min_ever"] >= R_MIN_THRESHOLD) & (d["label"] != -1)
    shape = d["shape_n"][mask]
    pi = d["pi_jacobi"][mask]
    labels = d["label"][mask].astype(np.int32)
    features_7d = np.concatenate([shape, pi], axis=-1).astype(np.float32)
    return features_7d, labels


def eval_f1(model, x_test, y_test):
    x_te = jnp.asarray(x_test)
    y_pred_parts = []
    for s in range(0, x_te.shape[0], 16384):
        p, _ = predict(model, x_te[s:s + 16384])
        y_pred_parts.append(np.asarray(p))
    y_pred = np.concatenate(y_pred_parts)
    y_true = y_test
    acc = float((y_pred == y_true).mean())
    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        confusion[int(t), int(p)] += 1
    f1s = []
    for i in range(N_CLASSES):
        tp = confusion[i, i]
        fn = confusion[i, :].sum() - tp
        fp = confusion[:, i].sum() - tp
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1s.append(2 * prec * rec / max(prec + rec, 1e-12))
    return acc, float(np.mean(f1s))


def main() -> int:
    out_dir = PROJECT_ROOT / "results" / "16_convergence_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"jax backend: {jax.default_backend()}")
    print(f"Training MLP-raw at N={N_TRAIN} for {EXTENDED_EPOCHS} epochs (10x Phase A)")
    print()

    features_7d, labels = load_6d()
    print(f"loaded {len(labels)} clean 6D samples")

    rng = np.random.default_rng(999)
    perm = rng.permutation(len(labels))
    test_idx = perm[:N_TEST]
    train_pool = perm[N_TEST:]

    x_test = features_7d[test_idx]
    y_test = labels[test_idx]

    rng_sub = np.random.default_rng(SEED + N_TRAIN)
    idx = rng_sub.choice(len(train_pool), size=N_TRAIN, replace=False)
    train_sel = train_pool[idx]
    x_train = features_7d[train_sel]
    y_train = labels[train_sel]

    key = jax.random.PRNGKey(SEED)
    model = BasinMLP(in_dim=7, hidden=384, n_layers=6, out_dim=N_CLASSES + 1, key=key)

    class_counts = np.array([int((y_train == i).sum()) for i in range(N_CLASSES)])
    class_w = 1.0 / np.maximum(class_counts, 1)
    class_w = class_w / class_w.mean()
    cw = jnp.asarray(class_w, dtype=jnp.float32)

    n = x_train.shape[0]
    total_steps = EXTENDED_EPOCHS * ((n + BATCH - 1) // BATCH)
    warmup = min(200, total_steps // 4)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=LR / 10, peak_value=LR, warmup_steps=warmup,
        decay_steps=total_steps, end_value=LR / 50)
    opt = optax.adamw(learning_rate=schedule, weight_decay=1e-4)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    x_tr = jnp.asarray(x_train)
    y_tr = jnp.asarray(y_train, dtype=jnp.int32)

    @eqx.filter_jit
    def step(model, opt_state, xb, yb):
        ltb = jnp.zeros(xb.shape[0], dtype=jnp.float32)
        mb = jnp.zeros(xb.shape[0], dtype=jnp.bool_)
        def f(m): return loss_fn(m, xb, yb, ltb, mb, cw)
        (_, metrics), grads = eqx.filter_value_and_grad(f, has_aux=True)(model)
        updates, opt_state_new = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        return eqx.apply_updates(model, updates), opt_state_new

    f1_vs_epoch = []
    t0 = time.perf_counter()

    for ep in range(EXTENDED_EPOCHS):
        key, ek = jax.random.split(key)
        idx_perm = jax.random.permutation(ek, n)
        for s in range(0, n, BATCH):
            sel = idx_perm[s:s + BATCH]
            model, opt_state = step(model, opt_state, x_tr[sel], y_tr[sel])

        if (ep + 1) % EVAL_EVERY == 0 or ep == 0 or (ep + 1) == PHASE_A_EPOCHS:
            acc, f1 = eval_f1(model, x_test, y_test)
            dt = time.perf_counter() - t0
            f1_vs_epoch.append({"epoch": ep + 1, "acc": acc, "f1": f1, "wall_time": round(dt, 1)})
            print(f"  epoch {ep+1:>3}/{EXTENDED_EPOCHS}  acc={acc:.4f}  F1={f1:.4f}  [{dt:.1f}s]")

    total_time = time.perf_counter() - t0

    f1_at_60 = None
    f1_at_600 = None
    for entry in f1_vs_epoch:
        if entry["epoch"] == PHASE_A_EPOCHS:
            f1_at_60 = entry["f1"]
        if entry["epoch"] == EXTENDED_EPOCHS:
            f1_at_600 = entry["f1"]

    if f1_at_60 is None:
        f1_at_60 = f1_vs_epoch[0]["f1"]
    if f1_at_600 is None:
        f1_at_600 = f1_vs_epoch[-1]["f1"]

    delta_f1 = f1_at_600 - f1_at_60
    passed = abs(delta_f1) <= 0.01

    results = {
        "f1_vs_epoch": f1_vs_epoch,
        "f1_at_60": f1_at_60,
        "f1_at_600": f1_at_600,
        "delta_f1": delta_f1,
        "pass": passed,
        "total_time": round(total_time, 1),
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== VERDICT (D5) ===")
    print(f"F1 at epoch 60:  {f1_at_60:.4f}")
    print(f"F1 at epoch 600: {f1_at_600:.4f}")
    print(f"Delta F1: {delta_f1:+.4f}")
    print(f"D5 PASS (|delta| <= 0.01): {passed}")
    print(f"Total time: {total_time:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
