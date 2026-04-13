"""Phase E Diagnostic: re-train 4 cells with intermediate logging.

Cells: {MLP, S3BodyNet} x {N=10k, N=300k}, seed=0.
Logs training loss and validation macro-F1 every 100 gradient updates.
Produces training curve figures and convergence classifications.
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

from mega3bp.equivariant import S3BodyNet
from mega3bp.ml import N_CLASSES, BasinMLP, loss_fn, predict

BATCH = 1024
WARMUP_STEPS = 500
WEIGHT_DECAY = 1e-4
TOTAL_UPDATES = (200 * 300_000) // BATCH
PEAK_LR = 0.003
R_MIN_THRESHOLD = 0.01
N_TEST = 100_000
LOG_EVERY = 100
SEED = 0

CELLS = [
    ("MLP", 10_000),
    ("MLP", 300_000),
    ("S3BodyNet", 10_000),
    ("S3BodyNet", 300_000),
]


def load_2d():
    path = PROJECT_ROOT / "results" / "01_dataset_regen" / "at_rest_with_diagnostics.npz"
    d = np.load(path)
    mask = (d["r_min_ever"] >= R_MIN_THRESHOLD) & (d["label"] != -1)
    shape = d["shape_n"][mask].astype(np.float32)
    labels = d["label"][mask].astype(np.int32)
    features_7d = np.concatenate([
        shape, np.zeros((len(shape), 4), dtype=np.float32)
    ], axis=-1)
    return shape, features_7d, labels


def eval_macro_f1(model, x, y):
    x_jax = jnp.asarray(x)
    parts = []
    for s in range(0, x_jax.shape[0], 8192):
        p, _ = predict(model, x_jax[s:s + 8192])
        parts.append(np.asarray(p))
    y_pred = np.concatenate(parts)
    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for t, p in zip(y, y_pred):
        confusion[int(t), int(p)] += 1
    f1s = []
    for i in range(N_CLASSES):
        tp = confusion[i, i]
        fn = confusion[i, :].sum() - tp
        fp = confusion[:, i].sum() - tp
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1s.append(2 * prec * rec / max(prec + rec, 1e-12))
    return float(np.mean(f1s))


def train_with_logging(model, x_train, y_train, x_val, y_val):
    key = jax.random.PRNGKey(SEED)
    class_counts = np.array([int((y_train == i).sum()) for i in range(N_CLASSES)])
    class_w = 1.0 / np.maximum(class_counts, 1)
    class_w = class_w / class_w.mean()
    cw = jnp.asarray(class_w, dtype=jnp.float32)

    n = x_train.shape[0]

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=PEAK_LR, warmup_steps=WARMUP_STEPS,
        decay_steps=TOTAL_UPDATES, end_value=0.0)
    opt = optax.adamw(learning_rate=schedule, b1=0.9, b2=0.999, eps=1e-8,
                      weight_decay=WEIGHT_DECAY)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    x_tr = jnp.asarray(x_train)
    y_tr = jnp.asarray(y_train, dtype=jnp.int32)

    @eqx.filter_jit
    def step(model, opt_state, xb, yb):
        ltb = jnp.zeros(xb.shape[0], dtype=jnp.float32)
        mb = jnp.zeros(xb.shape[0], dtype=jnp.bool_)
        def f(m): return loss_fn(m, xb, yb, ltb, mb, cw)
        (loss, metrics), grads = eqx.filter_value_and_grad(f, has_aux=True)(model)
        updates, new_opt = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        return eqx.apply_updates(model, updates), new_opt, loss

    log = {"step": [], "train_loss": [], "val_f1": []}
    global_step = 0
    running_loss = 0.0
    loss_count = 0

    while global_step < TOTAL_UPDATES:
        key, ek = jax.random.split(key)
        idx = jax.random.permutation(ek, n)
        for s in range(0, n, BATCH):
            if global_step >= TOTAL_UPDATES:
                break
            sel = idx[s:s + BATCH]
            model, opt_state, loss = step(model, opt_state, x_tr[sel], y_tr[sel])
            running_loss += float(loss)
            loss_count += 1
            global_step += 1

            if global_step % LOG_EVERY == 0:
                avg_loss = running_loss / loss_count
                val_f1 = eval_macro_f1(model, x_val, y_val)
                log["step"].append(global_step)
                log["train_loss"].append(avg_loss)
                log["val_f1"].append(val_f1)
                running_loss = 0.0
                loss_count = 0
                if global_step % (LOG_EVERY * 10) == 0:
                    print(f"    step {global_step:>6}/{TOTAL_UPDATES}  "
                          f"loss={avg_loss:.4f}  val_F1={val_f1:.4f}")

    return model, log


def main():
    out_dir = PROJECT_ROOT / "experiments" / "21_matched_protocol_final"
    out_dir.mkdir(parents=True, exist_ok=True)

    features_3d, features_7d, labels = load_2d()
    print(f"loaded {len(labels)} clean 2D samples")

    rng = np.random.default_rng(999)
    perm = rng.permutation(len(labels))
    test_idx = perm[:N_TEST]
    train_pool = perm[N_TEST:]

    all_logs = {}

    for arch_name, N in CELLS:
        use_7d = arch_name == "S3BodyNet"
        in_dim = 7 if use_7d else 3
        actual_n = min(N, len(train_pool))

        rng_sub = np.random.default_rng(SEED + N)
        idx = rng_sub.choice(len(train_pool), size=actual_n, replace=False)
        train_sel = train_pool[idx]

        n_val = max(1000, actual_n // 10)
        n_train = actual_n - n_val
        train_idx_local = train_sel[:n_train]
        val_idx_local = train_sel[n_train:]

        if use_7d:
            x_train = features_7d[train_idx_local]
            x_val = features_7d[val_idx_local]
        else:
            x_train = features_3d[train_idx_local]
            x_val = features_3d[val_idx_local]
        y_train = labels[train_idx_local]
        y_val = labels[val_idx_local]

        k = jax.random.PRNGKey(SEED)
        if arch_name == "S3BodyNet":
            model = S3BodyNet(body_hidden=256, body_layers=4, body_out=128, key=k)
        else:
            model = BasinMLP(in_dim=in_dim, hidden=384, n_layers=6,
                             out_dim=N_CLASSES + 1, key=k)

        cell_key = f"{arch_name}_N{N}"
        print(f"\n=== {cell_key}: {n_train} train, {n_val} val ===")
        t0 = time.perf_counter()
        model, log = train_with_logging(model, x_train, y_train, x_val, y_val)
        dt = time.perf_counter() - t0
        print(f"  done in {dt:.1f}s, {len(log['step'])} log points")

        all_logs[cell_key] = log

    with open(out_dir / "diagnostic_logs.json", "w") as f:
        json.dump(all_logs, f)
    print(f"\nWrote diagnostic_logs.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
