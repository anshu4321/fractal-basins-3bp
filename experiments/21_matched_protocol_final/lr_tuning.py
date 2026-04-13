"""Phase E Step 3.1: Peak-LR tuning.

For each architecture, run 4 trials at N=10k (8k train, 2k val) with
peak_lr in {1e-4, 3e-4, 1e-3, 3e-3}, seed=0, 60-epoch trial.
Pick peak_lr that maximizes validation macro-F1.
Writes lr_tuning.json.
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

# --- Protocol constants (pre-registered, do not modify) ---
LR_GRID = [1e-4, 3e-4, 1e-3, 3e-3]
N_TUNE = 10_000
TUNE_SEED = 0
TUNE_EPOCHS = 60
BATCH = 1024
WARMUP_STEPS = 500
WEIGHT_DECAY = 1e-4
R_MIN_THRESHOLD = 0.01
N_TEST = 100_000


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


def train_tuning_trial(model, x_train, y_train, x_val, y_val, peak_lr, seed):
    key = jax.random.PRNGKey(seed)
    class_counts = np.array([int((y_train == i).sum()) for i in range(N_CLASSES)])
    class_w = 1.0 / np.maximum(class_counts, 1)
    class_w = class_w / class_w.mean()
    cw = jnp.asarray(class_w, dtype=jnp.float32)

    n = x_train.shape[0]
    steps_per_epoch = (n + BATCH - 1) // BATCH
    total_steps = TUNE_EPOCHS * steps_per_epoch
    warmup = min(WARMUP_STEPS, total_steps // 4)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=peak_lr, warmup_steps=warmup,
        decay_steps=total_steps, end_value=0.0)
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
        (_, metrics), grads = eqx.filter_value_and_grad(f, has_aux=True)(model)
        updates, new_opt = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        return eqx.apply_updates(model, updates), new_opt

    for ep in range(TUNE_EPOCHS):
        key, ek = jax.random.split(key)
        idx = jax.random.permutation(ek, n)
        for s in range(0, n, BATCH):
            sel = idx[s:s + BATCH]
            model, opt_state = step(model, opt_state, x_tr[sel], y_tr[sel])

    val_f1 = eval_macro_f1(model, x_val, y_val)
    return val_f1


def main():
    out_dir = PROJECT_ROOT / "experiments" / "21_matched_protocol_final"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"jax backend: {jax.default_backend()}")
    print(f"Phase E LR tuning: grid={LR_GRID}, N={N_TUNE}, epochs={TUNE_EPOCHS}")

    features_3d, features_7d, labels = load_2d()
    print(f"loaded {len(labels)} clean 2D samples")

    rng = np.random.default_rng(999)
    perm = rng.permutation(len(labels))
    test_idx = perm[:N_TEST]
    pool = perm[N_TEST:]

    rng_tune = np.random.default_rng(TUNE_SEED)
    tune_perm = rng_tune.permutation(min(N_TUNE, len(pool)))
    tune_idx = pool[tune_perm[:N_TUNE]]
    n_val = int(0.2 * N_TUNE)
    n_train = N_TUNE - n_val
    tune_train_idx = tune_idx[:n_train]
    tune_val_idx = tune_idx[n_train:]

    print(f"LR tuning: {n_train} train, {n_val} val")

    results = {}

    for arch_name, in_dim, use_7d in [("MLP", 3, False), ("S3BodyNet", 7, True)]:
        print(f"\n=== {arch_name} LR tuning ===")
        best_lr = None
        best_f1 = -1.0
        arch_results = []

        if use_7d:
            x_train = features_7d[tune_train_idx]
            x_val = features_7d[tune_val_idx]
        else:
            x_train = features_3d[tune_train_idx]
            x_val = features_3d[tune_val_idx]
        y_train = labels[tune_train_idx]
        y_val_labels = labels[tune_val_idx]

        for lr in LR_GRID:
            k = jax.random.PRNGKey(TUNE_SEED)
            if arch_name == "S3BodyNet":
                model = S3BodyNet(body_hidden=256, body_layers=4, body_out=128, key=k)
            else:
                model = BasinMLP(in_dim=in_dim, hidden=384, n_layers=6,
                                 out_dim=N_CLASSES + 1, key=k)

            t0 = time.perf_counter()
            val_f1 = train_tuning_trial(model, x_train, y_train, x_val, y_val_labels,
                                        peak_lr=lr, seed=TUNE_SEED)
            dt = time.perf_counter() - t0
            print(f"  lr={lr:.0e}  val_F1={val_f1:.4f}  [{dt:.1f}s]")
            arch_results.append({"peak_lr": lr, "val_f1": val_f1, "time_s": round(dt, 1)})

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_lr = lr

        print(f"  -> best: lr={best_lr:.0e}, val_F1={best_f1:.4f}")
        results[arch_name] = {
            "trials": arch_results,
            "chosen_peak_lr": best_lr,
            "best_val_f1": best_f1,
        }

    with open(out_dir / "lr_tuning.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_dir / 'lr_tuning.json'}")
    print(f"MLP peak_lr: {results['MLP']['chosen_peak_lr']:.0e}")
    print(f"S3BodyNet peak_lr: {results['S3BodyNet']['chosen_peak_lr']:.0e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
