"""Phase E Step 3.2: Main sweep.

2 architectures x 4 N values x 5 seeds = 40 runs.
Fixed compute budget: total_updates = 200 * 300000 / 1024 = 58594.
No early stopping. Cosine decay from peak_lr to 0.

Reads peak_lr from lr_tuning.json.
Writes results.json with one row per (architecture, N, seed).
Supports resuming: skips cells already present in results.json.
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
N_VALUES = [10_000, 30_000, 100_000, 300_000]
SEEDS = [0, 1, 2, 3, 4]
N_TEST = 100_000
BATCH = 1024
WARMUP_STEPS = 500
WEIGHT_DECAY = 1e-4
EPOCHS_BASE = 200
N_BASE = 300_000
TOTAL_UPDATES = (EPOCHS_BASE * N_BASE) // BATCH  # 58594
R_MIN_THRESHOLD = 0.01

ALPHA_2D = 0.259
PREDICTED_SLOPE = -ALPHA_2D / 2


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


def eval_metrics(model, x, y):
    x_jax = jnp.asarray(x)
    parts = []
    for s in range(0, x_jax.shape[0], 8192):
        p, _ = predict(model, x_jax[s:s + 8192])
        parts.append(np.asarray(p))
    y_pred = np.concatenate(parts)
    acc = float((y_pred == y).mean())
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
    macro_f1 = float(np.mean(f1s))
    test_error = 1.0 - macro_f1
    return acc, macro_f1, test_error, confusion.tolist(), f1s


def train_fixed_budget(model, x_train, y_train, peak_lr, seed):
    key = jax.random.PRNGKey(seed)
    class_counts = np.array([int((y_train == i).sum()) for i in range(N_CLASSES)])
    class_w = 1.0 / np.maximum(class_counts, 1)
    class_w = class_w / class_w.mean()
    cw = jnp.asarray(class_w, dtype=jnp.float32)

    n = x_train.shape[0]
    steps_per_epoch = (n + BATCH - 1) // BATCH

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=peak_lr, warmup_steps=WARMUP_STEPS,
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
        (_, metrics), grads = eqx.filter_value_and_grad(f, has_aux=True)(model)
        updates, new_opt = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        return eqx.apply_updates(model, updates), new_opt

    global_step = 0
    epoch = 0
    while global_step < TOTAL_UPDATES:
        key, ek = jax.random.split(key)
        idx = jax.random.permutation(ek, n)
        for s in range(0, n, BATCH):
            if global_step >= TOTAL_UPDATES:
                break
            sel = idx[s:s + BATCH]
            model, opt_state = step(model, opt_state, x_tr[sel], y_tr[sel])
            global_step += 1
        epoch += 1

    return model, epoch, global_step


def main():
    exp_dir = PROJECT_ROOT / "experiments" / "21_matched_protocol_final"
    exp_dir.mkdir(parents=True, exist_ok=True)

    lr_path = exp_dir / "lr_tuning.json"
    if not lr_path.exists():
        print("ERROR: lr_tuning.json not found. Run lr_tuning.py first.")
        return 1
    with open(lr_path) as f:
        lr_config = json.load(f)

    results_path = exp_dir / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
    else:
        all_results = {"runs": [], "protocol": {
            "total_updates": TOTAL_UPDATES,
            "batch_size": BATCH,
            "warmup_steps": WARMUP_STEPS,
            "weight_decay": WEIGHT_DECAY,
            "n_values": N_VALUES,
            "seeds": SEEDS,
            "metric": "1 - macro_F1",
            "predicted_slope": PREDICTED_SLOPE,
            "mlp_peak_lr": lr_config["MLP"]["chosen_peak_lr"],
            "s3bodynet_peak_lr": lr_config["S3BodyNet"]["chosen_peak_lr"],
        }}

    done_keys = {(r["arch"], r["N"], r["seed"]) for r in all_results["runs"]}

    print(f"jax backend: {jax.default_backend()}")
    print(f"Phase E main sweep: {len(N_VALUES)} N values x {len(SEEDS)} seeds x 2 archs = 40 runs")
    print(f"Total gradient updates per run: {TOTAL_UPDATES}")
    print(f"Already completed: {len(done_keys)} / 40")
    print(f"MLP peak_lr: {lr_config['MLP']['chosen_peak_lr']:.0e}")
    print(f"S3BodyNet peak_lr: {lr_config['S3BodyNet']['chosen_peak_lr']:.0e}")

    features_3d, features_7d, labels = load_2d()
    print(f"loaded {len(labels)} clean 2D samples")

    rng = np.random.default_rng(999)
    perm = rng.permutation(len(labels))
    test_idx = perm[:N_TEST]
    train_pool = perm[N_TEST:]

    x_test_3d = features_3d[test_idx]
    x_test_7d = features_7d[test_idx]
    y_test = labels[test_idx]

    print(f"train pool: {len(train_pool)}, test: {N_TEST}")

    run_count = len(done_keys)
    for arch_name, in_dim, use_7d in [("MLP", 3, False), ("S3BodyNet", 7, True)]:
        peak_lr = lr_config[arch_name]["chosen_peak_lr"]
        print(f"\n=== {arch_name} (peak_lr={peak_lr:.0e}) ===")

        for N in N_VALUES:
            actual_n = min(N, len(train_pool))
            for seed in SEEDS:
                if (arch_name, actual_n, seed) in done_keys or (arch_name, N, seed) in done_keys:
                    print(f"  N={N:>7,}  seed={seed}  [skip, already done]")
                    continue

                rng_sub = np.random.default_rng(seed + N)
                idx = rng_sub.choice(len(train_pool), size=actual_n, replace=False)
                train_sel = train_pool[idx]

                if use_7d:
                    x_train = features_7d[train_sel]
                    x_t = x_test_7d
                else:
                    x_train = features_3d[train_sel]
                    x_t = x_test_3d
                y_train = labels[train_sel]

                k = jax.random.PRNGKey(seed)
                if arch_name == "S3BodyNet":
                    model = S3BodyNet(body_hidden=256, body_layers=4, body_out=128, key=k)
                else:
                    model = BasinMLP(in_dim=in_dim, hidden=384, n_layers=6,
                                     out_dim=N_CLASSES + 1, key=k)

                t0 = time.perf_counter()
                model, n_epochs, n_steps = train_fixed_budget(
                    model, x_train, y_train, peak_lr=peak_lr, seed=seed)
                acc, macro_f1, test_error, confusion, per_class_f1 = eval_metrics(model, x_t, y_test)
                dt = time.perf_counter() - t0

                run_count += 1
                row = {
                    "arch": arch_name, "N": actual_n, "seed": seed,
                    "accuracy": acc, "macro_f1": macro_f1, "test_error": test_error,
                    "confusion": confusion, "per_class_f1": per_class_f1,
                    "epochs": n_epochs, "gradient_steps": n_steps,
                    "peak_lr": peak_lr, "time_s": round(dt, 1),
                }
                all_results["runs"].append(row)

                with open(results_path, "w") as f:
                    json.dump(all_results, f, indent=2)

                print(f"  N={actual_n:>7,}  seed={seed}  err={test_error:.4f}  "
                      f"F1={macro_f1:.4f}  acc={acc:.4f}  ep={n_epochs}  "
                      f"[{dt:.1f}s]  ({run_count}/40)")

    print(f"\nAll 40 runs complete. Results in {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
