"""Experiment 13: S3BodyNet scaling law on 2D at-rest dataset (PROMPT_v2 Section 3.2).

Matches Phase A training setup exactly (fixed epochs, same split, same LR
schedule) but with 5 seeds and S3BodyNet added.

S3BodyNet adaptation: pad 3D shape-sphere input with zero Jacobi momenta
to create the 7D input S3BodyNet expects.

N in {10^4, 3*10^4, 10^5, 3*10^5}. N=10^6 skipped for 2D (only ~853k
training samples available, same as Phase A).
5 seeds per (architecture, N) cell.
Same 10^5 held-out test set as Phase A (seed 999).
Report slope +/- 2sigma via jackknife over seeds.

Pass criterion (pre-registered):
  - S3BodyNet measured 2D slope within 2sigma of GOY prediction -0.130, AND
  - 2sigma CI width <= 0.04.
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

N_VALUES = [10_000, 30_000, 100_000, 300_000]
SEEDS = [42, 137, 2024, 7, 314]
N_TEST = 100_000
EPOCHS = 60
BATCH = 4096
LR = 3e-4
R_MIN_THRESHOLD = 0.01

ALPHA_2D = 0.259
PREDICTED_SLOPE = -ALPHA_2D / 2  # -0.1295


def load_2d():
    path = PROJECT_ROOT / "results" / "01_dataset_regen" / "at_rest_with_diagnostics.npz"
    d = np.load(path)
    mask = (d["r_min_ever"] >= R_MIN_THRESHOLD) & (d["label"] != -1)
    shape = d["shape_n"][mask]
    labels = d["label"][mask].astype(np.int32)
    features_3d = shape.astype(np.float32)
    features_7d = np.concatenate([
        features_3d,
        np.zeros((len(features_3d), 4), dtype=np.float32)
    ], axis=-1)
    return features_3d, features_7d, labels


def train_and_eval(model, x_train, y_train, x_test, y_test, seed=0, epochs=None):
    """Train for fixed epochs with cosine LR schedule. Returns test error."""
    if epochs is None:
        epochs = EPOCHS
    key = jax.random.PRNGKey(seed)
    class_counts = np.array([int((y_train == i).sum()) for i in range(N_CLASSES)])
    class_w = 1.0 / np.maximum(class_counts, 1)
    class_w = class_w / class_w.mean()
    cw = jnp.asarray(class_w, dtype=jnp.float32)

    n = x_train.shape[0]
    steps_per_epoch = (n + BATCH - 1) // BATCH
    total_steps = max(epochs * steps_per_epoch, 1)
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
        updates, new_opt = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        return eqx.apply_updates(model, updates), new_opt

    for ep in range(epochs):
        key, ek = jax.random.split(key)
        idx = jax.random.permutation(ek, n)
        for s in range(0, n, BATCH):
            sel = idx[s:s + BATCH]
            model, opt_state = step(model, opt_state, x_tr[sel], y_tr[sel])

    x_te = jnp.asarray(x_test)
    y_pred_parts = []
    for s in range(0, x_te.shape[0], 16384):
        p, _ = predict(model, x_te[s:s + 16384])
        y_pred_parts.append(np.asarray(p))
    y_pred = np.concatenate(y_pred_parts)
    acc = float((y_pred == y_test).mean())

    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for t, p in zip(y_test, y_pred):
        confusion[int(t), int(p)] += 1
    f1s = []
    for i in range(N_CLASSES):
        tp = confusion[i, i]
        fn = confusion[i, :].sum() - tp
        fp = confusion[:, i].sum() - tp
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1s.append(2 * prec * rec / max(prec + rec, 1e-12))

    return 1.0 - acc, float(np.mean(f1s))


def jackknife_slope(N_vals, error_matrix):
    """Jackknife over seeds. Returns (slope, sigma, ci_width_2sigma)."""
    n_seeds = error_matrix.shape[1]
    mean_errors = np.mean(error_matrix, axis=1)
    log_N = np.log(np.array(N_vals, dtype=np.float64))
    log_e = np.log(mean_errors.astype(np.float64))
    slope_full, _ = np.polyfit(log_N, log_e, 1)

    jack_slopes = []
    for drop in range(n_seeds):
        kept = np.delete(error_matrix, drop, axis=1)
        me = np.mean(kept, axis=1)
        le = np.log(me.astype(np.float64))
        s, _ = np.polyfit(log_N, le, 1)
        jack_slopes.append(s)
    jack_slopes = np.array(jack_slopes)
    jack_var = (n_seeds - 1) / n_seeds * np.sum((jack_slopes - np.mean(jack_slopes)) ** 2)
    sigma = np.sqrt(jack_var)
    return float(slope_full), float(sigma), float(2 * sigma)


def main() -> int:
    out_dir = PROJECT_ROOT / "results" / "13_scaling_2d_s3bodynet"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"jax backend: {jax.default_backend()}")
    print(f"N values: {N_VALUES}, seeds: {SEEDS}, epochs: {EPOCHS}")
    print(f"GOY prediction: slope = {PREDICTED_SLOPE:.4f}")
    print()

    features_3d, features_7d, labels = load_2d()
    print(f"loaded {len(labels)} clean 2D samples")

    # Same split as Phase A: seed 999, first 100k = test, rest = train
    rng = np.random.default_rng(999)
    perm = rng.permutation(len(labels))
    test_idx = perm[:N_TEST]
    train_pool = perm[N_TEST:]

    x_test_3d = features_3d[test_idx]
    x_test_7d = features_7d[test_idx]
    y_test = labels[test_idx]

    all_results = {}

    for arch_name, in_dim, use_7d in [("MLP", 3, False), ("S3BodyNet", 7, True)]:
        print(f"\n=== {arch_name} ===")
        error_matrix = np.zeros((len(N_VALUES), len(SEEDS)))
        f1_matrix = np.zeros((len(N_VALUES), len(SEEDS)))
        runs = []

        for ni, N in enumerate(N_VALUES):
            for si, seed in enumerate(SEEDS):
                rng_sub = np.random.default_rng(seed + N)
                idx = rng_sub.choice(len(train_pool), size=min(N, len(train_pool)), replace=False)
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

                ep_count = 200 if arch_name == "S3BodyNet" else EPOCHS
                t0 = time.perf_counter()
                error, f1 = train_and_eval(model, x_train, y_train, x_t, y_test, seed=seed, epochs=ep_count)
                dt = time.perf_counter() - t0

                error_matrix[ni, si] = error
                f1_matrix[ni, si] = f1
                runs.append({"N": N, "seed": seed, "error": error, "f1": f1, "time": round(dt, 1)})
                print(f"  N={N:>7,}  seed={seed:>4}  err={error:.4f}  F1={f1:.4f}  [{dt:.1f}s]")

        slope, sigma, ci_2sigma = jackknife_slope(N_VALUES, error_matrix)
        print(f"\n  {arch_name} slope: {slope:.4f} +/- {sigma:.4f} (2sigma={ci_2sigma:.4f})")
        print(f"  GOY predicted: {PREDICTED_SLOPE:.4f}")
        print(f"  Ratio: {slope / PREDICTED_SLOPE:.3f}")

        all_results[arch_name] = {
            "runs": runs,
            "error_matrix": error_matrix.tolist(),
            "f1_matrix": f1_matrix.tolist(),
            "slope": slope,
            "slope_sigma": sigma,
            "slope_2sigma_ci": ci_2sigma,
            "predicted_slope": PREDICTED_SLOPE,
            "ratio": slope / PREDICTED_SLOPE,
            "epochs": EPOCHS,
        }

    # Verdict
    s3 = all_results["S3BodyNet"]
    mlp = all_results["MLP"]
    slope = s3["slope"]
    sigma = s3["slope_sigma"]
    ci = s3["slope_2sigma_ci"]
    within_2sigma = abs(slope - PREDICTED_SLOPE) < ci
    ci_narrow = ci <= 0.04

    verdict = {
        "s3bodynet_slope": f"{slope:.4f} +/- {sigma:.4f}",
        "s3bodynet_2sigma_ci_width": f"{ci:.4f}",
        "predicted_slope": f"{PREDICTED_SLOPE:.4f}",
        "mlp_slope": f"{mlp['slope']:.4f} +/- {mlp['slope_sigma']:.4f}",
        "within_2sigma": within_2sigma,
        "ci_narrow_enough": ci_narrow,
        "D2_pass": within_2sigma and ci_narrow,
    }
    all_results["verdict"] = verdict

    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n=== VERDICT (D2) ===")
    print(f"S3BodyNet slope: {slope:.4f} +/- {sigma:.4f} (2sigma CI width: {ci:.4f})")
    print(f"GOY predicted:   {PREDICTED_SLOPE:.4f}")
    print(f"Within 2sigma of GOY: {within_2sigma}")
    print(f"CI width <= 0.04: {ci_narrow}")
    print(f"D2 PASS: {verdict['D2_pass']}")
    print(f"\nMLP slope: {mlp['slope']:.4f} +/- {mlp['slope_sigma']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
