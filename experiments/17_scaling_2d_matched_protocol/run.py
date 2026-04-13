"""Experiment 17: 2D scaling law with matched training protocol (PROMPT_v3 2.1 + 2.3).

Both MLP and S3BodyNet trained under identical early-stopping criterion:
  Stop when val macro-F1 fails to improve by > 0.002 over a 20-epoch window.
  Cap at 300 epochs.

N in {10k, 30k, 100k, 300k, 600k}. 5 seeds per (architecture, N) cell.
Same 100k test set (seed 999). 50k validation set.
Reports slope +/- 2sigma by both jackknife and bootstrap (10^4 resamples).
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

N_VALUES = [10_000, 30_000, 100_000, 300_000, 600_000]
SEEDS = [42, 137, 2024, 7, 314]
N_TEST = 100_000
N_VAL = 50_000
MAX_EPOCHS = 300
BATCH = 4096
LR = 3e-4
R_MIN_THRESHOLD = 0.01
ES_DELTA = 0.002
ES_WINDOW = 20

ALPHA_2D = 0.259
PREDICTED_SLOPE = -ALPHA_2D / 2


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


def eval_f1(model, x, y):
    x_te = jnp.asarray(x)
    parts = []
    for s in range(0, x_te.shape[0], 16384):
        p, _ = predict(model, x_te[s:s + 16384])
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
    return acc, float(np.mean(f1s))


LR_DECAY_EPOCHS = 60


def train_with_early_stopping(model, x_train, y_train, x_val, y_val, seed=0):
    """Shared training protocol: 60-epoch cosine decay then flat at end_value.

    Both architectures use the same LR schedule (60-epoch cosine decay from
    LR to LR/50, then constant at LR/50). Early stopping: val F1 fails to
    improve by >ES_DELTA over a ES_WINDOW-epoch window, checked only after
    LR_DECAY_EPOCHS. Cap at MAX_EPOCHS.

    This gives fast-converging models (MLP) the benefit of cosine annealing
    and slow-converging models (S3BodyNet) extra training at low LR.
    """
    key = jax.random.PRNGKey(seed)
    class_counts = np.array([int((y_train == i).sum()) for i in range(N_CLASSES)])
    class_w = 1.0 / np.maximum(class_counts, 1)
    class_w = class_w / class_w.mean()
    cw = jnp.asarray(class_w, dtype=jnp.float32)

    n = x_train.shape[0]
    steps_per_epoch = (n + BATCH - 1) // BATCH
    decay_steps = LR_DECAY_EPOCHS * steps_per_epoch
    warmup = min(200, decay_steps // 4)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=LR / 10, peak_value=LR, warmup_steps=warmup,
        decay_steps=decay_steps, end_value=LR / 50)
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

    best_val_f1 = -1.0
    best_epoch = 0
    final_epoch = MAX_EPOCHS

    for ep in range(MAX_EPOCHS):
        key, ek = jax.random.split(key)
        idx = jax.random.permutation(ek, n)
        for s in range(0, n, BATCH):
            sel = idx[s:s + BATCH]
            model, opt_state = step(model, opt_state, x_tr[sel], y_tr[sel])

        _, vf1 = eval_f1(model, x_val, y_val)
        if vf1 > best_val_f1 + ES_DELTA:
            best_val_f1 = vf1
            best_epoch = ep

        if ep >= LR_DECAY_EPOCHS and ep - best_epoch >= ES_WINDOW:
            final_epoch = ep + 1
            break

    return model, final_epoch


def jackknife_slope(N_vals, error_matrix):
    n_seeds = error_matrix.shape[1]
    mean_errors = np.mean(error_matrix, axis=1)
    log_N = np.log(np.array(N_vals, dtype=np.float64))
    log_e = np.log(mean_errors.astype(np.float64))
    slope_full, intercept = np.polyfit(log_N, log_e, 1)
    ss_res = np.sum((log_e - (slope_full * log_N + intercept)) ** 2)
    ss_tot = np.sum((log_e - np.mean(log_e)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    jack_slopes = []
    for drop in range(n_seeds):
        kept = np.delete(error_matrix, drop, axis=1)
        me = np.mean(kept, axis=1)
        le = np.log(me.astype(np.float64))
        s, _ = np.polyfit(log_N, le, 1)
        jack_slopes.append(s)
    jack_var = (n_seeds - 1) / n_seeds * np.sum((np.array(jack_slopes) - np.mean(jack_slopes)) ** 2)
    return float(slope_full), float(np.sqrt(jack_var)), float(r2)


def bootstrap_slope(N_vals, error_matrix, n_boot=10_000):
    n_seeds = error_matrix.shape[1]
    log_N = np.log(np.array(N_vals, dtype=np.float64))
    rng = np.random.default_rng(12345)
    boot_slopes = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_seeds, size=n_seeds)
        resampled = error_matrix[:, idx]
        me = np.mean(resampled, axis=1)
        le = np.log(me.astype(np.float64))
        s, _ = np.polyfit(log_N, le, 1)
        boot_slopes.append(s)
    return float(np.std(boot_slopes))


def main() -> int:
    out_dir = PROJECT_ROOT / "results" / "17_scaling_2d_matched_protocol"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"jax backend: {jax.default_backend()}")
    print(f"N values: {N_VALUES}, seeds: {SEEDS}")
    print(f"Early stopping: delta={ES_DELTA}, window={ES_WINDOW}, cap={MAX_EPOCHS}")
    print(f"GOY prediction: slope = {PREDICTED_SLOPE:.4f}")
    print()

    features_3d, features_7d, labels = load_2d()
    print(f"loaded {len(labels)} clean 2D samples")

    rng = np.random.default_rng(999)
    perm = rng.permutation(len(labels))
    test_idx = perm[:N_TEST]
    val_idx = perm[N_TEST:N_TEST + N_VAL]
    train_pool = perm[N_TEST + N_VAL:]

    x_test_3d = features_3d[test_idx]
    x_test_7d = features_7d[test_idx]
    x_val_3d = features_3d[val_idx]
    x_val_7d = features_7d[val_idx]
    y_test = labels[test_idx]
    y_val = labels[val_idx]

    print(f"train pool: {len(train_pool)}, test: {N_TEST}, val: {N_VAL}")

    all_results = {}

    for arch_name, in_dim, use_7d in [("MLP", 3, False), ("S3BodyNet", 7, True)]:
        print(f"\n=== {arch_name} ===")
        error_matrix = np.zeros((len(N_VALUES), len(SEEDS)))
        f1_matrix = np.zeros((len(N_VALUES), len(SEEDS)))
        epoch_matrix = np.zeros((len(N_VALUES), len(SEEDS)), dtype=int)
        runs = []

        for ni, N in enumerate(N_VALUES):
            actual_n = min(N, len(train_pool))
            if actual_n < N:
                print(f"  N={N:,} capped to {actual_n:,} (train pool size)")

            for si, seed in enumerate(SEEDS):
                rng_sub = np.random.default_rng(seed + N)
                idx = rng_sub.choice(len(train_pool), size=actual_n, replace=False)
                train_sel = train_pool[idx]

                if use_7d:
                    x_train = features_7d[train_sel]
                    x_v, x_t = x_val_7d, x_test_7d
                else:
                    x_train = features_3d[train_sel]
                    x_v, x_t = x_val_3d, x_test_3d
                y_train = labels[train_sel]

                k = jax.random.PRNGKey(seed)
                if arch_name == "S3BodyNet":
                    model = S3BodyNet(body_hidden=256, body_layers=4, body_out=128, key=k)
                else:
                    model = BasinMLP(in_dim=in_dim, hidden=384, n_layers=6,
                                     out_dim=N_CLASSES + 1, key=k)

                t0 = time.perf_counter()
                model, n_ep = train_with_early_stopping(
                    model, x_train, y_train, x_v, y_val, seed=seed)
                acc, f1 = eval_f1(model, x_t, y_test)
                dt = time.perf_counter() - t0

                error = 1.0 - acc
                error_matrix[ni, si] = error
                f1_matrix[ni, si] = f1
                epoch_matrix[ni, si] = n_ep
                runs.append({"N": actual_n, "seed": seed, "error": error, "f1": f1,
                             "epochs": n_ep, "time": round(dt, 1)})
                print(f"  N={actual_n:>7,}  seed={seed:>4}  err={error:.4f}  F1={f1:.4f}  ep={n_ep:>3}  [{dt:.1f}s]")

        slope, jack_sigma, r2 = jackknife_slope(
            [min(N, len(train_pool)) for N in N_VALUES], error_matrix)
        boot_sigma = bootstrap_slope(
            [min(N, len(train_pool)) for N in N_VALUES], error_matrix)
        sigma = max(jack_sigma, boot_sigma)

        print(f"\n  {arch_name} slope: {slope:.4f}")
        print(f"    jackknife sigma: {jack_sigma:.4f}, bootstrap sigma: {boot_sigma:.4f}")
        print(f"    conservative 2sigma: {2*sigma:.4f}")
        print(f"    R2: {r2:.6f}")
        print(f"    GOY predicted: {PREDICTED_SLOPE:.4f}, ratio: {slope/PREDICTED_SLOPE:.3f}")
        print(f"    epoch distribution: {epoch_matrix.tolist()}")

        all_results[arch_name] = {
            "runs": runs,
            "error_matrix": error_matrix.tolist(),
            "f1_matrix": f1_matrix.tolist(),
            "epoch_matrix": epoch_matrix.tolist(),
            "slope": slope,
            "jack_sigma": jack_sigma,
            "boot_sigma": boot_sigma,
            "sigma_conservative": sigma,
            "r2": r2,
            "predicted_slope": PREDICTED_SLOPE,
            "ratio": slope / PREDICTED_SLOPE,
        }

    # Phase C original slopes for comparison
    phase_c_mlp_slope = -0.1306
    phase_c_mlp_2sigma = 0.0092
    phase_c_s3_slope = -0.1410
    phase_c_s3_2sigma = 0.0181

    # Check if slopes moved beyond original 2sigma
    mlp = all_results["MLP"]
    s3 = all_results["S3BodyNet"]
    mlp_moved = abs(mlp["slope"] - phase_c_mlp_slope) > phase_c_mlp_2sigma
    s3_moved = abs(s3["slope"] - phase_c_s3_slope) > phase_c_s3_2sigma

    sigma_mlp = mlp["sigma_conservative"]
    sigma_s3 = s3["sigma_conservative"]
    mlp_within = abs(mlp["slope"] - PREDICTED_SLOPE) < 2 * sigma_mlp
    s3_within = abs(s3["slope"] - PREDICTED_SLOPE) < 2 * sigma_s3
    mlp_ci_ok = 2 * sigma_mlp <= 0.04
    s3_ci_ok = 2 * sigma_s3 <= 0.04

    verdict = {
        "mlp_slope": f"{mlp['slope']:.4f} +/- {sigma_mlp:.4f} (2sigma={2*sigma_mlp:.4f})",
        "s3_slope": f"{s3['slope']:.4f} +/- {sigma_s3:.4f} (2sigma={2*sigma_s3:.4f})",
        "mlp_r2": f"{mlp['r2']:.6f}",
        "s3_r2": f"{s3['r2']:.6f}",
        "predicted": f"{PREDICTED_SLOPE:.4f}",
        "mlp_within_2sigma": mlp_within,
        "s3_within_2sigma": s3_within,
        "mlp_ci_width_ok": mlp_ci_ok,
        "s3_ci_width_ok": s3_ci_ok,
        "mlp_slope_moved_beyond_phase_c_ci": mlp_moved,
        "s3_slope_moved_beyond_phase_c_ci": s3_moved,
        "D2_pass": mlp_within and s3_within and mlp_ci_ok and s3_ci_ok,
    }
    all_results["verdict"] = verdict
    all_results["phase_c_comparison"] = {
        "mlp_old_slope": phase_c_mlp_slope, "mlp_old_2sigma": phase_c_mlp_2sigma,
        "s3_old_slope": phase_c_s3_slope, "s3_old_2sigma": phase_c_s3_2sigma,
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n=== VERDICT ===")
    for k, v in verdict.items():
        marker = ""
        if k.endswith("moved_beyond_phase_c_ci") and v:
            marker = " *** FLAGGED ***"
        print(f"  {k}: {v}{marker}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
