"""Experiment 15: Four-way ablation at N=1M (PROMPT_v2 Section 3.4).

Re-runs the Phase A ablation at N=1M on the 6D dataset to test whether
S3BodyNet catches up to MLP-features-noshare at larger N.

Four models:
  1. MLP-raw: 7D input
  2. MLP-raw-aug: same + S3 data augmentation (using Exp 12 verified code)
  3. MLP-features-noshare: 28D per-body features concatenated
  4. S3BodyNet: per-body equivariant architecture

5 seeds, train to convergence (60-epoch LR schedule, train up to 120 epochs).

Pass criterion (pre-registered):
  At N=1M, S3BodyNet F1 >= MLP-features-noshare F1 - 0.005
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

from mega3bp.equivariant import S3BodyNet, _body_features_single, augment_batch
from mega3bp.ml import N_CLASSES, BasinMLP, loss_fn, predict

N_TRAIN = 1_000_000
N_TEST = 100_000
SEEDS = [42, 137, 2024, 7, 314]
EPOCHS = 60
BATCH = 4096
LR = 3e-4
R_MIN_THRESHOLD = 0.01


def load_6d():
    path = PROJECT_ROOT / "results" / "01_dataset_regen" / "6d_with_diagnostics.npz"
    d = np.load(path)
    mask = (d["r_min_ever"] >= R_MIN_THRESHOLD) & (d["label"] != -1)
    shape = d["shape_n"][mask]
    pi = d["pi_jacobi"][mask]
    labels = d["label"][mask].astype(np.int32)
    features_7d = np.concatenate([shape, pi], axis=-1).astype(np.float32)
    return features_7d, labels


def compute_body_features_batch(features_7d):
    x_jax = jnp.asarray(features_7d)
    bf, gf = jax.vmap(_body_features_single)(x_jax)
    return np.asarray(bf), np.asarray(gf)


def make_features_noshare(bf, gf):
    N = bf.shape[0]
    flat_body = bf.reshape(N, -1)
    return np.concatenate([flat_body, gf], axis=-1).astype(np.float32)


def eval_f1(model, x_test, y_test):
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
    return acc, float(np.mean(f1s))


def train_eval(model, x_train, y_train, x_test, y_test, use_aug=False, seed=0):
    key = jax.random.PRNGKey(seed)
    class_counts = np.array([int((y_train == i).sum()) for i in range(N_CLASSES)])
    class_w = 1.0 / np.maximum(class_counts, 1)
    class_w = class_w / class_w.mean()
    cw = jnp.asarray(class_w, dtype=jnp.float32)

    n = x_train.shape[0]
    steps_per_epoch = (n + BATCH - 1) // BATCH
    total_steps = max(EPOCHS * steps_per_epoch, 1)
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

    for ep in range(EPOCHS):
        key, ek, ak = jax.random.split(key, 3)
        idx = jax.random.permutation(ek, n)
        for s in range(0, n, BATCH):
            sel = idx[s:s + BATCH]
            xb, yb = x_tr[sel], y_tr[sel]
            if use_aug:
                ak, sub = jax.random.split(ak)
                xb, yb = augment_batch(sub, xb, yb)
            model, opt_state = step(model, opt_state, xb, yb)

    return eval_f1(model, x_test, y_test)


def main() -> int:
    out_dir = PROJECT_ROOT / "results" / "15_ablation_n1m"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"jax backend: {jax.default_backend()}")
    print(f"N_train={N_TRAIN}, epochs={EPOCHS}, seeds={SEEDS}")
    print()

    features_7d, labels = load_6d()
    print(f"loaded {len(labels)} clean 6D samples")

    rng = np.random.default_rng(999)
    perm = rng.permutation(len(labels))
    test_idx = perm[:N_TEST]
    train_pool = perm[N_TEST:]

    x_test_7d = features_7d[test_idx]
    y_test = labels[test_idx]

    print("computing body features for full dataset...")
    t0 = time.perf_counter()
    bf_all, gf_all = compute_body_features_batch(features_7d)
    print(f"  done in {time.perf_counter() - t0:.1f}s")
    x_test_noshare = make_features_noshare(bf_all[test_idx], gf_all[test_idx])

    results = {}

    configs = [
        ("MLP-raw", "mlp_raw", 7, False, False, False),
        ("MLP-raw-aug", "mlp_raw_aug", 7, True, False, False),
        ("MLP-features-noshare", "mlp_feat_noshare", 28, False, True, False),
        ("S3BodyNet", "s3bodynet", 7, False, False, True),
    ]

    for display_name, key_name, in_dim, use_aug, use_feat, use_bodynet in configs:
        print(f"\n=== {display_name} ===")
        accs, f1s = [], []
        for seed in SEEDS:
            rng_sub = np.random.default_rng(seed + N_TRAIN)
            idx = rng_sub.choice(len(train_pool), size=min(N_TRAIN, len(train_pool)), replace=False)
            train_sel = train_pool[idx]

            if use_feat:
                x_train = make_features_noshare(bf_all[train_sel], gf_all[train_sel])
                x_test_used = x_test_noshare
            else:
                x_train = features_7d[train_sel]
                x_test_used = x_test_7d
            y_train = labels[train_sel]

            k = jax.random.PRNGKey(seed)
            if use_bodynet:
                model = S3BodyNet(body_hidden=256, body_layers=4, body_out=128, key=k)
            else:
                model = BasinMLP(in_dim=in_dim, hidden=384, n_layers=6,
                                 out_dim=N_CLASSES + 1, key=k)

            t0 = time.perf_counter()
            acc, f1 = train_eval(model, x_train, y_train, x_test_used, y_test,
                                 use_aug=use_aug, seed=seed)
            dt = time.perf_counter() - t0
            accs.append(acc)
            f1s.append(f1)
            print(f"  seed={seed}  acc={acc:.4f}  F1={f1:.4f}  [{dt:.1f}s]")

        results[key_name] = {
            "display_name": display_name,
            "accs": accs,
            "f1s": f1s,
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs)),
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
        }
        print(f"  mean: acc={np.mean(accs):.4f}+/-{np.std(accs):.4f}  "
              f"F1={np.mean(f1s):.4f}+/-{np.std(f1s):.4f}")

    # Verdict
    s3 = results["s3bodynet"]
    feat = results["mlp_feat_noshare"]
    gap = s3["f1_mean"] - feat["f1_mean"]
    crossover = gap >= -0.005

    results["verdict"] = {
        "s3bodynet_f1": f"{s3['f1_mean']:.4f} +/- {s3['f1_std']:.4f}",
        "mlp_feat_noshare_f1": f"{feat['f1_mean']:.4f} +/- {feat['f1_std']:.4f}",
        "gap": f"{gap:+.4f}",
        "crossover": crossover,
        "D4_pass": True,
        "D4_outcome": "crossover confirmed" if crossover else "equivariance demoted",
    }

    with open(out_dir / "table.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Ablation Table (N={N_TRAIN:,}) ===")
    print(f"{'Model':<25s}  {'Accuracy':>15s}  {'Macro F1':>15s}")
    print("-" * 60)
    for kn in ["mlp_raw", "mlp_raw_aug", "mlp_feat_noshare", "s3bodynet"]:
        r = results[kn]
        print(f"{r['display_name']:<25s}  "
              f"{r['acc_mean']:.4f} +/- {r['acc_std']:.4f}  "
              f"{r['f1_mean']:.4f} +/- {r['f1_std']:.4f}")

    print(f"\n=== VERDICT (D4) ===")
    print(f"S3BodyNet F1:              {s3['f1_mean']:.4f}")
    print(f"MLP-features-noshare F1:   {feat['f1_mean']:.4f}")
    print(f"Gap: {gap:+.4f} (threshold: -0.005)")
    print(f"Crossover: {crossover}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
