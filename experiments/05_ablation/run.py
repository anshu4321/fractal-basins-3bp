"""Experiment 05: Four-way ablation on 6D dataset.

Four models at N=300k, 60 epochs, 5 seeds each:
  1. MLP-raw: 7D input (n1,n2,n3,pi1x,pi1y,pi2x,pi2y), 384x6
  2. MLP-raw-aug: same + S3 data augmentation
  3. MLP-features-noshare: 3 bodies x 12D features concatenated (40D), no sharing
  4. S3BodyNet: 12D per-body with shared MLP

Reports mean +/- std of macro F1 and accuracy over 5 seeds.
Key comparison: (3) vs (4) isolates weight sharing from feature engineering.

Outputs:
    results/05_ablation/table.json
    figures/05_ablation.png / .pdf
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mega3bp.style import PALETTE, apply_dirac_style

N_TRAIN = 300_000
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
    """Compute per-body features for all samples (CPU, batched)."""
    x_jax = jnp.asarray(features_7d)
    bf, gf = jax.vmap(_body_features_single)(x_jax)
    bf_np = np.asarray(bf)  # (N, 3, 8)
    gf_np = np.asarray(gf)  # (N, 4)
    return bf_np, gf_np


def make_features_noshare(bf, gf):
    """Concatenate all 3 bodies' features + global = 3*8 + 4 = 28D."""
    N = bf.shape[0]
    flat_body = bf.reshape(N, -1)  # (N, 24)
    return np.concatenate([flat_body, gf], axis=-1).astype(np.float32)  # (N, 28)


def train_eval(model, x_train, y_train, x_test, y_test, epochs, batch, lr,
               use_aug=False, seed=0):
    key = jax.random.PRNGKey(seed)
    class_counts = np.array([int((y_train == i).sum()) for i in range(N_CLASSES)])
    class_w = 1.0 / np.maximum(class_counts, 1)
    class_w = class_w / class_w.mean()
    cw = jnp.asarray(class_w, dtype=jnp.float32)

    n = x_train.shape[0]
    total_steps = max(epochs * ((n + batch - 1) // batch), 1)
    warmup = min(200, total_steps // 4)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=lr/10, peak_value=lr, warmup_steps=warmup,
        decay_steps=total_steps, end_value=lr/50)
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
        updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        return eqx.apply_updates(model, updates), opt_state

    for ep in range(epochs):
        key, ek, ak = jax.random.split(key, 3)
        idx = jax.random.permutation(ek, n)
        for s in range(0, n, batch):
            sel = idx[s:s+batch]
            xb, yb = x_tr[sel], y_tr[sel]
            if use_aug:
                ak, sub = jax.random.split(ak)
                xb, yb = augment_batch(sub, xb, yb)
            model, opt_state = step(model, opt_state, xb, yb)

    x_te = jnp.asarray(x_test)
    y_pred_parts = []
    for s in range(0, x_te.shape[0], 16384):
        p, _ = predict(model, x_te[s:s+16384])
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
    macro_f1 = float(np.mean(f1s))
    return acc, macro_f1


def main() -> int:
    apply_dirac_style()
    out_results = PROJECT_ROOT / "results" / "05_ablation"
    out_figs = PROJECT_ROOT / "figures"
    out_results.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

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

    # Precompute body features for feature-based models
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
            idx = rng_sub.choice(len(train_pool), size=N_TRAIN, replace=False)
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
                                 out_dim=N_CLASSES+1, key=k)

            t0 = time.perf_counter()
            acc, f1 = train_eval(model, x_train, y_train, x_test_used, y_test,
                                 EPOCHS, BATCH, LR, use_aug=use_aug, seed=seed)
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
        print(f"  mean: acc={np.mean(accs):.4f}±{np.std(accs):.4f}  "
              f"F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}")

    with open(out_results / "table.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved table.json")

    # ---- Table summary ----
    print("\n=== Ablation Table ===")
    print(f"{'Model':<25s}  {'Accuracy':>15s}  {'Macro F1':>15s}")
    print("-" * 60)
    for key_name in ["mlp_raw", "mlp_raw_aug", "mlp_feat_noshare", "s3bodynet"]:
        r = results[key_name]
        print(f"{r['display_name']:<25s}  "
              f"{r['acc_mean']:.4f} ± {r['acc_std']:.4f}  "
              f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor(PALETTE["bg_deep"])
    names = []
    f1_means = []
    f1_stds = []
    colors = [PALETTE["text_mute"], PALETTE["cyan"], PALETTE["lavender"], PALETTE["lime"]]
    for i, key_name in enumerate(["mlp_raw", "mlp_raw_aug", "mlp_feat_noshare", "s3bodynet"]):
        r = results[key_name]
        names.append(r["display_name"])
        f1_means.append(r["f1_mean"])
        f1_stds.append(r["f1_std"])

    x_pos = np.arange(len(names))
    bars = ax.bar(x_pos, f1_means, yerr=f1_stds, capsize=5, color=colors,
                  alpha=0.85, edgecolor=PALETTE["bg_deep"], linewidth=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Macro F1")
    ax.set_title("Four-way ablation (6D, N=300k, 5 seeds)", color=PALETTE["text"])
    for bar, val, std in zip(bars, f1_means, f1_stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.005,
                f"{val:.3f}", ha="center", va="bottom", color=PALETTE["text"], fontsize=10)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_figs / f"05_ablation.{ext}", facecolor=PALETTE["bg_deep"])
    plt.close(fig)
    print("saved figures/05_ablation.png/pdf")

    return 0


if __name__ == "__main__":
    sys.exit(main())
