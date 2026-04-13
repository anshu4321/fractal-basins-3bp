"""Experiment 06: Matched-budget AL vs uniform comparison.

Two strategies at four budgets B in {200k, 500k, 1M}:
  Uniform: sample B points uniformly, train.
  AL: sample 0.7*B uniformly, train initial model, detect boundary,
      sample 0.3*B near boundary, merge, retrain.

Done in 2D (at-rest) and 6D, 3 seeds each.

Outputs:
    results/06_active_learning/curve.json
    figures/06_al_vs_uniform.png / .pdf
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
from scipy.spatial import cKDTree

from mega3bp.ml import N_CLASSES, BasinMLP, loss_fn, predict, predict_proba

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mega3bp.style import PALETTE, apply_dirac_style

BUDGETS = [200_000, 500_000, 900_000]
SEEDS = [42, 137, 2024]
N_TEST = 100_000
EPOCHS = 40
BATCH = 4096
LR = 3e-4
R_MIN_THRESHOLD = 0.01
K_NEIGHBORS = 12
SIGMA_SHAPE = 0.005
SIGMA_MOM = 0.1


def load_dataset(name):
    path = PROJECT_ROOT / "results" / "01_dataset_regen" / f"{name}_with_diagnostics.npz"
    d = np.load(path)
    mask = (d["r_min_ever"] >= R_MIN_THRESHOLD) & (d["label"] != -1)
    shape = d["shape_n"][mask]
    labels = d["label"][mask].astype(np.int32)
    if name == "6d":
        pi = d["pi_jacobi"][mask]
        features = np.concatenate([shape, pi], axis=-1).astype(np.float32)
    else:
        features = shape.astype(np.float32)
    return features, labels


def perturb_near_boundary(features, labels, n_new, sigma_shape, sigma_mom, rng):
    """Detect boundary via kNN, perturb nearby."""
    tree = cKDTree(features)
    _, idx = tree.query(features, k=K_NEIGHBORS + 1)
    neighbor_labels = labels[idx[:, 1:]]
    boundary = np.any(neighbor_labels != labels[:, None], axis=1)
    boundary_pts = features[boundary]
    if len(boundary_pts) == 0:
        return features[:0]  # empty

    n_per = max(1, (n_new + len(boundary_pts) - 1) // len(boundary_pts))
    parts = []
    in_dim = features.shape[1]
    for _ in range(n_per):
        if in_dim == 3:
            noise = rng.normal(0, sigma_shape, boundary_pts.shape)
            dot = np.sum(noise * boundary_pts, axis=-1, keepdims=True)
            tangent = noise - dot * boundary_pts
            moved = boundary_pts + tangent
            moved /= np.linalg.norm(moved, axis=-1, keepdims=True)
            parts.append(moved)
        else:
            shape_part = boundary_pts[:, :3]
            mom_part = boundary_pts[:, 3:]
            noise_s = rng.normal(0, sigma_shape, shape_part.shape)
            dot = np.sum(noise_s * shape_part, axis=-1, keepdims=True)
            tangent = noise_s - dot * shape_part
            moved_s = shape_part + tangent
            moved_s /= np.linalg.norm(moved_s, axis=-1, keepdims=True)
            noise_m = rng.normal(0, sigma_mom, mom_part.shape)
            moved_m = mom_part + noise_m
            parts.append(np.concatenate([moved_s, moved_m], axis=-1))

    out = np.concatenate(parts, axis=0)
    rng.shuffle(out)
    return out[:n_new].astype(np.float32)


def train_and_eval(features, labels, x_test, y_test, in_dim, seed):
    key = jax.random.PRNGKey(seed)
    n = len(labels)
    class_counts = np.array([int((labels == i).sum()) for i in range(N_CLASSES)])
    class_w = 1.0 / np.maximum(class_counts, 1)
    class_w = class_w / class_w.mean()
    cw = jnp.asarray(class_w, dtype=jnp.float32)

    model = BasinMLP(in_dim=in_dim, hidden=384, n_layers=6, out_dim=N_CLASSES+1, key=key)
    total_steps = max(EPOCHS * ((n + BATCH - 1) // BATCH), 1)
    warmup = min(200, total_steps // 4)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=LR/10, peak_value=LR, warmup_steps=warmup,
        decay_steps=total_steps, end_value=LR/50)
    opt = optax.adamw(learning_rate=schedule, weight_decay=1e-4)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    x_tr = jnp.asarray(features)
    y_tr = jnp.asarray(labels, dtype=jnp.int32)

    @eqx.filter_jit
    def step(model, opt_state, xb, yb):
        ltb = jnp.zeros(xb.shape[0], dtype=jnp.float32)
        mb = jnp.zeros(xb.shape[0], dtype=jnp.bool_)
        def f(m): return loss_fn(m, xb, yb, ltb, mb, cw)
        (_, metrics), grads = eqx.filter_value_and_grad(f, has_aux=True)(model)
        updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        return eqx.apply_updates(model, updates), opt_state

    for ep in range(EPOCHS):
        key, ek = jax.random.split(key)
        idx = jax.random.permutation(ek, n)
        for s in range(0, n, BATCH):
            sel = idx[s:s+BATCH]
            model, opt_state = step(model, opt_state, x_tr[sel], y_tr[sel])

    x_te = jnp.asarray(x_test)
    y_pred_parts = []
    for s in range(0, x_te.shape[0], 16384):
        p, _ = predict(model, x_te[s:s+16384])
        y_pred_parts.append(np.asarray(p))
    y_pred = np.concatenate(y_pred_parts)

    acc = float((y_pred == y_test).mean())
    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for t, p in zip(y_test, y_pred):
        confusion[int(t), int(p)] += 1
    f1s = []
    for i in range(N_CLASSES):
        tp = confusion[i, i]; fn = confusion[i,:].sum()-tp; fp = confusion[:,i].sum()-tp
        prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1)
        f1s.append(2*prec*rec/max(prec+rec,1e-12))
    return acc, float(np.mean(f1s)), model


def run_comparison(dim_name, features, labels, in_dim):
    rng_split = np.random.default_rng(999)
    perm = rng_split.permutation(len(labels))
    test_idx = perm[:N_TEST]
    pool_idx = perm[N_TEST:]
    x_test = features[test_idx]
    y_test = labels[test_idx]
    x_pool = features[pool_idx]
    y_pool = labels[pool_idx]

    results = {"uniform": [], "al": []}

    for B in BUDGETS:
        if B > len(x_pool):
            print(f"  B={B}: skipping (only {len(x_pool)} in pool)")
            continue
        for seed in SEEDS:
            rng = np.random.default_rng(seed + B)

            # Uniform
            idx_u = rng.choice(len(x_pool), size=B, replace=False)
            t0 = time.perf_counter()
            acc_u, f1_u, _ = train_and_eval(x_pool[idx_u], y_pool[idx_u],
                                             x_test, y_test, in_dim, seed)
            dt_u = time.perf_counter() - t0
            results["uniform"].append({"B": B, "seed": seed, "acc": acc_u, "f1": f1_u})
            print(f"  uniform B={B:>7,} seed={seed}  acc={acc_u:.4f} F1={f1_u:.4f} [{dt_u:.1f}s]")

            # AL: 70% uniform + 30% boundary
            n_initial = int(0.7 * B)
            n_al = B - n_initial
            idx_init = rng.choice(len(x_pool), size=n_initial, replace=False)
            t0 = time.perf_counter()
            acc_init, f1_init, model_init = train_and_eval(
                x_pool[idx_init], y_pool[idx_init], x_test, y_test, in_dim, seed)

            # Boundary detection + perturbation from pool
            new_pts = perturb_near_boundary(
                x_pool[idx_init], y_pool[idx_init], n_al,
                SIGMA_SHAPE, SIGMA_MOM, np.random.default_rng(seed + 1))

            # For these new points, we need labels. Use nearest-neighbor from pool.
            if len(new_pts) > 0:
                tree = cKDTree(x_pool)
                _, nn_idx = tree.query(new_pts, k=1)
                new_labels = y_pool[nn_idx]
                x_merged = np.concatenate([x_pool[idx_init], new_pts])
                y_merged = np.concatenate([y_pool[idx_init], new_labels])
            else:
                x_merged = x_pool[idx_init]
                y_merged = y_pool[idx_init]

            acc_al, f1_al, _ = train_and_eval(x_merged, y_merged,
                                               x_test, y_test, in_dim, seed + 100)
            dt_al = time.perf_counter() - t0
            results["al"].append({"B": B, "seed": seed, "acc": acc_al, "f1": f1_al})
            print(f"  AL     B={B:>7,} seed={seed}  acc={acc_al:.4f} F1={f1_al:.4f} [{dt_al:.1f}s]")

    return results


def main() -> int:
    apply_dirac_style()
    out_results = PROJECT_ROOT / "results" / "06_active_learning"
    out_figs = PROJECT_ROOT / "figures"
    out_results.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    print(f"jax backend: {jax.default_backend()}")
    print(f"budgets: {BUDGETS}, seeds: {SEEDS}")
    print()

    all_results = {}

    for dim_name in ["at_rest", "6d"]:
        print(f"=== {dim_name} ===")
        features, labels = load_dataset(dim_name)
        in_dim = features.shape[1]
        print(f"  loaded {len(labels)} samples, in_dim={in_dim}")
        res = run_comparison(dim_name, features, labels, in_dim)
        all_results[dim_name] = res
        print()

    with open(out_results / "curve.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    fig.patch.set_facecolor(PALETTE["bg_deep"])

    for ax, dim_name in [(axes[0], "at_rest"), (axes[1], "6d")]:
        res = all_results.get(dim_name, {})
        for strategy, color, marker in [("uniform", PALETTE["cyan"], "o"),
                                         ("al", PALETTE["lime"], "s")]:
            runs = res.get(strategy, [])
            if not runs:
                continue
            df = {}
            for r in runs:
                df.setdefault(r["B"], []).append(r["f1"])
            Bs = sorted(df.keys())
            means = [np.mean(df[b]) for b in Bs]
            stds = [np.std(df[b]) for b in Bs]
            ax.errorbar(Bs, means, yerr=stds, fmt=f"{marker}-", color=color,
                        lw=2, ms=7, capsize=3, label=strategy)

        ax.set_xscale("log")
        ax.set_xlabel("total budget $B$")
        ax.set_ylabel("test macro F1")
        title = "2D at-rest" if dim_name == "at_rest" else "6D phase space"
        ax.set_title(title, color=PALETTE["text"])
        ax.legend(fontsize=11)

    fig.suptitle("Active learning vs uniform at matched budget",
                 color=PALETTE["text"], fontsize=15, y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_figs / f"06_al_vs_uniform.{ext}",
                    facecolor=PALETTE["bg_deep"], bbox_inches="tight")
    plt.close(fig)
    print("saved figures/06_al_vs_uniform.png/pdf")

    # Summary
    print("\n=== Summary ===")
    for dim_name, res in all_results.items():
        print(f"\n  {dim_name}:")
        for strategy in ["uniform", "al"]:
            runs = res.get(strategy, [])
            df = {}
            for r in runs:
                df.setdefault(r["B"], []).append(r["f1"])
            for B in sorted(df.keys()):
                vals = df[B]
                print(f"    {strategy:>8s} B={B:>7,}  F1={np.mean(vals):.4f} ± {np.std(vals):.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
