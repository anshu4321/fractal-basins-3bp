"""Experiment 03: Scaling law test.

Train MLP and S3BodyNet at N in {10^4, 3*10^4, 10^5, 3*10^5, 10^6},
3 seeds each, in 2D and 6D. Measure test error = 1 - accuracy on a
fixed 10^5 held-out set. Fit log(error) vs log(N) and compare measured
slope to the GOY prediction: -alpha/d.

Predicted slopes (from Experiment 02 extended):
  2D: -alpha_2D / 2 = -0.259 / 2 = -0.130
  6D: -alpha_6D / 6 = -0.145 / 6 = -0.024

Outputs:
    results/03_scaling_law/{2d,6d}_{mlp,s3bodynet}.json
    results/03_scaling_law/summary.json
    figures/03_scaling_law.png / .pdf
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

from mega3bp.equivariant import S3BodyNet, augment_batch
from mega3bp.ml import N_CLASSES, BasinMLP, loss_fn, predict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mega3bp.style import PALETTE, apply_dirac_style

N_VALUES = [10_000, 30_000, 100_000, 300_000, 1_000_000]
SEEDS = [42, 137, 2024]
N_TEST = 100_000
EPOCHS = 40
BATCH = 4096
LR = 3e-4
R_MIN_THRESHOLD = 0.01

ALPHA_2D = 0.259
ALPHA_6D = 0.145


def load_dataset(name):
    """Load the diagnostic dataset, exclude close encounters."""
    path = PROJECT_ROOT / "results" / "01_dataset_regen" / f"{name}_with_diagnostics.npz"
    d = np.load(path)
    rmin = d["r_min_ever"]
    mask = rmin >= R_MIN_THRESHOLD
    labels = d["label"][mask]
    shape = d["shape_n"][mask]

    keep = labels != -1
    labels = labels[keep].astype(np.int32)
    shape = shape[keep]

    if name == "6d":
        pi = d["pi_jacobi"][mask][keep]
        features = np.concatenate([shape, pi], axis=-1).astype(np.float32)
    else:
        features = shape.astype(np.float32)

    return features, labels


def split_test(features, labels, n_test, seed):
    """Hold out a fixed test set, return (train_features, train_labels, test_features, test_labels)."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    perm = rng.permutation(n)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    return (features[train_idx], labels[train_idx],
            features[test_idx], labels[test_idx])


def train_and_eval(model, x_train, y_train, x_test, y_test, epochs, batch, lr, use_aug=False, seed=0):
    """Train model, return test error = 1 - accuracy."""
    key = jax.random.PRNGKey(seed)

    class_counts = np.array([int((y_train == i).sum()) for i in range(N_CLASSES)])
    class_w = 1.0 / np.maximum(class_counts, 1)
    class_w = class_w / class_w.mean()
    class_w_jax = jnp.asarray(class_w, dtype=jnp.float32)

    n_train = x_train.shape[0]
    dummy_lt = jnp.zeros(n_train, dtype=jnp.float32)
    dummy_mask = jnp.zeros(n_train, dtype=jnp.bool_)

    total_steps = max(epochs * ((n_train + batch - 1) // batch), 1)
    warmup = min(200, total_steps // 4)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=lr / 10, peak_value=lr, warmup_steps=warmup,
        decay_steps=total_steps, end_value=lr / 50)
    opt = optax.adamw(learning_rate=schedule, weight_decay=1e-4)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    x_tr = jnp.asarray(x_train)
    y_tr = jnp.asarray(y_train, dtype=jnp.int32)

    @eqx.filter_jit
    def step(model, opt_state, xb, yb):
        ltb = jnp.zeros(xb.shape[0], dtype=jnp.float32)
        mb = jnp.zeros(xb.shape[0], dtype=jnp.bool_)
        def f(m):
            return loss_fn(m, xb, yb, ltb, mb, class_w_jax)
        (_, metrics), grads = eqx.filter_value_and_grad(f, has_aux=True)(model)
        updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        return eqx.apply_updates(model, updates), opt_state, metrics

    for ep in range(epochs):
        key, ek, ak = jax.random.split(key, 3)
        idx = jax.random.permutation(ek, n_train)
        for s in range(0, n_train, batch):
            sel = idx[s:s + batch]
            xb, yb = x_tr[sel], y_tr[sel]
            if use_aug:
                ak, sub = jax.random.split(ak)
                xb, yb = augment_batch(sub, xb, yb)
            model, opt_state, _ = step(model, opt_state, xb, yb)

    x_te = jnp.asarray(x_test)
    y_pred_parts = []
    for s in range(0, x_te.shape[0], 16384):
        p, _ = predict(model, x_te[s:s+16384])
        y_pred_parts.append(np.asarray(p))
    y_pred = np.concatenate(y_pred_parts)
    acc = float((y_pred == y_test).mean())
    return 1.0 - acc


def run_scaling(dim_name, features, labels, in_dim, alpha_measured):
    """Run the full scaling law experiment for one dimensionality."""
    x_train_full, y_train_full, x_test, y_test = split_test(
        features, labels, N_TEST, seed=999)

    results = {}
    for model_name in ["mlp", "s3bodynet"]:
        runs = []
        for N in N_VALUES:
            if N > x_train_full.shape[0]:
                print(f"  {model_name} N={N}: skipping (only {x_train_full.shape[0]} available)")
                continue
            for seed in SEEDS:
                rng = np.random.default_rng(seed + N)
                idx = rng.choice(x_train_full.shape[0], size=N, replace=False)
                x_sub = x_train_full[idx]
                y_sub = y_train_full[idx]

                key = jax.random.PRNGKey(seed)
                if model_name == "mlp":
                    model = BasinMLP(in_dim=in_dim, hidden=384, n_layers=6,
                                     out_dim=N_CLASSES + 1, key=key)
                    use_aug = (dim_name == "6d")
                else:
                    model = S3BodyNet(body_hidden=256, body_layers=4,
                                      body_out=128, key=key)
                    use_aug = False

                t0 = time.perf_counter()
                if model_name == "s3bodynet" and dim_name == "at_rest":
                    print(f"  {model_name} N={N} seed={seed}: skipping (S3BodyNet needs 7D input)")
                    continue

                error = train_and_eval(model, x_sub, y_sub, x_test, y_test,
                                       EPOCHS, BATCH, LR, use_aug=use_aug, seed=seed)
                dt = time.perf_counter() - t0
                runs.append({"N": N, "seed": seed, "error": error, "time": round(dt, 1)})
                print(f"  {model_name} N={N:>7,} seed={seed}  error={error:.4f}  [{dt:.1f}s]")

        results[model_name] = runs

    d = 2 if dim_name == "at_rest" else 6
    predicted_slope = -alpha_measured / d

    for model_name in ["mlp", "s3bodynet"]:
        runs = results.get(model_name, [])
        if not runs:
            continue
        df = {}
        for r in runs:
            df.setdefault(r["N"], []).append(r["error"])
        Ns = sorted(df.keys())
        mean_errors = [np.mean(df[n]) for n in Ns]
        std_errors = [np.std(df[n]) for n in Ns]

        log_N = np.log(np.array(Ns))
        log_e = np.log(np.array(mean_errors))
        if len(Ns) >= 3:
            slope, intercept = np.polyfit(log_N, log_e, 1)
            results[f"{model_name}_slope"] = float(slope)
            results[f"{model_name}_intercept"] = float(intercept)
            print(f"\n  {model_name} measured slope: {slope:.4f} (predicted: {predicted_slope:.4f})")

    results["predicted_slope"] = float(predicted_slope)
    results["alpha"] = float(alpha_measured)
    results["d"] = d

    return results


def main() -> int:
    apply_dirac_style()
    out_results = PROJECT_ROOT / "results" / "03_scaling_law"
    out_figs = PROJECT_ROOT / "figures"
    out_results.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    print(f"jax backend: {jax.default_backend()}")
    print(f"N values: {N_VALUES}")
    print(f"Seeds: {SEEDS}")
    print(f"Predicted slopes: 2D={-ALPHA_2D/2:.4f}, 6D={-ALPHA_6D/6:.4f}")
    print()

    all_results = {}

    # ---- 2D ----
    print("=== 2D at-rest ===")
    feat_2d, lab_2d = load_dataset("at_rest")
    print(f"  loaded {len(lab_2d)} clean samples")
    res_2d = run_scaling("at_rest", feat_2d, lab_2d, in_dim=3, alpha_measured=ALPHA_2D)
    with open(out_results / "2d_mlp.json", "w") as f:
        json.dump(res_2d, f, indent=2)
    all_results["2d"] = res_2d
    print()

    # ---- 6D ----
    print("=== 6D ===")
    feat_6d, lab_6d = load_dataset("6d")
    print(f"  loaded {len(lab_6d)} clean samples")
    res_6d = run_scaling("6d", feat_6d, lab_6d, in_dim=7, alpha_measured=ALPHA_6D)
    with open(out_results / "6d_mlp_s3bodynet.json", "w") as f:
        json.dump(res_6d, f, indent=2)
    all_results["6d"] = res_6d
    print()

    # ---- Summary ----
    summary = {}
    for dim_name, res in all_results.items():
        s = {"predicted_slope": res["predicted_slope"], "alpha": res["alpha"], "d": res["d"]}
        for model_name in ["mlp", "s3bodynet"]:
            k = f"{model_name}_slope"
            if k in res:
                s[f"{model_name}_measured_slope"] = res[k]
                s[f"{model_name}_ratio"] = res[k] / res["predicted_slope"] if res["predicted_slope"] != 0 else None
        summary[dim_name] = s
    with open(out_results / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    fig.patch.set_facecolor(PALETTE["bg_deep"])

    for ax, dim_name, res in [(axes[0], "2d", all_results.get("2d", {})),
                               (axes[1], "6d", all_results.get("6d", {}))]:
        pred_slope = res.get("predicted_slope", 0)
        d = res.get("d", 2)

        for model_name, color, marker in [("mlp", PALETTE["cyan"], "o"),
                                           ("s3bodynet", PALETTE["lime"], "s")]:
            runs = res.get(model_name, [])
            if not runs:
                continue
            df = {}
            for r in runs:
                df.setdefault(r["N"], []).append(r["error"])
            Ns = sorted(df.keys())
            means = [np.mean(df[n]) for n in Ns]
            stds = [np.std(df[n]) for n in Ns]

            ax.errorbar(Ns, means, yerr=stds, fmt=f"{marker}-", color=color,
                        lw=2, ms=7, capsize=3, label=model_name)

        if pred_slope != 0:
            N_range = np.array([N_VALUES[0], N_VALUES[-1]])
            ref_error = 0.1
            ref_N = 1e5
            pred_errors = ref_error * (N_range / ref_N) ** pred_slope
            ax.plot(N_range, pred_errors, "--", color=PALETTE["coral"], lw=1.5,
                    label=rf"GOY: $N^{{{pred_slope:.3f}}}$")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("training set size $N$")
        ax.set_ylabel("test error $(1 - \\mathrm{accuracy})$")
        title = f"{'2D at-rest' if dim_name == '2d' else '6D phase space'}"
        ax.set_title(title, color=PALETTE["text"], fontsize=13)
        ax.legend(fontsize=10)

    fig.suptitle("Scaling law: test error vs training set size",
                 color=PALETTE["text"], fontsize=15, y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_figs / f"03_scaling_law.{ext}",
                    facecolor=PALETTE["bg_deep"], bbox_inches="tight")
    plt.close(fig)

    print("=== Final Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nsaved figures/03_scaling_law.png/pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
