"""Train the Phase 2 baseline MLP classifier on the 1M-IC basin dataset.

Reads data/phase2_basin_*.parquet, excludes ambiguous rows, stratifies an
80/10/10 split, trains the BasinMLP with class-weighted cross-entropy plus
masked log-escape-time regression, evaluates on the held-out test set, and
saves:

    data/baseline_model.eqx        serialized equinox weights
    data/baseline_history.npz      per-epoch metrics
    figures/baseline_curves.png    loss / accuracy / R^2 over training
    figures/baseline_confusion.png per-class confusion matrix
    figures/baseline_mollweide.png predicted vs true basin side-by-side
    figures/baseline_errors.png    where the model is wrong, colored by true class
    figures/baseline_summary.png   dashboard with all metrics

Target gates from the scoping contract:
    pointwise accuracy >= 0.90 (easy — trivial baseline already gets 0.935;
                                real metric is per-class F1)
    log escape time R^2 >= 0.80 on escape subset
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax

jax.config.update("jax_enable_x64", False)  # fp32 is fine for MLP training

import equinox as eqx
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd

from mega3bp.ml import (
    N_CLASSES,
    BasinMLP,
    loss_fn,
    predict,
    stratified_split,
)
from mega3bp.style import BASIN_COLORS, PALETTE, apply_dirac_style

CLASS_NAMES = ["bound", "body1_escape", "body2_escape", "body3_escape"]


def log_escape_target(escape_time: np.ndarray, finite_mask: np.ndarray) -> np.ndarray:
    """log escape_time where finite, 0 elsewhere (contribution masked out)."""
    out = np.zeros_like(escape_time, dtype=np.float32)
    out[finite_mask] = np.log(escape_time[finite_mask])
    return out


def main() -> int:
    apply_dirac_style()
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--features", choices=["raw", "fourier"], default="raw",
                        help="raw (n1,n2,n3) or fourier harmonics of them")
    parser.add_argument("--n_freqs", type=int, default=8,
                        help="number of frequency bands when features=fourier")
    parser.add_argument("--weighting", choices=["none", "sqrt", "inverse"], default="inverse",
                        help="class-imbalance weighting")
    args = parser.parse_args()

    print(f"jax backend : {jax.default_backend()}")
    print(f"devices     : {jax.devices()}")
    print()

    df = pd.read_parquet(args.parquet)
    print(f"loaded {len(df):,} rows from {args.parquet}")

    # Drop ambiguous, map label to [0..3]
    mask_keep = df.label.values != -1
    df = df.loc[mask_keep].reset_index(drop=True)

    # Remap labels: 0 (bound), 1/2/3 (escapes) are already consecutive
    labels = df.label.to_numpy().astype(np.int32)
    n = len(df)
    print(f"kept {n:,} non-ambiguous rows")

    raw = df[["shape_n1", "shape_n2", "shape_n3"]].to_numpy(dtype=np.float32)
    if args.features == "fourier":
        # [sin(2^k pi n), cos(2^k pi n)] for k=0..n_freqs-1, stacked along last axis.
        freqs = (2.0 ** np.arange(args.n_freqs, dtype=np.float32)) * np.pi
        args_grid = raw[:, :, None] * freqs  # (N, 3, n_freqs)
        sins = np.sin(args_grid).reshape(raw.shape[0], -1)
        coss = np.cos(args_grid).reshape(raw.shape[0], -1)
        inputs = np.concatenate([sins, coss], axis=-1).astype(np.float32)
        in_dim = inputs.shape[1]
        print(f"fourier features    : n_freqs={args.n_freqs}, in_dim={in_dim}")
    else:
        inputs = raw
        in_dim = 3
        print(f"raw features        : in_dim=3")
    etimes = df.escape_time.to_numpy(dtype=np.float32)
    finite = np.isfinite(etimes)
    log_t = log_escape_target(etimes, finite)
    print(f"finite escape times : {int(finite.sum()):,} ({100*finite.mean():.2f}%)")
    print()

    # Stratified split
    key = jax.random.PRNGKey(args.seed)
    key, split_key = jax.random.split(key)
    train_idx, val_idx, test_idx = stratified_split(jnp.asarray(labels), split_key)
    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)
    test_idx = np.asarray(test_idx)
    print(f"train / val / test : {train_idx.size:,} / {val_idx.size:,} / {test_idx.size:,}")

    x_train = jnp.asarray(inputs[train_idx])
    y_train = jnp.asarray(labels[train_idx], dtype=jnp.int32)
    lt_train = jnp.asarray(log_t[train_idx])
    m_train = jnp.asarray(finite[train_idx])

    x_val = jnp.asarray(inputs[val_idx])
    y_val = jnp.asarray(labels[val_idx], dtype=jnp.int32)
    lt_val = jnp.asarray(log_t[val_idx])
    m_val = jnp.asarray(finite[val_idx])

    x_test = jnp.asarray(inputs[test_idx])
    y_test = jnp.asarray(labels[test_idx], dtype=jnp.int32)
    lt_test = jnp.asarray(log_t[test_idx])
    m_test = jnp.asarray(finite[test_idx])

    # Class weights
    class_counts = np.array([int((labels[train_idx] == i).sum()) for i in range(N_CLASSES)])
    if args.weighting == "inverse":
        class_w = 1.0 / np.maximum(class_counts, 1)
    elif args.weighting == "sqrt":
        class_w = 1.0 / np.sqrt(np.maximum(class_counts, 1))
    else:  # none
        class_w = np.ones(N_CLASSES)
    class_w = class_w / class_w.mean()
    print(f"class counts (train): {class_counts.tolist()}")
    print(f"class weights       : {np.round(class_w, 3).tolist()}  ({args.weighting})")
    print()

    class_w_jax = jnp.asarray(class_w, dtype=jnp.float32)

    key, model_key = jax.random.split(key)
    model = BasinMLP(
        in_dim=in_dim,
        hidden=args.hidden,
        n_layers=args.layers,
        out_dim=N_CLASSES + 1,
        key=model_key,
    )

    opt = optax.adamw(learning_rate=args.lr, weight_decay=1e-4)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, opt_state, x, y, lt, mask):
        def loss_closure(m):
            return loss_fn(m, x, y, lt, mask, class_w_jax)

        (loss_val, metrics), grads = eqx.filter_value_and_grad(
            loss_closure, has_aux=True
        )(model)
        updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_val, metrics

    @eqx.filter_jit
    def eval_batch(model, x, y, lt, mask):
        _, metrics = loss_fn(model, x, y, lt, mask, class_w_jax)
        return metrics

    # Shuffle-based mini-batching
    def iterate_batches(x_, y_, lt_, m_, key_):
        n_ = x_.shape[0]
        idx = jax.random.permutation(key_, n_)
        for start in range(0, n_, args.batch):
            sel = idx[start : start + args.batch]
            yield x_[sel], y_[sel], lt_[sel], m_[sel]

    print("=== training ===")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "val_cls": [], "val_mse": []}

    # Track best model by validation accuracy
    best_val_acc = -1.0
    best_epoch = 0
    best_model = model

    t_train = time.perf_counter()
    for epoch in range(args.epochs):
        key, epoch_key = jax.random.split(key)
        ep_losses = []
        ep_accs = []
        t0 = time.perf_counter()
        for xb, yb, ltb, mb in iterate_batches(x_train, y_train, lt_train, m_train, epoch_key):
            model, opt_state, loss_val, metrics = train_step(model, opt_state, xb, yb, ltb, mb)
            ep_losses.append(float(metrics["total_loss"]))
            ep_accs.append(float(metrics["accuracy"]))

        # Validation
        val_metrics = eval_batch(model, x_val, y_val, lt_val, m_val)
        dt = time.perf_counter() - t0
        tr_loss = float(np.mean(ep_losses))
        tr_acc = float(np.mean(ep_accs))
        val_loss = float(val_metrics["total_loss"])
        val_acc = float(val_metrics["accuracy"])
        val_cls = float(val_metrics["cls_loss"])
        val_mse = float(val_metrics["mse_loss"])
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_cls"].append(val_cls)
        history["val_mse"].append(val_mse)

        star = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            # Snapshot current weights via equinox tree copy
            best_model = eqx.tree_at(lambda m: m.layers, model, model.layers)
            star = " *"

        print(
            f"  epoch {epoch + 1:3d}/{args.epochs}  "
            f"loss {tr_loss:.4f}  acc {tr_acc:.4f}  |  "
            f"val_loss {val_loss:.4f}  val_acc {val_acc:.4f}  "
            f"val_mse {val_mse:.4f}  [{dt:.2f}s]{star}"
        )

    total_train = time.perf_counter() - t_train
    print(f"trained for {total_train:.1f}s")
    print(f"best val acc {best_val_acc:.4f} at epoch {best_epoch} — using best checkpoint")
    model = best_model  # restore best for evaluation
    print()

    # ---- Test set evaluation --------------------------------------------
    print("=== test set evaluation ===")

    # Large batch split to avoid blowing memory
    def full_predict(x_all, batch=16384):
        ys = []
        lts = []
        for start in range(0, x_all.shape[0], batch):
            p, lt = predict(model, x_all[start : start + batch])
            ys.append(np.asarray(p))
            lts.append(np.asarray(lt))
        return np.concatenate(ys), np.concatenate(lts)

    y_pred, log_t_pred = full_predict(x_test)
    y_true = np.asarray(y_test)

    overall_acc = float((y_pred == y_true).mean())
    trivial_acc = float((y_true == 0).mean())  # predict bound
    print(f"  overall accuracy : {overall_acc:.4f}")
    print(f"  trivial (bound)  : {trivial_acc:.4f}")
    print()

    # Confusion matrix
    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        confusion[int(t), int(p)] += 1

    # Per-class precision/recall/F1
    print("  per-class metrics:")
    per_class = []
    for i, name in enumerate(CLASS_NAMES):
        tp = confusion[i, i]
        fn = confusion[i, :].sum() - tp
        fp = confusion[:, i].sum() - tp
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        per_class.append({"class": name, "precision": prec, "recall": rec, "f1": f1,
                          "support": int(tp + fn)})
        print(f"    {name:14s}  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  N={tp+fn}")
    macro_f1 = float(np.mean([c["f1"] for c in per_class]))
    print(f"  macro F1         : {macro_f1:.4f}")

    # Log escape time R^2 on escape subset
    escape_mask_test = np.asarray(m_test)
    lt_true_esc = np.asarray(lt_test)[escape_mask_test]
    lt_pred_esc = log_t_pred[escape_mask_test]
    ss_res = float(np.sum((lt_true_esc - lt_pred_esc) ** 2))
    ss_tot = float(np.sum((lt_true_esc - lt_true_esc.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    print(f"  log-escape R^2   : {r2:.4f}  (on {escape_mask_test.sum():,} escape samples)")

    # Contract gates
    print()
    print("  contract gates (see .gpd/state.json):")
    gate_acc = overall_acc >= 0.90
    gate_r2 = r2 >= 0.80
    print(f"    accuracy >= 0.90 : {'PASS' if gate_acc else 'FAIL'}  ({overall_acc:.4f})")
    print(f"    R^2      >= 0.80 : {'PASS' if gate_r2 else 'FAIL'}  ({r2:.4f})")
    print(f"    (note: 0.935 trivial baseline makes accuracy gate near-trivial;")
    print(f"     macro F1 {macro_f1:.3f} is the meaningful signal)")
    print()

    # ---- Save artifacts --------------------------------------------------
    out_data = PROJECT_ROOT / "data"
    out_fig = PROJECT_ROOT / "figures"
    out_data.mkdir(exist_ok=True)
    out_fig.mkdir(exist_ok=True)

    model_path = out_data / "baseline_model.eqx"
    eqx.tree_serialise_leaves(str(model_path), model)
    print(f"saved {model_path}")

    hist_path = out_data / "baseline_history.npz"
    np.savez(
        hist_path,
        train_loss=np.array(history["train_loss"]),
        train_acc=np.array(history["train_acc"]),
        val_loss=np.array(history["val_loss"]),
        val_acc=np.array(history["val_acc"]),
        confusion=confusion,
        per_class_f1=np.array([c["f1"] for c in per_class]),
        overall_acc=overall_acc,
        trivial_acc=trivial_acc,
        macro_f1=macro_f1,
        log_escape_r2=r2,
        shape_test=np.asarray(raw[test_idx]),
        y_test=y_true,
        y_pred=y_pred,
        log_t_true=np.asarray(lt_test),
        log_t_pred=log_t_pred,
        escape_mask_test=escape_mask_test,
    )
    print(f"saved {hist_path}")
    print()

    # ---- Plots -----------------------------------------------------------
    print("=== plotting ===")

    # 1. Training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    fig.patch.set_facecolor(PALETTE["bg_deep"])
    epochs_axis = np.arange(1, args.epochs + 1)
    axes[0].plot(epochs_axis, history["train_loss"], color=PALETTE["cyan"], lw=1.8, label="train")
    axes[0].plot(epochs_axis, history["val_loss"], color=PALETTE["lime"], lw=1.8, label="val")
    axes[0].set_title("loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()

    axes[1].plot(epochs_axis, history["train_acc"], color=PALETTE["cyan"], lw=1.8, label="train")
    axes[1].plot(epochs_axis, history["val_acc"], color=PALETTE["lime"], lw=1.8, label="val")
    axes[1].axhline(trivial_acc, color=PALETTE["coral"], ls="--", lw=1.1,
                    label=f"trivial bound = {trivial_acc:.3f}")
    axes[1].axhline(0.90, color=PALETTE["amber"], ls=":", lw=1.0,
                    label=r"contract gate = 0.90")
    axes[1].set_title("accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0.85, 1.0)
    axes[1].legend()

    axes[2].plot(epochs_axis, history["val_mse"], color=PALETTE["lavender"], lw=1.8)
    axes[2].set_title(r"val masked log-escape MSE")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("MSE")

    for ax in axes:
        for leg in [ax.get_legend()]:
            if leg is not None:
                leg.get_frame().set_facecolor(PALETTE["bg_elevate"])
                leg.get_frame().set_edgecolor(PALETTE["spine"])
                for t in leg.get_texts():
                    t.set_color(PALETTE["text"])

    fig.suptitle(
        f"Baseline MLP training  ·  hidden={args.hidden}×{args.layers}  ·  "
        f"epochs={args.epochs}  ·  lr={args.lr}",
        color=PALETTE["text"], fontsize=14, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_fig / "baseline_curves.png", facecolor=PALETTE["bg_deep"])
    print(f"  saved {out_fig / 'baseline_curves.png'}")
    plt.close(fig)

    # 2. Confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    fig.patch.set_facecolor(PALETTE["bg_deep"])
    ax.set_facecolor(PALETTE["bg_panel"])
    norm = confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(norm, cmap="magma", vmin=0, vmax=1.0, aspect="auto")
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            val = confusion[i, j]
            col = PALETTE["text"] if norm[i, j] < 0.5 else PALETTE["bg_deep"]
            ax.text(
                j, i, f"{val:,}\n{norm[i,j]*100:.1f}%",
                ha="center", va="center", color=col, fontsize=9,
            )
    ax.set_xticks(range(N_CLASSES), labels=CLASS_NAMES, rotation=25)
    ax.set_yticks(range(N_CLASSES), labels=CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(
        f"Confusion matrix (row-normalized)  ·  macro F1 = {macro_f1:.3f}",
        color=PALETTE["text"],
    )
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.outline.set_edgecolor(PALETTE["spine"])
    cb.ax.yaxis.set_tick_params(colors=PALETTE["text_mute"])
    fig.tight_layout()
    fig.savefig(out_fig / "baseline_confusion.png", facecolor=PALETTE["bg_deep"])
    print(f"  saved {out_fig / 'baseline_confusion.png'}")
    plt.close(fig)

    # 3. Predicted vs true basin Mollweide side-by-side (use raw shape coords)
    n_test = raw[test_idx]
    lon = np.arctan2(n_test[:, 1], n_test[:, 0])
    lat = np.arcsin(np.clip(n_test[:, 2], -1, 1))

    def mollweide_basin(ax, lab, title):
        ax.set_facecolor(PALETTE["bg_deep"])
        ax.grid(color=PALETTE["grid"], alpha=0.4, lw=0.3)
        ax.tick_params(colors=PALETTE["text_mute"], labelsize=6)
        names = CLASS_NAMES
        size = max(0.4, 9000 / len(lab))
        bound = lab == 0
        ax.scatter(lon[bound], lat[bound], s=size, c=BASIN_COLORS["bound"],
                   alpha=0.25, edgecolors="none", rasterized=True)
        for i in range(1, 4):
            m = lab == i
            if m.any():
                ax.scatter(lon[m], lat[m], s=size * 3.0,
                           c=BASIN_COLORS[names[i]], alpha=0.9,
                           edgecolors="none", rasterized=True)
        ax.set_title(title, color=PALETTE["text"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), dpi=150,
                             subplot_kw={"projection": "mollweide"})
    fig.patch.set_facecolor(PALETTE["bg_deep"])
    mollweide_basin(axes[0], y_true, "ground truth")
    mollweide_basin(axes[1], y_pred, "model prediction")
    fig.suptitle(
        f"Baseline MLP · predicted vs true basin on {len(y_true):,} held-out ICs",
        color=PALETTE["text"], fontsize=14, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_fig / "baseline_mollweide.png", facecolor=PALETTE["bg_deep"])
    print(f"  saved {out_fig / 'baseline_mollweide.png'}")
    plt.close(fig)

    # 4. Error map: only misclassified points, colored by TRUE class
    wrong = y_pred != y_true
    fig = plt.figure(figsize=(12, 6.5), dpi=150)
    ax = fig.add_subplot(111, projection="mollweide")
    ax.set_facecolor(PALETTE["bg_deep"])
    ax.grid(color=PALETTE["grid"], alpha=0.3, lw=0.3)
    ax.tick_params(colors=PALETTE["text_mute"], labelsize=7)
    # Faint true boundaries as reference
    ax.scatter(lon[~wrong], lat[~wrong], s=1.2, c=PALETTE["grid"], alpha=0.25,
               edgecolors="none", rasterized=True)
    for i in range(N_CLASSES):
        m = wrong & (y_true == i)
        if m.any():
            ax.scatter(lon[m], lat[m], s=10, c=BASIN_COLORS[CLASS_NAMES[i]],
                       alpha=0.92, edgecolors="none", rasterized=True,
                       label=f"true={CLASS_NAMES[i]} ({int(m.sum())})")
    ax.set_title(
        f"Where the model is wrong  ·  {int(wrong.sum()):,} / {len(y_true):,} "
        f"= {100*wrong.mean():.2f}% error",
        color=PALETTE["text"], pad=16,
    )
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0.0, -0.25), ncol=2, fontsize=9)
    leg.get_frame().set_facecolor(PALETTE["bg_elevate"])
    leg.get_frame().set_edgecolor(PALETTE["spine"])
    for t in leg.get_texts():
        t.set_color(PALETTE["text"])
    fig.tight_layout()
    fig.savefig(out_fig / "baseline_errors.png", facecolor=PALETTE["bg_deep"])
    print(f"  saved {out_fig / 'baseline_errors.png'}")
    plt.close(fig)

    # 5. Summary dashboard
    fig = plt.figure(figsize=(16, 10), dpi=150, constrained_layout=True)
    fig.patch.set_facecolor(PALETTE["bg_deep"])
    gs = fig.add_gridspec(2, 3)

    ax1 = fig.add_subplot(gs[0, :2], projection="mollweide")
    mollweide_basin(ax1, y_pred, "predicted basin (held-out)")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(PALETTE["bg_panel"])
    f1_vals = [c["f1"] for c in per_class]
    bars = ax2.barh(np.arange(N_CLASSES), f1_vals,
                    color=[BASIN_COLORS[n] for n in CLASS_NAMES], alpha=0.92,
                    edgecolor=PALETTE["bg_deep"], linewidth=0.7)
    ax2.set_yticks(range(N_CLASSES), labels=CLASS_NAMES, color=PALETTE["text"])
    ax2.set_xlim(0, 1)
    ax2.invert_yaxis()
    ax2.set_xlabel("F1")
    ax2.set_title("per-class F1", color=PALETTE["text"])
    for bar, val in zip(bars, f1_vals):
        ax2.text(val, bar.get_y() + bar.get_height() / 2, f" {val:.3f}",
                 va="center", color=PALETTE["text"], fontsize=9)

    ax3 = fig.add_subplot(gs[1, 0])
    epochs_axis = np.arange(1, args.epochs + 1)
    ax3.plot(epochs_axis, history["train_acc"], color=PALETTE["cyan"], lw=1.8, label="train")
    ax3.plot(epochs_axis, history["val_acc"], color=PALETTE["lime"], lw=1.8, label="val")
    ax3.axhline(trivial_acc, color=PALETTE["coral"], ls="--", lw=1.0, label="trivial bound")
    ax3.axhline(0.90, color=PALETTE["amber"], ls=":", lw=0.9, label="contract gate")
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("accuracy")
    ax3.set_ylim(0.85, 1.0)
    ax3.set_title("accuracy over training")
    leg = ax3.legend(fontsize=8)
    leg.get_frame().set_facecolor(PALETTE["bg_elevate"])
    leg.get_frame().set_edgecolor(PALETTE["spine"])
    for t in leg.get_texts():
        t.set_color(PALETTE["text"])

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(PALETTE["bg_panel"])
    im2 = ax4.imshow(norm, cmap="magma", vmin=0, vmax=1.0, aspect="auto")
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            col = PALETTE["text"] if norm[i, j] < 0.5 else PALETTE["bg_deep"]
            ax4.text(j, i, f"{norm[i,j]*100:.0f}", ha="center", va="center",
                     color=col, fontsize=8)
    ax4.set_xticks(range(N_CLASSES), labels=CLASS_NAMES, rotation=25, fontsize=8)
    ax4.set_yticks(range(N_CLASSES), labels=CLASS_NAMES, fontsize=8)
    ax4.set_title("confusion (row %)", color=PALETTE["text"])

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    lines = [
        r"$\bf{Baseline\ MLP\ results}$", "",
        f"N test             : {len(y_true):,}",
        f"overall accuracy   : {overall_acc:.4f}",
        f"trivial baseline   : {trivial_acc:.4f}",
        f"macro F1           : {macro_f1:.4f}",
        "",
        "per-class F1:",
    ]
    for c in per_class:
        lines.append(f"  {c['class']:14s}: {c['f1']:.3f}")
    lines.extend([
        "",
        f"log-escape R^2     : {r2:.4f}",
        f"  on {int(escape_mask_test.sum()):,} escape samples",
        "",
        "contract gates:",
        f"  acc >= 0.90      : {'PASS' if gate_acc else 'FAIL'}",
        f"  R^2 >= 0.80      : {'PASS' if gate_r2 else 'FAIL'}",
        "",
        f"model              : MLP [{args.hidden}x{args.layers}]",
        f"training time      : {total_train:.1f}s",
    ])
    for i, line in enumerate(lines):
        ax5.text(0.02, 0.97 - i * 0.047, line, transform=ax5.transAxes,
                 ha="left", va="top", fontsize=10,
                 color=PALETTE["text"], family="monospace")

    fig.suptitle(
        f"Baseline MLP on 1M-IC basin dataset",
        color=PALETTE["text"], fontsize=16, y=1.005,
    )
    fig.savefig(out_fig / "baseline_summary.png", facecolor=PALETTE["bg_deep"])
    print(f"  saved {out_fig / 'baseline_summary.png'}")
    plt.close(fig)

    print()
    print("=== done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
