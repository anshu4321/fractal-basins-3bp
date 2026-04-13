"""Train baseline MLP on the 6D (non-zero momentum) basin dataset.

Reads a parquet with columns shape_n1..n3 + pi1x/pi1y/pi2x/pi2y,
trains BasinMLP with 7-dim input (or Fourier-featurized), evaluates,
and saves artifacts.  Largely follows train_baseline.py but for the
extended feature set.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax

jax.config.update("jax_enable_x64", False)

import equinox as eqx
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd

from mega3bp.ml import N_CLASSES, BasinMLP, loss_fn, predict, stratified_split
from mega3bp.style import BASIN_COLORS, PALETTE, apply_dirac_style

CLASS_NAMES = ["bound", "body1_escape", "body2_escape", "body3_escape"]
SHAPE_COLS = ["shape_n1", "shape_n2", "shape_n3"]
MOM_COLS = ["pi1x", "pi1y", "pi2x", "pi2y"]
FEATURE_COLS = SHAPE_COLS + MOM_COLS


def log_escape_target(escape_time, finite_mask):
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
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--features", choices=["raw", "fourier"], default="raw")
    parser.add_argument("--n_freqs", type=int, default=8)
    parser.add_argument("--weighting", choices=["none", "sqrt", "inverse"], default="sqrt")
    parser.add_argument("--tag", type=str, default="6d")
    args = parser.parse_args()

    print(f"jax backend : {jax.default_backend()}")
    print(f"devices     : {jax.devices()}")
    print()

    df = pd.read_parquet(args.parquet)
    print(f"loaded {len(df):,} rows from {args.parquet}")

    for col in FEATURE_COLS:
        assert col in df.columns, f"missing column {col}"

    mask_keep = df.label.values != -1
    df = df.loc[mask_keep].reset_index(drop=True)
    labels = df.label.to_numpy().astype(np.int32)
    n = len(df)
    print(f"kept {n:,} non-ambiguous rows")

    raw = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    if args.features == "fourier":
        freqs = (2.0 ** np.arange(args.n_freqs, dtype=np.float32)) * np.pi
        args_grid = raw[:, :, None] * freqs
        sins = np.sin(args_grid).reshape(n, -1)
        coss = np.cos(args_grid).reshape(n, -1)
        inputs = np.concatenate([sins, coss], axis=-1).astype(np.float32)
        in_dim = inputs.shape[1]
        print(f"fourier features: n_freqs={args.n_freqs}, in_dim={in_dim}")
    else:
        inputs = raw
        in_dim = 7
        print(f"raw features: in_dim={in_dim}")

    etimes = df.escape_time.to_numpy(dtype=np.float32)
    finite = np.isfinite(etimes)
    log_t = log_escape_target(etimes, finite)

    if "energy" in df.columns:
        E = df.energy.values
        print(f"energy stats: mean={E.mean():.3f} std={E.std():.3f} "
              f"bound(E<0)={100*(E<0).mean():.1f}%")
    print(f"finite escape times: {int(finite.sum()):,} ({100*finite.mean():.2f}%)")
    print()

    key = jax.random.PRNGKey(args.seed)
    key, split_key = jax.random.split(key)
    train_idx, val_idx, test_idx = stratified_split(jnp.asarray(labels), split_key)
    train_idx, val_idx, test_idx = np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)
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

    class_counts = np.array([int((labels[train_idx] == i).sum()) for i in range(N_CLASSES)])
    if args.weighting == "inverse":
        class_w = 1.0 / np.maximum(class_counts, 1)
    elif args.weighting == "sqrt":
        class_w = 1.0 / np.sqrt(np.maximum(class_counts, 1))
    else:
        class_w = np.ones(N_CLASSES)
    class_w = class_w / class_w.mean()
    print(f"class counts (train): {class_counts.tolist()}")
    print(f"class weights       : {np.round(class_w, 3).tolist()}  ({args.weighting})")
    print()

    class_w_jax = jnp.asarray(class_w, dtype=jnp.float32)

    key, model_key = jax.random.split(key)
    model = BasinMLP(in_dim=in_dim, hidden=args.hidden, n_layers=args.layers,
                     out_dim=N_CLASSES + 1, key=model_key)

    opt = optax.adamw(learning_rate=args.lr, weight_decay=1e-4)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, opt_state, x, y, lt, mask):
        def loss_closure(m):
            return loss_fn(m, x, y, lt, mask, class_w_jax)
        (loss_val, metrics), grads = eqx.filter_value_and_grad(
            loss_closure, has_aux=True)(model)
        updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_val, metrics

    @eqx.filter_jit
    def eval_batch(model, x, y, lt, mask):
        _, metrics = loss_fn(model, x, y, lt, mask, class_w_jax)
        return metrics

    def iterate_batches(x_, y_, lt_, m_, key_):
        n_ = x_.shape[0]
        idx = jax.random.permutation(key_, n_)
        for start in range(0, n_, args.batch):
            sel = idx[start : start + args.batch]
            yield x_[sel], y_[sel], lt_[sel], m_[sel]

    print("=== training ===")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "val_cls": [], "val_mse": []}
    best_val_acc = -1.0
    best_epoch = 0
    best_model = model

    t_train = time.perf_counter()
    for epoch in range(args.epochs):
        key, epoch_key = jax.random.split(key)
        ep_losses, ep_accs = [], []
        t0 = time.perf_counter()
        for xb, yb, ltb, mb in iterate_batches(x_train, y_train, lt_train, m_train, epoch_key):
            model, opt_state, _, metrics = train_step(model, opt_state, xb, yb, ltb, mb)
            ep_losses.append(float(metrics["total_loss"]))
            ep_accs.append(float(metrics["accuracy"]))

        val_metrics = eval_batch(model, x_val, y_val, lt_val, m_val)
        dt = time.perf_counter() - t0
        tr_loss, tr_acc = float(np.mean(ep_losses)), float(np.mean(ep_accs))
        val_loss = float(val_metrics["total_loss"])
        val_acc = float(val_metrics["accuracy"])
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_cls"].append(float(val_metrics["cls_loss"]))
        history["val_mse"].append(float(val_metrics["mse_loss"]))

        star = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model = eqx.tree_at(lambda m: m.layers, model, model.layers)
            star = " *"
        print(f"  epoch {epoch+1:3d}/{args.epochs}  loss {tr_loss:.4f}  acc {tr_acc:.4f}  |  "
              f"val_loss {val_loss:.4f}  val_acc {val_acc:.4f}  [{dt:.2f}s]{star}")

    total_train = time.perf_counter() - t_train
    print(f"trained for {total_train:.1f}s")
    print(f"best val acc {best_val_acc:.4f} at epoch {best_epoch}")
    model = best_model
    print()

    # ---- Test evaluation ----
    print("=== test set evaluation ===")
    def full_predict(x_all, batch=16384):
        ys, lts = [], []
        for start in range(0, x_all.shape[0], batch):
            p, lt = predict(model, x_all[start : start + batch])
            ys.append(np.asarray(p))
            lts.append(np.asarray(lt))
        return np.concatenate(ys), np.concatenate(lts)

    y_pred, log_t_pred = full_predict(x_test)
    y_true = np.asarray(y_test)
    overall_acc = float((y_pred == y_true).mean())
    trivial_acc = float((y_true == 0).mean())
    print(f"  overall accuracy : {overall_acc:.4f}")
    print(f"  trivial (bound)  : {trivial_acc:.4f}")

    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        confusion[int(t), int(p)] += 1

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

    escape_mask_test = np.asarray(m_test)
    lt_true_esc = np.asarray(lt_test)[escape_mask_test]
    lt_pred_esc = log_t_pred[escape_mask_test]
    ss_res = float(np.sum((lt_true_esc - lt_pred_esc) ** 2))
    ss_tot = float(np.sum((lt_true_esc - lt_true_esc.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    print(f"  log-escape R^2   : {r2:.4f}")
    print()

    # ---- Save ----
    out_data = PROJECT_ROOT / "data"
    out_fig = PROJECT_ROOT / "figures"
    out_data.mkdir(exist_ok=True)
    out_fig.mkdir(exist_ok=True)

    tag = args.tag
    model_path = out_data / f"{tag}_model.eqx"
    eqx.tree_serialise_leaves(str(model_path), model)
    print(f"saved {model_path}")

    np.savez(
        out_data / f"{tag}_history.npz",
        train_loss=np.array(history["train_loss"]),
        train_acc=np.array(history["train_acc"]),
        val_loss=np.array(history["val_loss"]),
        val_acc=np.array(history["val_acc"]),
        confusion=confusion,
        per_class_f1=np.array([c["f1"] for c in per_class]),
        overall_acc=overall_acc, macro_f1=macro_f1, log_escape_r2=r2,
    )

    # ---- Plots ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    fig.patch.set_facecolor(PALETTE["bg_deep"])
    epochs_axis = np.arange(1, args.epochs + 1)
    axes[0].plot(epochs_axis, history["train_loss"], color=PALETTE["cyan"], lw=1.8, label="train")
    axes[0].plot(epochs_axis, history["val_loss"], color=PALETTE["lime"], lw=1.8, label="val")
    axes[0].set_title("loss"); axes[0].set_xlabel("epoch"); axes[0].legend()

    axes[1].plot(epochs_axis, history["train_acc"], color=PALETTE["cyan"], lw=1.8, label="train")
    axes[1].plot(epochs_axis, history["val_acc"], color=PALETTE["lime"], lw=1.8, label="val")
    axes[1].axhline(trivial_acc, color=PALETTE["coral"], ls="--", lw=1.0, label="trivial")
    axes[1].set_title("accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()

    axes[2].plot(epochs_axis, history["val_mse"], color=PALETTE["lavender"], lw=1.8)
    axes[2].set_title("val MSE (log-escape)"); axes[2].set_xlabel("epoch")

    fig.suptitle(f"6D Baseline MLP [{args.hidden}x{args.layers}]  macro-F1={macro_f1:.3f}",
                 color=PALETTE["text"], fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_fig / f"{tag}_curves.png", facecolor=PALETTE["bg_deep"])
    plt.close(fig)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    fig.patch.set_facecolor(PALETTE["bg_deep"])
    ax.set_facecolor(PALETTE["bg_panel"])
    norm = confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(norm, cmap="magma", vmin=0, vmax=1.0, aspect="auto")
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            val = confusion[i, j]
            col = PALETTE["text"] if norm[i, j] < 0.5 else PALETTE["bg_deep"]
            ax.text(j, i, f"{val:,}\n{norm[i,j]*100:.1f}%", ha="center", va="center",
                    color=col, fontsize=9)
    ax.set_xticks(range(N_CLASSES), labels=CLASS_NAMES, rotation=25)
    ax.set_yticks(range(N_CLASSES), labels=CLASS_NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Confusion (6D) · macro F1 = {macro_f1:.3f}", color=PALETTE["text"])
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(out_fig / f"{tag}_confusion.png", facecolor=PALETTE["bg_deep"])
    plt.close(fig)

    print(f"  saved {out_fig / f'{tag}_curves.png'}")
    print(f"  saved {out_fig / f'{tag}_confusion.png'}")
    print(f"\n=== done  acc={overall_acc:.4f}  F1={macro_f1:.4f}  R2={r2:.4f} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
