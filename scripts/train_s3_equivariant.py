"""Train S₃-equivariant models on the 6D basin dataset.

Supports three model types:
  --model mlp       : plain BasinMLP on 7D input (baseline)
  --model reynolds  : S3ReynoldsNet wrapping BasinMLP (exact equivariance)
  --model bodynet   : S3BodyNet with per-body features (built-in equivariance)

Data augmentation (random S₃ permutation per sample) is used for mlp and
reynolds during training.  bodynet is equivariant by construction so
augmentation is optional.
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

from mega3bp.equivariant import S3BodyNet, S3ReynoldsNet, augment_batch
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


def make_model(model_type, in_dim, hidden, layers, key):
    if model_type == "mlp":
        return BasinMLP(in_dim=in_dim, hidden=hidden, n_layers=layers,
                        out_dim=N_CLASSES + 1, key=key)
    elif model_type == "reynolds":
        base = BasinMLP(in_dim=in_dim, hidden=hidden, n_layers=layers,
                        out_dim=N_CLASSES + 1, key=key)
        return S3ReynoldsNet(base=base)
    elif model_type == "bodynet":
        return S3BodyNet(body_hidden=hidden, body_layers=layers, body_out=hidden // 2,
                         key=key)
    else:
        raise ValueError(f"unknown model type: {model_type}")


def main() -> int:
    apply_dirac_style()
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, required=True)
    parser.add_argument("--model", choices=["mlp", "reynolds", "bodynet"], default="bodynet")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--augment", action="store_true",
                        help="S₃ data augmentation (auto-on for mlp/reynolds)")
    parser.add_argument("--weighting", choices=["none", "sqrt", "inverse"], default="inverse")
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    if args.tag is None:
        args.tag = f"s3_{args.model}"
    use_augment = args.augment or args.model in ("mlp", "reynolds")

    print(f"model       : {args.model}")
    print(f"augmentation: {use_augment}")
    print(f"jax backend : {jax.default_backend()}")
    print(f"devices     : {jax.devices()}")
    print()

    df = pd.read_parquet(args.parquet)
    print(f"loaded {len(df):,} rows")
    mask_keep = df.label.values != -1
    df = df.loc[mask_keep].reset_index(drop=True)
    labels = df.label.to_numpy().astype(np.int32)
    n = len(df)
    print(f"kept {n:,} non-ambiguous")

    raw = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    inputs = raw
    in_dim = 7

    etimes = df.escape_time.to_numpy(dtype=np.float32)
    finite = np.isfinite(etimes)
    log_t = log_escape_target(etimes, finite)

    key = jax.random.PRNGKey(args.seed)
    key, split_key = jax.random.split(key)
    train_idx, val_idx, test_idx = stratified_split(jnp.asarray(labels), split_key)
    train_idx, val_idx, test_idx = np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)
    print(f"train/val/test: {train_idx.size:,}/{val_idx.size:,}/{test_idx.size:,}")

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
    print(f"class weights: {np.round(class_w, 3).tolist()} ({args.weighting})")

    class_w_jax = jnp.asarray(class_w, dtype=jnp.float32)

    key, model_key = jax.random.split(key)
    model = make_model(args.model, in_dim, args.hidden, args.layers, model_key)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=args.lr / 10, peak_value=args.lr,
        warmup_steps=500, decay_steps=args.epochs * (len(train_idx) // args.batch),
        end_value=args.lr / 50,
    )
    opt = optax.adamw(learning_rate=schedule, weight_decay=1e-4)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, opt_state, x, y, lt, mask):
        def loss_closure(m):
            return loss_fn(m, x, y, lt, mask, class_w_jax)
        (_, metrics), grads = eqx.filter_value_and_grad(
            loss_closure, has_aux=True)(model)
        updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, metrics

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

    print(f"\n=== training ({args.model}) ===")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_epoch = 0
    best_model = model

    t_train = time.perf_counter()
    for epoch in range(args.epochs):
        key, epoch_key, aug_key = jax.random.split(key, 3)
        ep_losses, ep_accs = [], []
        t0 = time.perf_counter()
        for xb, yb, ltb, mb in iterate_batches(x_train, y_train, lt_train, m_train, epoch_key):
            if use_augment:
                aug_key, sub = jax.random.split(aug_key)
                xb, yb = augment_batch(sub, xb, yb)
            model, opt_state, metrics = train_step(model, opt_state, xb, yb, ltb, mb)
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

        star = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model = jax.tree.map(lambda x: x, model)
            star = " *"
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}/{args.epochs}  loss {tr_loss:.4f}  acc {tr_acc:.4f}  |  "
                  f"val {val_loss:.4f} / {val_acc:.4f}  [{dt:.1f}s]{star}")

    total_train = time.perf_counter() - t_train
    print(f"trained {total_train:.1f}s, best val acc {best_val_acc:.4f} @ epoch {best_epoch}")
    model = best_model

    # ---- Test evaluation ----
    print("\n=== test set ===")
    def full_predict(x_all, batch=8192):
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
    print(f"  accuracy  : {overall_acc:.4f}  (trivial={trivial_acc:.4f})")

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
        per_class.append({"class": name, "f1": f1, "support": int(tp + fn)})
        print(f"    {name:14s}  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  N={tp+fn}")
    macro_f1 = float(np.mean([c["f1"] for c in per_class]))
    print(f"  macro F1  : {macro_f1:.4f}")

    escape_mask_test = np.asarray(m_test)
    lt_true_esc = np.asarray(lt_test)[escape_mask_test]
    lt_pred_esc = log_t_pred[escape_mask_test]
    ss_res = float(np.sum((lt_true_esc - lt_pred_esc) ** 2))
    ss_tot = float(np.sum((lt_true_esc - lt_true_esc.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    print(f"  R²        : {r2:.4f}")

    # ---- Save ----
    out_data = PROJECT_ROOT / "data"
    out_fig = PROJECT_ROOT / "figures"
    out_data.mkdir(exist_ok=True)
    out_fig.mkdir(exist_ok=True)

    tag = args.tag
    model_path = out_data / f"{tag}_model.eqx"
    eqx.tree_serialise_leaves(str(model_path), model)
    print(f"\nsaved {model_path}")

    np.savez(out_data / f"{tag}_history.npz",
             train_loss=np.array(history["train_loss"]),
             val_loss=np.array(history["val_loss"]),
             train_acc=np.array(history["train_acc"]),
             val_acc=np.array(history["val_acc"]),
             confusion=confusion,
             per_class_f1=np.array([c["f1"] for c in per_class]),
             overall_acc=overall_acc, macro_f1=macro_f1, log_escape_r2=r2)

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    fig.patch.set_facecolor(PALETTE["bg_deep"])
    ep = np.arange(1, args.epochs + 1)
    axes[0].plot(ep, history["train_loss"], color=PALETTE["cyan"], lw=1.8, label="train")
    axes[0].plot(ep, history["val_loss"], color=PALETTE["lime"], lw=1.8, label="val")
    axes[0].set_title("loss"); axes[0].legend()
    axes[1].plot(ep, history["train_acc"], color=PALETTE["cyan"], lw=1.8, label="train")
    axes[1].plot(ep, history["val_acc"], color=PALETTE["lime"], lw=1.8, label="val")
    axes[1].axhline(trivial_acc, color=PALETTE["coral"], ls="--", lw=1.0, label="trivial")
    axes[1].set_title("accuracy"); axes[1].legend()
    fig.suptitle(f"{tag}  acc={overall_acc:.3f}  F1={macro_f1:.3f}",
                 color=PALETTE["text"], fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_fig / f"{tag}_curves.png", facecolor=PALETTE["bg_deep"])
    plt.close(fig)
    print(f"saved {out_fig / f'{tag}_curves.png'}")

    print(f"\n=== done  {tag}  acc={overall_acc:.4f}  F1={macro_f1:.4f}  R²={r2:.4f} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
