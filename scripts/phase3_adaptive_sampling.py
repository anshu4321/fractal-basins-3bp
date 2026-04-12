"""Phase 3: adaptive sampling near detected basin boundaries.

Uses the boundary-point detection from the fractal analysis (k=12 nearest-
neighbor label disagreement) to place NEW initial conditions in a narrow
band around the basin boundaries, integrate them, and merge with the
original 1M uniform dataset. The resulting boundary-enriched dataset should
have higher information density in the fractal zone, testing whether the
MLP/SIREN plateau from Phase 2 chunks 3-4 was an information bottleneck
(as the D=1.592 measurement from chunk 5 predicts) or something else.

Steps:
  1. Load original 1M parquet, detect boundary points (same as fractal script).
  2. For each boundary point, generate N_PER perturbations by adding a small
     Gaussian displacement in the tangent plane of S^2 and re-projecting.
     σ = 0.005 radians (half the original nearest-neighbor spacing).
  3. Integrate the new ICs at rest (same params as Phase 2 generation).
  4. Merge with the original dataset. Save as a new parquet.
  5. Retrain the MLP baseline on the enriched dataset.
  6. Evaluate on ORIGINAL held-out test set (same split, same seed) so the
     comparison with Phase 2 chunk 3 is apples-to-apples.
  7. Produce comparison figures.

Outputs:
    data/phase3_adaptive_500k.parquet
    data/phase3_enriched_1.5M.parquet
    figures/phase3_adaptive_mollweide.png
    figures/phase3_enriched_vs_uniform.png
    figures/phase3_summary.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

jax.config.update("jax_enable_x64", False)  # fp32 for ML training

import equinox as eqx
import optax

from mega3bp.batch import integrate_batch
from mega3bp.escape import LABEL_NAMES
from mega3bp.ml import (
    N_CLASSES,
    BasinMLP,
    fourier_features,
    loss_fn,
    predict,
    stratified_split,
)
from mega3bp.shape_sphere import sample_shape_sphere, shape_to_config
from mega3bp.style import BASIN_COLORS, PALETTE, apply_dirac_style

K_NEIGHBORS = 12
SIGMA = 0.005        # perturbation radius on S^2 (radians, ~half NN spacing)
N_PER = 2            # perturbations per boundary point (~500k from 262k bdry)
SEED_ADAPTIVE = 0xADA97
SEED_TRAIN = 1337    # same as Phase 2 for apples-to-apples test split

# Integration params (same as Phase 2)
H = 0.01
T_MAX = 500.0
N_STEPS = int(round(T_MAX / H))
R_ESCAPE = 5.0
BINARY_FACTOR = 2.0
R_CLOSE = 1e-3
CHUNK = 65536

CLASS_NAMES = ["bound", "body1_escape", "body2_escape", "body3_escape"]


def perturb_on_sphere(pts: np.ndarray, sigma: float, n_per: int, rng) -> np.ndarray:
    """For each point on S^2, generate n_per tangent-plane perturbations and re-project."""
    N = pts.shape[0]
    all_new = []
    for _ in range(n_per):
        noise = rng.normal(0, sigma, size=(N, 3)).astype(np.float64)
        # Project noise to tangent plane: noise_tangent = noise - (noise . n) * n
        dot = np.sum(noise * pts, axis=-1, keepdims=True)
        tangent = noise - dot * pts
        # Move and re-normalize to S^2
        moved = pts + tangent
        moved /= np.linalg.norm(moved, axis=-1, keepdims=True)
        all_new.append(moved)
    return np.concatenate(all_new, axis=0)


def log_escape_target(etimes, finite):
    out = np.zeros_like(etimes, dtype=np.float32)
    out[finite] = np.log(etimes[finite])
    return out


def run_integration_chunked(configs, p0, chunk_size=CHUNK):
    """Run the batch integrator in chunks, return arrays."""
    N = configs.shape[0]
    n_chunks = (N + chunk_size - 1) // chunk_size
    all_labels = np.empty(N, dtype=np.int8)
    all_etimes = np.empty(N, dtype=np.float64)
    all_amb = np.empty(N, dtype=np.bool_)
    all_rmin = np.empty(N, dtype=np.float64)
    all_maxsep = np.empty(N, dtype=np.float64)

    for i in range(n_chunks):
        lo = i * chunk_size
        hi = min(lo + chunk_size, N)
        actual = hi - lo
        q_chunk = configs[lo:hi]
        p_chunk = p0[lo:hi]
        if actual < chunk_size:
            pad_n = chunk_size - actual
            q_chunk = jnp.concatenate([q_chunk, jnp.repeat(q_chunk[:1], pad_n, axis=0)])
            p_chunk = jnp.concatenate([p_chunk, jnp.repeat(p_chunk[:1], pad_n, axis=0)])
        out = integrate_batch(q_chunk, p_chunk, H, N_STEPS,
                              r_escape=R_ESCAPE, binary_factor=BINARY_FACTOR, r_close=R_CLOSE)
        jax.block_until_ready(out["label"])
        for k, arr in [("label", all_labels), ("escape_time", all_etimes),
                        ("ambiguous", all_amb), ("r_min_ever", all_rmin), ("max_sep", all_maxsep)]:
            arr[lo:hi] = np.asarray(out[k])[:actual]
        print(f"  chunk {i+1}/{n_chunks}  [{lo}:{hi}]")

    return all_labels, all_etimes, all_amb, all_rmin, all_maxsep


def main() -> int:
    apply_dirac_style()
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--n_freqs", type=int, default=8)
    args = parser.parse_args()

    print(f"jax backend : {jax.default_backend()}")
    print(f"devices     : {jax.devices()}")
    print()

    # ---- 1. Load original dataset and detect boundaries ----------------------
    print("=== loading original dataset ===")
    df_orig = pd.read_parquet(args.parquet)
    df_clean = df_orig.loc[df_orig.label != -1].reset_index(drop=True)
    pts = df_clean[["shape_n1", "shape_n2", "shape_n3"]].to_numpy(dtype=np.float64)
    labels = df_clean.label.to_numpy(dtype=np.int64)
    print(f"  {len(df_clean):,} non-ambiguous rows")

    print(f"\n=== detecting boundary (k={K_NEIGHBORS}) ===")
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=K_NEIGHBORS + 1)
    neighbor_labels = labels[idx[:, 1:]]
    is_boundary = np.any(neighbor_labels != labels[:, None], axis=1)
    bdry_pts = pts[is_boundary]
    print(f"  {int(is_boundary.sum()):,} boundary points ({100*is_boundary.mean():.2f}%)")

    # ---- 2. Generate adaptive ICs near boundaries ----------------------------
    print(f"\n=== generating adaptive ICs ===")
    print(f"  sigma = {SIGMA}  n_per = {N_PER}")
    rng = np.random.default_rng(SEED_ADAPTIVE)
    adaptive_pts = perturb_on_sphere(bdry_pts, SIGMA, N_PER, rng)
    N_adaptive = adaptive_pts.shape[0]
    print(f"  generated {N_adaptive:,} new shape-sphere points")

    # ---- 3. Integrate adaptive ICs -------------------------------------------
    print(f"\n=== integrating {N_adaptive:,} adaptive ICs ===")
    jax.config.update("jax_enable_x64", True)
    adaptive_configs = shape_to_config(jnp.asarray(adaptive_pts), inertia=1.0)
    adaptive_p0 = jnp.zeros_like(adaptive_configs)
    t0 = time.perf_counter()
    ad_labels, ad_etimes, ad_amb, ad_rmin, ad_maxsep = run_integration_chunked(
        adaptive_configs, adaptive_p0
    )
    dt = time.perf_counter() - t0
    print(f"  done in {dt:.1f}s ({N_adaptive * N_STEPS / dt / 1e6:.1f} Mstep/s)")

    # Summary
    unique, counts = np.unique(ad_labels, return_counts=True)
    for lab, cnt in zip(unique, counts):
        print(f"    {LABEL_NAMES[int(lab)]:16s}: {cnt:>8,}  ({100*cnt/N_adaptive:5.2f}%)")
    print(f"  ambiguous: {ad_amb.mean():.3%}")

    # ---- 4. Save adaptive + enriched parquets --------------------------------
    out_data = PROJECT_ROOT / "data"
    out_fig = PROJECT_ROOT / "figures"
    out_data.mkdir(exist_ok=True)
    out_fig.mkdir(exist_ok=True)

    df_adaptive = pd.DataFrame({
        "shape_n1": adaptive_pts[:, 0],
        "shape_n2": adaptive_pts[:, 1],
        "shape_n3": adaptive_pts[:, 2],
        "label": ad_labels,
        "escape_time": ad_etimes,
        "ambiguous": ad_amb,
        "r_min_ever": ad_rmin,
        "max_sep": ad_maxsep,
    })
    ad_path = out_data / "phase3_adaptive_500k.parquet"
    df_adaptive.to_parquet(ad_path, engine="pyarrow", compression="snappy")
    print(f"\n  saved {ad_path} ({ad_path.stat().st_size/1e6:.1f} MB)")

    df_enriched = pd.concat([df_orig, df_adaptive], ignore_index=True)
    en_path = out_data / "phase3_enriched_1.5M.parquet"
    df_enriched.to_parquet(en_path, engine="pyarrow", compression="snappy")
    print(f"  saved {en_path} ({en_path.stat().st_size/1e6:.1f} MB)")

    # ---- 5. Train MLP on enriched dataset ------------------------------------
    print(f"\n=== training MLP on enriched dataset ===")
    jax.config.update("jax_enable_x64", False)  # fp32 for training

    # Use the ORIGINAL test set for apples-to-apples comparison.
    # Split the ORIGINAL clean data first, then add adaptive to TRAINING only.
    key = jax.random.PRNGKey(SEED_TRAIN)
    key, split_key = jax.random.split(key)
    orig_labels_jax = jnp.asarray(df_clean.label.to_numpy().astype(np.int32))
    train_idx, val_idx, test_idx = stratified_split(orig_labels_jax, split_key)
    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)
    test_idx = np.asarray(test_idx)

    # Build enriched training set: original train + all non-ambiguous adaptive
    orig_raw = df_clean[["shape_n1", "shape_n2", "shape_n3"]].to_numpy(dtype=np.float32)
    orig_labels_np = df_clean.label.to_numpy().astype(np.int32)
    orig_etimes = df_clean.escape_time.to_numpy(dtype=np.float32)
    orig_finite = np.isfinite(orig_etimes)

    ad_keep = df_adaptive.label.values != -1
    ad_raw = df_adaptive.loc[ad_keep, ["shape_n1", "shape_n2", "shape_n3"]].to_numpy(dtype=np.float32)
    ad_labels_np = df_adaptive.loc[ad_keep, "label"].to_numpy().astype(np.int32)
    ad_etimes_np = df_adaptive.loc[ad_keep, "escape_time"].to_numpy(dtype=np.float32)
    ad_finite = np.isfinite(ad_etimes_np)

    # Fourier features
    def make_fourier(raw):
        freqs = (2.0 ** np.arange(args.n_freqs, dtype=np.float32)) * np.pi
        args_grid = raw[:, :, None] * freqs
        sins = np.sin(args_grid).reshape(raw.shape[0], -1)
        coss = np.cos(args_grid).reshape(raw.shape[0], -1)
        return np.concatenate([sins, coss], axis=-1).astype(np.float32)

    orig_feat = make_fourier(orig_raw)
    ad_feat = make_fourier(ad_raw)
    in_dim = orig_feat.shape[1]

    # Enriched training: original train + adaptive
    x_train = jnp.asarray(np.concatenate([orig_feat[train_idx], ad_feat], axis=0))
    y_train = jnp.asarray(np.concatenate([orig_labels_np[train_idx], ad_labels_np], axis=0), dtype=jnp.int32)
    lt_orig = log_escape_target(orig_etimes, orig_finite)
    lt_ad = log_escape_target(ad_etimes_np, ad_finite)
    lt_train = jnp.asarray(np.concatenate([lt_orig[train_idx], lt_ad], axis=0))
    m_train = jnp.asarray(np.concatenate([orig_finite[train_idx], ad_finite], axis=0))

    # Val and test from ORIGINAL only (same split as Phase 2 chunk 3)
    x_val = jnp.asarray(orig_feat[val_idx])
    y_val = jnp.asarray(orig_labels_np[val_idx], dtype=jnp.int32)
    lt_val = jnp.asarray(lt_orig[val_idx])
    m_val = jnp.asarray(orig_finite[val_idx])

    x_test = jnp.asarray(orig_feat[test_idx])
    y_test = jnp.asarray(orig_labels_np[test_idx], dtype=jnp.int32)
    lt_test = jnp.asarray(lt_orig[test_idx])
    m_test = jnp.asarray(orig_finite[test_idx])

    print(f"  train : {x_train.shape[0]:,} ({train_idx.size:,} orig + {ad_feat.shape[0]:,} adaptive)")
    print(f"  val   : {x_val.shape[0]:,} (original)")
    print(f"  test  : {x_test.shape[0]:,} (original — apples-to-apples)")
    print(f"  in_dim: {in_dim}")

    # Class weights (sqrt inverse on TRAINING set)
    class_counts = np.array([int((np.asarray(y_train) == i).sum()) for i in range(N_CLASSES)])
    class_w = 1.0 / np.sqrt(np.maximum(class_counts, 1))
    class_w = class_w / class_w.mean()
    print(f"  class counts: {class_counts.tolist()}")
    print(f"  class weights: {np.round(class_w, 3).tolist()}")
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
        (_, metrics), grads = eqx.filter_value_and_grad(loss_closure, has_aux=True)(model)
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

    best_val_acc = -1.0
    best_epoch = 0
    best_model = model
    history = {"train_acc": [], "val_acc": []}

    t_train = time.perf_counter()
    for epoch in range(args.epochs):
        key, epoch_key = jax.random.split(key)
        ep_accs = []
        for xb, yb, ltb, mb in iterate_batches(x_train, y_train, lt_train, m_train, epoch_key):
            model, opt_state, metrics = train_step(model, opt_state, xb, yb, ltb, mb)
            ep_accs.append(float(metrics["accuracy"]))
        val_metrics = eval_batch(model, x_val, y_val, lt_val, m_val)
        tr_acc = float(np.mean(ep_accs))
        val_acc = float(val_metrics["accuracy"])
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        star = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model = model
            star = " *"
        if (epoch + 1) % 10 == 0 or star:
            print(f"  epoch {epoch+1:3d}/{args.epochs}  acc {tr_acc:.4f}  val {val_acc:.4f}{star}")

    total_train = time.perf_counter() - t_train
    print(f"  trained {total_train:.1f}s, best val {best_val_acc:.4f} at epoch {best_epoch}")
    model = best_model

    # ---- 6. Evaluate on ORIGINAL test set ------------------------------------
    print(f"\n=== test evaluation (original held-out) ===")
    def full_predict(x_all, batch=16384):
        ys, lts = [], []
        for start in range(0, x_all.shape[0], batch):
            p, lt = predict(model, x_all[start:start+batch])
            ys.append(np.asarray(p))
            lts.append(np.asarray(lt))
        return np.concatenate(ys), np.concatenate(lts)

    y_pred, log_t_pred = full_predict(x_test)
    y_true = np.asarray(y_test)

    overall_acc = float((y_pred == y_true).mean())
    trivial_acc = float((y_true == 0).mean())
    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        confusion[int(t), int(p)] += 1
    per_class = []
    for i in range(N_CLASSES):
        tp = confusion[i, i]
        fn = confusion[i, :].sum() - tp
        fp = confusion[:, i].sum() - tp
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        per_class.append({"class": CLASS_NAMES[i], "precision": prec, "recall": rec, "f1": f1})
    macro_f1 = float(np.mean([c["f1"] for c in per_class]))

    escape_mask_test = np.asarray(m_test)
    lt_true_esc = np.asarray(lt_test)[escape_mask_test]
    lt_pred_esc = log_t_pred[escape_mask_test]
    ss_res = float(np.sum((lt_true_esc - lt_pred_esc) ** 2))
    ss_tot = float(np.sum((lt_true_esc - lt_true_esc.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    print(f"  overall accuracy : {overall_acc:.4f}")
    print(f"  trivial baseline : {trivial_acc:.4f}")
    print(f"  macro F1         : {macro_f1:.4f}")
    for c in per_class:
        print(f"    {c['class']:14s}  P={c['precision']:.3f}  R={c['recall']:.3f}  F1={c['f1']:.3f}")
    print(f"  log-escape R^2   : {r2:.4f}")
    print()

    # Load Phase 2 baseline for comparison
    baseline_path = PROJECT_ROOT / "data" / "baseline_history.npz"
    if baseline_path.exists():
        bl = np.load(baseline_path, allow_pickle=True)
        bl_acc = float(bl["overall_acc"])
        bl_f1 = float(bl["macro_f1"])
        bl_r2 = float(bl["log_escape_r2"])
    else:
        bl_acc = 0.9128
        bl_f1 = 0.4153
        bl_r2 = 0.3141

    delta_acc = overall_acc - bl_acc
    delta_f1 = macro_f1 - bl_f1
    print(f"  vs Phase 2 baseline (1M uniform):")
    print(f"    Δ accuracy : {delta_acc:+.4f}")
    print(f"    Δ macro F1 : {delta_f1:+.4f}")
    print(f"    Δ R^2      : {r2 - bl_r2:+.4f}")

    # ---- 7. Figures ----------------------------------------------------------
    print(f"\n=== plotting ===")
    n_test_raw = orig_raw[test_idx]
    lon = np.arctan2(n_test_raw[:, 1], n_test_raw[:, 0])
    lat = np.arcsin(np.clip(n_test_raw[:, 2], -1, 1))

    # Mollweide: enriched model prediction vs ground truth
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), dpi=150,
                             subplot_kw={"projection": "mollweide"})
    fig.patch.set_facecolor(PALETTE["bg_deep"])
    def moll(ax, lab, title):
        ax.set_facecolor(PALETTE["bg_deep"])
        ax.grid(color=PALETTE["grid"], alpha=0.4, lw=0.3)
        ax.tick_params(colors=PALETTE["text_mute"], labelsize=6)
        size = max(0.4, 9000 / len(lab))
        bound = lab == 0
        ax.scatter(lon[bound], lat[bound], s=size, c=BASIN_COLORS["bound"],
                   alpha=0.25, edgecolors="none", rasterized=True)
        for i in range(1, 4):
            m = lab == i
            if m.any():
                ax.scatter(lon[m], lat[m], s=size * 3.0,
                           c=BASIN_COLORS[CLASS_NAMES[i]], alpha=0.9,
                           edgecolors="none", rasterized=True)
        ax.set_title(title, color=PALETTE["text"])
    moll(axes[0], y_true, "ground truth")
    moll(axes[1], y_pred, f"enriched MLP (acc {overall_acc:.3f})")
    fig.suptitle(f"Phase 3 adaptive sampling — enriched MLP vs ground truth",
                 color=PALETTE["text"], fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_fig / "phase3_adaptive_mollweide.png", facecolor=PALETTE["bg_deep"])
    print(f"  saved phase3_adaptive_mollweide.png")
    plt.close(fig)

    # Error map
    wrong = y_pred != y_true
    fig = plt.figure(figsize=(12, 6.5), dpi=150)
    ax = fig.add_subplot(111, projection="mollweide")
    ax.set_facecolor(PALETTE["bg_deep"])
    ax.grid(color=PALETTE["grid"], alpha=0.3, lw=0.3)
    ax.scatter(lon[~wrong], lat[~wrong], s=1.2, c=PALETTE["grid"], alpha=0.25,
               edgecolors="none", rasterized=True)
    for i in range(N_CLASSES):
        m = wrong & (y_true == i)
        if m.any():
            ax.scatter(lon[m], lat[m], s=10, c=BASIN_COLORS[CLASS_NAMES[i]],
                       alpha=0.92, edgecolors="none", rasterized=True,
                       label=f"true={CLASS_NAMES[i]} ({int(m.sum())})")
    ax.set_title(f"Enriched MLP errors  ·  {int(wrong.sum()):,} / {len(y_true):,} = {100*wrong.mean():.2f}%",
                 color=PALETTE["text"], pad=16)
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0.0, -0.25), ncol=2, fontsize=9)
    leg.get_frame().set_facecolor(PALETTE["bg_elevate"])
    for t in leg.get_texts():
        t.set_color(PALETTE["text"])
    fig.tight_layout()
    fig.savefig(out_fig / "phase3_errors.png", facecolor=PALETTE["bg_deep"])
    print(f"  saved phase3_errors.png")
    plt.close(fig)

    # Comparison bar chart: Phase 2 vs Phase 3
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    fig.patch.set_facecolor(PALETTE["bg_deep"])
    labels_bar = ["accuracy", "macro F1", "log-escape R²"]
    bl_vals = [bl_acc, bl_f1, bl_r2]
    en_vals = [overall_acc, macro_f1, r2]
    x = np.arange(len(labels_bar))
    w = 0.35
    axes[0].bar(x - w/2, bl_vals, w, color=PALETTE["coral"], alpha=0.9, label="1M uniform")
    axes[0].bar(x + w/2, en_vals, w, color=PALETTE["cyan"], alpha=0.9, label="1.5M enriched")
    axes[0].set_xticks(x, labels=labels_bar, color=PALETTE["text"])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("key metrics", color=PALETTE["text"])
    leg = axes[0].legend()
    leg.get_frame().set_facecolor(PALETTE["bg_elevate"])
    for t in leg.get_texts():
        t.set_color(PALETTE["text"])

    # Per-class F1 comparison
    bl_f1_pc = [0.961, 0.208, 0.231, 0.261]  # from Phase 2 chunk 3
    en_f1_pc = [c["f1"] for c in per_class]
    x4 = np.arange(N_CLASSES)
    axes[1].bar(x4 - w/2, bl_f1_pc, w, color=PALETTE["coral"], alpha=0.9, label="1M")
    axes[1].bar(x4 + w/2, en_f1_pc, w, color=PALETTE["cyan"], alpha=0.9, label="1.5M enriched")
    axes[1].set_xticks(range(N_CLASSES), labels=CLASS_NAMES, rotation=20, fontsize=9)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("per-class F1", color=PALETTE["text"])

    # Val accuracy curves
    epochs_axis = np.arange(1, args.epochs + 1)
    axes[2].plot(epochs_axis, history["val_acc"], color=PALETTE["cyan"], lw=1.8, label="enriched")
    axes[2].axhline(bl_acc, color=PALETTE["coral"], ls="--", lw=1.2, label=f"baseline ({bl_acc:.3f})")
    axes[2].axhline(trivial_acc, color=PALETTE["text_mute"], ls=":", lw=0.9, label="trivial")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("val accuracy")
    axes[2].set_ylim(0.88, 0.96)
    axes[2].set_title("training progress", color=PALETTE["text"])
    leg = axes[2].legend(fontsize=9)
    leg.get_frame().set_facecolor(PALETTE["bg_elevate"])
    for t in leg.get_texts():
        t.set_color(PALETTE["text"])

    fig.suptitle("Phase 3: adaptive sampling comparison — 1M uniform vs 1.5M boundary-enriched",
                 color=PALETTE["text"], fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_fig / "phase3_enriched_vs_uniform.png", facecolor=PALETTE["bg_deep"])
    print(f"  saved phase3_enriched_vs_uniform.png")
    plt.close(fig)

    # Summary dashboard
    fig = plt.figure(figsize=(16, 8), dpi=150, constrained_layout=True)
    fig.patch.set_facecolor(PALETTE["bg_deep"])
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
    axM = fig.add_subplot(gs[0, 0], projection="mollweide")
    moll(axM, y_pred, f"enriched MLP prediction (acc {overall_acc:.3f})")
    axT = fig.add_subplot(gs[0, 1])
    axT.axis("off")
    lines = [
        r"$\bf{Phase\ 3:\ adaptive\ sampling}$", "",
        f"original dataset      : 1,048,576 uniform on S²",
        f"adaptive supplement   : {N_adaptive:,} near-boundary",
        f"  sigma              : {SIGMA}  rad",
        f"  source             : {int(is_boundary.sum()):,} boundary pts",
        f"enriched total       : {len(df_enriched):,}",
        "",
        r"$\bf{Test\ metrics\ (original\ held-out)}$", "",
        f"{'metric':<22}{'1M':>8}{'1.5M':>8}{'Δ':>8}",
        f"{'-'*46}",
        f"{'accuracy':<22}{bl_acc:>8.4f}{overall_acc:>8.4f}{delta_acc:>+8.4f}",
        f"{'macro F1':<22}{bl_f1:>8.4f}{macro_f1:>8.4f}{delta_f1:>+8.4f}",
        f"{'log-escape R²':<22}{bl_r2:>8.4f}{r2:>8.4f}{r2-bl_r2:>+8.4f}",
        f"{'trivial baseline':<22}{trivial_acc:>8.4f}",
    ]
    for i, line in enumerate(lines):
        axT.text(0.02, 0.97 - i * 0.055, line, transform=axT.transAxes,
                 ha="left", va="top", fontsize=10,
                 color=PALETTE["text"], family="monospace")
    fig.suptitle("Phase 3 summary", color=PALETTE["text"], fontsize=16, y=1.005)
    fig.savefig(out_fig / "phase3_summary.png", facecolor=PALETTE["bg_deep"])
    print(f"  saved phase3_summary.png")
    plt.close(fig)

    print("\n=== done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
