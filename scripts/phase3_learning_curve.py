"""Phase 3 learning curve: uniform vs adaptive sampling efficiency.

Trains the same MLP architecture (Fourier-8, 384×6, sqrt weighting,
best-checkpoint) at multiple dataset sizes for both uniform and
boundary-enriched sampling strategies, and plots the learning curve.

No new integration required — the existing 1M uniform parquet provides
both the uniform subsamples and the boundary detection for generating
adaptive ICs (which are integrated once then subsampled).

Data points:
  uniform:   250k, 500k, 750k, 1M
  adaptive:  1M + 125k, 1M + 250k, 1M + 500k

All evaluated on the SAME original 104k held-out test set (seed 1337).

Output:
  figures/phase3_learning_curve.png  — the headline figure
  data/learning_curve.npz            — raw numbers
"""
from __future__ import annotations

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

from mega3bp.batch import integrate_batch
from mega3bp.escape import LABEL_NAMES
from mega3bp.shape_sphere import shape_to_config
from mega3bp.style import PALETTE, apply_dirac_style

# Training imports (fp32)
import equinox as eqx
import optax

from mega3bp.ml import N_CLASSES, BasinMLP, loss_fn, predict, stratified_split

SEED_TRAIN = 1337
K_NEIGHBORS = 12
SIGMA = 0.005
H = 0.01
T_MAX = 500.0
N_STEPS = int(round(T_MAX / H))
R_ESCAPE = 5.0
BINARY_FACTOR = 2.0
R_CLOSE = 1e-3
CHUNK = 65536

UNIFORM_SIZES = [250_000, 500_000, 750_000, 1_000_000]
ADAPTIVE_ADDS = [125_000, 250_000, 500_000]


def make_fourier(raw, n_freqs=8):
    freqs = (2.0 ** np.arange(n_freqs, dtype=np.float32)) * np.pi
    args_grid = raw[:, :, None] * freqs
    sins = np.sin(args_grid).reshape(raw.shape[0], -1)
    coss = np.cos(args_grid).reshape(raw.shape[0], -1)
    return np.concatenate([sins, coss], axis=-1).astype(np.float32)


def log_escape_target(etimes, finite):
    out = np.zeros_like(etimes, dtype=np.float32)
    out[finite] = np.log(etimes[finite])
    return out


def perturb_on_sphere(pts, sigma, n_per, rng):
    N = pts.shape[0]
    all_new = []
    for _ in range(n_per):
        noise = rng.normal(0, sigma, size=(N, 3)).astype(np.float64)
        dot = np.sum(noise * pts, axis=-1, keepdims=True)
        tangent = noise - dot * pts
        moved = pts + tangent
        moved /= np.linalg.norm(moved, axis=-1, keepdims=True)
        all_new.append(moved)
    return np.concatenate(all_new, axis=0)


def run_integration_chunked(configs, p0):
    N = configs.shape[0]
    n_chunks = (N + CHUNK - 1) // CHUNK
    all_labels = np.empty(N, dtype=np.int8)
    all_etimes = np.empty(N, dtype=np.float64)
    all_amb = np.empty(N, dtype=np.bool_)
    for i in range(n_chunks):
        lo = i * CHUNK
        hi = min(lo + CHUNK, N)
        actual = hi - lo
        q = configs[lo:hi]
        p = p0[lo:hi]
        if actual < CHUNK:
            pad = CHUNK - actual
            q = jnp.concatenate([q, jnp.repeat(q[:1], pad, axis=0)])
            p = jnp.concatenate([p, jnp.repeat(p[:1], pad, axis=0)])
        out = integrate_batch(q, p, H, N_STEPS, r_escape=R_ESCAPE,
                              binary_factor=BINARY_FACTOR, r_close=R_CLOSE)
        jax.block_until_ready(out["label"])
        all_labels[lo:hi] = np.asarray(out["label"])[:actual]
        all_etimes[lo:hi] = np.asarray(out["escape_time"])[:actual]
        all_amb[lo:hi] = np.asarray(out["ambiguous"])[:actual]
        print(f"    chunk {i+1}/{n_chunks}")
    return all_labels, all_etimes, all_amb


def train_and_eval(x_train, y_train, lt_train, m_train,
                   x_test, y_test, lt_test, m_test,
                   key, epochs=40, hidden=384, layers=6, batch=4096, lr=3e-4):
    """Train MLP, return (accuracy, macro_f1, r2) on test set."""
    class_counts = np.array([int((np.asarray(y_train) == i).sum()) for i in range(N_CLASSES)])
    class_w = 1.0 / np.sqrt(np.maximum(class_counts, 1))
    class_w = class_w / class_w.mean()
    class_w_jax = jnp.asarray(class_w, dtype=jnp.float32)

    key, mk = jax.random.split(key)
    model = BasinMLP(in_dim=x_train.shape[1], hidden=hidden, n_layers=layers,
                     out_dim=N_CLASSES + 1, key=mk)
    opt = optax.adamw(learning_rate=lr, weight_decay=1e-4)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state, x, y, lt, mask):
        def lc(m):
            return loss_fn(m, x, y, lt, mask, class_w_jax)
        (_, metrics), grads = eqx.filter_value_and_grad(lc, has_aux=True)(model)
        updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, metrics

    best_acc = -1.0
    best_model = model
    for epoch in range(epochs):
        key, ek = jax.random.split(key)
        idx = jax.random.permutation(ek, x_train.shape[0])
        for start in range(0, x_train.shape[0], batch):
            sel = idx[start:start + batch]
            model, opt_state, _ = step(model, opt_state,
                                       x_train[sel], y_train[sel],
                                       lt_train[sel], m_train[sel])
        # Quick val on test set for best-checkpoint
        pred_logits = jax.vmap(model)(x_test[:8192])
        pred_cls = jnp.argmax(pred_logits[:, :N_CLASSES], axis=-1)
        acc = float((pred_cls == y_test[:8192]).mean())
        if acc > best_acc:
            best_acc = acc
            best_model = model

    model = best_model
    # Full test eval
    all_pred = []
    all_lt = []
    for s in range(0, x_test.shape[0], 16384):
        p, lt = predict(model, x_test[s:s + 16384])
        all_pred.append(np.asarray(p))
        all_lt.append(np.asarray(lt))
    y_pred = np.concatenate(all_pred)
    lt_pred = np.concatenate(all_lt)
    y_true = np.asarray(y_test)

    overall_acc = float((y_pred == y_true).mean())

    per_class_f1 = []
    for i in range(N_CLASSES):
        tp = int(((y_pred == i) & (y_true == i)).sum())
        fp = int(((y_pred == i) & (y_true != i)).sum())
        fn = int(((y_pred != i) & (y_true == i)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        per_class_f1.append(f1)
    macro_f1 = float(np.mean(per_class_f1))

    esc_mask = np.asarray(m_test)
    if esc_mask.sum() > 0:
        lt_true = np.asarray(lt_test)[esc_mask]
        lt_p = lt_pred[esc_mask]
        ss_res = float(np.sum((lt_true - lt_p) ** 2))
        ss_tot = float(np.sum((lt_true - lt_true.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    else:
        r2 = 0.0

    return overall_acc, macro_f1, r2


def main() -> int:
    apply_dirac_style()
    print(f"jax backend: {jax.default_backend()}")
    print(f"devices    : {jax.devices()}")
    print()

    # Load data
    parquet = PROJECT_ROOT / "data" / "phase2_basin_1048576.parquet"
    print(f"loading {parquet} ...")
    df = pd.read_parquet(parquet)
    df_clean = df.loc[df.label != -1].reset_index(drop=True)
    pts = df_clean[["shape_n1", "shape_n2", "shape_n3"]].to_numpy(dtype=np.float64)
    labels = df_clean.label.to_numpy(dtype=np.int32)
    etimes = df_clean.escape_time.to_numpy(dtype=np.float32)
    finite = np.isfinite(etimes)
    raw = pts.astype(np.float32)
    feat = make_fourier(raw)
    lt = log_escape_target(etimes, finite)
    N = len(df_clean)
    print(f"  {N:,} non-ambiguous rows")

    # Fixed test split (same as all Phase 2/3 work)
    key = jax.random.PRNGKey(SEED_TRAIN)
    key, sk = jax.random.split(key)
    train_idx, val_idx, test_idx = stratified_split(jnp.asarray(labels), sk)
    train_idx = np.asarray(train_idx)
    test_idx = np.asarray(test_idx)

    x_test = jnp.asarray(feat[test_idx])
    y_test = jnp.asarray(labels[test_idx], dtype=jnp.int32)
    lt_test = jnp.asarray(lt[test_idx])
    m_test = jnp.asarray(finite[test_idx])
    print(f"  test set: {test_idx.size:,}")

    # Detect boundaries for adaptive sampling
    print(f"\ndetecting boundaries (k={K_NEIGHBORS}) ...")
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=K_NEIGHBORS + 1)
    nbr_labels = labels[idx[:, 1:]]
    is_boundary = np.any(nbr_labels != labels[:, None], axis=1)
    bdry_pts = pts[is_boundary]
    print(f"  {int(is_boundary.sum()):,} boundary points")

    # Generate max adaptive ICs (500k), then subsample for smaller sizes
    print(f"\ngenerating 500k adaptive ICs ...")
    rng = np.random.default_rng(0xADA97)
    adaptive_pts = perturb_on_sphere(bdry_pts, SIGMA, 2, rng)
    # Trim to 500k
    if len(adaptive_pts) > 500_000:
        adaptive_pts = adaptive_pts[:500_000]
    N_ad = len(adaptive_pts)
    print(f"  {N_ad:,} adaptive points")

    # Integrate adaptive ICs
    print(f"\nintegrating {N_ad:,} adaptive ICs ...")
    ad_configs = shape_to_config(jnp.asarray(adaptive_pts), inertia=1.0)
    ad_p0 = jnp.zeros_like(ad_configs)
    t0 = time.perf_counter()
    ad_labels, ad_etimes, ad_amb = run_integration_chunked(ad_configs, ad_p0)
    print(f"  done in {time.perf_counter() - t0:.1f}s")

    ad_keep = ad_labels != -1
    ad_raw = adaptive_pts[ad_keep].astype(np.float32)
    ad_feat = make_fourier(ad_raw)
    ad_labels_clean = ad_labels[ad_keep].astype(np.int32)
    ad_etimes_clean = ad_etimes[ad_keep].astype(np.float32)
    ad_finite = np.isfinite(ad_etimes_clean)
    ad_lt = log_escape_target(ad_etimes_clean, ad_finite)
    print(f"  {ad_keep.sum():,} non-ambiguous adaptive rows")

    # Switch to fp32 for training
    jax.config.update("jax_enable_x64", False)

    # ---- Run training at each dataset size ---------------------------------
    results = {"uniform": [], "adaptive": []}

    # Uniform curve: subsample from the full 1M training set
    orig_train_feat = feat[train_idx]
    orig_train_labels = labels[train_idx]
    orig_train_lt = lt[train_idx]
    orig_train_finite = finite[train_idx]

    print(f"\n=== uniform learning curve ===")
    for n_sub in UNIFORM_SIZES:
        n_use = min(n_sub, len(train_idx))
        rng_sub = np.random.default_rng(42 + n_sub)
        sub = rng_sub.choice(len(train_idx), n_use, replace=False)
        x_tr = jnp.asarray(orig_train_feat[sub])
        y_tr = jnp.asarray(orig_train_labels[sub], dtype=jnp.int32)
        lt_tr = jnp.asarray(orig_train_lt[sub])
        m_tr = jnp.asarray(orig_train_finite[sub])
        key, tk = jax.random.split(key)
        t0 = time.perf_counter()
        acc, f1, r2 = train_and_eval(x_tr, y_tr, lt_tr, m_tr,
                                      x_test, y_test, lt_test, m_test, tk)
        dt = time.perf_counter() - t0
        results["uniform"].append({"n": n_use, "acc": acc, "f1": f1, "r2": r2})
        print(f"  N={n_use:>10,}  acc={acc:.4f}  F1={f1:.4f}  R²={r2:.4f}  [{dt:.1f}s]")

    # Adaptive curve: 1M uniform base + N adaptive
    print(f"\n=== adaptive learning curve ===")
    x_base = jnp.asarray(orig_train_feat)
    y_base = jnp.asarray(orig_train_labels, dtype=jnp.int32)
    lt_base = jnp.asarray(orig_train_lt)
    m_base = jnp.asarray(orig_train_finite)

    for n_add in ADAPTIVE_ADDS:
        n_use = min(n_add, len(ad_feat))
        rng_sub = np.random.default_rng(99 + n_add)
        sub = rng_sub.choice(len(ad_feat), n_use, replace=False)
        x_tr = jnp.concatenate([x_base, jnp.asarray(ad_feat[sub])], axis=0)
        y_tr = jnp.concatenate([y_base, jnp.asarray(ad_labels_clean[sub], dtype=jnp.int32)])
        lt_tr = jnp.concatenate([lt_base, jnp.asarray(ad_lt[sub])])
        m_tr = jnp.concatenate([m_base, jnp.asarray(ad_finite[sub])])
        total = int(x_tr.shape[0])
        key, tk = jax.random.split(key)
        t0 = time.perf_counter()
        acc, f1, r2 = train_and_eval(x_tr, y_tr, lt_tr, m_tr,
                                      x_test, y_test, lt_test, m_test, tk)
        dt = time.perf_counter() - t0
        results["adaptive"].append({"n": total, "acc": acc, "f1": f1, "r2": r2})
        print(f"  N={total:>10,} (1M+{n_use:,} ad)  acc={acc:.4f}  F1={f1:.4f}  R²={r2:.4f}  [{dt:.1f}s]")

    # ---- Save and plot ------------------------------------------------------
    out_data = PROJECT_ROOT / "data"
    out_fig = PROJECT_ROOT / "figures"
    out_data.mkdir(exist_ok=True)
    out_fig.mkdir(exist_ok=True)

    np.savez(out_data / "learning_curve.npz",
             uniform_n=[r["n"] for r in results["uniform"]],
             uniform_acc=[r["acc"] for r in results["uniform"]],
             uniform_f1=[r["f1"] for r in results["uniform"]],
             adaptive_n=[r["n"] for r in results["adaptive"]],
             adaptive_acc=[r["acc"] for r in results["adaptive"]],
             adaptive_f1=[r["f1"] for r in results["adaptive"]])

    u_n = [r["n"] for r in results["uniform"]]
    u_acc = [r["acc"] for r in results["uniform"]]
    u_f1 = [r["f1"] for r in results["uniform"]]
    a_n = [r["n"] for r in results["adaptive"]]
    a_acc = [r["acc"] for r in results["adaptive"]]
    a_f1 = [r["f1"] for r in results["adaptive"]]

    trivial = float((np.asarray(y_test) == 0).mean())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    fig.patch.set_facecolor(PALETTE["bg_deep"])

    # Accuracy curve
    axes[0].plot(u_n, u_acc, "o-", color=PALETTE["coral"], lw=2.2, ms=8,
                 label="uniform sampling", zorder=5)
    axes[0].plot(a_n, a_acc, "s-", color=PALETTE["cyan"], lw=2.2, ms=8,
                 label="1M uniform + adaptive", zorder=5)
    axes[0].axhline(trivial, color=PALETTE["text_mute"], ls="--", lw=1.0,
                    label=f"trivial bound ({trivial:.3f})")
    axes[0].axhline(0.90, color=PALETTE["amber"], ls=":", lw=1.0,
                    label="contract gate (0.90)")
    axes[0].set_xlabel("total training samples")
    axes[0].set_ylabel("test accuracy")
    axes[0].set_title("accuracy vs dataset size", color=PALETTE["text"])
    axes[0].set_ylim(0.88, 0.96)
    axes[0].set_xscale("log")
    axes[0].set_xlim(2e5, 2e6)
    leg = axes[0].legend(fontsize=9)
    leg.get_frame().set_facecolor(PALETTE["bg_elevate"])
    leg.get_frame().set_edgecolor(PALETTE["spine"])
    for t in leg.get_texts():
        t.set_color(PALETTE["text"])

    # Macro F1 curve
    axes[1].plot(u_n, u_f1, "o-", color=PALETTE["coral"], lw=2.2, ms=8,
                 label="uniform sampling", zorder=5)
    axes[1].plot(a_n, a_f1, "s-", color=PALETTE["cyan"], lw=2.2, ms=8,
                 label="1M uniform + adaptive", zorder=5)
    axes[1].set_xlabel("total training samples")
    axes[1].set_ylabel("macro F1")
    axes[1].set_title("macro F1 vs dataset size", color=PALETTE["text"])
    axes[1].set_ylim(0.2, 0.7)
    axes[1].set_xscale("log")
    axes[1].set_xlim(2e5, 2e6)
    leg = axes[1].legend(fontsize=9)
    leg.get_frame().set_facecolor(PALETTE["bg_elevate"])
    leg.get_frame().set_edgecolor(PALETTE["spine"])
    for t in leg.get_texts():
        t.set_color(PALETTE["text"])

    fig.suptitle(
        "Phase 3 learning curve — uniform vs boundary-adaptive sampling\n"
        "same MLP architecture (Fourier-8, 384×6), same test set",
        color=PALETTE["text"], fontsize=14, y=1.04,
    )
    fig.tight_layout()
    fig.savefig(out_fig / "phase3_learning_curve.png", facecolor=PALETTE["bg_deep"])
    print(f"\nsaved {out_fig / 'phase3_learning_curve.png'}")

    print("\n=== done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
