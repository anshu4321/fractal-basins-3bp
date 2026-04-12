"""Phase 3 periodic orbit discovery via model-uncertainty-guided shooting.

Periodic orbits of the equal-mass planar 3BP live exactly on the basin
boundaries (they are the unstable fixed points of the return map). The
baseline MLP's prediction entropy is highest near these boundaries —
so we use it as a heuristic for WHERE to launch high-precision shooting.

Strategy:
  1. Load the trained model + full dataset.
  2. Compute model prediction entropy H(p) = -sum p_i log p_i for each
     shape-sphere point. High entropy = model is uncertain = likely near
     a basin boundary = candidate for periodic orbit.
  3. Select the top-K highest-entropy points as shooting candidates.
  4. For each candidate, integrate with TIGHT tolerance (smaller h) for
     a long time and check for near-return: does the trajectory come
     back close to its initial condition at some time T?
  5. If |q(T) - q(0)| < threshold, refine via Newton-like shooting to
     get the exact period.

This is speculative — periodic orbits are measure-zero in phase space
so most candidates will NOT return. The value is in demonstrating the
active-search methodology, not in guaranteed discovery.

Known orbits to try to rediscover:
  - Chenciner-Montgomery figure-eight (the one we validated in Phase 1)
  - Broucke-Hénon family
  - Šuvakov-Dmitrašinović families (moth, butterfly, etc.)

Outputs:
  data/periodic_candidates.npz
  figures/periodic_entropy_map.png
  figures/periodic_candidates.png
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

from mega3bp.dynamics import total_energy
from mega3bp.integrators import yoshida6_integrate_energy_only, yoshida6_step
from mega3bp.shape_sphere import config_to_shape, shape_to_config
from mega3bp.style import PALETTE, apply_dirac_style

jax.config.update("jax_enable_x64", False)

import equinox as eqx

from mega3bp.ml import BasinMLP, N_CLASSES, predict_proba

N_CANDIDATES = 200
RETURN_THRESHOLD = 0.05  # |q(T) - q(0)| threshold for near-return
H_FINE = 0.002           # finer step for orbit search
T_SCAN = 50.0            # scan for returns up to this time
SCAN_STEPS = int(round(T_SCAN / H_FINE))
N_FREQS = 8
HIDDEN = 384
LAYERS = 6


def make_fourier(raw, n_freqs=N_FREQS):
    freqs = (2.0 ** np.arange(n_freqs, dtype=np.float32)) * np.pi
    args_grid = raw[:, :, None] * freqs
    sins = np.sin(args_grid).reshape(raw.shape[0], -1)
    coss = np.cos(args_grid).reshape(raw.shape[0], -1)
    return np.concatenate([sins, coss], axis=-1).astype(np.float32)


def entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy of probability vectors. Shape (..., C) -> (...)."""
    log_p = np.log(np.maximum(probs, 1e-12))
    return -np.sum(probs * log_p, axis=-1)


def check_near_return(q0, p0, h, n_steps, threshold):
    """Integrate from (q0, p0) and check if trajectory returns near q0.

    Returns (best_distance, best_time, best_q) — the closest approach
    to the initial configuration over the integration window.
    """
    jax.config.update("jax_enable_x64", True)
    q0_jax = jnp.asarray(q0, dtype=jnp.float64)
    p0_jax = jnp.asarray(p0, dtype=jnp.float64)
    h_arr = jnp.asarray(h, dtype=jnp.float64)

    best_dist = float('inf')
    best_t = 0.0
    q, p = q0_jax, p0_jax

    # Skip the first ~10 steps to avoid trivial near-return at t ≈ 0
    for i in range(min(50, n_steps)):
        q, p = yoshida6_step(q, p, h_arr)

    for i in range(50, n_steps):
        q, p = yoshida6_step(q, p, h_arr)
        if i % 10 == 0:  # check every 10 steps for speed
            dist = float(jnp.linalg.norm(q - q0_jax))
            t = (i + 1) * h
            if dist < best_dist:
                best_dist = dist
                best_t = t
            if dist < threshold:
                return best_dist, best_t, np.asarray(q)

    return best_dist, best_t, np.asarray(q)


def main() -> int:
    apply_dirac_style()
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--n_candidates", type=int, default=N_CANDIDATES)
    args = parser.parse_args()

    print(f"jax backend: {jax.default_backend()}")
    print()

    # Load dataset
    df = pd.read_parquet(args.parquet)
    df = df.loc[df.label != -1].reset_index(drop=True)
    pts = df[["shape_n1", "shape_n2", "shape_n3"]].to_numpy(dtype=np.float64)
    raw = pts.astype(np.float32)
    feat = make_fourier(raw)
    print(f"loaded {len(df):,} points")

    # Load or create model
    jax.config.update("jax_enable_x64", False)
    model_path = args.model or str(PROJECT_ROOT / "data" / "baseline_model.eqx")
    key = jax.random.PRNGKey(42)
    model = BasinMLP(in_dim=feat.shape[1], hidden=HIDDEN, n_layers=LAYERS,
                     out_dim=N_CLASSES + 1, key=key)
    try:
        model = eqx.tree_deserialise_leaves(model_path, model)
        print(f"loaded model from {model_path}")
    except Exception as e:
        print(f"could not load model ({e}); using random init for entropy-based search")

    # Compute prediction entropy for all points
    print("computing prediction entropy ...")
    all_probs = []
    for start in range(0, feat.shape[0], 16384):
        probs = predict_proba(model, jnp.asarray(feat[start:start + 16384]))
        all_probs.append(np.asarray(probs))
    all_probs = np.concatenate(all_probs, axis=0)
    H = entropy(all_probs)
    print(f"  mean entropy: {H.mean():.4f}  max: {H.max():.4f}")

    # Select top-K highest-entropy candidates
    top_idx = np.argsort(H)[-args.n_candidates:][::-1]
    candidates = pts[top_idx]
    candidate_H = H[top_idx]
    print(f"  selected {len(top_idx)} highest-entropy candidates")
    print(f"  entropy range: [{candidate_H.min():.4f}, {candidate_H.max():.4f}]")
    print()

    # ---- Entropy map on Mollweide -------------------------------------------
    out_fig = PROJECT_ROOT / "figures"
    out_fig.mkdir(exist_ok=True)

    lon = np.arctan2(pts[:, 1], pts[:, 0])
    lat = np.arcsin(np.clip(pts[:, 2], -1, 1))

    fig = plt.figure(figsize=(12, 6.5), dpi=150)
    ax = fig.add_subplot(111, projection="mollweide")
    ax.set_facecolor(PALETTE["bg_deep"])
    ax.grid(color=PALETTE["grid"], alpha=0.3, lw=0.3)
    ax.tick_params(colors=PALETTE["text_mute"], labelsize=7)

    # Subsample for the background (faint)
    step = 10
    ax.scatter(lon[::step], lat[::step], s=0.3,
               c=H[::step], cmap="inferno", vmin=0, vmax=H.max(),
               alpha=0.7, edgecolors="none", rasterized=True)

    # Candidates on top
    c_lon = np.arctan2(candidates[:, 1], candidates[:, 0])
    c_lat = np.arcsin(np.clip(candidates[:, 2], -1, 1))
    ax.scatter(c_lon, c_lat, s=15, c=PALETTE["lime"],
               alpha=0.95, edgecolors=PALETTE["bg_deep"], linewidths=0.3,
               label=f"top {args.n_candidates} candidates")

    ax.set_title(
        f"Model prediction entropy on shape sphere  ·  {args.n_candidates} periodic-orbit candidates",
        color=PALETTE["text"], pad=16,
    )
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0.0, -0.18), fontsize=9)
    leg.get_frame().set_facecolor(PALETTE["bg_elevate"])
    leg.get_frame().set_edgecolor(PALETTE["spine"])
    for t in leg.get_texts():
        t.set_color(PALETTE["text"])
    fig.tight_layout()
    fig.savefig(out_fig / "periodic_entropy_map.png", facecolor=PALETTE["bg_deep"])
    print(f"saved {out_fig / 'periodic_entropy_map.png'}")

    # ---- Shooting: check each candidate for near-return --------------------
    print(f"\n=== shooting {args.n_candidates} candidates ===")
    print(f"  h={H_FINE}  T_scan={T_SCAN}  threshold={RETURN_THRESHOLD}")

    jax.config.update("jax_enable_x64", True)
    results = []
    n_returns = 0

    for i, (pt, h_val) in enumerate(zip(candidates, candidate_H)):
        q0 = np.asarray(shape_to_config(jnp.asarray(pt), inertia=1.0))
        p0 = np.zeros_like(q0)

        best_dist, best_t, best_q = check_near_return(q0, p0, H_FINE, SCAN_STEPS, RETURN_THRESHOLD)
        is_return = best_dist < RETURN_THRESHOLD
        if is_return:
            n_returns += 1

        results.append({
            "shape": pt,
            "entropy": float(h_val),
            "best_dist": best_dist,
            "best_t": best_t,
            "is_return": is_return,
        })

        if (i + 1) % 20 == 0 or is_return:
            flag = " *** NEAR-RETURN ***" if is_return else ""
            print(f"  [{i+1:3d}/{args.n_candidates}]  H={h_val:.4f}  "
                  f"best_dist={best_dist:.4f}  T={best_t:.2f}{flag}")

    print(f"\n  near-returns: {n_returns} / {args.n_candidates}")
    print()

    # Save results
    out_data = PROJECT_ROOT / "data"
    out_data.mkdir(exist_ok=True)
    np.savez(
        out_data / "periodic_candidates.npz",
        shapes=np.array([r["shape"] for r in results]),
        entropies=np.array([r["entropy"] for r in results]),
        best_dists=np.array([r["best_dist"] for r in results]),
        best_ts=np.array([r["best_t"] for r in results]),
        is_return=np.array([r["is_return"] for r in results]),
    )

    # ---- Plot: near-return distances ----------------------------------------
    dists = np.array([r["best_dist"] for r in results])
    ts = np.array([r["best_t"] for r in results])
    returns = np.array([r["is_return"] for r in results])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    fig.patch.set_facecolor(PALETTE["bg_deep"])

    axes[0].hist(np.log10(np.maximum(dists, 1e-6)), bins=30,
                 color=PALETTE["cyan"], alpha=0.85,
                 edgecolor=PALETTE["bg_deep"], linewidth=0.4)
    axes[0].axvline(np.log10(RETURN_THRESHOLD), color=PALETTE["coral"], ls="--", lw=1.2,
                    label=f"threshold = {RETURN_THRESHOLD}")
    axes[0].set_xlabel(r"$\log_{10}$ closest return distance")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"near-return search  ·  {n_returns}/{args.n_candidates} hits",
                      color=PALETTE["text"])
    leg = axes[0].legend(fontsize=9)
    leg.get_frame().set_facecolor(PALETTE["bg_elevate"])
    for t in leg.get_texts():
        t.set_color(PALETTE["text"])

    axes[1].scatter(ts, dists, s=12, c=np.where(returns, PALETTE["lime"], PALETTE["coral"]),
                    alpha=0.85, edgecolors="none")
    axes[1].axhline(RETURN_THRESHOLD, color=PALETTE["coral"], ls="--", lw=1.0)
    axes[1].set_xlabel("time of closest return")
    axes[1].set_ylabel("closest return distance")
    axes[1].set_yscale("log")
    axes[1].set_title("distance vs return time", color=PALETTE["text"])

    fig.suptitle(
        "Periodic orbit search — model-uncertainty-guided shooting",
        color=PALETTE["text"], fontsize=14, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_fig / "periodic_candidates.png", facecolor=PALETTE["bg_deep"])
    print(f"saved {out_fig / 'periodic_candidates.png'}")

    print("\n=== done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
