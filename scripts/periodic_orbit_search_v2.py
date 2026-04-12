"""Periodic orbit search v2: vectorized shooting with lax.scan + vmap.

v1 used a Python for-loop over 25k steps × 200 candidates = 5M tiny GPU
kernel launches. This v2 integrates all candidates simultaneously using
the existing batch integrator infrastructure and a custom lax.scan that
tracks the minimum return distance.

Strategy:
  1. Compute model prediction entropy, select top-K candidates.
  2. Integrate ALL candidates in one vectorized lax.scan, tracking
     min |q(t) - q(0)| and the time it occurs (skip first 50 steps
     to avoid trivial t≈0 returns).
  3. Sort by closest return distance, report + plot.
  4. For the best near-returns, produce trajectory animations.
"""
from __future__ import annotations

import sys
import time
from functools import partial
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
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb

from mega3bp.integrators import yoshida6_step
from mega3bp.shape_sphere import config_to_shape, shape_to_config
from mega3bp.style import BODY_COLORS, BODY_GLOW, PALETTE, apply_dirac_style

N_CANDIDATES = 200
H_FINE = 0.002
T_SCAN = 100.0          # scan to 100 time units (~16 figure-eight periods)
N_STEPS = int(round(T_SCAN / H_FINE))  # 50,000 steps
SKIP_INITIAL = 1000     # skip first 1000 steps (T > 2.0) to avoid trivial near-returns
N_FREQS = 8
HIDDEN = 384
LAYERS = 6


def make_fourier(raw, n_freqs=N_FREQS):
    freqs = (2.0 ** np.arange(n_freqs, dtype=np.float32)) * np.pi
    args_grid = raw[:, :, None] * freqs
    sins = np.sin(args_grid).reshape(raw.shape[0], -1)
    coss = np.cos(args_grid).reshape(raw.shape[0], -1)
    return np.concatenate([sins, coss], axis=-1).astype(np.float32)


@partial(jax.jit, static_argnames=("n_steps", "skip"))
def scan_min_return(q0_batch, p0_batch, h, n_steps, skip=SKIP_INITIAL):
    """Integrate a batch and track minimum SHAPE-SPACE return distance.

    Uses config_to_shape to measure distance on the Montgomery shape sphere
    rather than in Cartesian coordinates. This factors out the overall
    rotation and scaling that change during the orbit — two configurations
    that look the same on the shape sphere ARE the same orbit.

    Also tracks |p| at the minimum-distance point as a secondary check
    (a true periodic orbit from rest must also have p → 0 at return).
    """
    h_arr = jnp.asarray(h, dtype=jnp.float64)
    shape0 = config_to_shape(q0_batch)  # (B, 3)

    def step(carry, i):
        q, p, best_dist, best_t, best_q, best_pnorm = carry
        q_new, p_new = yoshida6_step(q, p, h_arr)

        # Shape-space distance (rotation/scale invariant)
        shape_now = config_to_shape(q_new)
        dist = jnp.sqrt(jnp.sum((shape_now - shape0) ** 2, axis=-1))
        # Momentum norm (for periodic orbit from rest, should be ~0 at return)
        p_norm = jnp.sqrt(jnp.sum(p_new ** 2, axis=(-2, -1)))
        t_now = (i.astype(jnp.float64) + 1.0) * h_arr

        is_past_skip = i >= skip
        is_better = (dist < best_dist) & is_past_skip
        best_dist_new = jnp.where(is_better, dist, best_dist)
        best_t_new = jnp.where(is_better, t_now, best_t)
        best_q_new = jnp.where(is_better[:, None, None], q_new, best_q)
        best_pnorm_new = jnp.where(is_better, p_norm, best_pnorm)

        return (q_new, p_new, best_dist_new, best_t_new, best_q_new, best_pnorm_new), None

    B = q0_batch.shape[0]
    init = (
        q0_batch,
        p0_batch,
        jnp.full(B, jnp.inf, dtype=jnp.float64),  # best_dist
        jnp.zeros(B, dtype=jnp.float64),            # best_t
        q0_batch.copy(),                              # best_q
        jnp.full(B, jnp.inf, dtype=jnp.float64),    # best_pnorm
    )

    (q_f, p_f, min_dist, min_t, min_q, min_pnorm), _ = jax.lax.scan(
        step, init, jnp.arange(n_steps)
    )
    return min_dist, min_t, min_q, min_pnorm


@partial(jax.jit, static_argnames=("n_steps",))
def integrate_with_trace(q0, p0, h, n_steps):
    """Integrate a single trajectory and return full q trace."""
    h_arr = jnp.asarray(h, dtype=jnp.float64)

    def step(carry, _):
        q, p = carry
        q_new, p_new = yoshida6_step(q, p, h_arr)
        return (q_new, p_new), q_new

    (q_f, p_f), q_trace = jax.lax.scan(step, (q0, p0), None, length=n_steps)
    q_trace = jnp.concatenate([q0[None], q_trace], axis=0)
    return q_trace


def main() -> int:
    apply_dirac_style()
    print(f"jax backend: {jax.default_backend()}")
    print(f"devices    : {jax.devices()}")
    print()

    parquet = PROJECT_ROOT / "data" / "phase2_basin_1048576.parquet"
    print(f"loading {parquet} ...")
    df = pd.read_parquet(parquet)
    df = df.loc[df.label != -1].reset_index(drop=True)
    pts = df[["shape_n1", "shape_n2", "shape_n3"]].to_numpy(dtype=np.float64)
    raw = pts.astype(np.float32)
    feat = make_fourier(raw)
    print(f"  {len(df):,} rows")

    # Compute model entropy
    jax.config.update("jax_enable_x64", False)
    import equinox as eqx
    from mega3bp.ml import BasinMLP, N_CLASSES, predict_proba

    model_path = str(PROJECT_ROOT / "data" / "baseline_model.eqx")
    key = jax.random.PRNGKey(42)
    model = BasinMLP(in_dim=feat.shape[1], hidden=HIDDEN, n_layers=LAYERS,
                     out_dim=N_CLASSES + 1, key=key)
    try:
        model = eqx.tree_deserialise_leaves(model_path, model)
        print(f"loaded model from {model_path}")
    except Exception as e:
        print(f"model load failed ({e}); using random init")

    print("computing prediction entropy ...")
    all_probs = []
    for s in range(0, feat.shape[0], 16384):
        probs = predict_proba(model, jnp.asarray(feat[s:s + 16384]))
        all_probs.append(np.asarray(probs))
    all_probs = np.concatenate(all_probs)
    H = -np.sum(all_probs * np.log(np.maximum(all_probs, 1e-12)), axis=-1)
    print(f"  mean H: {H.mean():.4f}  max: {H.max():.4f}")

    top_idx = np.argsort(H)[-N_CANDIDATES:][::-1]
    candidates = pts[top_idx]
    candidate_H = H[top_idx]
    print(f"  selected {N_CANDIDATES} candidates (H range [{candidate_H.min():.3f}, {candidate_H.max():.3f}])")

    # ---- Vectorized shooting ------------------------------------------------
    jax.config.update("jax_enable_x64", True)
    print(f"\n=== vectorized shooting ({N_CANDIDATES} candidates, {N_STEPS} steps) ===")

    q0_batch = shape_to_config(jnp.asarray(candidates), inertia=1.0)
    p0_batch = jnp.zeros_like(q0_batch)

    print("  compiling + running ...")
    print(f"  skip first {SKIP_INITIAL} steps (T > {SKIP_INITIAL * H_FINE:.1f})")
    print(f"  scan to T = {T_SCAN} ({N_STEPS} steps)")
    print(f"  distance metric: SHAPE SPACE (rotation/scale invariant)")
    t0 = time.perf_counter()
    min_dist, min_t, min_q, min_pnorm = scan_min_return(q0_batch, p0_batch, H_FINE, N_STEPS)
    min_dist.block_until_ready()
    dt = time.perf_counter() - t0
    print(f"  done in {dt:.2f}s  ({N_CANDIDATES * N_STEPS / dt / 1e6:.1f} Mstep/s)")

    dists = np.asarray(min_dist)
    times = np.asarray(min_t)
    pnorms = np.asarray(min_pnorm)
    print(f"  closest shape-return: {dists.min():.6f} at T={times[dists.argmin()]:.3f} (|p|={pnorms[dists.argmin()]:.4f})")
    print(f"  median shape-return:  {np.median(dists):.4f}")

    thresholds = [0.01, 0.02, 0.05, 0.1, 0.2]
    for thr in thresholds:
        n_hits = int((dists < thr).sum())
        if n_hits > 0:
            print(f"  shape dist < {thr}: {n_hits} candidates")

    # Sort by distance
    order = np.argsort(dists)
    top_n = min(10, N_CANDIDATES)
    print(f"\n  top {top_n} closest shape-space returns:")
    for i in range(top_n):
        idx = order[i]
        print(f"    #{i+1}  shape_dist={dists[idx]:.6f}  T={times[idx]:.3f}  "
              f"|p|={pnorms[idx]:.4f}  H={candidate_H[idx]:.3f}  "
              f"n=({candidates[idx,0]:.4f},{candidates[idx,1]:.4f},{candidates[idx,2]:.4f})")

    # ---- Save ---------------------------------------------------------------
    out_data = PROJECT_ROOT / "data"
    out_fig = PROJECT_ROOT / "figures"
    out_data.mkdir(exist_ok=True)
    out_fig.mkdir(exist_ok=True)

    np.savez(out_data / "periodic_candidates.npz",
             shapes=candidates, entropies=candidate_H,
             best_dists=dists, best_ts=times,
             order=order)

    # ---- Entropy map --------------------------------------------------------
    lon_all = np.arctan2(pts[:, 1], pts[:, 0])
    lat_all = np.arcsin(np.clip(pts[:, 2], -1, 1))

    fig = plt.figure(figsize=(12, 6.5), dpi=150)
    ax = fig.add_subplot(111, projection="mollweide")
    ax.set_facecolor(PALETTE["bg_deep"])
    ax.grid(color=PALETTE["grid"], alpha=0.3, lw=0.3)
    ax.tick_params(colors=PALETTE["text_mute"], labelsize=7)
    step = 10
    sc = ax.scatter(lon_all[::step], lat_all[::step], s=0.3,
                    c=H[::step], cmap="inferno", vmin=0, vmax=H.max(),
                    alpha=0.7, edgecolors="none", rasterized=True)
    c_lon = np.arctan2(candidates[:, 1], candidates[:, 0])
    c_lat = np.arcsin(np.clip(candidates[:, 2], -1, 1))

    # Color candidates by return distance
    near = dists < 0.1
    far = ~near
    if far.any():
        ax.scatter(c_lon[far], c_lat[far], s=12, c=PALETTE["text_dim"],
                   alpha=0.6, edgecolors="none")
    if near.any():
        ax.scatter(c_lon[near], c_lat[near], s=40, c=PALETTE["lime"],
                   alpha=0.95, edgecolors=PALETTE["bg_deep"], linewidths=0.5,
                   label=f"near-return < 0.1 ({near.sum()})")

    # Mark the single best
    best = order[0]
    ax.scatter([c_lon[best]], [c_lat[best]], s=120, c=PALETTE["amber"],
               marker="*", edgecolors=PALETTE["bg_deep"], linewidths=0.8,
               zorder=10, label=f"best: dist={dists[best]:.4f}")

    ax.set_title("Model entropy + periodic orbit candidates",
                 color=PALETTE["text"], pad=16)
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0.0, -0.22), ncol=2, fontsize=9)
    leg.get_frame().set_facecolor(PALETTE["bg_elevate"])
    leg.get_frame().set_edgecolor(PALETTE["spine"])
    for t in leg.get_texts():
        t.set_color(PALETTE["text"])
    fig.tight_layout()
    fig.savefig(out_fig / "periodic_entropy_map.png", facecolor=PALETTE["bg_deep"])
    print(f"\nsaved periodic_entropy_map.png")
    plt.close(fig)

    # ---- Return distance histogram ------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    fig.patch.set_facecolor(PALETTE["bg_deep"])
    axes[0].hist(np.log10(np.maximum(dists, 1e-6)), bins=30,
                 color=PALETTE["cyan"], alpha=0.85,
                 edgecolor=PALETTE["bg_deep"], linewidth=0.4)
    for thr in [0.01, 0.05, 0.1]:
        axes[0].axvline(np.log10(thr), color=PALETTE["coral"], ls="--", lw=0.9, alpha=0.7)
    axes[0].set_xlabel(r"$\log_{10}$ min return distance")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"return distance distribution ({N_CANDIDATES} candidates)",
                      color=PALETTE["text"])

    axes[1].scatter(times, dists, s=15,
                    c=np.where(dists < 0.1, PALETTE["lime"], PALETTE["coral"]),
                    alpha=0.85, edgecolors="none")
    axes[1].axhline(0.1, color=PALETTE["coral"], ls="--", lw=1.0, alpha=0.7)
    axes[1].set_xlabel("return time T")
    axes[1].set_ylabel("min return distance")
    axes[1].set_yscale("log")
    axes[1].set_title("distance vs return time", color=PALETTE["text"])
    fig.suptitle("Periodic orbit search — vectorized shooting results",
                 color=PALETTE["text"], fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_fig / "periodic_candidates.png", facecolor=PALETTE["bg_deep"])
    print(f"saved periodic_candidates.png")
    plt.close(fig)

    # ---- Animate best near-returns -----------------------------------------
    n_animate = min(4, int((dists < 0.2).sum()))
    if n_animate == 0:
        n_animate = min(4, len(order))
        print(f"\nno near-returns < 0.2; animating {n_animate} closest candidates as stills")

    print(f"\n=== generating trajectory plots for top {n_animate} candidates ===")

    for rank in range(n_animate):
        idx = order[rank]
        pt = candidates[idx]
        d = dists[idx]
        T = times[idx]

        q0 = shape_to_config(jnp.asarray(pt), inertia=1.0)
        p0 = jnp.zeros_like(q0)

        # Integrate with full trace
        n_trace = int(round(T * 1.2 / H_FINE)) if T > 0.5 else N_STEPS
        n_trace = min(n_trace, N_STEPS)
        t0_t = time.perf_counter()
        q_trace = integrate_with_trace(q0, p0, H_FINE, n_trace)
        q_trace.block_until_ready()
        q_np = np.asarray(q_trace)
        print(f"  candidate #{rank+1}: dist={d:.6f} T={T:.3f} ({n_trace} steps in {time.perf_counter()-t0_t:.2f}s)")

        # Static trajectory plot (aesthetic)
        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
        fig.patch.set_facecolor(PALETTE["bg_deep"])
        ax.set_facecolor(PALETTE["bg_deep"])
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

        # Draw each body's path with fading trail
        for body in range(3):
            xs = q_np[:, body, 0]
            ys = q_np[:, body, 1]
            segs = np.stack(
                [np.column_stack([xs[:-1], ys[:-1]]),
                 np.column_stack([xs[1:], ys[1:]])],
                axis=1,
            )
            n_segs = len(segs)
            alphas = np.linspace(0.08, 0.85, n_segs)
            rgba = np.zeros((n_segs, 4))
            rgba[:, :3] = to_rgb(BODY_COLORS[body])
            rgba[:, 3] = alphas
            lc = LineCollection(segs, colors=rgba, linewidths=np.linspace(0.4, 2.0, n_segs))
            ax.add_collection(lc)

            # Start dot
            ax.scatter(xs[0], ys[0], s=80, c=BODY_COLORS[body], alpha=0.95,
                       edgecolors=PALETTE["bg_deep"], linewidths=0.5, zorder=8)
            # End dot (with glow)
            ax.scatter(xs[-1], ys[-1], s=250, c=BODY_GLOW[body], alpha=0.25,
                       edgecolors="none", zorder=7)
            ax.scatter(xs[-1], ys[-1], s=60, c=BODY_COLORS[body], alpha=0.95,
                       edgecolors="white", linewidths=0.4, zorder=9)

        pad = 0.15
        xs_all = q_np[..., 0].flatten()
        ys_all = q_np[..., 1].flatten()
        lim = max(np.abs(xs_all).max(), np.abs(ys_all).max()) + pad
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

        is_near = d < 0.1
        status = "NEAR-PERIODIC" if is_near else "candidate"
        ax.text(0.5, 0.97,
                f"periodic orbit {status}  #{rank+1}",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=14, color=PALETTE["text"])
        ax.text(0.5, 0.935,
                f"return dist = {d:.6f}  ·  T = {T:.3f}  ·  entropy = {candidate_H[idx]:.3f}",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=10, color=PALETTE["text_mute"])
        ax.text(0.5, 0.905,
                f"shape = ({pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.4f})",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=9, color=PALETTE["text_dim"])

        fig.tight_layout()
        fig.savefig(out_fig / f"periodic_orbit_{rank+1}.png", facecolor=PALETTE["bg_deep"])
        print(f"  saved periodic_orbit_{rank+1}.png")
        plt.close(fig)

        # MP4 animation for near-returns
        if is_near or rank == 0:
            import matplotlib.animation as animation

            stride = max(1, n_trace // 400)
            frames_data = q_np[::stride]
            n_anim = frames_data.shape[0]
            TRAIL = min(200, n_anim)

            fig_a, ax_a = plt.subplots(figsize=(8, 8), dpi=120)
            fig_a.patch.set_facecolor(PALETTE["bg_deep"])
            ax_a.set_facecolor(PALETTE["bg_deep"])
            ax_a.set_aspect("equal")
            for spine in ax_a.spines.values():
                spine.set_visible(False)
            ax_a.set_xticks([])
            ax_a.set_yticks([])
            ax_a.set_xlim(-lim, lim)
            ax_a.set_ylim(-lim, lim)

            # Faint full path as reference
            for body in range(3):
                ax_a.plot(q_np[:, body, 0], q_np[:, body, 1],
                          color=PALETTE["text_dim"], lw=0.2, alpha=0.15)

            trail_lcs = []
            halos = []
            cores = []
            for body in range(3):
                lc = LineCollection([], linewidths=np.linspace(0.3, 2.0, TRAIL), zorder=5)
                ax_a.add_collection(lc)
                trail_lcs.append(lc)
                halo = ax_a.scatter([], [], s=400, c=BODY_GLOW[body], alpha=0.2,
                                    edgecolors="none", zorder=6)
                core = ax_a.scatter([], [], s=80, c=BODY_COLORS[body], alpha=0.95,
                                    edgecolors="white", linewidths=0.4, zorder=7)
                halos.append(halo)
                cores.append(core)

            title_text = ax_a.text(0.5, 0.97, "", transform=ax_a.transAxes,
                                   ha="center", va="top", fontsize=13, color=PALETTE["text"])
            time_text = ax_a.text(0.98, 0.03, "", transform=ax_a.transAxes,
                                  ha="right", va="bottom", fontsize=9,
                                  color=PALETTE["cyan_glow"], family="monospace")
            title_text.set_text(f"periodic orbit {status}  #{rank+1}")

            rgb_colors = [np.array(to_rgb(c)) for c in BODY_COLORS]

            def update(fi):
                arts = []
                for body in range(3):
                    lo = max(0, fi - TRAIL)
                    win = frames_data[lo:fi + 1, body, :]
                    if win.shape[0] >= 2:
                        segs = np.stack([win[:-1], win[1:]], axis=1)
                        ns = segs.shape[0]
                        rgba = np.zeros((ns, 4))
                        rgba[:, :3] = rgb_colors[body]
                        rgba[:, 3] = np.linspace(0.0, 0.9, ns)
                        trail_lcs[body].set_segments(segs)
                        trail_lcs[body].set_color(rgba)
                        trail_lcs[body].set_linewidths(np.linspace(0.3, 2.0, ns))
                    cur = frames_data[fi, body, :]
                    halos[body].set_offsets([[cur[0], cur[1]]])
                    cores[body].set_offsets([[cur[0], cur[1]]])
                    arts.extend([trail_lcs[body], halos[body], cores[body]])
                t_now = fi * stride * H_FINE
                time_text.set_text(f"t = {t_now:.2f}")
                arts.append(time_text)
                return arts

            ani = animation.FuncAnimation(fig_a, update, frames=n_anim,
                                          interval=1000/60, blit=True)
            mp4_path = out_fig / f"periodic_orbit_{rank+1}.mp4"
            try:
                writer = animation.FFMpegWriter(fps=60, bitrate=4000, codec="libx264")
                ani.save(str(mp4_path), writer=writer, dpi=120,
                         savefig_kwargs={"facecolor": PALETTE["bg_deep"]})
                print(f"  saved {mp4_path.name}")
            except Exception as e:
                print(f"  mp4 failed: {e}")
            plt.close(fig_a)

    print("\n=== done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
