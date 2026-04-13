"""Enhanced visualization: gallery panels, better GIFs, shape-sphere trajectories.

Produces:
  1. Gallery panel: all 10 Suvakov orbits in one figure (paper + blog)
  2. Single-orbit detail: trajectory + energy trace (paper + blog)
  3. High-quality GIFs with multi-period traces and glow effects
  4. Shape-sphere trajectory (Mollweide projection)
  5. Three-body synchronization plot (x(t), y(t) of each body)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from functools import partial

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

from mega3bp.integrators import yoshida6_step
from mega3bp.dynamics import total_energy
from mega3bp.shape_sphere import config_to_shape

OUT_DIR = Path(__file__).parent
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

Q_COLLINEAR = jnp.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 0.0]], dtype=jnp.float64)

# All 10 unique Suvakov orbits (skip duplicates like Butterfly III, II variants)
SUVAKOV_GALLERY = [
    ("Figure-eight", 0.3471168881, 0.5327249454, 6.3259139829),
    ("Butterfly I", 0.3068934205, 0.1255065670, 6.2346748391),
    ("Butterfly II", 0.392955223941802, 0.0975792352080344, 7.003707),
    ("Butterfly III", 0.4059155671, 0.2301631260, 13.8671234361),
    ("Moth I", 0.4644451728, 0.3960600146, 14.8943051743),
    ("Moth II", 0.4391659182, 0.4529676431, 28.6692709402),
    ("Moth III", 0.3834435199, 0.3773636946, 25.8392363356),
    ("Goggles", 0.0833000718, 0.1278892555, 10.4648495256),
    ("Dragonfly", 0.0805842255, 0.5888360898, 21.2723373956),
    ("Yin-Yang Ib", 0.2826986823, 0.3272087861, 10.9633031497),
]


# ============================================================
# Integration
# ============================================================

@partial(jax.jit, static_argnames=("n_steps",))
def trace_orbit_full(v1, v2, T, n_steps=3000):
    """Trace orbit: returns (q_trace, p_trace, E_trace)."""
    q0 = Q_COLLINEAR
    p0 = jnp.array([[v1, v2], [v1, v2], [-2*v1, -2*v2]], dtype=jnp.float64)
    h = T / n_steps

    def step_fn(carry, _):
        q, p = carry
        q, p = yoshida6_step(q, p, jnp.float64(h))
        E = total_energy(q, p)
        return (q, p), (q, p, E)

    (_, _), (q_tr, p_tr, E_tr) = jax.lax.scan(step_fn, (q0, p0), None, length=n_steps)
    E0 = total_energy(q0, p0)
    q_tr = jnp.concatenate([q0[None], q_tr], axis=0)
    p_tr = jnp.concatenate([p0[None], p_tr], axis=0)
    E_tr = jnp.concatenate([E0[None], E_tr], axis=0)
    return q_tr, p_tr, E_tr


# ============================================================
# Styles
# ============================================================

def set_paper_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 0.6,
        "axes.grid": False,
    })


def set_blog_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica", "Arial"],
        "font.weight": "300",
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "axes.labelcolor": "#b0b0c0",
        "axes.titlecolor": "#f0f0ff",
        "xtick.color": "#707090",
        "ytick.color": "#707090",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "#080810",
        "axes.facecolor": "#080810",
        "axes.edgecolor": "#303048",
        "axes.linewidth": 0.4,
        "axes.grid": False,
    })


# ============================================================
# Gallery panel (all 10 orbits in one figure)
# ============================================================

def plot_gallery(mode="paper"):
    """Grid of all 10 Suvakov orbits."""
    if mode == "paper":
        set_paper_style()
        colors = ["#1a1a1a", "#606060", "#a0a0a0"]
        lw = 0.7
    else:
        set_blog_style()
        colors = ["#ff4080", "#40ff90", "#60b0ff"]
        lw = 1.0

    n_orbits = len(SUVAKOV_GALLERY)
    ncols = 5
    nrows = (n_orbits + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.4, nrows * 2.4))
    axes = axes.flatten()

    for i, (name, v1, v2, T) in enumerate(SUVAKOV_GALLERY):
        ax = axes[i]
        q_tr, _, _ = trace_orbit_full(v1, v2, T, n_steps=2500)
        q_np = np.array(q_tr)

        for b in range(3):
            ax.plot(q_np[:, b, 0], q_np[:, b, 1], color=colors[b], lw=lw, alpha=0.85)
            ax.plot(q_np[0, b, 0], q_np[0, b, 1], "o", color=colors[b], ms=3,
                    mec="white" if mode == "blog" else "black", mew=0.5)

        ax.set_aspect("equal")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        title_color = "#f0f0ff" if mode == "blog" else "black"
        ax.set_title(name, color=title_color, fontsize=9 if mode == "paper" else 10)

        # Subtle frame
        for spine in ax.spines.values():
            spine.set_edgecolor("#303048" if mode == "blog" else "#c0c0c0")

    # Hide extra axes
    for j in range(n_orbits, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Periodic orbits of the three-body problem (Suvakov-Dmitrasinovic 2013)",
                 fontsize=11, color="white" if mode == "blog" else "black", y=0.99)

    fname = f"gallery_{mode}.pdf" if mode == "paper" else f"gallery_{mode}.png"
    plt.savefig(FIG_DIR / fname)
    plt.close()
    print(f"  Saved {fname}")


# ============================================================
# Detailed single-orbit figure with energy trace
# ============================================================

def plot_orbit_detailed(name, v1, v2, T, mode="paper", n_periods=2):
    """Four-panel figure: trajectory + energy + x(t) + y(t)."""
    if mode == "paper":
        set_paper_style()
        colors = ["#1a1a1a", "#606060", "#a0a0a0"]
        bg = "white"
        lw = 0.8
    else:
        set_blog_style()
        colors = ["#ff4080", "#40ff90", "#60b0ff"]
        bg = "#080810"
        lw = 1.2

    n_steps = 3000 * n_periods
    T_total = T * n_periods
    q_tr, p_tr, E_tr = trace_orbit_full(v1, v2, T_total, n_steps=n_steps)
    q_np = np.array(q_tr)
    E_np = np.array(E_tr)
    t_arr = np.linspace(0, T_total, n_steps + 1)

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1], height_ratios=[2, 1, 1],
                            hspace=0.4, wspace=0.3)

    # Main trajectory (large, left)
    ax_traj = fig.add_subplot(gs[0, 0])
    for b in range(3):
        ax_traj.plot(q_np[:, b, 0], q_np[:, b, 1], color=colors[b], lw=lw, alpha=0.85,
                      label=f"Body {b+1}")
        ax_traj.plot(q_np[0, b, 0], q_np[0, b, 1], "o", color=colors[b], ms=6,
                      mec="white" if mode == "blog" else "black", mew=1)
    ax_traj.set_aspect("equal")
    ax_traj.set_xlim(-1.8, 1.8)
    ax_traj.set_ylim(-1.8, 1.8)
    ax_traj.set_xlabel(r"$x$")
    ax_traj.set_ylabel(r"$y$")
    ax_traj.set_title(f"Trajectory ({n_periods} periods)")
    ax_traj.legend(loc="upper right", framealpha=0.8, fontsize=8)

    # Shape sphere (small, right top)
    ax_shape = fig.add_subplot(gs[0, 1], projection="3d")
    n_tr = np.array(jax.vmap(config_to_shape)(jnp.array(q_tr)))
    ax_shape.plot(n_tr[:, 0], n_tr[:, 1], n_tr[:, 2],
                  color="#ff4080" if mode == "blog" else "black", lw=lw, alpha=0.8)
    # Draw unit sphere
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax_shape.plot_surface(xs, ys, zs, alpha=0.08,
                           color="#6080c0" if mode == "blog" else "grey")
    ax_shape.set_xlabel(r"$n_1$", fontsize=8)
    ax_shape.set_ylabel(r"$n_2$", fontsize=8)
    ax_shape.set_zlabel(r"$n_3$", fontsize=8)
    ax_shape.set_title("Shape sphere path")
    if mode == "blog":
        ax_shape.set_facecolor("#080810")

    # Energy conservation
    ax_E = fig.add_subplot(gs[1, :])
    E0 = E_np[0]
    dE_rel = (E_np - E0) / abs(E0)
    ax_E.plot(t_arr, dE_rel * 1e10, color="#ff4080" if mode == "blog" else "black", lw=0.8)
    ax_E.set_xlabel(r"$t$")
    ax_E.set_ylabel(r"$\Delta E / |E|\ \times 10^{10}$")
    ax_E.set_title(f"Energy conservation (drift: {np.max(np.abs(dE_rel)):.2e})")
    ax_E.grid(True, alpha=0.2)
    # Period markers
    for k in range(1, n_periods):
        ax_E.axvline(k * T, color="#606060" if mode == "paper" else "#6080a0",
                      ls="--", lw=0.5, alpha=0.6)

    # Body x(t), y(t)
    ax_xy = fig.add_subplot(gs[2, :])
    for b in range(3):
        ax_xy.plot(t_arr, q_np[:, b, 0], color=colors[b], lw=0.7, alpha=0.8,
                    label=f"$x_{b+1}$")
        ax_xy.plot(t_arr, q_np[:, b, 1], color=colors[b], lw=0.7, alpha=0.8,
                    ls="--", label=f"$y_{b+1}$")
    ax_xy.set_xlabel(r"$t$")
    ax_xy.set_ylabel(r"position")
    ax_xy.set_title("Position vs time")
    ax_xy.legend(ncol=6, loc="upper right", fontsize=7, framealpha=0.7)
    for k in range(1, n_periods):
        ax_xy.axvline(k * T, color="#606060" if mode == "paper" else "#6080a0",
                       ls="--", lw=0.5, alpha=0.6)

    fig.suptitle(f"{name}  ($v_1 = {v1:.4f}$, $v_2 = {v2:.4f}$, $T = {T:.4f}$)",
                 fontsize=12, color="white" if mode == "blog" else "black", y=0.995)

    fname_safe = name.replace(" ", "_").replace("-", "")
    fname = f"detailed_{fname_safe}_{mode}.pdf" if mode == "paper" else f"detailed_{fname_safe}_{mode}.png"
    plt.savefig(FIG_DIR / fname)
    plt.close()
    print(f"  Saved {fname}")


# ============================================================
# High-quality multi-period GIF with glow
# ============================================================

def make_orbit_gif_hq(name, v1, v2, T, n_frames=120, n_periods=1):
    """High-quality GIF with multi-period trail and glow effect."""
    import matplotlib.animation as animation

    set_blog_style()
    colors = ["#ff4080", "#40ff90", "#60b0ff"]
    glow_colors = ["#ff80b0", "#80ffb0", "#90c0ff"]

    n_steps = 400 * n_periods  # one period per 400 frames baseline
    T_total = T * n_periods
    q_tr, _, _ = trace_orbit_full(v1, v2, T_total, n_steps=n_steps)
    q_np = np.array(q_tr)

    fig, ax = plt.subplots(figsize=(7, 7))
    margin = 0.2
    xlim = (q_np[..., 0].min() - margin, q_np[..., 0].max() + margin)
    ylim = (q_np[..., 1].min() - margin, q_np[..., 1].max() + margin)
    size = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
    xmid = 0.5 * (xlim[0] + xlim[1])
    ymid = 0.5 * (ylim[0] + ylim[1])
    ax.set_xlim(xmid - size/2, xmid + size/2)
    ax.set_ylim(ymid - size/2, ymid + size/2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(name, fontsize=14, pad=12)

    # Main trail lines + glow underlay
    glows = [ax.plot([], [], color=glow_colors[i], lw=5, alpha=0.3, solid_capstyle="round")[0]
             for i in range(3)]
    trails = [ax.plot([], [], color=colors[i], lw=1.8, alpha=0.9, solid_capstyle="round")[0]
              for i in range(3)]
    points = [ax.plot([], [], "o", color=colors[i], ms=15,
                      mec="white", mew=1.5, zorder=10)[0] for i in range(3)]
    halos = [ax.plot([], [], "o", color=glow_colors[i], ms=30, alpha=0.35, zorder=9)[0]
             for i in range(3)]

    frame_step = max(1, (n_steps + 1) // n_frames)

    def init():
        for t in trails + glows + points + halos:
            t.set_data([], [])
        return trails + glows + points + halos

    def animate(frame):
        idx = min((frame + 1) * frame_step, n_steps + 1)
        for i in range(3):
            trails[i].set_data(q_np[:idx, i, 0], q_np[:idx, i, 1])
            glows[i].set_data(q_np[:idx, i, 0], q_np[:idx, i, 1])
            points[i].set_data([q_np[idx - 1, i, 0]], [q_np[idx - 1, i, 1]])
            halos[i].set_data([q_np[idx - 1, i, 0]], [q_np[idx - 1, i, 1]])
        return trails + glows + points + halos

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                                    interval=40, blit=True)

    fname_safe = name.replace(" ", "_").replace("-", "")
    fname = f"hq_{fname_safe}.gif"
    try:
        ani.save(FIG_DIR / fname, writer="pillow", fps=25)
        print(f"  Saved {fname}")
    except Exception as e:
        print(f"  Failed: {e}")
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("ENHANCED VISUALIZATIONS")
    print(f"Output: {FIG_DIR}")
    print("=" * 60)

    # Gallery panels
    print("\n=== Gallery panels (10 orbits) ===")
    plot_gallery(mode="paper")
    plot_gallery(mode="blog")

    # Detailed per-orbit figures for the most famous ones
    print("\n=== Detailed figures ===")
    famous = [
        ("Figure-eight", 0.3471168881, 0.5327249454, 6.3259139829),
        ("Butterfly I", 0.3068934205, 0.1255065670, 6.2346748391),
        ("Moth I", 0.4644451728, 0.3960600146, 14.8943051743),
        ("Goggles", 0.0833000718, 0.1278892555, 10.4648495256),
    ]
    for name, v1, v2, T in famous:
        print(f"\n{name}:")
        plot_orbit_detailed(name, v1, v2, T, mode="paper", n_periods=2)
        plot_orbit_detailed(name, v1, v2, T, mode="blog", n_periods=2)

    # High-quality GIFs with glow
    print("\n=== High-quality GIFs with glow effects ===")
    for name, v1, v2, T in famous:
        print(f"\n{name}:")
        make_orbit_gif_hq(name, v1, v2, T, n_frames=120)

    print(f"\nAll done. See {FIG_DIR}")


if __name__ == "__main__":
    main()
