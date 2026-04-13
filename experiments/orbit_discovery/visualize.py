"""Visualization for velocity-space basin search results.

Generates two versions of each plot:
  - Paper: black-and-white, publication style
  - Blog: dark theme, glowing trails, animated

Produces:
  1. Velocity-space basin map (fractal landscape in (v1, v2))
  2. Entropy heatmap with Suvakov orbit locations
  3. Orbit trajectory plots (physical space)
  4. Animated GIFs of orbits
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
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from mega3bp.integrators import yoshida6_step
from mega3bp.shape_sphere import pairwise_distances

OUT_DIR = Path(__file__).parent
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Collinear configuration
Q_COLLINEAR = jnp.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 0.0]], dtype=jnp.float64)

SUVAKOV = [
    ("Figure-eight", 0.3471168881, 0.5327249454, 6.3259139829),
    ("Butterfly I", 0.3068934205, 0.1255065670, 6.2346748391),
    ("Butterfly II", 0.392955223941802, 0.0975792352080344, 7.003707),
    ("Moth I", 0.4644451728, 0.3960600146, 14.8943051743),
    ("Moth II", 0.4391659182, 0.4529676431, 28.6692709402),
    ("Moth III", 0.3834435199, 0.3773636946, 25.8392363356),
    ("Goggles", 0.0833000718, 0.1278892555, 10.4648495256),
    ("Dragonfly", 0.0805842255, 0.5888360898, 21.2723373956),
    ("Yarn", 0.559064247131347, 0.349191558837891, 14.894307),
    ("Yin-Yang Ib", 0.2826986823, 0.3272087861, 10.9633031497),
]


# ============================================================
# PAPER STYLE (black, white, grey)
# ============================================================

def set_paper_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
    })


# ============================================================
# BLOG STYLE (dark, glowing)
# ============================================================

def set_blog_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Arial"],
        "font.weight": "300",
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "axes.labelcolor": "#e0e0e0",
        "axes.titlecolor": "#ffffff",
        "xtick.color": "#a0a0a0",
        "ytick.color": "#a0a0a0",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "#0a0a12",
        "axes.facecolor": "#0a0a12",
        "axes.edgecolor": "#404060",
        "axes.linewidth": 0.5,
        "axes.grid": False,
    })


# ============================================================
# Integrate orbit for visualization
# ============================================================

@partial(jax.jit, static_argnames=("n_steps",))
def trace_orbit(v1, v2, T, n_steps=2000):
    """Integrate orbit and return full trajectory of body positions."""
    q0 = Q_COLLINEAR
    p0 = jnp.array([[v1, v2], [v1, v2], [-2*v1, -2*v2]], dtype=jnp.float64)
    h = T / n_steps

    def step_fn(carry, _):
        q, p = carry
        q, p = yoshida6_step(q, p, jnp.float64(h))
        return (q, p), q

    (_, _), q_trace = jax.lax.scan(step_fn, (q0, p0), None, length=n_steps)
    return jnp.concatenate([q0[None], q_trace], axis=0)  # (n_steps+1, 3, 2)


# ============================================================
# Plot 1: Velocity-space basin map
# ============================================================

def plot_basin_map(labels_grid, v1_grid, v2_grid, mode="paper"):
    """Plot the basin map as a 2D image colored by escape outcome."""
    if mode == "paper":
        set_paper_style()
        # Greyscale: 0=bound=black, 1=light grey, 2=medium grey, 3=dark grey, -1=white
        cmap = ListedColormap(["white", "black", "#606060", "#a0a0a0", "#d0d0d0"])
        vmin, vmax = -1.5, 3.5
        bg_color = "white"
        fg_color = "black"
    else:
        set_blog_style()
        # Dark theme: 0=bound=blue, escape=warm colors
        cmap = ListedColormap(["#202030", "#4080ff", "#ff4060", "#ffaa20", "#a0ff40"])
        vmin, vmax = -1.5, 3.5
        bg_color = "#0a0a12"
        fg_color = "#ffffff"

    fig, ax = plt.subplots(figsize=(7, 7))

    im = ax.imshow(labels_grid, extent=[v1_grid[0], v1_grid[-1], v2_grid[0], v2_grid[-1]],
                    origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal",
                    interpolation="nearest")

    # Mark Suvakov orbits
    for name, v1, v2, T in SUVAKOV:
        if mode == "paper":
            ax.plot(v1, v2, "o", ms=6, mfc="white", mec="black", mew=1.2)
            ax.annotate(name, (v1, v2), xytext=(5, 5), textcoords="offset points",
                        fontsize=7, color="black",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5))
        else:
            ax.plot(v1, v2, "o", ms=8, mfc="#ffff60", mec="#ffffff", mew=1.5)
            ax.annotate(name, (v1, v2), xytext=(6, 6), textcoords="offset points",
                        fontsize=8, color="#ffff60", fontweight="300",
                        bbox=dict(boxstyle="round,pad=0.3", fc="#0a0a12", ec="#ffff60", lw=0.5, alpha=0.8))

    ax.set_xlabel(r"$v_1$")
    ax.set_ylabel(r"$v_2$")
    title = "Velocity-Space Basin Map" if mode == "paper" else "Escape basins in velocity space"
    ax.set_title(title)

    fname = f"basin_map_{mode}.pdf" if mode == "paper" else f"basin_map_{mode}.png"
    plt.savefig(FIG_DIR / fname)
    plt.close()
    print(f"  Saved {fname}")


# ============================================================
# Plot 2: Entropy heatmap
# ============================================================

def plot_entropy_map(entropy_grid, v1_grid, v2_grid, mode="paper"):
    """Plot classifier entropy as a heatmap with Suvakov orbits overlaid."""
    if mode == "paper":
        set_paper_style()
        cmap = "Greys"
    else:
        set_blog_style()
        # Custom vibrant gradient: dark blue -> cyan -> yellow -> hot pink
        cmap = LinearSegmentedColormap.from_list(
            "glow", ["#0a0a12", "#1a3050", "#20a0c0", "#ffd040", "#ff2080"])

    fig, ax = plt.subplots(figsize=(7, 7))

    im = ax.imshow(entropy_grid, extent=[v1_grid[0], v1_grid[-1], v2_grid[0], v2_grid[-1]],
                    origin="lower", cmap=cmap, aspect="equal")

    cbar = plt.colorbar(im, ax=ax, label="Entropy H(x)" if mode == "paper" else "classifier entropy")

    for name, v1, v2, T in SUVAKOV:
        if mode == "paper":
            ax.plot(v1, v2, "o", ms=6, mfc="white", mec="black", mew=1.2)
            ax.annotate(name, (v1, v2), xytext=(5, 5), textcoords="offset points",
                        fontsize=7, color="black",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5))
        else:
            ax.plot(v1, v2, "o", ms=10, mfc="#ffffff", mec="#ff2080", mew=2)

    ax.set_xlabel(r"$v_1$")
    ax.set_ylabel(r"$v_2$")
    title = "Classifier Entropy in Velocity Space" if mode == "paper" else "where the model is uncertain"
    ax.set_title(title)

    fname = f"entropy_map_{mode}.pdf" if mode == "paper" else f"entropy_map_{mode}.png"
    plt.savefig(FIG_DIR / fname)
    plt.close()
    print(f"  Saved {fname}")


# ============================================================
# Plot 3: Orbit trajectory (known Suvakov orbits)
# ============================================================

def plot_orbit(name, v1, v2, T, mode="paper", n_steps=5000):
    """Plot the trajectory of three bodies for a given orbit."""
    q_trace = trace_orbit(v1, v2, T, n_steps)
    q_np = np.array(q_trace)  # (n_steps+1, 3, 2)

    if mode == "paper":
        set_paper_style()
        colors = ["black", "#606060", "#b0b0b0"]
        bg = "white"
        lw = 0.8
    else:
        set_blog_style()
        colors = ["#ff4060", "#40ff80", "#60a0ff"]
        bg = "#0a0a12"
        lw = 1.2

    fig, ax = plt.subplots(figsize=(7, 7))

    for i in range(3):
        ax.plot(q_np[:, i, 0], q_np[:, i, 1], color=colors[i], lw=lw, alpha=0.8,
                label=f"Body {i+1}")
        # Starting point
        ax.plot(q_np[0, i, 0], q_np[0, i, 1], "o", color=colors[i], ms=7,
                mec="white" if mode == "blog" else "black", mew=1)

    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    title_text = f"{name} (T = {T:.3f})" if mode == "paper" else f"{name}"
    ax.set_title(title_text)
    ax.legend(loc="upper right", framealpha=0.7)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    fname = f"orbit_{name.replace(' ', '_')}_{mode}.pdf" if mode == "paper" else f"orbit_{name.replace(' ', '_')}_{mode}.png"
    plt.savefig(FIG_DIR / fname)
    plt.close()
    print(f"  Saved {fname}")


# ============================================================
# Plot 4: GIF animation
# ============================================================

def make_orbit_gif(name, v1, v2, T, mode="blog", n_steps=500, n_frames=100):
    """Animated GIF of an orbit with glowing trails (blog mode only)."""
    try:
        import matplotlib.animation as animation
        from matplotlib.colors import to_rgba
    except ImportError:
        print(f"  Skipping GIF for {name} (matplotlib animation not available)")
        return

    q_trace = trace_orbit(v1, v2, T, n_steps)
    q_np = np.array(q_trace)

    if mode == "paper":
        set_paper_style()
        colors = ["black", "#606060", "#b0b0b0"]
        bg = "white"
    else:
        set_blog_style()
        colors = ["#ff4060", "#40ff80", "#60a0ff"]
        bg = "#0a0a12"

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(name)

    # Lines for trails, points for bodies
    trails = [ax.plot([], [], color=colors[i], lw=1.5, alpha=0.7)[0] for i in range(3)]
    points = [ax.plot([], [], "o", color=colors[i], ms=12,
                      mec="white" if mode == "blog" else "black", mew=1.5)[0]
              for i in range(3)]

    frame_step = max(1, n_steps // n_frames)

    def init():
        for t in trails:
            t.set_data([], [])
        for p in points:
            p.set_data([], [])
        return trails + points

    def animate(frame):
        idx = min((frame + 1) * frame_step, n_steps + 1)
        for i in range(3):
            trails[i].set_data(q_np[:idx, i, 0], q_np[:idx, i, 1])
            points[i].set_data([q_np[idx - 1, i, 0]], [q_np[idx - 1, i, 1]])
        return trails + points

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                                    interval=50, blit=True)

    fname = f"orbit_{name.replace(' ', '_')}_{mode}.gif"
    try:
        ani.save(FIG_DIR / fname, writer="pillow", fps=20)
        print(f"  Saved {fname}")
    except Exception as e:
        print(f"  Failed to save GIF: {e}")

    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("VISUALIZATION: Paper and Blog figures")
    print(f"Output: {FIG_DIR}")
    print("=" * 60)

    # Check for data from velocity search
    results_path = OUT_DIR / "velocity_space_results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        print(f"Loaded velocity search results: {results['n_verified']} verified orbits")
    else:
        results = None
        print("No velocity_space_results.json yet -- making Suvakov plots only")

    # Plot known Suvakov orbits (always works)
    print("\n=== Known Suvakov orbits ===")
    for name, v1, v2, T in SUVAKOV[:6]:  # First 6 to save time
        print(f"\n{name}:")
        plot_orbit(name, v1, v2, T, mode="paper")
        plot_orbit(name, v1, v2, T, mode="blog")

    # Make GIFs for top 3 famous orbits (takes longer)
    print("\n=== Animated GIFs (blog mode) ===")
    for name, v1, v2, T in SUVAKOV[:3]:  # Figure-eight, Butterfly I, II
        print(f"\n{name} GIF:")
        make_orbit_gif(name, v1, v2, T, mode="blog")

    # If basin map data exists, plot it
    basin_path = OUT_DIR / "basin_map.npz"
    if basin_path.exists():
        print("\n=== Velocity-space basin map ===")
        d = np.load(basin_path)
        labels_grid = d["labels"].reshape(500, 500)
        entropy_grid = d["entropy"].reshape(500, 500)
        v1_grid = d["v1_grid"]
        v2_grid = d["v2_grid"]
        plot_basin_map(labels_grid, v1_grid, v2_grid, mode="paper")
        plot_basin_map(labels_grid, v1_grid, v2_grid, mode="blog")
        plot_entropy_map(entropy_grid, v1_grid, v2_grid, mode="paper")
        plot_entropy_map(entropy_grid, v1_grid, v2_grid, mode="blog")

    print(f"\nDone. Figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
