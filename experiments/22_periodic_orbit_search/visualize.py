"""Visualize discovered periodic orbits.

Produces two figures per orbit:
  1. Physical-space trajectory: three colored trails showing each body's path
  2. Shape-sphere trajectory: the closed (or near-closed) loop on S^2

Also produces a summary panel of all discovered orbits.
Reads results.json from the search script.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

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
from mpl_toolkits.mplot3d import Axes3D

from mega3bp.shape_sphere import shape_to_config, config_to_shape, pairwise_distances
from mega3bp.integrators import yoshida6_step
from mega3bp.dynamics import total_energy

WONG = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "cyan": "#56B4E9",
}
BODY_COLORS = [WONG["blue"], WONG["orange"], WONG["green"]]

H = 0.003


def integrate_full_trace(n0, n_steps):
    """Integrate from shape n0 at rest, return full (q, p, shape) trace."""
    q0 = shape_to_config(jnp.array(n0, dtype=jnp.float64), inertia=1.0)
    p0 = jnp.zeros_like(q0)
    h = jnp.float64(H)

    @jax.jit
    def run(q0, p0):
        def step_fn(carry, _):
            q, p = carry
            q, p = yoshida6_step(q, p, h)
            n = config_to_shape(q)
            return (q, p), (q, n)
        (qf, pf), (q_trace, n_trace) = jax.lax.scan(
            step_fn, (q0, p0), None, length=n_steps)
        return q_trace, n_trace

    q_trace, n_trace = run(q0, p0)
    q_all = np.concatenate([np.array(q0)[None], np.array(q_trace)], axis=0)
    n_all = np.concatenate([np.array(n0)[None], np.array(n_trace)], axis=0)
    return q_all, n_all


def plot_physical_trajectory(q_trace, period_step, orbit_info, save_path):
    """Plot three body trajectories in the plane, colored by body."""
    q = q_trace[:period_step + 1]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_aspect("equal")

    for body in range(3):
        x = q[:, body, 0]
        y = q[:, body, 1]
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        t = np.linspace(0, 1, len(segments))
        colors = plt.cm.viridis(t) if body == 0 else (
            plt.cm.plasma(t) if body == 1 else plt.cm.cividis(t))
        lc = LineCollection(segments, colors=colors, linewidths=0.8, alpha=0.8)
        ax.add_collection(lc)

        ax.plot(x[0], y[0], "o", color=BODY_COLORS[body], markersize=8,
                markeredgecolor="white", markeredgewidth=0.5, zorder=5)
        ax.plot(x[-1], y[-1], "s", color=BODY_COLORS[body], markersize=6,
                markeredgecolor="white", markeredgewidth=0.5, zorder=5,
                alpha=0.6)

    all_x = q[:, :, 0].flatten()
    all_y = q[:, :, 1].flatten()
    pad = 0.15 * max(np.ptp(all_x), np.ptp(all_y))
    ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
    ax.set_ylim(all_y.min() - pad, all_y.max() + pad)

    cls = orbit_info.get("classification", "?")
    dist = orbit_info.get("return_dist_refined", orbit_info.get("return_dist_original", 0))
    T = orbit_info.get("period", 0)
    E = orbit_info.get("energy", 0)

    ax.set_title(f"{cls}  |  T = {T:.3f}  |  return dist = {dist:.5f}  |  E = {E:.4f}",
                 fontsize=9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.2)

    for body in range(3):
        ax.plot([], [], color=BODY_COLORS[body], linewidth=2,
                label=f"Body {body+1}")
    ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_shape_sphere_trajectory(n_trace, period_step, orbit_info, save_path):
    """Plot the trajectory on the shape sphere (3D view)."""
    n = n_trace[:period_step + 1]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    xs = np.outer(np.cos(u), np.sin(v)) * 0.99
    ys = np.outer(np.sin(u), np.sin(v)) * 0.99
    zs = np.outer(np.ones_like(u), np.cos(v)) * 0.99
    ax.plot_surface(xs, ys, zs, alpha=0.05, color="gray")

    t = np.linspace(0, 1, len(n))
    colors = plt.cm.plasma(t)
    for i in range(len(n) - 1):
        ax.plot(n[i:i+2, 0], n[i:i+2, 1], n[i:i+2, 2],
                color=colors[i], linewidth=1.2, alpha=0.8)

    ax.scatter(*n[0], color=WONG["green"], s=60, zorder=5,
              edgecolors="white", linewidths=0.5, label="Start")
    ax.scatter(*n[-1], color=WONG["red"], s=40, zorder=5, marker="s",
              edgecolors="white", linewidths=0.5, label="End")

    collisions = np.array([[-1, 0, 0], [0.5, -0.866, 0], [0.5, 0.866, 0]])
    for i, (c, lbl) in enumerate(zip(collisions, ["$C_{12}$", "$C_{13}$", "$C_{23}$"])):
        ax.scatter(*c, color=WONG["red"], s=30, marker="x", zorder=4)
        ax.text(c[0]*1.15, c[1]*1.15, c[2]*1.15, lbl, fontsize=7)

    ax.scatter(0, 0, 1, color=WONG["cyan"], s=30, marker="^", zorder=4)
    ax.scatter(0, 0, -1, color=WONG["cyan"], s=30, marker="v", zorder=4)
    ax.text(0.05, 0.05, 1.1, "$L_+$", fontsize=7)
    ax.text(0.05, 0.05, -1.15, "$L_-$", fontsize=7)

    cls = orbit_info.get("classification", "?")
    T = orbit_info.get("period", 0)
    ax.set_title(f"Shape sphere: {cls}, T = {T:.3f}", fontsize=9)
    ax.legend(fontsize=7)
    ax.set_xlabel("$n_1$", fontsize=8)
    ax.set_ylabel("$n_2$", fontsize=8)
    ax.set_zlabel("$n_3$", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_summary(all_orbits, save_path):
    """Summary panel: one subplot per orbit showing physical trajectory."""
    n_orbits = len(all_orbits)
    if n_orbits == 0:
        print("No orbits to plot")
        return

    cols = min(4, n_orbits)
    rows = (n_orbits + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if n_orbits == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for i, orb in enumerate(all_orbits):
        ax = axes[i // cols, i % cols]
        ax.set_aspect("equal")

        q = orb["q_trace"]
        step = orb["period_step"]
        q_sub = q[:step + 1]

        for body in range(3):
            ax.plot(q_sub[:, body, 0], q_sub[:, body, 1],
                    color=BODY_COLORS[body], linewidth=0.6, alpha=0.7)
            ax.plot(q_sub[0, body, 0], q_sub[0, body, 1], "o",
                    color=BODY_COLORS[body], markersize=5,
                    markeredgecolor="white", markeredgewidth=0.3)

        cls = orb["info"].get("classification", "?")
        T = orb["info"].get("period", 0)
        dist = orb["info"].get("return_dist_refined", 0)
        ax.set_title(f"#{i+1} {cls}\nT={T:.2f}  d={dist:.4f}", fontsize=8)
        ax.grid(True, alpha=0.15)
        ax.tick_params(labelsize=6)

    for i in range(n_orbits, rows * cols):
        axes[i // cols, i % cols].set_visible(False)

    fig.suptitle(f"Discovered Periodic Orbits ({n_orbits} found)", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    out_dir = PROJECT_ROOT / "experiments" / "22_periodic_orbit_search"
    fig_dir = PROJECT_ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)

    results_path = out_dir / "results.json"
    if not results_path.exists():
        print("ERROR: results.json not found. Run run.py first.")
        return 1

    with open(results_path) as f:
        data = json.load(f)

    refined = data.get("refined", [])
    if not refined:
        print("No refined orbits found.")
        return 1

    print(f"Visualizing {len(refined)} refined orbits")

    all_orbit_data = []
    for i, orb in enumerate(refined):
        n0 = np.array(orb["n0_refined"])
        period_step = int(round(orb["period"] / H))

        extra_steps = min(int(period_step * 1.1), period_step + 200)
        print(f"\n  Orbit {i+1}: {orb['classification']}, T={orb['period']:.3f}, "
              f"integrating {extra_steps} steps...")

        q_trace, n_trace = integrate_full_trace(n0, extra_steps)

        plot_physical_trajectory(
            q_trace, period_step, orb,
            fig_dir / f"periodic_orbit_{i+1}_physical.png")

        plot_shape_sphere_trajectory(
            n_trace, period_step, orb,
            fig_dir / f"periodic_orbit_{i+1}_shape_sphere.png")

        all_orbit_data.append({
            "q_trace": q_trace, "n_trace": n_trace,
            "period_step": period_step, "info": orb,
        })
        print(f"  Saved figures for orbit {i+1}")

    plot_summary(all_orbit_data, fig_dir / "periodic_orbits_summary.png")
    print(f"\nSaved summary to figures/periodic_orbits_summary.png")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
