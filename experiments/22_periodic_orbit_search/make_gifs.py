"""Generate beautiful orbit GIFs for discovered periodic orbits.

Each GIF shows three colored bodies moving through space with fading trails.
The start configuration is marked, and the near-return is highlighted.
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
from matplotlib.patches import Circle
from PIL import Image
import io

from mega3bp.shape_sphere import shape_to_config, config_to_shape
from mega3bp.integrators import yoshida6_step

H = 0.003
TRAIL_LENGTH = 80
FPS = 30
BODY_COLORS = ["#0072B2", "#E69F00", "#009E73"]
BODY_NAMES = ["Body 1", "Body 2", "Body 3"]
BG_COLOR = "#0a0a1a"
GRID_COLOR = "#1a1a3a"


def integrate_trajectory(n0, n_steps):
    """Integrate from shape n0 at rest, return full q trace."""
    q0 = shape_to_config(jnp.array(n0, dtype=jnp.float64), inertia=1.0)
    p0 = jnp.zeros_like(q0)
    h = jnp.float64(H)

    @jax.jit
    def run(q0, p0):
        def step_fn(carry, _):
            q, p = carry
            q, p = yoshida6_step(q, p, h)
            return (q, p), q
        (_, _), q_trace = jax.lax.scan(step_fn, (q0, p0), None, length=n_steps)
        return q_trace

    q_trace = run(q0, p0)
    q_all = np.concatenate([np.array(q0)[None], np.array(q_trace)], axis=0)
    return q_all


def make_orbit_gif(q_trace, orbit_info, save_path, max_frames=200):
    """Create a beautiful GIF of a three-body orbit."""
    total_steps = len(q_trace)
    step_skip = max(1, total_steps // max_frames)
    frame_indices = list(range(0, total_steps, step_skip))
    if frame_indices[-1] != total_steps - 1:
        frame_indices.append(total_steps - 1)

    # Compute bounds
    all_x = q_trace[:, :, 0].flatten()
    all_y = q_trace[:, :, 1].flatten()
    cx, cy = np.mean(all_x), np.mean(all_y)
    span = max(np.ptp(all_x), np.ptp(all_y)) * 0.65
    span = max(span, 0.3)

    cls = orbit_info.get("classification", "?")
    T = orbit_info.get("period", 0)
    dist = orbit_info.get("return_dist_refined", 0)

    frames = []
    for fi, step in enumerate(frame_indices):
        fig, ax = plt.subplots(figsize=(5, 5), facecolor=BG_COLOR)
        ax.set_facecolor(BG_COLOR)

        ax.set_xlim(cx - span, cx + span)
        ax.set_ylim(cy - span, cy + span)
        ax.set_aspect("equal")
        ax.grid(True, color=GRID_COLOR, linewidth=0.3, alpha=0.5)
        ax.tick_params(colors="#555555", labelsize=6)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)

        # Draw trails with fading alpha
        trail_start = max(0, step - TRAIL_LENGTH * step_skip)
        trail = q_trace[trail_start:step + 1]

        for body in range(3):
            if len(trail) > 1:
                x = trail[:, body, 0]
                y = trail[:, body, 1]
                n_seg = len(x) - 1
                for i in range(n_seg):
                    alpha = 0.1 + 0.7 * (i / max(n_seg, 1))
                    ax.plot(x[i:i+2], y[i:i+2], color=BODY_COLORS[body],
                            linewidth=1.5, alpha=alpha)

        # Draw current body positions as glowing dots
        for body in range(3):
            x, y = q_trace[step, body, 0], q_trace[step, body, 1]
            # Outer glow
            ax.plot(x, y, "o", color=BODY_COLORS[body], markersize=12,
                    alpha=0.3, markeredgewidth=0)
            # Inner body
            ax.plot(x, y, "o", color=BODY_COLORS[body], markersize=7,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=5)

        # Draw starting positions as small diamonds
        if step > TRAIL_LENGTH * step_skip:
            for body in range(3):
                x0, y0 = q_trace[0, body, 0], q_trace[0, body, 1]
                ax.plot(x0, y0, "D", color=BODY_COLORS[body], markersize=4,
                        alpha=0.4, markeredgecolor="white", markeredgewidth=0.3)

        # Title
        progress = step / max(total_steps - 1, 1)
        t_now = step * H
        ax.set_title(f"{cls}   T = {t_now:.2f} / {T:.2f}",
                      color="white", fontsize=10, pad=8)

        # Legend (first frame only shows labels)
        if fi == 0:
            for body in range(3):
                ax.plot([], [], "o", color=BODY_COLORS[body], label=BODY_NAMES[body])
            ax.legend(facecolor=BG_COLOR, edgecolor=GRID_COLOR,
                      labelcolor="white", fontsize=7, loc="upper right")

        fig.tight_layout()

        # Render to PIL Image
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, facecolor=BG_COLOR)
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()
        plt.close(fig)

    # Add a few frozen frames at the end to show the return
    for _ in range(10):
        frames.append(frames[-1])

    # Save GIF
    frames[0].save(
        save_path, save_all=True, append_images=frames[1:],
        duration=1000 // FPS, loop=0, optimize=True)

    print(f"  Saved {save_path.name} ({len(frames)} frames)")


def main():
    results_path = PROJECT_ROOT / "experiments" / "22_periodic_orbit_search" / "results.json"
    gif_dir = PROJECT_ROOT / "figures" / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)

    with open(results_path) as f:
        data = json.load(f)

    refined = data["refined"]

    # Only animate the near-periodic orbits + best candidates
    orbits_to_animate = []
    for i, orb in enumerate(refined):
        if orb["classification"] == "NEAR-PERIODIC" or orb["return_dist_refined"] < 0.02:
            orbits_to_animate.append((i, orb))

    print(f"Animating {len(orbits_to_animate)} orbits")

    for idx, (i, orb) in enumerate(orbits_to_animate):
        n0 = np.array(orb["n0_refined"])
        period_step = int(round(orb["period"] / H))
        # Integrate for 1.1 periods to show the near-return
        n_steps = int(period_step * 1.1)

        print(f"\nOrbit #{i+1}: {orb['classification']}, T={orb['period']:.3f}")
        print(f"  Integrating {n_steps} steps...")
        q_trace = integrate_trajectory(n0, n_steps)

        make_orbit_gif(
            q_trace, orb,
            gif_dir / f"orbit_{i+1}_{orb['classification'].lower()}.gif",
            max_frames=200)

    print(f"\nAll GIFs saved to {gif_dir}")


if __name__ == "__main__":
    main()
