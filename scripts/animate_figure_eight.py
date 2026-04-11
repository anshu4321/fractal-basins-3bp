"""Aesthetic animation of the Chenciner-Montgomery figure-eight orbit.

Produces:
    figures/figure_eight.mp4   (if ffmpeg present)
    figures/figure_eight.gif   (always, for universal playback)
    figures/figure_eight.png   (the final still)

Visual language:
  - Deep navy background with a subtle radial vignette
  - Three bodies rendered as glowing dots: each a layered scatter
      (big halo with low alpha + solid core)
  - A fading trail behind each body (LineCollection with per-segment alpha
      decaying along the history)
  - Thin time scale strip and elapsed-time counter
  - Dirac palette (cyan / lime / lavender)
  - 60 fps, ~8 second loop covering one full period of the figure-eight
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
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgb

from mega3bp.integrators import yoshida6_integrate
from mega3bp.orbits import FIGURE_EIGHT_PERIOD, figure_eight_state
from mega3bp.style import BODY_COLORS, BODY_GLOW, PALETTE, apply_dirac_style


# --------------- settings ---------------------------------------------------

N_PERIODS = 1.0          # integrate this many figure-eight periods
H = 0.002                 # integrator step size (fine so the trail is smooth)
FPS = 60                  # animation frame rate
DURATION_S = 8.0          # total animation length
TRAIL_LENGTH = 900         # points held in each fading trail
BODY_DOT_SIZE = 150        # core dot area
BODY_HALO_SIZE = 1200      # halo area


def add_vignette(ax: plt.Axes, alpha: float = 0.55) -> None:
    """Soft radial darkening to focus attention."""
    nx = ny = 512
    xv, yv = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny))
    r = np.sqrt(xv ** 2 + yv ** 2)
    mask = np.clip((r - 0.55) / 0.55, 0.0, 1.0) ** 2
    rgba = np.zeros((ny, nx, 4))
    rgba[..., 3] = mask * alpha
    ax.imshow(
        rgba,
        extent=ax.get_xlim() + ax.get_ylim(),
        origin="lower",
        zorder=0.5,
        interpolation="bilinear",
    )


def build_fading_colormap(hex_base: str, hex_glow: str) -> LinearSegmentedColormap:
    """Colormap from fully transparent -> glow -> core."""
    base = np.array(to_rgb(hex_base))
    glow = np.array(to_rgb(hex_glow))
    stops = [
        (0.00, (*glow, 0.00)),
        (0.45, (*glow, 0.25)),
        (0.80, (*base, 0.55)),
        (1.00, (*base, 0.95)),
    ]
    return LinearSegmentedColormap.from_list("fade_" + hex_base.lstrip("#"), stops)


def main() -> int:
    apply_dirac_style()
    print(f"jax backend: {jax.default_backend()}")
    q0, p0 = figure_eight_state()

    n_steps = int(round(N_PERIODS * FIGURE_EIGHT_PERIOD / H))
    print(f"integrating {n_steps} steps ({N_PERIODS} periods)")
    t_start = time.perf_counter()
    q_trace, p_trace, energies = yoshida6_integrate(q0, p0, H, n_steps)
    q_trace.block_until_ready()
    print(f"  done in {time.perf_counter() - t_start:.2f}s")

    q_np = np.asarray(q_trace)   # shape (n_steps + 1, 3, 2)
    n_frames_total = q_np.shape[0]
    n_anim_frames = int(round(FPS * DURATION_S))
    stride = max(1, n_frames_total // n_anim_frames)
    frames = q_np[::stride]
    n_anim = frames.shape[0]
    print(f"animation: {n_anim} frames at {FPS} fps")

    # --- figure --------------------------------------------------------------
    fig = plt.figure(figsize=(8, 7), dpi=150)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor(PALETTE["bg_deep"])
    ax.set_facecolor(PALETTE["bg_deep"])  # override panel bg for clean look

    # Axis extent: pad around the full orbit
    xs = q_np[..., 0].flatten()
    ys = q_np[..., 1].flatten()
    pad = 0.3
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_ylim(ys.min() - pad, ys.max() + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    add_vignette(ax)

    # Faint reference orbit (full figure-eight), drawn once as a guide
    ref_path, = ax.plot(
        q_np[:, 0, 0], q_np[:, 0, 1],
        color=PALETTE["text_mute"], lw=0.3, alpha=0.25, zorder=1,
    )
    # Draw all three body paths very faintly as a reference
    for i in range(3):
        ax.plot(
            q_np[:, i, 0], q_np[:, i, 1],
            color=PALETTE["text_dim"], lw=0.25, alpha=0.15, zorder=1,
        )

    # --- Trails (LineCollection per body, per-segment alpha gradient) --------
    trail_collections = []
    for i in range(3):
        lc = LineCollection(
            [],
            linewidths=np.linspace(0.3, 2.3, TRAIL_LENGTH),
            zorder=5,
        )
        ax.add_collection(lc)
        trail_collections.append(lc)

    # --- Dots (halo + core, one scatter each, sized in points^2) -------------
    halos = []
    cores = []
    for i in range(3):
        halo = ax.scatter(
            [q_np[0, i, 0]], [q_np[0, i, 1]],
            s=BODY_HALO_SIZE, c=BODY_GLOW[i], alpha=0.22, zorder=6,
            edgecolors="none",
        )
        core = ax.scatter(
            [q_np[0, i, 0]], [q_np[0, i, 1]],
            s=BODY_DOT_SIZE, c=BODY_COLORS[i], alpha=0.95, zorder=7,
            edgecolors="white", linewidths=0.4,
        )
        halos.append(halo)
        cores.append(core)

    # --- Title / annotation bar ----------------------------------------------
    ax.text(
        0.5, 0.97,
        "Chenciner-Montgomery figure-eight",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=15, color=PALETTE["text"], fontweight="normal",
    )
    ax.text(
        0.5, 0.935,
        r"planar equal-mass 3-body choreography",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=9, color=PALETTE["text_mute"],
    )
    time_text = ax.text(
        0.98, 0.035,
        "",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, color=PALETTE["cyan_glow"],
        family="monospace",
    )
    body_text = ax.text(
        0.02, 0.035,
        "body 1  ·  body 2  ·  body 3",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=9, color=PALETTE["text_dim"],
        family="monospace",
    )

    # Pre-compute the per-segment alpha for trails: fades along the history
    # Segment widths and alphas fade from 0 (oldest) -> 1 (newest)
    alpha_gradient = np.linspace(0.0, 0.95, TRAIL_LENGTH - 1)

    def segs_from_window(win: np.ndarray) -> np.ndarray:
        """Build line segments from a (N, 2) point array. Returns (N-1, 2, 2)."""
        return np.stack([win[:-1], win[1:]], axis=1)

    rgb_colors = [np.array(to_rgb(c)) for c in BODY_COLORS]

    def update(frame_idx: int):
        art = []
        cur = frames[frame_idx]  # shape (3, 2)
        for i in range(3):
            # Window indices into the frames history
            lo = max(0, frame_idx - TRAIL_LENGTH)
            win = frames[lo:frame_idx + 1, i, :]
            if win.shape[0] >= 2:
                segs = segs_from_window(win)
                n_segs = segs.shape[0]
                alphas = alpha_gradient[-n_segs:]
                rgba = np.zeros((n_segs, 4))
                rgba[:, :3] = rgb_colors[i]
                rgba[:, 3] = alphas
                trail_collections[i].set_segments(segs)
                trail_collections[i].set_color(rgba)
                trail_collections[i].set_linewidths(np.linspace(0.3, 2.3, n_segs))
            else:
                trail_collections[i].set_segments([])

            halos[i].set_offsets([[cur[i, 0], cur[i, 1]]])
            cores[i].set_offsets([[cur[i, 0], cur[i, 1]]])
            art.append(trail_collections[i])
            art.append(halos[i])
            art.append(cores[i])

        t_now = frame_idx * stride * H
        time_text.set_text(f"t / T = {t_now / FIGURE_EIGHT_PERIOD:6.3f}")
        art.append(time_text)
        return art

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=n_anim,
        interval=1000 / FPS,
        blit=True,
    )

    out = PROJECT_ROOT / "figures"
    out.mkdir(exist_ok=True)

    # Save MP4 if ffmpeg is available
    mp4_path = out / "figure_eight.mp4"
    gif_path = out / "figure_eight.gif"
    png_path = out / "figure_eight.png"

    try:
        writer = animation.FFMpegWriter(fps=FPS, bitrate=6000, codec="libx264")
        print(f"writing {mp4_path} ...")
        t0 = time.perf_counter()
        ani.save(str(mp4_path), writer=writer, dpi=150, savefig_kwargs={"facecolor": PALETTE["bg_deep"]})
        print(f"  done in {time.perf_counter() - t0:.1f}s")
    except Exception as e:
        print(f"  ffmpeg not available ({e}); skipping mp4")

    # Gif skipped by default — PillowWriter is 10-20x slower than FFMpegWriter
    # for the same quality and bloats output. Pass --gif to force it.
    if "--gif" in sys.argv:
        try:
            writer = animation.PillowWriter(fps=max(FPS // 2, 15))
            print(f"writing {gif_path} ...")
            t0 = time.perf_counter()
            ani.save(str(gif_path), writer=writer, dpi=100, savefig_kwargs={"facecolor": PALETTE["bg_deep"]})
            print(f"  done in {time.perf_counter() - t0:.1f}s")
        except Exception as e:
            print(f"  gif save failed ({e})")

    # Final still
    update(n_anim - 1)
    fig.savefig(png_path, facecolor=PALETTE["bg_deep"])
    print(f"wrote {png_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
