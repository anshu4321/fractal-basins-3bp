"""Animation helpers: comet-tail trajectory writer and dual GIF+MP4 output."""
from pathlib import Path
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from .palette import BODY


def save_gif_mp4(anim, stem, outdir, fps=30, colors=128,
                 max_size_mb=8):
    """Save a matplotlib animation as both `<stem>.gif` AND `<stem>.mp4`.

    GIF is optimised with gifsicle (if available) or Pillow (fallback).
    MP4 is H.264 CRF 18 via ffmpeg (must be on PATH).
    If the GIF exceeds `max_size_mb`, prints a WARNING (does not error).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    gif_path = outdir / f"{stem}.gif"
    mp4_path = outdir / f"{stem}.mp4"

    # --- GIF via Pillow (matplotlib backend), then gifsicle if present
    anim.save(str(gif_path), writer="pillow", fps=fps)

    # Optimise with gifsicle if available
    try:
        subprocess.run(
            ["gifsicle", "-O3", f"--colors={colors}", "-b", str(gif_path)],
            check=True, capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass  # fallback: unoptimised Pillow output

    size_mb = gif_path.stat().st_size / 1024 / 1024
    if size_mb > max_size_mb:
        print(f"WARNING: {gif_path} is {size_mb:.1f}MB (target {max_size_mb}MB)")

    # --- MP4 via ffmpeg
    try:
        writer = animation.FFMpegWriter(fps=fps, codec="libx264",
                                        extra_args=["-crf", "18",
                                                    "-pix_fmt", "yuv420p"])
        anim.save(str(mp4_path), writer=writer)
    except Exception as e:
        print(f"WARNING: could not write MP4 ({e}); GIF-only for {stem}")

    return gif_path, mp4_path


def comet_tail_positions(traj, t_idx, tail_len):
    """Return (N_tail, 2) positions and per-point alpha values for body trail.

    traj: (T_steps, 2) position array for one body.
    t_idx: current step index.
    tail_len: how many prior steps to draw as fading trail.
    Returns (pts, alphas) where alphas linearly ramp from 0.1 to 1.0.
    """
    start = max(0, t_idx - tail_len)
    pts = traj[start:t_idx + 1]
    n = len(pts)
    alphas = np.linspace(0.1, 1.0, n)
    return pts, alphas


def make_orbit_trail_frames(traj3, T, n_frames=240, tail_frac=0.125):
    """Generate (n_frames) frames for a 3-body comet-tail orbit trail.

    traj3 : ndarray, shape (T_steps, 3, 2)
        Positions of the 3 bodies along one period.
    T : float
        The period (for title/time display).
    n_frames : int
        Number of animation frames to subsample.
    tail_frac : float
        Fraction of one period used as tail length (0.125 = T/8).

    Yields (fig, ax) per frame for the caller to write via FuncAnimation.
    (The caller typically uses FuncAnimation + this helper's update function.)
    """
    T_steps = traj3.shape[0]
    tail_len = int(tail_frac * T_steps)
    frame_idxs = np.linspace(0, T_steps - 1, n_frames).astype(int)
    return frame_idxs, tail_len


def update_orbit_trail(ax, traj3, t_idx, tail_len, T,
                       xlim=None, ylim=None):
    """Redraw one frame of the comet-tail animation on `ax`.

    Call from FuncAnimation's update() with the current t_idx. Clears the
    axes and re-plots trails + body positions + time indicator.
    """
    ax.cla()
    for i in range(3):
        pts, alphas = comet_tail_positions(traj3[:, i, :], t_idx, tail_len)
        for j in range(len(pts) - 1):
            ax.plot(pts[j:j+2, 0], pts[j:j+2, 1],
                    color=BODY[i+1], alpha=alphas[j], linewidth=1.6)
        ax.scatter(pts[-1, 0], pts[-1, 1], s=120, c=BODY[i+1],
                   edgecolors="black", linewidths=0.7, zorder=10)
    ax.set_aspect("equal")
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)
    t_frac = t_idx / (traj3.shape[0] - 1)
    ax.text(0.98, 0.02, f"t/T = {t_frac:.2f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10, color="#566573")
