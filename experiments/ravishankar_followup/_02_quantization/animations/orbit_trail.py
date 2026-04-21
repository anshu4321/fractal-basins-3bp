"""4 orbit-trail GIFs (comet-tail, 8s loop each) for A, B, C, D."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from experiments.ravishankar_followup._00_style import (
    save_gif_mp4, update_orbit_trail, BODY, ORBIT,
)
from experiments.ravishankar_followup._01_census.figures.scaling_families import (
    integrate_one_period, ICS,
)

HERE = Path(__file__).resolve().parent.parent


def build_orbit_trail(name, fps=30, duration_s=8):
    traj = integrate_one_period(name)  # (500, 3, 2)
    T = ICS[name]["T"]
    n_frames = fps * duration_s
    tail_len = max(5, traj.shape[0] // 8)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    max_r = np.abs(traj).max() * 1.15
    xlim = (-max_r, max_r); ylim = (-max_r, max_r)

    def update(f):
        # Map f in [0, n_frames) -> t_idx in [0, traj.shape[0])
        t_idx = int((f / (n_frames - 1)) * (traj.shape[0] - 1))
        update_orbit_trail(ax, traj, t_idx, tail_len, T, xlim=xlim, ylim=ylim)
        ax.set_title(f"Orbit {name}: T = {T:.2f}", color=ORBIT[name],
                     fontweight="bold", fontsize=14)

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False, interval=1000/fps)
    return fig, anim


if __name__ == "__main__":
    for name in "ABCD":
        print(f"rendering orbit_trail_{name} ...")
        fig, anim = build_orbit_trail(name)
        save_gif_mp4(anim, f"orbit_trail_{name}", HERE / "animations", fps=30)
        plt.close(fig)
    print("orbit_trail_all_ok")
