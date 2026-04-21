"""Emit alpha_sweep_{A,B,C,D}.gif + .mp4 - 6s loop of each orbit rescaling."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from experiments.ravishankar_followup._00_style import (
    save_gif_mp4, BODY, ORBIT,
)
from experiments.ravishankar_followup._01_census.figures.scaling_families import (
    integrate_one_period, ICS,
)

HERE = Path(__file__).resolve().parent.parent


def build_anim(orbit_name, fps=30, duration_s=6):
    traj = integrate_one_period(orbit_name)  # (500, 3, 2)
    T, E = ICS[orbit_name]["T"], ICS[orbit_name]["E"]
    Ts = T * abs(E) ** 1.5
    n_frames = fps * duration_s

    alphas = np.concatenate([
        np.linspace(0.3, 3.0, n_frames // 2),
        np.linspace(3.0, 0.3, n_frames - n_frames // 2),
    ])

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    max_r = 3.0 * np.abs(traj).max() * 1.1
    ax.set_xlim(-max_r, max_r); ax.set_ylim(-max_r, max_r)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(f"{orbit_name}: α-scaling family", fontsize=14,
                 color=ORBIT[orbit_name], fontweight="bold")
    info = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
                   fontsize=10, color="#2c3e50",
                   bbox=dict(facecolor="white", edgecolor="none", alpha=0.75))
    trails = [ax.plot([], [], color=BODY[i+1], linewidth=1.8)[0]
              for i in range(3)]

    def update(frame_idx):
        alpha = alphas[frame_idx]
        for i in range(3):
            trails[i].set_data(alpha * traj[:, i, 0], alpha * traj[:, i, 1])
        T_a = alpha**1.5 * T
        E_a = E / alpha
        info.set_text(f"α = {alpha:.2f}\nT(α) = {T_a:.2f}\nE(α) = {E_a:.3f}\n"
                      f"T* = {Ts:.2f} (invariant)")
        return trails + [info]

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False,
                         interval=1000 / fps)
    return fig, anim


if __name__ == "__main__":
    outdir = HERE / "animations"
    for name in "ABCD":
        print(f"Rendering alpha_sweep_{name}...")
        fig, anim = build_anim(name)
        save_gif_mp4(anim, f"alpha_sweep_{name}", outdir, fps=30)
        plt.close(fig)
    print("alpha_sweep_all_ok")
