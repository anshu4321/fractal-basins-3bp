"""8s smooth morph from Euler r=(-1,0),(1,0),(0,0) to equilateral vertices."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from experiments.ravishankar_followup._00_style import save_gif_mp4, BODY

HERE = Path(__file__).resolve().parent.parent


def build():
    r_euler = np.array([[-1, 0], [1, 0], [0, 0]], dtype=float)
    R = 1 / np.sqrt(3)
    r_eq = np.array([[R*np.cos(2*np.pi*k/3), R*np.sin(2*np.pi*k/3)]
                     for k in range(3)])

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Euler section → equilateral section", fontsize=12)
    for spine in ax.spines.values(): spine.set_visible(False)

    bodies = [ax.scatter([r_euler[i, 0]], [r_euler[i, 1]], s=400, c=BODY[i+1],
                         edgecolors="black", linewidths=0.8, zorder=10)
              for i in range(3)]
    label_txt = ax.text(0.02, 0.98, "Euler section", transform=ax.transAxes,
                        va="top", ha="left", fontsize=11,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))

    n_frames = 240
    hold_frames = 30

    def update(f):
        if f < hold_frames:
            t = 0.0
        elif f > n_frames - hold_frames:
            t = 1.0
        else:
            t = (f - hold_frames) / (n_frames - 2 * hold_frames)
            t = 0.5 - 0.5 * np.cos(np.pi * t)
        r_now = (1 - t) * r_euler + t * r_eq
        for i in range(3):
            bodies[i].set_offsets([r_now[i]])
        label_txt.set_text(("Euler section" if t < 0.5 else
                             "Equilateral section") + f" (morph={t:.2f})")
        return bodies + [label_txt]

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False, interval=1000/30)
    return fig, anim


if __name__ == "__main__":
    fig, anim = build()
    save_gif_mp4(anim, "equilateral_vs_euler", HERE / "animations", fps=30)
    plt.close(fig)
    print("equilateral_vs_euler_ok")
