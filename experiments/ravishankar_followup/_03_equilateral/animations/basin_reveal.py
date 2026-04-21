"""10s column-by-column reveal of the 256x256 basin."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from experiments.ravishankar_followup._00_style import save_gif_mp4

HERE = Path(__file__).resolve().parent.parent


def build():
    data = np.load(HERE / "basin_map.npz")
    labels = data["labels"]
    N = labels.shape[0]
    cmap = plt.cm.colors.ListedColormap(["#F1C40F", "#3498DB", "#E74C3C", "#27AE60"])
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    ax.set_title("Equilateral basin reveal (256² on A100)", fontsize=12)

    im = ax.imshow(np.zeros_like(labels).T, origin="lower", cmap=cmap,
                   aspect="equal", extent=(-2, 2, -2, 2),
                   vmin=0, vmax=3, interpolation="nearest")

    n_frames = 300

    def update(f):
        cols_revealed = int((f + 1) / n_frames * N)
        canvas = np.full_like(labels, -1, dtype=np.int8)
        canvas[:cols_revealed, :] = labels[:cols_revealed, :]
        im.set_data(np.where(canvas >= 0, canvas, 3).T)
        return [im]

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False, interval=1000/30)
    return fig, anim


if __name__ == "__main__":
    fig, anim = build()
    save_gif_mp4(anim, "basin_reveal", HERE / "animations", fps=30)
    plt.close(fig)
    print("basin_reveal_ok")
