"""8s loop: LD entropy field at classifier training epochs."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from experiments.ravishankar_followup._00_style import save_gif_mp4, seq_cmap

HERE = Path(__file__).resolve().parent.parent


def compute_ld_from_probs(probs):
    """Entropy from softmax probs (shape 256, 256, 4)."""
    return -np.sum(probs * np.log(probs + 1e-12), axis=-1)


def build():
    snap = np.load(HERE / "classifier_snapshots.npz")
    epochs = sorted(int(k.split("_")[1]) for k in snap.keys()
                    if k.startswith("epoch_"))
    ld_per_epoch = {
        e: compute_ld_from_probs(snap[f"epoch_{e}"]) for e in epochs
    }

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    im = ax.imshow(ld_per_epoch[epochs[0]].T, origin="lower", cmap=seq_cmap(),
                   aspect="equal", extent=(-2, 2, -2, 2),
                   vmin=0, vmax=np.log(4))
    ttl = ax.set_title(f"LD field growth: epoch {epochs[0]}", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.7, label="entropy")

    n_frames = 240

    def update(f):
        epoch_idx = int(f / n_frames * len(epochs))
        epoch_idx = min(epoch_idx, len(epochs) - 1)
        e = epochs[epoch_idx]
        im.set_data(ld_per_epoch[e].T)
        ttl.set_text(f"LD field growth: epoch {e}")
        return [im, ttl]

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False, interval=1000/30)
    return fig, anim


if __name__ == "__main__":
    fig, anim = build()
    save_gif_mp4(anim, "ld_growth", HERE / "animations", fps=30)
    plt.close(fig)
    print("ld_growth_ok")
