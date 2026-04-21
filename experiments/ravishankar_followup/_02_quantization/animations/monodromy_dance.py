"""6s loop showing monodromy eigenvalues on the unit circle.

For elliptic orbits C, D: the non-trivial eigenvalues rotate around the unit
circle at their known omega_perp frequencies (decorative -- not the true Floquet path).
For hyperbolic A, B: the non-trivial eigenvalue sits static at its real value,
while the other 10 trivial ones stay at +1.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from experiments.ravishankar_followup._00_style import save_gif_mp4, ORBIT

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.resolve().parents[2]


def build():
    ab = json.loads((ROOT / "experiments/orbit_verification/11_monodromy/monodromy_results.json").read_text())
    cd = json.loads((HERE / "monodromy_CD.json").read_text())
    mono = {**ab, **cd}

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), dpi=100)
    for ax, name in zip(axes.flat, "ABCD"):
        ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), color="#888", linewidth=0.8)
        ax.set_title(f"{name}: {mono[name].get('classification', '?')}",
                     color=ORBIT[name], fontweight="bold", fontsize=11)
        ax.axhline(0, color="#CCCCCC", linewidth=0.4)
        ax.axvline(0, color="#CCCCCC", linewidth=0.4)

    fig.suptitle("Monodromy eigenvalues on the complex plane (decorative time evolution)",
                 fontsize=13)
    fig.tight_layout()

    # Per-orbit scatter handles
    handles = {}
    for ax, name in zip(axes.flat, "ABCD"):
        entry = mono[name]
        re = np.array(entry["eigenvalues_real"])
        im = np.array(entry["eigenvalues_imag"])
        sc = ax.scatter(re, im, s=30, c=ORBIT[name], edgecolors="black",
                        linewidths=0.6, zorder=10)
        handles[name] = (sc, re, im, entry)

    n_frames = 180  # 6s at 30 fps

    def update(f):
        phase = 2 * np.pi * f / n_frames
        for name, (sc, re0, im0, entry) in handles.items():
            # Rotate non-trivial pair; leave trivial at (+1, 0)
            mag = np.asarray(entry["eigenvalue_magnitudes"])
            log_mag = np.log(np.maximum(mag, 1e-15))
            new_re = re0.copy().astype(float); new_im = im0.copy().astype(float)
            if entry.get("classification") == "linearly stable":
                # Rotate the non-trivial (|arg|>1e-4) pairs by `phase`
                for i, (r, im_) in enumerate(zip(re0, im0)):
                    if abs(np.arctan2(im_, r)) > 1e-4:
                        new_r = r * np.cos(phase) - im_ * np.sin(phase)
                        new_i = r * np.sin(phase) + im_ * np.cos(phase)
                        new_re[i] = new_r; new_im[i] = new_i
            # Hyperbolic: leave static (already plotted correctly)
            sc.set_offsets(np.c_[new_re, new_im])
        return []

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False, interval=1000/30)
    return fig, anim


if __name__ == "__main__":
    fig, anim = build()
    save_gif_mp4(anim, "monodromy_dance_AB", HERE / "animations", fps=30)
    plt.close(fig)
    print("monodromy_dance_ok")
