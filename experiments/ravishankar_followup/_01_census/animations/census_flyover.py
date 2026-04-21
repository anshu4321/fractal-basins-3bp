"""census_flyover.gif: 8s reveal of the (T*, E) scatter bucket-by-bucket,
with the final frames highlighting A, B, C, D with marker-grow."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from experiments.ravishankar_followup._00_style import (
    save_gif_mp4, TOL_BRIGHT, ORBIT,
)

HERE = Path(__file__).resolve().parent.parent

OUR = {"A": (-1.51880093609, 36.94001257740),
       "B": (-1.57045059002, 64.64865733920),
       "C": (-1.50430210713, 64.65542233190),
       "D": (-0.93930544448, 73.81478296716)}

BIN_EDGES = [10.0, 30.0, 50.0, 70.0, 100.0, float("inf")]
BIN_LABELS = [
    "T* < 30",
    "30 ≤ T* < 50",
    "50 ≤ T* < 70",
    "70 ≤ T* < 100",
    "T* ≥ 100",
]


def _bucket(T_star):
    for i, hi in enumerate(BIN_EDGES[1:]):
        if T_star < hi:
            return i
    return len(BIN_EDGES) - 2


def build():
    inv = json.loads((HERE / "hristov_2024_invariants.json").read_text())
    E_all = np.array([r["E"] for r in inv])
    TS_all = np.array([r["T_star"] for r in inv])
    mask = (E_all < 0) & np.isfinite(TS_all) & (TS_all > 0)
    E_all, TS_all = E_all[mask], TS_all[mask]
    bin_ids = np.array([_bucket(t) for t in TS_all])

    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    ax.set_xlim(9.0, 300.0); ax.set_xscale("log")
    y_lo = float(E_all.min()) * 1.05
    ax.set_ylim(y_lo, 0.0)
    ax.set_xlabel(r"$T^{\star}$"); ax.set_ylabel(r"$E$")
    ax.set_title("Hristov 2024 census (24,582 orbits): T* reveal")

    # Reveal schedule: 30 frames per bucket × 5 buckets = 150 frames,
    # then 60 frames showcasing A/B/C/D markers = 210 frames total (~7 s).
    FRAMES_PER_BUCKET = 30
    N_BUCKETS = len(BIN_LABELS)
    N_MARKER_FRAMES = 60
    n_frames = FRAMES_PER_BUCKET * N_BUCKETS + N_MARKER_FRAMES

    # Pre-create one scatter per bucket (empty to start)
    scats = []
    for i in range(N_BUCKETS):
        s = ax.scatter([], [], s=4, c=TOL_BRIGHT[i % len(TOL_BRIGHT)],
                       alpha=0.6, edgecolors="none",
                       label=BIN_LABELS[i])
        scats.append(s)
    orbit_scats = {name: ax.scatter([], [], s=0, c=ORBIT[name],
                                    edgecolors="black", linewidths=0.7,
                                    zorder=10, label=name)
                   for name in OUR}
    ax.legend(loc="upper left", fontsize=8, markerscale=2)

    def update(f):
        # Which bucket is currently being revealed?
        bucket_idx = min(f // FRAMES_PER_BUCKET, N_BUCKETS - 1)
        progress_in = (f % FRAMES_PER_BUCKET + 1) / FRAMES_PER_BUCKET
        for b in range(bucket_idx + 1):
            m = bin_ids == b
            n_pts = int(m.sum())
            if b < bucket_idx:
                keep = n_pts
            else:
                keep = int(n_pts * progress_in)
            idx = np.where(m)[0][:keep]
            scats[b].set_offsets(np.c_[TS_all[idx], E_all[idx]])

        # After all buckets revealed, grow A/B/C/D markers
        if f >= FRAMES_PER_BUCKET * N_BUCKETS:
            growth = min(1.0, (f - FRAMES_PER_BUCKET * N_BUCKETS + 1) / 30)
            marker_size = (16 * growth) ** 2
            for name, (e, ts) in OUR.items():
                orbit_scats[name].set_offsets([[ts, e]])
                orbit_scats[name].set_sizes([marker_size])

        return list(scats) + list(orbit_scats.values())

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False, interval=1000/30)
    return fig, anim


if __name__ == "__main__":
    fig, anim = build()
    save_gif_mp4(anim, "census_flyover", HERE / "animations", fps=30)
    plt.close(fig)
    print("census_flyover_ok")
