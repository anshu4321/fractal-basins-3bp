"""Render census_scatter_paper.pdf + census_scatter_blog.png.

Scatter of Hristov 2024 entries in (T*, E) with density heatmap coloring
(hexbin), overlaid with our 4 orbits A, B, C, D and (optionally) Hristov 2025
stable orbits as open circles.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from experiments.ravishankar_followup._00_style import (
    make_fig, save_both, add_orbit_marker, seq_cmap,
)

HERE = Path(__file__).resolve().parent.parent

# Our 4 orbits' (E, T*) at 50-digit precision (from
# experiments/orbit_verification/15_scaling_and_invariants/invariants.json)
OUR_ORBITS = {
    "A": (-1.51880093609, 36.94001257740),
    "B": (-1.57045059002, 64.64865733920),
    "C": (-1.50430210713, 64.65542233190),
    "D": (-0.93930544448, 73.81478296716),
}


def render(kind="paper"):
    inv = json.loads((HERE / "hristov_2024_invariants.json").read_text())
    E = np.array([r["E"] for r in inv])
    T_star = np.array([r["T_star"] for r in inv])

    # Drop unbound or invalid points
    mask = (E < 0) & np.isfinite(T_star) & (T_star > 0)
    E = E[mask]; T_star = T_star[mask]

    figsize = (7.0, 5.0) if kind == "blog" else (5.0, 3.5)
    fig, ax = make_fig(kind=kind, figsize=figsize)

    # Hexbin density
    hb = ax.hexbin(T_star, E, gridsize=80, xscale="log",
                   cmap=seq_cmap(), mincnt=1)
    cbar = fig.colorbar(hb, ax=ax, label="orbits / hex")
    cbar.ax.tick_params(labelsize=8 if kind == "paper" else 10)

    # Overlay our 4 orbits
    for name, (e, ts) in OUR_ORBITS.items():
        add_orbit_marker(ax, ts, e, name, size=16)

    # Optional: Hristov 2025 stable overlay if the invariants JSON exists
    h25_path = HERE / "hristov_2025_invariants.json"
    if h25_path.exists():
        h25 = json.loads(h25_path.read_text())
        TS25 = [r["T_star"] for r in h25 if r.get("E", 0) < 0]
        E25 = [r["E"] for r in h25 if r.get("E", 0) < 0]
        ax.scatter(TS25, E25, s=16, facecolors="none",
                   edgecolors="#444444", linewidths=0.6,
                   label="Hristov 2025 stable")
        ax.legend(loc="upper right", fontsize=8 if kind == "paper" else 10)

    ax.set_xlabel(r"$T^{\star} = T \cdot |E|^{3/2}$")
    ax.set_ylabel(r"$E$")
    ax.set_title("Equal-mass 3BP: Hristov 2024 (density) + our 4 orbits")

    save_both(fig, "census_scatter", HERE / "figures")
    plt.close(fig)


if __name__ == "__main__":
    for kind in ("paper", "blog"):
        render(kind)
    print("scatter_ok")
