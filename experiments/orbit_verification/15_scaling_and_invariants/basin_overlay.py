"""
Overlay A's and B's 4 Klein copies on the ML basin-label map.
Also mark C, D and the 3 ML-verified equivalence classes for context.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

HERE = Path(__file__).parent
HP_JSON   = HERE.parent / "06_high_precision" / "hp_heyoka_newton_results.json"
BASIN     = HERE.parent.parent / "orbit_discovery" / "basin_map.npz"
VERIFIED  = HERE.parent.parent / "orbit_discovery" / "velocity_space_results.json"


def klein(v1, v2):
    return [(v1, v2), (-v1, v2), (v1, -v2), (-v1, -v2)]


def main():
    orbits = {o["name"]: o for o in json.loads(HP_JSON.read_text())}
    basin = np.load(BASIN, allow_pickle=True)
    verified = json.loads(VERIFIED.read_text())["verified"]

    v1_grid = basin["v1_grid"]
    v2_grid = basin["v2_grid"]
    labels = basin["labels"]   # shape (N_v1, N_v2)

    fig, ax = plt.subplots(figsize=(8, 8))
    # imshow with labels (transposed so v1 on x-axis, v2 on y-axis)
    cmap = ListedColormap(["#e5d8ff", "#ffd8a8", "#a8e6cf", "#b2ccff"])
    ax.imshow(labels.T, origin="lower",
              extent=[v1_grid.min(), v1_grid.max(),
                      v2_grid.min(), v2_grid.max()],
              cmap=cmap, alpha=0.85, aspect="equal",
              interpolation="nearest")

    # Plot Klein copies of A, B, C, D
    orbit_style = {
        "A": {"color": "red",     "marker": "*", "size": 320, "edge": "white"},
        "B": {"color": "black",   "marker": "*", "size": 320, "edge": "white"},
        "C": {"color": "#d62728", "marker": "o", "size": 140, "edge": "white"},
        "D": {"color": "#1f77b4", "marker": "o", "size": 140, "edge": "white"},
    }
    for name in ["A", "B", "C", "D"]:
        o = orbits[name]
        v1 = float(o["v1_HP"]); v2 = float(o["v2_HP"])
        pts = klein(v1, v2)
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        st = orbit_style[name]
        ax.scatter(xs, ys, c=st["color"], marker=st["marker"], s=st["size"],
                   edgecolors=st["edge"], linewidths=1.8,
                   label=f"{name} (4 Klein copies)", zorder=5)
        # Annotate the original copy
        ax.annotate(name, (v1, v2), color=st["color"], fontsize=14,
                    fontweight="bold",
                    xytext=(8, 8), textcoords="offset points", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec=st["color"], alpha=0.9))

    # Plot ML-verified candidates
    xs_v = [r["v1"] for r in verified]
    ys_v = [r["v2"] for r in verified]
    ax.scatter(xs_v, ys_v, c="black", marker="x", s=40, linewidths=1.2,
               label=f"ML-verified ({len(verified)})", zorder=4, alpha=0.75)

    ax.set_xlabel("$v_1$")
    ax.set_ylabel("$v_2$")
    ax.set_title("Basin of attraction (100$\\times$100) with Klein copies of "
                 "A, B, C, D\n"
                 "Labels 1, 2 swap under $\\sigma_y$; labels 0, 3 are $D_2$-fixed")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)
    ax.set_xlim(v1_grid.min(), v1_grid.max())
    ax.set_ylim(v2_grid.min(), v2_grid.max())

    out = HERE / "basin_klein_overlay.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
