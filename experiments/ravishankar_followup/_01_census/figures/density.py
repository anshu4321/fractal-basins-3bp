"""Overall density of T* across Hristov 2024, with A/B/C/D reference lines.

B and C have nearly identical T* (64.649 vs 64.655); their labels are
staggered vertically so both are legible.

When per-entry syzygy-word lengths become available (from a future topology
sweep), this figure will be extended to overlay per-class densities.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from experiments.ravishankar_followup._00_style import (
    make_fig, save_both, ORBIT,
)

HERE = Path(__file__).resolve().parent.parent

OUR_TSTAR = {"A": 36.94001257740, "B": 64.64865733920,
             "C": 64.65542233190, "D": 73.81478296716}

# Vertical placement of labels (fraction of y_top).
# B and C are staggered so both are visible despite nearly identical T*.
LABEL_Y = {"A": 0.97, "B": 0.72, "C": 0.97, "D": 0.97}


def _kde(x, xs):
    from scipy.stats import gaussian_kde
    if len(x) < 2:
        return np.zeros_like(xs)
    return gaussian_kde(x, bw_method="silverman")(xs)


def render(kind="paper"):
    inv = json.loads((HERE / "hristov_2024_invariants.json").read_text())
    T_star = np.array([r["T_star"] for r in inv if r["T_star"] > 0])
    T_star = T_star[np.isfinite(T_star)]

    figsize = (7.0, 5.0) if kind == "blog" else (5.0, 3.5)
    fig, ax = make_fig(kind=kind, figsize=figsize)

    # Overall KDE on log-x
    ts_grid = np.logspace(np.log10(max(T_star.min(), 1.0)),
                          np.log10(T_star.max()), 500)
    kde = _kde(np.log10(T_star), np.log10(ts_grid))
    ax.fill_between(ts_grid, 0, kde, color="#4477AA", alpha=0.6,
                    label=f"Hristov 2024 density (n={len(T_star)})")

    # Reference lines at A, B, C, D positions
    y_top = kde.max() * 1.08
    ax.set_ylim(0, y_top)
    for name, ts in OUR_TSTAR.items():
        ax.axvline(ts, color=ORBIT[name], linestyle="--",
                   linewidth=1.6, alpha=0.85)
        y_frac = LABEL_Y.get(name, 0.97)
        ax.text(ts, y_top * y_frac, name, color=ORBIT[name],
                ha="center", va="top", fontsize=12, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel(r"$T^{\star}$")
    ax.set_ylabel("density (KDE on $\\log T^\\star$)")
    ax.set_title(r"$T^{\star}$ distribution across Hristov 2024")
    ax.legend(loc="upper left", fontsize=8 if kind == "paper" else 10)

    save_both(fig, "census_density", HERE / "figures")
    plt.close(fig)


if __name__ == "__main__":
    for kind in ("paper", "blog"):
        render(kind)
    print("density_ok")
