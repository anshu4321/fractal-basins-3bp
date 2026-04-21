"""Unit-circle eigenvalue plot for A, B, C, D."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from experiments.ravishankar_followup._00_style import ORBIT
from experiments.ravishankar_followup._00_style.figure_helpers import (
    PAPER_STYLE, BLOG_STYLE, save_both,
)

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.resolve().parents[2]


def _load_all_monodromy():
    ab = json.loads((ROOT / "experiments/orbit_verification/11_monodromy/monodromy_results.json").read_text())
    cd = json.loads((HERE / "monodromy_CD.json").read_text())
    return {**ab, **cd}


def render(kind="paper"):
    plt.style.use(str(PAPER_STYLE if kind == "paper" else BLOG_STYLE))
    fig, axes = plt.subplots(
        nrows=2, ncols=2,
        figsize=(8, 8) if kind == "blog" else (6.0, 6.0),
    )
    mono = _load_all_monodromy()
    theta = np.linspace(0, 2*np.pi, 200)
    for ax, name in zip(axes.flat, "ABCD"):
        entry = mono[name]
        re = entry["eigenvalues_real"]; im = entry["eigenvalues_imag"]
        mag = entry["eigenvalue_magnitudes"]
        # Unit circle
        ax.plot(np.cos(theta), np.sin(theta), color="#888888", linewidth=0.8)
        # All eigenvalues
        ax.scatter(re, im, s=40, c=ORBIT[name], edgecolors="black",
                   linewidths=0.7, zorder=10)
        # Dominant hyperbolic pair (if any): mark with large ring
        m = np.asarray(mag)
        log_m = np.log(np.maximum(m, 1e-15))
        if np.max(np.abs(log_m)) > 0.01:
            idx_dom = int(np.argmax(np.abs(log_m)))
            ax.scatter([re[idx_dom]], [im[idx_dom]], s=100,
                       facecolors="none", edgecolors="red", linewidths=1.5,
                       label=fr"$|\lambda|={m[idx_dom]:.2f}$")
            ax.legend(loc="best", fontsize=8)

        max_extent = max(1.3, max(abs(x) for x in re + im) * 1.1)
        ax.set_xlim(-max_extent, max_extent); ax.set_ylim(-max_extent, max_extent)
        ax.set_aspect("equal")
        ax.set_title(f"Orbit {name}: {entry.get('classification', '?')}",
                     color=ORBIT[name], fontweight="bold")
        ax.set_xlabel(r"Re $\lambda$"); ax.set_ylabel(r"Im $\lambda$")
        ax.axhline(0, color="#CCCCCC", linewidth=0.5)
        ax.axvline(0, color="#CCCCCC", linewidth=0.5)

    fig.suptitle("Monodromy eigenvalues on the complex plane",
                 fontsize=13 if kind == "paper" else 15)
    fig.tight_layout()
    save_both(fig, "monodromy_circle", HERE / "figures")
    plt.close(fig)


if __name__ == "__main__":
    for kind in ("paper", "blog"):
        render(kind)
    print("monodromy_circle_ok")
