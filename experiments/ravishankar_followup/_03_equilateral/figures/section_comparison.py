"""Schematic: Euler section vs equilateral-triangle section side by side."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from experiments.ravishankar_followup._00_style import BODY
from experiments.ravishankar_followup._00_style.figure_helpers import (
    PAPER_STYLE, BLOG_STYLE, save_both,
)

HERE = Path(__file__).resolve().parent.parent


def render(kind="paper"):
    plt.style.use(str(PAPER_STYLE if kind == "paper" else BLOG_STYLE))
    fig, (ax1, ax2) = plt.subplots(
        nrows=1, ncols=2,
        figsize=(10, 4.5) if kind == "blog" else (7.0, 3.2)
    )
    # Euler section: r=(-1,0), (+1,0), (0,0)
    r_euler = [(-1, 0), (1, 0), (0, 0)]
    for i, (x, y) in enumerate(r_euler):
        ax1.scatter([x], [y], s=500, c=BODY[i+1],
                    edgecolors="black", linewidths=0.8, zorder=10)
        ax1.annotate(f"{i+1}", (x, y), ha="center", va="center",
                     fontsize=12, fontweight="bold", color="white")
    ax1.set_xlim(-2, 2); ax1.set_ylim(-1.5, 1.5); ax1.set_aspect("equal")
    ax1.set_title("Euler velocity section\n$r_1=(-1,0)$, $r_2=(+1,0)$, $r_3=(0,0)$\n"
                  "$p_1=p_2=(v_1,v_2),\\ p_3=-2(v_1,v_2)$\n$L_z = 0$ identically",
                  fontsize=10 if kind == "paper" else 12)
    ax1.set_xticks([]); ax1.set_yticks([])
    for spine in ax1.spines.values(): spine.set_visible(False)

    # Equilateral section: r at 3 vertices of equilateral triangle at distance 1/sqrt(3)
    R = 1 / np.sqrt(3)
    for k in range(3):
        theta = 2 * np.pi * k / 3
        x, y = R * np.cos(theta), R * np.sin(theta)
        ax2.scatter([x], [y], s=500, c=BODY[k+1],
                    edgecolors="black", linewidths=0.8, zorder=10)
        ax2.annotate(f"{k+1}", (x, y), ha="center", va="center",
                     fontsize=12, fontweight="bold", color="white")
    ax2.set_xlim(-1.2, 1.2); ax2.set_ylim(-1.2, 1.2); ax2.set_aspect("equal")
    ax2.set_title("Equilateral section (this work)\n$r_k = \\hat{n}_k/\\sqrt{3}$ at 2$\\pi$/3 spacing\n"
                  "$v_k = R(2\\pi k/3) u$, $u \\in \\mathbb{R}^2$\n"
                  "$L_z = \\sqrt{3}\\,u_y$ (zero only on $u_y=0$ axis)",
                  fontsize=10 if kind == "paper" else 12)
    ax2.set_xticks([]); ax2.set_yticks([])
    for spine in ax2.spines.values(): spine.set_visible(False)

    fig.tight_layout()
    save_both(fig, "section_comparison", HERE / "figures")
    plt.close(fig)


if __name__ == "__main__":
    for kind in ("paper", "blog"):
        render(kind)
    print("section_comparison_ok")
