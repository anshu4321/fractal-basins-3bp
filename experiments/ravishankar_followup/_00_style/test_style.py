"""Renders a sanity panel verifying palette + typography + line styling.

Writes sanity_panel.pdf into this directory.
Run once after any change to palette.py or the stylesheets to eyeball results.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from .palette import BODY, ORBIT, TOL_BRIGHT
from .figure_helpers import PAPER_STYLE, BLOG_STYLE, save_both


def render_sanity_panel(kind="paper"):
    style_path = PAPER_STYLE if kind == "paper" else BLOG_STYLE
    plt.style.use(str(style_path))
    fig, axes = plt.subplots(
        nrows=2, ncols=2,
        figsize=(10, 7) if kind == "blog" else (6.5, 4.5),
    )
    (ax1, ax2), (ax3, ax4) = axes

    # Panel 1: body colors
    for i in range(3):
        ax1.scatter([i], [0], s=400, c=BODY[i+1],
                    edgecolors="black", linewidths=0.7)
        ax1.text(i, -0.4, f"body {i+1}", ha="center")
    ax1.set_xlim(-0.5, 2.5); ax1.set_ylim(-0.8, 0.5)
    ax1.set_title("Body palette"); ax1.set_xticks([]); ax1.set_yticks([])

    # Panel 2: orbit colors
    for j, name in enumerate("ABCD"):
        ax2.scatter([j], [0], s=400, c=ORBIT[name],
                    edgecolors="black", linewidths=0.7)
        ax2.text(j, -0.4, name, ha="center")
    ax2.set_xlim(-0.5, 3.5); ax2.set_ylim(-0.8, 0.5)
    ax2.set_title("Orbit palette"); ax2.set_xticks([]); ax2.set_yticks([])

    # Panel 3: Tol bright qualitative
    for k, c in enumerate(TOL_BRIGHT):
        ax3.bar([k], [1], color=c, edgecolor="black", linewidth=0.5)
    ax3.set_title("Paul Tol bright"); ax3.set_xticks([]); ax3.set_yticks([])

    # Panel 4: typography sample + math
    ax4.text(0.5, 0.7, "Typography sample", ha="center", fontsize=14,
             transform=ax4.transAxes)
    ax4.text(0.5, 0.45, r"$T^\star = T \cdot |E|^{3/2}$",
             ha="center", fontsize=16, transform=ax4.transAxes)
    ax4.text(0.5, 0.2, "body 10pt, label 12pt, title 14pt",
             ha="center", fontsize=10, transform=ax4.transAxes)
    ax4.set_xticks([]); ax4.set_yticks([])

    fig.tight_layout()
    return fig


def main():
    outdir = Path(__file__).parent
    # Render paper version
    fig_paper = render_sanity_panel("paper")
    save_both(fig_paper, "sanity_panel", outdir)
    # Save the canonical "sanity_panel.pdf" (paper version at 600 DPI, for git)
    fig_paper.savefig(str(outdir / "sanity_panel.pdf"),
                      format="pdf", dpi=600, bbox_inches="tight")
    plt.close(fig_paper)
    # Render blog version
    fig_blog = render_sanity_panel("blog")
    save_both(fig_blog, "sanity_panel", outdir)  # overwrites _blog.png
    plt.close(fig_blog)
    print(f"Rendered sanity panel to {outdir}")


if __name__ == "__main__":
    main()
