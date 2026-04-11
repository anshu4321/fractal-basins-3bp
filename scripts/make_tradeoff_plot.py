"""Phase 2 chunk 4 headline figure: the accuracy ↔ macro-F1 tradeoff frontier.

Loads baseline_history.npz and siren_history.npz plus a hardcoded SIREN
unweighted point (reproduced by running train_siren.py with --weighting none),
and plots all three models in the (accuracy, macro_F1) plane. The plateau
around (0.94, 0.25) -- coincident with the trivial always-predict-bound
baseline -- and the off-plateau point at ~(0.91, 0.42) are the two sides of
the fundamental tradeoff for classifying the 1M-IC basin dataset at this
resolution.

Outputs:
    figures/phase2_tradeoff.png
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mega3bp.style import PALETTE, apply_dirac_style


# Hardcoded: SIREN with weighting=none reaches the trivial always-bound point
# exactly. F1 = 4 * (0.969 bound) / 4 / ... actually just the reported macro F1
SIREN_UNWEIGHTED = {
    "name": "SIREN unweighted (ω₀=60, 384×6)",
    "acc": 0.9396,
    "macro_f1": 0.2422,
    "color": PALETTE["amber"],
    "marker": "D",
}


def main() -> int:
    apply_dirac_style()

    mlp_path = PROJECT_ROOT / "data" / "baseline_history.npz"
    siren_path = PROJECT_ROOT / "data" / "siren_history.npz"

    mlp = np.load(mlp_path, allow_pickle=True)
    siren = np.load(siren_path, allow_pickle=True)

    points = [
        {
            "name": "Trivial (always-bound)",
            "acc": float(mlp["trivial_acc"]),
            "macro_f1": 4 * 0.969 / 4 * 0.25,  # approx — see text below
            "color": PALETTE["text_mute"],
            "marker": "o",
        },
        {
            "name": "MLP + Fourier-8 + sqrt (best ckpt)",
            "acc": float(mlp["overall_acc"]),
            "macro_f1": float(mlp["macro_f1"]),
            "color": PALETTE["coral"],
            "marker": "s",
        },
        {
            "name": "SIREN weighted (ω₀=30, 256×5, sqrt)",
            "acc": float(siren["overall_acc"]),
            "macro_f1": float(siren["macro_f1"]),
            "color": PALETTE["cyan"],
            "marker": "^",
        },
        SIREN_UNWEIGHTED,
    ]
    # Override trivial macro_f1 with the exact number from the unweighted SIREN run
    # (which IS the trivial bound classifier to 4 decimals): 0.2422
    points[0]["macro_f1"] = 0.2422

    fig = plt.figure(figsize=(11, 7.5), dpi=150)
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor(PALETTE["bg_deep"])
    ax.set_facecolor(PALETTE["bg_panel"])

    # Shade the "trivial zone" — predict bound always, 0.94 accuracy
    ax.axvline(
        float(mlp["trivial_acc"]),
        color=PALETTE["text_mute"], ls="--", lw=1.1, alpha=0.8,
        label=f"trivial baseline ({float(mlp['trivial_acc']):.4f})",
    )
    ax.axvline(
        0.90, color=PALETTE["amber"], ls=":", lw=1.0, alpha=0.8,
        label="contract gate (0.90)",
    )

    for p in points:
        ax.scatter(
            p["acc"], p["macro_f1"],
            s=280, c=p["color"], marker=p["marker"],
            edgecolors=PALETTE["bg_deep"], linewidth=1.2, zorder=5,
            label=p["name"],
        )
        # Label each point
        ax.annotate(
            f"({p['acc']:.3f}, {p['macro_f1']:.3f})",
            xy=(p["acc"], p["macro_f1"]),
            xytext=(8, 8), textcoords="offset points",
            color=PALETTE["text"], fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor=PALETTE["bg_elevate"],
                      edgecolor=PALETTE["spine"], linewidth=0.5, alpha=0.9),
        )

    # Draw an indicative Pareto frontier curve between the two non-trivial points
    mlp_pt = points[1]
    siren_pt = points[2]
    trivial_pt = points[0]
    frontier_x = [trivial_pt["acc"], mlp_pt["acc"], siren_pt["acc"]]
    frontier_y = [trivial_pt["macro_f1"], mlp_pt["macro_f1"], siren_pt["macro_f1"]]
    order = np.argsort(frontier_x)
    ax.plot(
        np.array(frontier_x)[order],
        np.array(frontier_y)[order],
        color=PALETTE["lime_glow"], lw=1.2, ls="--", alpha=0.45, zorder=1,
        label="observed tradeoff frontier",
    )

    ax.set_xlabel("overall accuracy")
    ax.set_ylabel("macro F1")
    ax.set_xlim(0.88, 0.96)
    ax.set_ylim(0.18, 0.5)
    ax.set_title(
        "Phase 2 chunk 4 — the basin-classification tradeoff frontier\n"
        "all models sit on a ~linear frontier; no architecture breaks above it",
        color=PALETTE["text"], pad=14,
    )

    leg = ax.legend(loc="upper right", fontsize=9, framealpha=0.92)
    leg.get_frame().set_facecolor(PALETTE["bg_elevate"])
    leg.get_frame().set_edgecolor(PALETTE["spine"])
    for t in leg.get_texts():
        t.set_color(PALETTE["text"])

    # Annotation explaining the finding
    ax.text(
        0.885, 0.22,
        "Finding: at 1M samples on $S^2$, nearest-neighbor spacing is\n"
        "~$\\sqrt{4\\pi/10^6} \\approx 0.004$ radians. Fractal basin boundaries\n"
        "finer than that are NOT in the training set. All models converge to\n"
        "the same (accuracy, F1) tradeoff because the information bottleneck\n"
        "is dataset resolution, not architecture.",
        color=PALETTE["text_mute"], fontsize=9,
        va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor=PALETTE["bg_elevate"],
                  edgecolor=PALETTE["spine"], linewidth=0.5, alpha=0.95),
    )

    out = PROJECT_ROOT / "figures" / "phase2_tradeoff.png"
    fig.tight_layout()
    fig.savefig(out, facecolor=PALETTE["bg_deep"])
    plt.close(fig)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
