"""Basin map visualizations — the fractal landscape of three-body outcomes.

Reads basin_map.npz (from extract_basin_map.py) and produces:
  1. Basin map (escape classification by (v1, v2))
  2. Entropy heatmap with Suvakov + discovered orbits overlaid
  3. Combined side-by-side: basin | entropy | overlay
  4. Zoom-in on the region where our new orbits were discovered
  5. Escape time map (log scale)
  6. Min-distance map (close encounter hot spots)
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm

OUT_DIR = Path(__file__).parent
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

SUVAKOV = [
    ("Figure-8",      0.3471168881, 0.5327249454, 6.33),
    ("Butterfly I",   0.3068934205, 0.1255065670, 6.23),
    ("Butterfly II",  0.3930, 0.0976, 7.00),
    ("Butterfly III", 0.4059, 0.2302, 13.87),
    ("Moth I",        0.4644, 0.3961, 14.89),
    ("Moth II",       0.4392, 0.4530, 28.67),
    ("Moth III",      0.3834, 0.3774, 25.84),
    ("Goggles",       0.0833, 0.1279, 10.46),
    ("Dragonfly",     0.0806, 0.5888, 21.27),
    ("Yarn",          0.5591, 0.3492, 14.89),
    ("Yin-Yang Ib",   0.2827, 0.3272, 10.96),
]

OURS = [
    ("D1", 0.559519, 0.431263, 79.53),
    ("D2", 0.556313, 0.434469, 79.07),
]


def paper():
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm",
        "axes.labelsize": 11, "axes.titlesize": 12,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.facecolor": "white", "axes.facecolor": "white",
    })


def blog():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.weight": "300",
        "axes.labelsize": 12, "axes.titlesize": 14,
        "axes.labelcolor": "#b0b0c0", "axes.titlecolor": "#f0f0ff",
        "xtick.color": "#707090", "ytick.color": "#707090",
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
        "savefig.facecolor": "#080810", "axes.facecolor": "#080810",
    })


# ============================================================
# 1. Basin map
# ============================================================

def plot_basin_map(labels, v1_grid, v2_grid, mode="blog"):
    (blog if mode == "blog" else paper)()

    if mode == "paper":
        # Greyscale: bound=black, body 1=light grey, 2=mid grey, 3=darker grey
        cmap = ListedColormap(["white", "#202020", "#d0d0d0", "#808080", "#404040"])
    else:
        # Dark theme: bound=deep blue, escapes=warm colors
        cmap = ListedColormap(["#101018", "#2040a0", "#ff4080", "#40ffa0", "#ffa040"])
    vmin, vmax = -1.5, 3.5

    fig, ax = plt.subplots(figsize=(9, 9))
    extent = [v1_grid[0], v1_grid[-1], v2_grid[0], v2_grid[-1]]
    im = ax.imshow(labels, extent=extent, origin="lower", cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest")

    # Overlay Suvakov orbits
    for name, v1, v2, T in SUVAKOV:
        if mode == "paper":
            ax.plot(v1, v2, "o", ms=8, mfc="white", mec="black", mew=1.5, zorder=10)
            ax.annotate(name, (v1, v2), xytext=(6, 6), textcoords="offset points",
                         fontsize=7, color="black",
                         bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="black", lw=0.5, alpha=0.9))
        else:
            ax.plot(v1, v2, "o", ms=10, mfc="#ffff80", mec="#ffffff", mew=1.5, zorder=10)
            ax.annotate(name, (v1, v2), xytext=(6, 6), textcoords="offset points",
                         fontsize=8, color="#ffff80",
                         bbox=dict(boxstyle="round,pad=0.2", fc="#0a0a18",
                                  ec="#ffff80", lw=0.5, alpha=0.85))

    # Overlay OUR discovered orbits as stars
    for name, v1, v2, T in OURS:
        if mode == "paper":
            ax.plot(v1, v2, "*", ms=18, mfc="white", mec="black", mew=2, zorder=11)
            ax.annotate(f"NEW: {name} (T={T:.1f})", (v1, v2), xytext=(8, 8),
                         textcoords="offset points", fontsize=9, color="black",
                         fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  ec="black", lw=1, alpha=1))
        else:
            ax.plot(v1, v2, "*", ms=22, mfc="#ff40a0", mec="#ffffff", mew=2, zorder=11)
            ax.annotate(f"NEW: {name} (T={T:.1f})", (v1, v2), xytext=(8, 8),
                         textcoords="offset points", fontsize=10, color="#ff40a0",
                         fontweight="500",
                         bbox=dict(boxstyle="round,pad=0.3", fc="#0a0a18",
                                  ec="#ff40a0", lw=1, alpha=0.95))

    ax.set_xlabel(r"$v_1$")
    ax.set_ylabel(r"$v_2$")
    title = ("Velocity-space basin map" if mode == "paper" else
             "which body gets ejected? (velocity-space basin map)")
    ax.set_title(title)

    # Manual legend
    legend_elements = []
    if mode == "paper":
        labels_map = ["ambiguous", "bound", "body 1 escapes", "body 2 escapes", "body 3 escapes"]
        cols = ["white", "#202020", "#d0d0d0", "#808080", "#404040"]
    else:
        labels_map = ["ambiguous", "bound", "body 1 escapes", "body 2 escapes", "body 3 escapes"]
        cols = ["#101018", "#2040a0", "#ff4080", "#40ffa0", "#ffa040"]
    for col, lbl in zip(cols, labels_map):
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc=col, ec="grey", lw=0.5, label=lbl))
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9, fontsize=8)

    fname = f"basin_map_{mode}.pdf" if mode == "paper" else f"basin_map_{mode}.png"
    plt.savefig(FIG_DIR / fname)
    plt.close()
    print(f"  Saved {fname}")


# ============================================================
# 2. Entropy heatmap
# ============================================================

def plot_entropy_map(entropy, v1_grid, v2_grid, mode="blog"):
    (blog if mode == "blog" else paper)()

    if mode == "paper":
        cmap = plt.cm.Greys
    else:
        cmap = LinearSegmentedColormap.from_list(
            "glow", ["#080810", "#1a3050", "#20a0c0", "#ffd040", "#ff20a0"])

    fig, ax = plt.subplots(figsize=(9, 9))
    extent = [v1_grid[0], v1_grid[-1], v2_grid[0], v2_grid[-1]]
    im = ax.imshow(entropy, extent=extent, origin="lower", cmap=cmap,
                   aspect="equal", interpolation="bilinear")
    plt.colorbar(im, ax=ax, label=r"classifier entropy $H(v_1, v_2)$", shrink=0.7)

    # Overlays
    for name, v1, v2, T in SUVAKOV:
        if mode == "paper":
            ax.plot(v1, v2, "o", ms=8, mfc="white", mec="black", mew=1.5, zorder=10)
            ax.annotate(name, (v1, v2), xytext=(6, 6), textcoords="offset points",
                         fontsize=7, color="black",
                         bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="black", lw=0.5, alpha=0.9))
        else:
            ax.plot(v1, v2, "o", ms=10, mfc="#ffff80", mec="#ffffff", mew=1.5, zorder=10)
            ax.annotate(name, (v1, v2), xytext=(6, 6), textcoords="offset points",
                         fontsize=8, color="#ffff80",
                         bbox=dict(boxstyle="round,pad=0.2", fc="#080810",
                                  ec="#ffff80", lw=0.5, alpha=0.85))

    for name, v1, v2, T in OURS:
        if mode == "paper":
            ax.plot(v1, v2, "*", ms=18, mfc="white", mec="black", mew=2, zorder=11)
        else:
            ax.plot(v1, v2, "*", ms=22, mfc="#ff40a0", mec="#ffffff", mew=2, zorder=11)

    ax.set_xlabel(r"$v_1$")
    ax.set_ylabel(r"$v_2$")
    title = ("Classifier entropy in velocity space" if mode == "paper" else
             "where the model is uncertain")
    ax.set_title(title)

    fname = f"entropy_map_{mode}.pdf" if mode == "paper" else f"entropy_map_{mode}.png"
    plt.savefig(FIG_DIR / fname)
    plt.close()
    print(f"  Saved {fname}")


# ============================================================
# 3. Side-by-side comparison
# ============================================================

def plot_combined(labels, entropy, v1_grid, v2_grid, mode="blog"):
    (blog if mode == "blog" else paper)()

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    extent = [v1_grid[0], v1_grid[-1], v2_grid[0], v2_grid[-1]]

    if mode == "paper":
        basin_cmap = ListedColormap(["white", "#202020", "#d0d0d0", "#808080", "#404040"])
        entropy_cmap = plt.cm.Greys
    else:
        basin_cmap = ListedColormap(["#101018", "#2040a0", "#ff4080", "#40ffa0", "#ffa040"])
        entropy_cmap = LinearSegmentedColormap.from_list(
            "glow", ["#080810", "#1a3050", "#20a0c0", "#ffd040", "#ff20a0"])

    ax = axes[0]
    ax.imshow(labels, extent=extent, origin="lower", cmap=basin_cmap,
              vmin=-1.5, vmax=3.5, aspect="equal", interpolation="nearest")
    for name, v1, v2, T in SUVAKOV:
        mc = "#ffff80" if mode == "blog" else "white"
        mec = "#ffffff" if mode == "blog" else "black"
        ax.plot(v1, v2, "o", ms=8, mfc=mc, mec=mec, mew=1)
    for name, v1, v2, T in OURS:
        ax.plot(v1, v2, "*", ms=20,
                mfc="#ff40a0" if mode == "blog" else "white",
                mec="#ffffff" if mode == "blog" else "black", mew=1.5)
    ax.set_xlabel(r"$v_1$")
    ax.set_ylabel(r"$v_2$")
    ax.set_title("Basin structure (which body escapes)")

    ax = axes[1]
    im = ax.imshow(entropy, extent=extent, origin="lower", cmap=entropy_cmap,
                   aspect="equal", interpolation="bilinear")
    plt.colorbar(im, ax=ax, label="entropy", shrink=0.7)
    for name, v1, v2, T in SUVAKOV:
        mc = "#ffff80" if mode == "blog" else "white"
        mec = "#ffffff" if mode == "blog" else "black"
        ax.plot(v1, v2, "o", ms=8, mfc=mc, mec=mec, mew=1)
    for name, v1, v2, T in OURS:
        ax.plot(v1, v2, "*", ms=20,
                mfc="#ff40a0" if mode == "blog" else "white",
                mec="#ffffff" if mode == "blog" else "black", mew=1.5)
    ax.set_xlabel(r"$v_1$")
    ax.set_ylabel(r"$v_2$")
    ax.set_title("Classifier uncertainty (entropy)")

    fig.suptitle("Velocity-space basin structure and classifier prior\n"
                  "Periodic orbits (circles/stars) lie on or near basin boundaries",
                  fontsize=14, color="white" if mode == "blog" else "black")

    fname = f"basin_combined_{mode}.pdf" if mode == "paper" else f"basin_combined_{mode}.png"
    plt.savefig(FIG_DIR / fname)
    plt.close()
    print(f"  Saved {fname}")


# ============================================================
# 4. Zoom on discovered orbit region
# ============================================================

def plot_zoom_discovery(labels, entropy, v1_grid, v2_grid, mode="blog"):
    (blog if mode == "blog" else paper)()

    # Find indices for zoom region around our discoveries
    v1_center = 0.558
    v2_center = 0.433
    window = 0.08

    i1 = np.searchsorted(v1_grid, v1_center - window)
    i2 = np.searchsorted(v1_grid, v1_center + window)
    j1 = np.searchsorted(v2_grid, v2_center - window)
    j2 = np.searchsorted(v2_grid, v2_center + window)

    labels_zoom = labels[j1:j2, i1:i2]
    entropy_zoom = entropy[j1:j2, i1:i2]
    v1_z = v1_grid[i1:i2]
    v2_z = v2_grid[j1:j2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    extent = [v1_z[0], v1_z[-1], v2_z[0], v2_z[-1]]

    if mode == "paper":
        basin_cmap = ListedColormap(["white", "#202020", "#d0d0d0", "#808080", "#404040"])
        entropy_cmap = plt.cm.Greys
    else:
        basin_cmap = ListedColormap(["#101018", "#2040a0", "#ff4080", "#40ffa0", "#ffa040"])
        entropy_cmap = LinearSegmentedColormap.from_list(
            "glow", ["#080810", "#1a3050", "#20a0c0", "#ffd040", "#ff20a0"])

    axes[0].imshow(labels_zoom, extent=extent, origin="lower", cmap=basin_cmap,
                    vmin=-1.5, vmax=3.5, aspect="equal", interpolation="nearest")
    axes[0].set_title("Basin structure (zoomed)")

    im = axes[1].imshow(entropy_zoom, extent=extent, origin="lower", cmap=entropy_cmap,
                         aspect="equal", interpolation="bilinear")
    plt.colorbar(im, ax=axes[1], label="entropy", shrink=0.7)
    axes[1].set_title("Entropy (zoomed)")

    for ax in axes:
        for name, v1, v2, T in OURS:
            if v1_z[0] <= v1 <= v1_z[-1] and v2_z[0] <= v2 <= v2_z[-1]:
                mfc = "#ff40a0" if mode == "blog" else "white"
                mec = "#ffffff" if mode == "blog" else "black"
                ax.plot(v1, v2, "*", ms=28, mfc=mfc, mec=mec, mew=2)
                ax.annotate(f"{name}\nT={T:.1f}", (v1, v2),
                             xytext=(12, 12), textcoords="offset points",
                             fontsize=11, fontweight="500",
                             color="#ff40a0" if mode == "blog" else "black",
                             bbox=dict(boxstyle="round,pad=0.3",
                                      fc="#080810" if mode == "blog" else "white",
                                      ec="#ff40a0" if mode == "blog" else "black",
                                      lw=1, alpha=0.95))
        for name, v1, v2, T in SUVAKOV:
            if v1_z[0] <= v1 <= v1_z[-1] and v2_z[0] <= v2 <= v2_z[-1]:
                ax.plot(v1, v2, "o", ms=10,
                        mfc="#ffff80" if mode == "blog" else "white",
                        mec="#ffffff" if mode == "blog" else "black", mew=1.5)

        ax.set_xlabel(r"$v_1$")
        ax.set_ylabel(r"$v_2$")

    fig.suptitle(f"Zoom: discovery region around $(v_1, v_2) \\approx ({v1_center:.2f}, {v2_center:.2f})$",
                  fontsize=14, color="white" if mode == "blog" else "black")

    fname = f"zoom_discovery_{mode}.pdf" if mode == "paper" else f"zoom_discovery_{mode}.png"
    plt.savefig(FIG_DIR / fname)
    plt.close()
    print(f"  Saved {fname}")


# ============================================================
# 5. Escape time heatmap (log scale)
# ============================================================

def plot_escape_time(esc_step, v1_grid, v2_grid, mode="blog"):
    (blog if mode == "blog" else paper)()

    # Escape time: step * h = step * 0.002
    esc_time = esc_step * 0.002
    # Set non-escapes to NaN for transparency
    max_step = esc_step.max()
    esc_time_masked = np.where(esc_step >= max_step, np.nan, esc_time)

    fig, ax = plt.subplots(figsize=(9, 9))
    extent = [v1_grid[0], v1_grid[-1], v2_grid[0], v2_grid[-1]]

    if mode == "paper":
        cmap = plt.cm.Greys
        bg = "white"
    else:
        cmap = LinearSegmentedColormap.from_list(
            "time", ["#ffff40", "#ff8040", "#ff4080", "#a040ff", "#4080ff"])
        bg = "#080810"

    cmap.set_bad(bg)
    im = ax.imshow(esc_time_masked, extent=extent, origin="lower", cmap=cmap,
                    aspect="equal", interpolation="bilinear",
                    norm=LogNorm(vmin=1, vmax=100))
    plt.colorbar(im, ax=ax, label="escape time (log scale)", shrink=0.7)

    for name, v1, v2, T in SUVAKOV:
        ax.plot(v1, v2, "o", ms=8,
                mfc="#ffff80" if mode == "blog" else "white",
                mec="black", mew=1)
    for name, v1, v2, T in OURS:
        ax.plot(v1, v2, "*", ms=22,
                mfc="#ff40a0" if mode == "blog" else "white",
                mec="white" if mode == "blog" else "black", mew=2)

    ax.set_xlabel(r"$v_1$")
    ax.set_ylabel(r"$v_2$")
    title = ("Escape time map" if mode == "paper" else
             "how fast does the system come apart?")
    ax.set_title(title + "\n(dark = bound or slow escape; bright = fast escape)")

    fname = f"escape_time_{mode}.pdf" if mode == "paper" else f"escape_time_{mode}.png"
    plt.savefig(FIG_DIR / fname)
    plt.close()
    print(f"  Saved {fname}")


# ============================================================
# 6. Close-encounter map (r_min over trajectory)
# ============================================================

def plot_rmin_map(r_min, v1_grid, v2_grid, mode="blog"):
    (blog if mode == "blog" else paper)()

    fig, ax = plt.subplots(figsize=(9, 9))
    extent = [v1_grid[0], v1_grid[-1], v2_grid[0], v2_grid[-1]]

    if mode == "paper":
        cmap = plt.cm.Greys_r
    else:
        # Red = very close (danger of collision), blue = wide
        cmap = LinearSegmentedColormap.from_list(
            "close", ["#ff2040", "#ffa020", "#40a0c0", "#202060"])

    im = ax.imshow(r_min, extent=extent, origin="lower", cmap=cmap,
                   aspect="equal", interpolation="bilinear",
                   norm=LogNorm(vmin=max(r_min[r_min > 0].min(), 1e-4), vmax=min(r_min.max(), 10)))
    plt.colorbar(im, ax=ax, label=r"min pairwise distance $r_{\min}$ (log)", shrink=0.7)

    for name, v1, v2, T in SUVAKOV:
        ax.plot(v1, v2, "o", ms=8,
                mfc="#ffff80" if mode == "blog" else "white",
                mec="black", mew=1)
    for name, v1, v2, T in OURS:
        ax.plot(v1, v2, "*", ms=22,
                mfc="#ff40a0" if mode == "blog" else "white",
                mec="white" if mode == "blog" else "black", mew=2)

    ax.set_xlabel(r"$v_1$")
    ax.set_ylabel(r"$v_2$")
    title = ("Minimum-distance map" if mode == "paper" else
             "where do bodies get dangerously close?")
    ax.set_title(title + "\n(red = very close approach; blue = widely separated)")

    fname = f"rmin_map_{mode}.pdf" if mode == "paper" else f"rmin_map_{mode}.png"
    plt.savefig(FIG_DIR / fname)
    plt.close()
    print(f"  Saved {fname}")


# ============================================================
# Main
# ============================================================

def main():
    basin_path = OUT_DIR / "basin_map.npz"
    if not basin_path.exists():
        print(f"ERROR: {basin_path} not found.")
        print(f"Run extract_basin_map.py first (takes ~47 min on A100).")
        return 1

    print("Loading basin map data...")
    d = np.load(basin_path)
    labels = d["labels"]
    entropy = d["entropy"]
    esc_step = d["esc_step"]
    r_min = d["r_min"]
    v1_grid = d["v1_grid"]
    v2_grid = d["v2_grid"]

    print(f"Grid: {labels.shape}")
    print(f"Labels: {np.unique(labels, return_counts=True)}")
    print(f"Entropy range: [{entropy.min():.4f}, {entropy.max():.4f}]")
    print(f"r_min range: [{r_min.min():.4f}, {r_min.max():.4f}]")

    print("\n=== Generating visualizations ===")
    for mode in ["paper", "blog"]:
        print(f"\n-- {mode.upper()} --")
        plot_basin_map(labels, v1_grid, v2_grid, mode=mode)
        plot_entropy_map(entropy, v1_grid, v2_grid, mode=mode)
        plot_combined(labels, entropy, v1_grid, v2_grid, mode=mode)
        plot_zoom_discovery(labels, entropy, v1_grid, v2_grid, mode=mode)
        plot_escape_time(esc_step, v1_grid, v2_grid, mode=mode)
        plot_rmin_map(r_min, v1_grid, v2_grid, mode=mode)

    print(f"\nDone. See {FIG_DIR}")


if __name__ == "__main__":
    main()
