"""Dense basin map visualization from a saved Phase 2 dataset.

Loads data/phase2_basin_*.parquet and produces the showcase figures:

  figures/basin_mollweide.png     2D Mollweide projection of the shape sphere
                                   with every IC colored by escape outcome.
                                   Collision and Lagrange landmarks annotated.
  figures/basin_3d.png            3D scatter on the unit sphere.
  figures/basin_stereographic.png Three stereographic panels centered on the
                                   three binary-collision points.
  figures/basin_density.png       Per-outcome density heatmap on the Mollweide
                                   grid (binned counts of body-i-escape).
  figures/basin_summary.png       Dashboard combining the above with stats.

All in Dirac palette.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb

from mega3bp.escape import LABEL_NAMES
from mega3bp.shape_sphere import (
    COLLISION_12, COLLISION_13, COLLISION_23,
    LAGRANGE_NORTH, LAGRANGE_SOUTH,
)
from mega3bp.style import BASIN_COLORS, PALETTE, apply_dirac_style


def label_name(lab: int) -> str:
    return LABEL_NAMES[int(lab)]


def shape_to_lonlat(n1: np.ndarray, n2: np.ndarray, n3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert (n1,n2,n3) on S^2 to (longitude, latitude) in radians."""
    lat = np.arcsin(np.clip(n3, -1, 1))  # [-pi/2, pi/2]
    lon = np.arctan2(n2, n1)              # [-pi, pi]
    return lon, lat


def stereographic(n: np.ndarray, center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stereographic projection of points n (shape (N, 3)) from the antipode of `center`.

    Returns (x, y) planar coordinates. The `center` point maps to (0, 0); its
    antipode maps to infinity.
    """
    # Build a local frame with z-axis = -center (so center is at the north pole of the projection)
    z_axis = np.asarray(center) / np.linalg.norm(center)
    if abs(z_axis[2]) < 0.99:
        tmp = np.array([0.0, 0.0, 1.0])
    else:
        tmp = np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(tmp, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    # Express each point in this basis
    nx = n @ x_axis
    ny = n @ y_axis
    nz = n @ z_axis
    denom = 1.0 + nz  # stereographic from the antipode
    # Where denom is 0 (the antipode), project to a large value
    denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    return nx / denom, ny / denom


def add_mollweide_landmarks(ax, which: str = "all") -> None:
    """Draw collision and Lagrange landmark points on a Mollweide axis."""
    landmarks_3 = [
        ("coll(1,2)", COLLISION_12, PALETTE["coral"],   "X",  36),
        ("coll(1,3)", COLLISION_13, PALETTE["coral"],   "X",  36),
        ("coll(2,3)", COLLISION_23, PALETTE["coral"],   "X",  36),
    ]
    landmarks_l = [
        ("L (N)",     LAGRANGE_NORTH, PALETTE["amber"], "*", 110),
        ("L (S)",     LAGRANGE_SOUTH, PALETTE["amber"], "*", 110),
    ]
    for name, pt, col, marker, size in landmarks_3 + landmarks_l:
        n = np.asarray(pt, dtype=np.float64)
        lon, lat = shape_to_lonlat(n[0:1], n[1:2], n[2:3])
        ax.scatter(lon, lat, s=size, c=col, marker=marker,
                   edgecolors=PALETTE["bg_deep"], linewidths=0.7, zorder=10)


def make_mollweide_scatter(
    df: pd.DataFrame,
    out: Path,
    title: str,
) -> None:
    lon, lat = shape_to_lonlat(df.shape_n1.to_numpy(), df.shape_n2.to_numpy(), df.shape_n3.to_numpy())
    labels = df.label.to_numpy()

    fig = plt.figure(figsize=(12, 6.5), dpi=150)
    ax = fig.add_subplot(111, projection="mollweide")
    ax.set_facecolor(PALETTE["bg_deep"])
    ax.grid(color=PALETTE["grid"], alpha=0.5, lw=0.4)
    ax.tick_params(colors=PALETTE["text_mute"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["spine"])

    # Point size scales inversely with sample count so dense regions don't saturate
    N = len(df)
    base_size = max(0.4, 12000 / N)

    # Bound first (faint), then escapes on top
    bound = labels == 0
    amb = labels == -1
    ax.scatter(lon[bound], lat[bound], s=base_size, c=BASIN_COLORS["bound"],
               alpha=0.28, edgecolors="none", rasterized=True)
    for lab in (1, 2, 3):
        m = labels == lab
        if not m.any():
            continue
        ax.scatter(lon[m], lat[m], s=base_size * 3.2, c=BASIN_COLORS[label_name(lab)],
                   alpha=0.9, edgecolors="none", rasterized=True)
    if amb.any():
        ax.scatter(lon[amb], lat[amb], s=base_size * 6, c=BASIN_COLORS["ambiguous"],
                   alpha=0.95, edgecolors=PALETTE["text"], linewidths=0.3,
                   marker="X", rasterized=True)

    add_mollweide_landmarks(ax)

    ax.set_title(title, color=PALETTE["text"], pad=18, fontsize=14)
    ax.text(
        0.5, -0.08,
        "longitude = $\\mathrm{atan2}(n_2, n_1)$  ·  latitude = $\\arcsin(n_3)$  ·  "
        "coral X = binary collision  ·  amber ★ = Lagrange equilateral",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=9, color=PALETTE["text_mute"],
    )
    fig.tight_layout()
    fig.savefig(out, facecolor=PALETTE["bg_deep"])
    plt.close(fig)
    print(f"  saved {out}")


def make_3d_scatter(df: pd.DataFrame, out: Path, title: str) -> None:
    labels = df.label.to_numpy()
    n = df[["shape_n1", "shape_n2", "shape_n3"]].to_numpy()
    N = len(df)

    fig = plt.figure(figsize=(9, 9), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(PALETTE["bg_deep"])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_color(PALETTE["spine"])
        axis.set_pane_color((0, 0, 0, 0))
        axis.set_tick_params(colors=PALETTE["text_mute"])
        axis.label.set_color(PALETTE["text"])

    u = np.linspace(0, 2 * np.pi, 48)
    v = np.linspace(0, np.pi, 24)
    sx = np.outer(np.cos(u), np.sin(v))
    sy = np.outer(np.sin(u), np.sin(v))
    sz = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(sx, sy, sz, color=PALETTE["spine"], lw=0.2, alpha=0.35)

    base_size = max(1.0, 20000 / N)
    bound = labels == 0
    amb = labels == -1
    if bound.any():
        ax.scatter(n[bound, 0], n[bound, 1], n[bound, 2],
                   s=base_size, c=BASIN_COLORS["bound"], alpha=0.18,
                   edgecolors="none", depthshade=False)
    for lab in (1, 2, 3):
        m = labels == lab
        if not m.any():
            continue
        ax.scatter(n[m, 0], n[m, 1], n[m, 2],
                   s=base_size * 3.0, c=BASIN_COLORS[label_name(lab)], alpha=0.92,
                   edgecolors="none", depthshade=False)
    if amb.any():
        ax.scatter(n[amb, 0], n[amb, 1], n[amb, 2],
                   s=base_size * 6, c=BASIN_COLORS["ambiguous"], alpha=0.95,
                   edgecolors=PALETTE["text"], linewidths=0.3,
                   marker="X", depthshade=False)

    ax.set_xlabel(r"$n_1$")
    ax.set_ylabel(r"$n_2$")
    ax.set_zlabel(r"$n_3$")
    ax.set_title(title, color=PALETTE["text"], pad=12, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, facecolor=PALETTE["bg_deep"])
    plt.close(fig)
    print(f"  saved {out}")


def make_stereographic(df: pd.DataFrame, out: Path) -> None:
    labels = df.label.to_numpy()
    n = df[["shape_n1", "shape_n2", "shape_n3"]].to_numpy()
    N = len(df)

    centers = {
        "collision (1,2)": COLLISION_12,
        "collision (1,3)": COLLISION_13,
        "collision (2,3)": COLLISION_23,
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), dpi=150)
    fig.patch.set_facecolor(PALETTE["bg_deep"])

    base_size = max(0.6, 18000 / N)

    for ax, (name, ctr) in zip(axes, centers.items()):
        ctr_np = np.asarray(ctr, dtype=np.float64)
        x, y = stereographic(n, ctr_np)
        # Clamp to a view radius so the antipode region doesn't explode
        r2 = x * x + y * y
        view_mask = r2 < 4.0

        bound = view_mask & (labels == 0)
        amb = view_mask & (labels == -1)
        ax.set_facecolor(PALETTE["bg_deep"])
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_color(PALETTE["spine"])
        ax.tick_params(colors=PALETTE["text_mute"], labelsize=8)

        ax.scatter(x[bound], y[bound], s=base_size, c=BASIN_COLORS["bound"],
                   alpha=0.25, edgecolors="none", rasterized=True)
        for lab in (1, 2, 3):
            m = view_mask & (labels == lab)
            if not m.any():
                continue
            ax.scatter(x[m], y[m], s=base_size * 3.2,
                       c=BASIN_COLORS[label_name(lab)], alpha=0.9,
                       edgecolors="none", rasterized=True)
        if amb.any():
            ax.scatter(x[amb], y[amb], s=base_size * 6,
                       c=BASIN_COLORS["ambiguous"], alpha=0.95,
                       marker="X", linewidths=0.3,
                       edgecolors=PALETTE["text"], rasterized=True)

        # Unit-circle outline (equator in the projection frame)
        theta = np.linspace(0, 2 * np.pi, 256)
        ax.plot(np.cos(theta), np.sin(theta), color=PALETTE["spine"], lw=0.6, alpha=0.7)

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_title(f"stereographic from {name}", color=PALETTE["text"], fontsize=11)

    fig.suptitle(
        "Phase 2 basin map · stereographic projections centered on the three binary collisions",
        color=PALETTE["text"], fontsize=14, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out, facecolor=PALETTE["bg_deep"])
    plt.close(fig)
    print(f"  saved {out}")


def make_density_map(df: pd.DataFrame, out: Path) -> None:
    """Per-outcome density heatmap on the Mollweide grid."""
    lon, lat = shape_to_lonlat(
        df.shape_n1.to_numpy(),
        df.shape_n2.to_numpy(),
        df.shape_n3.to_numpy(),
    )
    labels = df.label.to_numpy()

    NLON, NLAT = 80, 40
    lon_edges = np.linspace(-np.pi, np.pi, NLON + 1)
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, NLAT + 1)

    def hist_for_mask(mask):
        h, _, _ = np.histogram2d(lon[mask], lat[mask], bins=[lon_edges, lat_edges])
        return h.T

    h_body1 = hist_for_mask(labels == 1)
    h_body2 = hist_for_mask(labels == 2)
    h_body3 = hist_for_mask(labels == 3)
    h_bound = hist_for_mask(labels == 0)
    h_amb = hist_for_mask(labels == -1)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=150,
                             subplot_kw={"projection": "mollweide"})
    fig.patch.set_facecolor(PALETTE["bg_deep"])

    panels = [
        ("body1 escape", h_body1, PALETTE["cyan_glow"]),
        ("body2 escape", h_body2, PALETTE["lime_glow"]),
        ("body3 escape", h_body3, PALETTE["lav_glow"]),
        ("bound",         h_bound, PALETTE["coral_glow"]),
        ("ambiguous",     h_amb,   "#9aa3b3"),
    ]

    axes_flat = axes.flatten()

    # For each panel, plot a heatmap with a dark->accent colormap
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    LON_G, LAT_G = np.meshgrid(lon_centers, lat_centers)

    for ax, (name, h, glow) in zip(axes_flat[:5], panels):
        cmap = LinearSegmentedColormap.from_list(
            f"heat_{name}",
            [(0.0, PALETTE["bg_deep"]), (0.3, "#162036"),
             (0.6, glow), (1.0, PALETTE["text"])],
        )
        # log-scale-friendly display
        vmax = float(h.max()) if h.max() > 0 else 1.0
        ax.set_facecolor(PALETTE["bg_deep"])
        ax.grid(color=PALETTE["grid"], alpha=0.4, lw=0.3)
        ax.tick_params(colors=PALETTE["text_mute"], labelsize=6)
        ax.pcolormesh(LON_G, LAT_G, h, cmap=cmap, shading="auto", vmin=0, vmax=vmax, rasterized=True)
        add_mollweide_landmarks(ax)
        ax.set_title(f"{name}   (total {int(h.sum())})", color=PALETTE["text"], fontsize=10)

    # Sixth panel: empty
    axes_flat[5].axis("off")
    axes_flat[5].set_facecolor(PALETTE["bg_deep"])

    fig.suptitle(
        "Phase 2 density maps — Mollweide binned counts by outcome",
        color=PALETTE["text"], fontsize=15, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out, facecolor=PALETTE["bg_deep"])
    plt.close(fig)
    print(f"  saved {out}")


def make_summary(df: pd.DataFrame, out: Path, meta: dict) -> None:
    fig = plt.figure(figsize=(16, 10), dpi=150, constrained_layout=True)
    fig.patch.set_facecolor(PALETTE["bg_deep"])
    gs = fig.add_gridspec(2, 3)

    labels = df.label.to_numpy()
    N = len(df)

    # Panel 1: Mollweide scatter (inline)
    ax1 = fig.add_subplot(gs[0, :2], projection="mollweide")
    ax1.set_facecolor(PALETTE["bg_deep"])
    ax1.grid(color=PALETTE["grid"], alpha=0.4, lw=0.3)
    ax1.tick_params(colors=PALETTE["text_mute"], labelsize=7)
    lon, lat = shape_to_lonlat(df.shape_n1.to_numpy(), df.shape_n2.to_numpy(), df.shape_n3.to_numpy())
    base_size = max(0.3, 10000 / N)
    bound = labels == 0
    amb = labels == -1
    ax1.scatter(lon[bound], lat[bound], s=base_size, c=BASIN_COLORS["bound"],
                alpha=0.25, edgecolors="none", rasterized=True)
    for lab in (1, 2, 3):
        m = labels == lab
        if m.any():
            ax1.scatter(lon[m], lat[m], s=base_size * 3.2,
                        c=BASIN_COLORS[label_name(lab)], alpha=0.9,
                        edgecolors="none", rasterized=True)
    if amb.any():
        ax1.scatter(lon[amb], lat[amb], s=base_size * 6, c=BASIN_COLORS["ambiguous"],
                    alpha=0.95, marker="X", linewidths=0.2,
                    edgecolors=PALETTE["text"], rasterized=True)
    add_mollweide_landmarks(ax1)
    ax1.set_title(f"Basin map  ·  {N:,} ICs", color=PALETTE["text"], pad=12)

    # Panel 2: outcome bar chart
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(PALETTE["bg_panel"])
    names_ord = ["body1_escape", "body2_escape", "body3_escape", "bound", "ambiguous"]
    codes = {-1: "ambiguous", 0: "bound", 1: "body1_escape", 2: "body2_escape", 3: "body3_escape"}
    count_map = {name: 0 for name in names_ord}
    for lab, cnt in zip(*np.unique(labels, return_counts=True)):
        count_map[codes[int(lab)]] = int(cnt)
    vals = [count_map[n] for n in names_ord]
    cols = [BASIN_COLORS[n] for n in names_ord]
    ypos = np.arange(len(names_ord))
    bars = ax2.barh(ypos, vals, color=cols, alpha=0.92, edgecolor=PALETTE["bg_deep"], linewidth=0.7)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels(names_ord, color=PALETTE["text"])
    ax2.set_xlabel("count")
    ax2.set_title("outcome counts", color=PALETTE["text"])
    ax2.invert_yaxis()
    for bar, val in zip(bars, vals):
        if val > 0:
            ax2.text(val, bar.get_y() + bar.get_height() / 2,
                     f" {val:,}", va="center", color=PALETTE["text"], fontsize=9)

    # Panel 3: escape time histogram (log x)
    ax3 = fig.add_subplot(gs[1, 0])
    finite = np.isfinite(df.escape_time.to_numpy())
    et = df.escape_time.to_numpy()
    t_lo = max(float(et[finite & (et > 0)].min()) * 0.5, 0.3) if finite.any() else 0.5
    t_hi = max(float(et[finite].max()) * 1.2, 10.0) if finite.any() else 100.0
    bins = np.logspace(np.log10(t_lo), np.log10(t_hi), 40)
    for lab in (1, 2, 3):
        m = (labels == lab) & finite
        if m.any():
            ax3.hist(et[m], bins=bins, color=BASIN_COLORS[label_name(lab)],
                     alpha=0.7, edgecolor=PALETTE["bg_deep"], linewidth=0.3,
                     label=label_name(lab))
    ax3.set_xscale("log")
    ax3.set_xlim(t_lo, t_hi)
    ax3.set_xlabel(r"escape time $t_{\mathrm{esc}}$  (log)")
    ax3.set_ylabel("count")
    ax3.set_title("escape time distribution")
    leg = ax3.legend()
    leg.get_frame().set_facecolor(PALETTE["bg_elevate"])
    leg.get_frame().set_edgecolor(PALETTE["spine"])
    for t in leg.get_texts():
        t.set_color(PALETTE["text"])

    # Panel 4: r_min histogram
    ax4 = fig.add_subplot(gs[1, 1])
    log_rmin = np.log10(np.maximum(df.r_min_ever.to_numpy(), 1e-6))
    ax4.hist(log_rmin, bins=60, color=PALETTE["cyan"], alpha=0.85,
             edgecolor=PALETTE["bg_deep"], linewidth=0.3)
    ax4.axvline(-3.0, color=PALETTE["coral"], ls="--", lw=1.2,
                label=r"$r_{\mathrm{close}} = 10^{-3}$")
    ax4.set_xlabel(r"$\log_{10} r_{\mathrm{min}}$")
    ax4.set_ylabel("count")
    ax4.set_title("minimum pair distance")
    leg = ax4.legend()
    leg.get_frame().set_facecolor(PALETTE["bg_elevate"])
    leg.get_frame().set_edgecolor(PALETTE["spine"])
    for t in leg.get_texts():
        t.set_color(PALETTE["text"])

    # Panel 5: metadata
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    lines = [
        r"$\bf{Phase\ 2\ dataset}$", "",
        f"N ICs            : {N:,}",
        f"h                : {meta.get('h', 0.01)}",
        f"t_max            : {meta.get('t_max', 500)}",
        f"r_escape         : {meta.get('r_escape', 5.0)}",
        f"r_close          : {meta.get('r_close', 1e-3)}",
        "",
        f"bound            : {count_map['bound']:,} ({100*count_map['bound']/N:.2f}%)",
        f"body1 escape     : {count_map['body1_escape']:,} ({100*count_map['body1_escape']/N:.2f}%)",
        f"body2 escape     : {count_map['body2_escape']:,} ({100*count_map['body2_escape']/N:.2f}%)",
        f"body3 escape     : {count_map['body3_escape']:,} ({100*count_map['body3_escape']/N:.2f}%)",
        f"ambiguous        : {count_map['ambiguous']:,} ({100*count_map['ambiguous']/N:.2f}%)",
        "",
        f"amb fraction     : {100 * df.ambiguous.mean():.3f}%",
        f"min r_min        : {df.r_min_ever.min():.2e}",
        f"source           : {meta.get('source', '—')}",
    ]
    for i, line in enumerate(lines):
        ax5.text(0.02, 0.97 - i * 0.055, line, transform=ax5.transAxes,
                 ha="left", va="top", fontsize=10,
                 color=PALETTE["text"], family="monospace")

    fig.suptitle(
        f"Phase 2 basin dataset — {N:,} shape-sphere ICs at rest",
        color=PALETTE["text"], fontsize=16, y=1.005,
    )
    fig.savefig(out, facecolor=PALETTE["bg_deep"])
    plt.close(fig)
    print(f"  saved {out}")


def main() -> int:
    apply_dirac_style()
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, required=True,
                        help="path to phase2 parquet dataset")
    args = parser.parse_args()

    path = Path(args.parquet)
    print(f"loading {path} ...")
    df = pd.read_parquet(path)
    print(f"  {len(df):,} rows")
    print()

    out_fig = PROJECT_ROOT / "figures"
    out_fig.mkdir(exist_ok=True)

    meta = {
        "h": 0.01,
        "t_max": 500.0,
        "r_escape": 5.0,
        "r_close": 1e-3,
        "source": path.name,
    }

    print("=== plotting ===")
    make_mollweide_scatter(df, out_fig / "basin_mollweide.png",
                           title=f"Phase 2 basin map — Mollweide projection  ·  {len(df):,} ICs")
    make_3d_scatter(df, out_fig / "basin_3d.png",
                    title=f"Phase 2 basin map — 3D shape sphere  ·  {len(df):,} ICs")
    make_stereographic(df, out_fig / "basin_stereographic.png")
    make_density_map(df, out_fig / "basin_density.png")
    make_summary(df, out_fig / "basin_summary.png", meta)
    print()
    print("=== done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
