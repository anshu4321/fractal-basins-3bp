"""Phase 2 chunk 5: estimate the fractal dimension of the 3BP basin boundaries
on the Montgomery shape sphere via nearest-neighbor boundary detection +
3D voxel box-counting.

Method
------

1. Load the 1M labeled dataset (shape sphere coords + escape label).
   Exclude ambiguous rows.

2. For each sphere point, find its k=12 nearest neighbors using a
   Euclidean k-d tree in R^3 (monotone with great-circle distance for
   small ε, which is all we care about for boundary detection). Mark the
   point as "boundary" if any neighbor carries a different label.

3. Define four boundary sets:
     B_all    — any neighbor-label disagreement (all escape classes vs bound)
     B_esc_i  — bound vs body-i escape only (one per i in 1,2,3)
   so we can test whether the three escape regions have the same fractal
   dimension (S_3 symmetry of the equal-mass 3BP) and whether the bound
   set's overall outline differs from its per-class outlines.

4. Box-count each boundary set on a 3D voxel grid with sides ε in
   logspace(-2.3, -0.3, 25) (i.e. ε from ~0.005 to ~0.5). Count the number
   of unique voxels containing at least one boundary point.

5. Fit log(N) vs log(1/ε) with a straight line on the inner region
   (avoiding sample saturation at small ε and full-sphere saturation at
   large ε). The slope IS the box-counting (= Hausdorff for self-similar
   sets) dimension of the boundary.

   For a smooth 1D curve on S^2 in R^3, box-counting gives D -> 1.
   For a fractal curve, D ∈ (1, 2].
   For chaotic-scattering 3BP basins, literature expects D in ~1.3..1.8.

Outputs
-------

figures/fractal_scaling.png        log-log N vs 1/ε for all 4 boundary sets
figures/fractal_3d.png             3D scatter on the sphere with boundary
                                    points highlighted
figures/fractal_mollweide.png      Mollweide scatter with B_all visible
figures/fractal_summary.png        dashboard: dimensions, histograms, table
data/fractal_dimension.npz         per-ε counts + fitted slopes
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from mega3bp.style import BASIN_COLORS, PALETTE, apply_dirac_style

K_NEIGHBORS = 12
N_EPS = 25
EPS_LO = 0.005
EPS_HI = 0.5

# Fit region: avoid saturation at both ends. These are empirical for 1M samples.
FIT_EPS_LO = 0.008
FIT_EPS_HI = 0.15


def detect_boundary(pts: np.ndarray, labels: np.ndarray, k: int = K_NEIGHBORS) -> np.ndarray:
    """Returns a bool array: True iff any of the k nearest neighbors disagrees on label.

    Neighbor search is Euclidean in R^3. This is equivalent to great-circle
    nearest neighbors on S^2 up to a monotone transformation of distance.
    """
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=k + 1)
    neighbor_labels = labels[idx[:, 1:]]
    return np.any(neighbor_labels != labels[:, None], axis=1)


def box_count(boundary_pts: np.ndarray, eps_list: np.ndarray) -> np.ndarray:
    """For each ε in eps_list, return the number of unique 3D voxels that contain a boundary point.

    For each scale ε we discretize each boundary point to integer voxel coords
    `floor(pt / ε)` and count the unique triples. numpy's `unique(axis=0)` is
    fast enough for 1M points × ~25 scales (total ~1-2 seconds on a modern CPU).
    """
    counts = np.zeros_like(eps_list, dtype=np.int64)
    for i, eps in enumerate(eps_list):
        coords = np.floor(boundary_pts / eps).astype(np.int32)
        counts[i] = np.unique(coords, axis=0).shape[0]
    return counts


def fit_slope(eps: np.ndarray, counts: np.ndarray, fit_lo: float, fit_hi: float) -> tuple[float, float, np.ndarray]:
    """Linear fit log(N) ~ -D * log(eps) on the inner region. Returns (D, intercept, fit_mask)."""
    log_eps = np.log(eps)
    log_n = np.log(np.maximum(counts, 1))
    mask = (eps >= fit_lo) & (eps <= fit_hi) & (counts > 0)
    if mask.sum() < 3:
        return float("nan"), float("nan"), mask
    slope, intercept = np.polyfit(log_eps[mask], log_n[mask], 1)
    return -slope, intercept, mask


def shape_to_lonlat(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon = np.arctan2(n[:, 1], n[:, 0])
    lat = np.arcsin(np.clip(n[:, 2], -1, 1))
    return lon, lat


def main() -> int:
    apply_dirac_style()
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, required=True)
    args = parser.parse_args()

    print(f"loading {args.parquet} ...")
    df = pd.read_parquet(args.parquet)
    df = df.loc[df.label != -1].reset_index(drop=True)
    n = len(df)
    pts = df[["shape_n1", "shape_n2", "shape_n3"]].to_numpy(dtype=np.float64)
    labels = df.label.to_numpy(dtype=np.int64)
    print(f"  {n:,} non-ambiguous rows")
    print()

    # --- 1. Build neighbor tree once and detect boundaries --------------------
    print(f"=== nearest-neighbor boundary detection (k={K_NEIGHBORS}) ===")
    t0 = time.perf_counter()
    tree = cKDTree(pts)
    dists, idx = tree.query(pts, k=K_NEIGHBORS + 1)
    print(f"  kdtree + {K_NEIGHBORS}-NN query : {time.perf_counter() - t0:.2f}s")

    neighbor_labels = labels[idx[:, 1:]]

    # Boundary set B_all: any disagreement
    B_all = np.any(neighbor_labels != labels[:, None], axis=1)
    # Boundary set B_i: only flagged if a neighbor has THIS specific escape label
    # AND the point is currently a different class (i.e. this is a bound/i boundary).
    B_by_class = {}
    for lab in (1, 2, 3):
        self_is_bound = labels == 0
        nbr_is_i = np.any(neighbor_labels == lab, axis=1)
        self_is_i = labels == lab
        nbr_is_bound = np.any(neighbor_labels == 0, axis=1)
        # Symmetric: either I'm bound and a neighbor is body-i, or I'm body-i and a neighbor is bound
        B_by_class[lab] = (self_is_bound & nbr_is_i) | (self_is_i & nbr_is_bound)

    print(f"  |B_all|                 : {int(B_all.sum()):>10,}  ({100*B_all.mean():5.2f}%)")
    for lab in (1, 2, 3):
        print(f"  |B_bound-vs-body{lab}|      : {int(B_by_class[lab].sum()):>10,}"
              f"  ({100*B_by_class[lab].mean():5.2f}%)")
    print()

    # --- 2. Box-count each boundary set --------------------------------------
    eps_list = np.logspace(np.log10(EPS_LO), np.log10(EPS_HI), N_EPS)
    print(f"=== box counting at {N_EPS} ε values from {EPS_LO} to {EPS_HI} ===")

    results: dict[str, dict] = {}

    def do_one(name: str, mask: np.ndarray, color: str) -> None:
        bpts = pts[mask]
        t0 = time.perf_counter()
        counts = box_count(bpts, eps_list)
        D, intercept, fit_mask = fit_slope(eps_list, counts, FIT_EPS_LO, FIT_EPS_HI)
        dt = time.perf_counter() - t0
        print(f"  {name:<24}  {len(bpts):>10,} pts   D = {D:.3f}   [{dt:.1f}s]")
        results[name] = {
            "counts": counts,
            "D": D,
            "intercept": intercept,
            "fit_mask": fit_mask,
            "n_points": len(bpts),
            "color": color,
        }

    do_one("all boundaries",     B_all,          PALETTE["cyan"])
    do_one("bound vs body1",     B_by_class[1],  BASIN_COLORS["body1_escape"])
    do_one("bound vs body2",     B_by_class[2],  BASIN_COLORS["body2_escape"])
    do_one("bound vs body3",     B_by_class[3],  BASIN_COLORS["body3_escape"])
    print()

    # ---- Save numerics -------------------------------------------------------
    out_data = PROJECT_ROOT / "data"
    out_fig = PROJECT_ROOT / "figures"
    out_data.mkdir(exist_ok=True)
    out_fig.mkdir(exist_ok=True)
    np.savez(
        out_data / "fractal_dimension.npz",
        eps=eps_list,
        all_counts=results["all boundaries"]["counts"],
        b1_counts=results["bound vs body1"]["counts"],
        b2_counts=results["bound vs body2"]["counts"],
        b3_counts=results["bound vs body3"]["counts"],
        D_all=results["all boundaries"]["D"],
        D_body1=results["bound vs body1"]["D"],
        D_body2=results["bound vs body2"]["D"],
        D_body3=results["bound vs body3"]["D"],
        fit_lo=FIT_EPS_LO,
        fit_hi=FIT_EPS_HI,
    )

    # ---- Plot 1: scaling plot log N vs log(1/ε) -----------------------------
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)
    fig.patch.set_facecolor(PALETTE["bg_deep"])
    ax.set_facecolor(PALETTE["bg_panel"])
    for name, info in results.items():
        eps = eps_list
        cnt = info["counts"]
        ax.plot(
            1.0 / eps, cnt,
            marker="o", ms=4, lw=1.4, color=info["color"], alpha=0.9,
            label=f"{name}  ·  $D = {info['D']:.3f}$",
        )
        # Draw the fitted line on the fit region
        log_eps = np.log(eps)
        D = info["D"]
        b = info["intercept"]
        if info["fit_mask"].any():
            xs = eps[info["fit_mask"]]
            ys = np.exp(b + (-D) * np.log(xs))
            ax.plot(1.0 / xs, ys, ls="--", lw=1.0,
                    color=info["color"], alpha=0.55)

    ax.axvline(1.0 / FIT_EPS_LO, color=PALETTE["text_mute"], ls=":", lw=0.8, alpha=0.7)
    ax.axvline(1.0 / FIT_EPS_HI, color=PALETTE["text_mute"], ls=":", lw=0.8, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$1 / \varepsilon$")
    ax.set_ylabel(r"number of non-empty voxels  $N(\varepsilon)$")
    ax.set_title(
        "Box-counting scaling for 3BP basin boundaries\n"
        r"slope of the log-log fit = fractal dimension $D$",
        color=PALETTE["text"], pad=14,
    )
    ax.grid(True, which="both", alpha=0.4)
    leg = ax.legend(loc="upper left", fontsize=9)
    leg.get_frame().set_facecolor(PALETTE["bg_elevate"])
    leg.get_frame().set_edgecolor(PALETTE["spine"])
    for t in leg.get_texts():
        t.set_color(PALETTE["text"])
    fig.tight_layout()
    fig.savefig(out_fig / "fractal_scaling.png", facecolor=PALETTE["bg_deep"])
    print(f"  saved {out_fig / 'fractal_scaling.png'}")
    plt.close(fig)

    # ---- Plot 2: 3D sphere with boundary points highlighted -----------------
    fig = plt.figure(figsize=(9, 9), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(PALETTE["bg_deep"])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_color(PALETTE["spine"])
        axis.set_pane_color((0, 0, 0, 0))
        axis.set_tick_params(colors=PALETTE["text_mute"])
        axis.label.set_color(PALETTE["text"])

    # Wireframe sphere
    u = np.linspace(0, 2 * np.pi, 48)
    v = np.linspace(0, np.pi, 24)
    sx = np.outer(np.cos(u), np.sin(v))
    sy = np.outer(np.sin(u), np.sin(v))
    sz = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(sx, sy, sz, color=PALETTE["spine"], lw=0.2, alpha=0.3)

    # Interior (non-boundary) points faintly
    interior = ~B_all
    sub = np.random.default_rng(0).choice(int(interior.sum()), 30000, replace=False)
    ixs = np.where(interior)[0][sub]
    ax.scatter(pts[ixs, 0], pts[ixs, 1], pts[ixs, 2],
               s=1.0, c=PALETTE["text_dim"], alpha=0.15,
               edgecolors="none", depthshade=False)

    # Boundary points prominent
    bs = np.where(B_all)[0]
    if len(bs) > 120000:
        bs = np.random.default_rng(1).choice(bs, 120000, replace=False)
    ax.scatter(pts[bs, 0], pts[bs, 1], pts[bs, 2],
               s=2.4, c=PALETTE["lime"], alpha=0.9,
               edgecolors="none", depthshade=False,
               label=f"{int(B_all.sum()):,} boundary points")

    ax.set_xlabel(r"$n_1$")
    ax.set_ylabel(r"$n_2$")
    ax.set_zlabel(r"$n_3$")
    ax.set_title(
        f"3BP basin boundary on the shape sphere  ·  $D = {results['all boundaries']['D']:.3f}$",
        color=PALETTE["text"], pad=12,
    )
    leg = ax.legend(loc="upper right", fontsize=9)
    leg.get_frame().set_facecolor(PALETTE["bg_elevate"])
    leg.get_frame().set_edgecolor(PALETTE["spine"])
    for t in leg.get_texts():
        t.set_color(PALETTE["text"])
    fig.tight_layout()
    fig.savefig(out_fig / "fractal_3d.png", facecolor=PALETTE["bg_deep"])
    print(f"  saved {out_fig / 'fractal_3d.png'}")
    plt.close(fig)

    # ---- Plot 3: Mollweide with boundary points -----------------------------
    lon, lat = shape_to_lonlat(pts)
    fig = plt.figure(figsize=(12, 6.5), dpi=150)
    ax = fig.add_subplot(111, projection="mollweide")
    ax.set_facecolor(PALETTE["bg_deep"])
    ax.grid(color=PALETTE["grid"], alpha=0.35, lw=0.3)
    ax.tick_params(colors=PALETTE["text_mute"], labelsize=7)

    # Faint interior
    step = 25
    sub_int = np.where(~B_all)[0][::step]
    ax.scatter(lon[sub_int], lat[sub_int], s=0.35,
               c=PALETTE["text_dim"], alpha=0.22, edgecolors="none", rasterized=True)
    # Boundaries
    for lab, mask in B_by_class.items():
        col = BASIN_COLORS[f"body{lab}_escape"]
        ax.scatter(lon[mask], lat[mask], s=0.8, c=col,
                   alpha=0.85, edgecolors="none", rasterized=True,
                   label=f"bound↔body{lab} ({int(mask.sum()):,})")
    ax.set_title(
        f"Basin boundary points on Mollweide  ·  $D_{{\\rm all}} = {results['all boundaries']['D']:.3f}$",
        color=PALETTE["text"], pad=16,
    )
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0.0, -0.25), ncol=3, fontsize=9)
    leg.get_frame().set_facecolor(PALETTE["bg_elevate"])
    leg.get_frame().set_edgecolor(PALETTE["spine"])
    for t in leg.get_texts():
        t.set_color(PALETTE["text"])
    fig.tight_layout()
    fig.savefig(out_fig / "fractal_mollweide.png", facecolor=PALETTE["bg_deep"])
    print(f"  saved {out_fig / 'fractal_mollweide.png'}")
    plt.close(fig)

    # ---- Plot 4: summary dashboard ------------------------------------------
    fig = plt.figure(figsize=(16, 9), dpi=150, constrained_layout=True)
    fig.patch.set_facecolor(PALETTE["bg_deep"])
    gs = fig.add_gridspec(2, 3)

    # Panel A: scaling plot recap
    axA = fig.add_subplot(gs[0, :2])
    axA.set_facecolor(PALETTE["bg_panel"])
    for name, info in results.items():
        axA.plot(1.0 / eps_list, info["counts"], marker="o", ms=4, lw=1.4,
                 color=info["color"], alpha=0.9,
                 label=f"{name}  ·  $D={info['D']:.3f}$")
    axA.set_xscale("log")
    axA.set_yscale("log")
    axA.set_xlabel(r"$1 / \varepsilon$")
    axA.set_ylabel(r"$N(\varepsilon)$")
    axA.set_title("box-counting scaling", color=PALETTE["text"])
    leg = axA.legend(fontsize=9, loc="upper left")
    leg.get_frame().set_facecolor(PALETTE["bg_elevate"])
    leg.get_frame().set_edgecolor(PALETTE["spine"])
    for t in leg.get_texts():
        t.set_color(PALETTE["text"])

    # Panel B: dimensions as a bar chart
    axB = fig.add_subplot(gs[0, 2])
    axB.set_facecolor(PALETTE["bg_panel"])
    names_ord = ["all boundaries", "bound vs body1", "bound vs body2", "bound vs body3"]
    Ds = [results[n]["D"] for n in names_ord]
    cols = [results[n]["color"] for n in names_ord]
    ypos = np.arange(len(names_ord))
    bars = axB.barh(ypos, Ds, color=cols, alpha=0.92,
                    edgecolor=PALETTE["bg_deep"], linewidth=0.7)
    axB.set_yticks(ypos, labels=names_ord, color=PALETTE["text"], fontsize=9)
    axB.set_xlabel("fractal dimension D")
    axB.set_xlim(0, 3)
    axB.invert_yaxis()
    axB.axvline(1.0, color=PALETTE["text_mute"], ls="--", lw=1.0, alpha=0.7)
    axB.axvline(2.0, color=PALETTE["text_mute"], ls="--", lw=1.0, alpha=0.7)
    axB.text(1.0, -0.5, "smooth curve", color=PALETTE["text_mute"], fontsize=8,
             ha="center", va="bottom")
    axB.text(2.0, -0.5, "plane", color=PALETTE["text_mute"], fontsize=8,
             ha="center", va="bottom")
    for bar, d in zip(bars, Ds):
        if np.isfinite(d):
            axB.text(d, bar.get_y() + bar.get_height()/2, f" {d:.3f}",
                     va="center", color=PALETTE["text"], fontsize=9)
    axB.set_title("measured dimensions", color=PALETTE["text"])

    # Panel C: boundary count histogram per class (bar)
    axC = fig.add_subplot(gs[1, 0])
    axC.set_facecolor(PALETTE["bg_panel"])
    ns = [results[n]["n_points"] for n in names_ord]
    axC.barh(ypos, ns, color=cols, alpha=0.92,
             edgecolor=PALETTE["bg_deep"], linewidth=0.7)
    axC.set_yticks(ypos, labels=names_ord, color=PALETTE["text"], fontsize=9)
    axC.set_xlabel("boundary points")
    axC.invert_yaxis()
    axC.set_title("|B| per boundary definition", color=PALETTE["text"])

    # Panel D: 3D-on-sphere recap (smaller)
    axD = fig.add_subplot(gs[1, 1], projection="3d")
    axD.set_facecolor(PALETTE["bg_deep"])
    for axis in (axD.xaxis, axD.yaxis, axD.zaxis):
        axis.line.set_color(PALETTE["spine"])
        axis.set_pane_color((0, 0, 0, 0))
        axis.set_tick_params(colors=PALETTE["text_mute"])
        axis.label.set_color(PALETTE["text"])
    axD.plot_wireframe(sx, sy, sz, color=PALETTE["spine"], lw=0.15, alpha=0.25)
    if len(bs) > 40000:
        bs_small = np.random.default_rng(2).choice(bs, 40000, replace=False)
    else:
        bs_small = bs
    axD.scatter(pts[bs_small, 0], pts[bs_small, 1], pts[bs_small, 2],
                s=1.4, c=PALETTE["lime"], alpha=0.9,
                edgecolors="none", depthshade=False)
    axD.set_xticks([])
    axD.set_yticks([])
    axD.set_zticks([])
    axD.set_title("boundary points on S²", color=PALETTE["text"])

    # Panel E: text summary
    axE = fig.add_subplot(gs[1, 2])
    axE.axis("off")
    lines = [
        r"$\bf{Phase\ 2\ chunk\ 5}$", "",
        f"N ICs (non-amb)   : {n:,}",
        f"k nearest         : {K_NEIGHBORS}",
        f"ε range           : {EPS_LO}..{EPS_HI}",
        f"fit region        : {FIT_EPS_LO}..{FIT_EPS_HI}",
        "",
        r"$\bf{Measured\ fractal\ dimensions}$",
    ]
    for name in names_ord:
        lines.append(f"  {name:<18}: D = {results[name]['D']:.3f}")
    lines.extend([
        "",
        "interpretation:",
        f"  D ≈ 1  → smooth 1-D curve",
        f"  D ≈ 2  → area-filling",
        f"  D in (1, 2) → fractal curve",
        "",
        f"S_3 symmetry check:",
    ])
    D_esc = [results[f"bound vs body{i}"]["D"] for i in (1, 2, 3)]
    lines.append(f"  std over 3 classes: {np.std(D_esc):.4f}")
    lines.append(f"  mean:              {np.mean(D_esc):.4f}")

    for i, line in enumerate(lines):
        axE.text(0.02, 0.97 - i * 0.045, line, transform=axE.transAxes,
                 ha="left", va="top", fontsize=10,
                 color=PALETTE["text"], family="monospace")

    fig.suptitle(
        "Phase 2 chunk 5 — fractal dimension of the 3BP basin boundaries",
        color=PALETTE["text"], fontsize=16, y=1.005,
    )
    fig.savefig(out_fig / "fractal_summary.png", facecolor=PALETTE["bg_deep"])
    print(f"  saved {out_fig / 'fractal_summary.png'}")
    plt.close(fig)

    print()
    print("=== done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
