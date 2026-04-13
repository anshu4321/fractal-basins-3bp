"""Experiment 04: Wada basin test and basin entropy.

Daza-Wagemakers-Sanjuan 2018 merging test:
  For each pair of basins (i,j), merge them into one class. Compute
  box-counting dimension of the boundary between {i,j} and the remaining
  basin k. If all three such dimensions equal the full boundary dimension,
  the basin is Wada.

Basin entropy (Daza et al. 2016):
  S_b = mean over grid boxes of -sum p_i log p_i
  where p_i is the fraction of points in the box belonging to basin i.
  S_bb = same but restricted to boundary boxes.

Uses the at-rest 2D dataset (shape sphere).

Outputs:
    results/04_wada_basin_entropy/results.json
    figures/04_basin_entropy.png / .pdf
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mega3bp.style import PALETTE, apply_dirac_style

R_MIN_THRESHOLD = 0.01
GRID_RESOLUTIONS = [20, 30, 50, 75, 100, 150, 200, 300]


def load_clean_2d():
    path = PROJECT_ROOT / "results" / "01_dataset_regen" / "at_rest_with_diagnostics.npz"
    d = np.load(path)
    mask = (d["r_min_ever"] >= R_MIN_THRESHOLD) & (d["label"] != -1)
    shape = d["shape_n"][mask]
    labels = d["label"][mask]
    lon = np.arctan2(shape[:, 1], shape[:, 0])
    lat = np.arcsin(np.clip(shape[:, 2], -1, 1))
    return lon, lat, labels


def box_counting_boundary(lon, lat, labels, resolutions, merge_map=None):
    """Count boundary boxes at each resolution.

    If merge_map is provided, remap labels before checking boundaries.
    A box is a boundary box if it contains more than one label.
    Returns (resolutions, N_boundary) arrays.
    """
    if merge_map is not None:
        labels = np.array([merge_map.get(int(l), int(l)) for l in labels])

    N_boundary = []
    for res in resolutions:
        lon_bins = np.linspace(-np.pi, np.pi, res + 1)
        lat_bins = np.linspace(-np.pi / 2, np.pi / 2, res + 1)
        i_lon = np.digitize(lon, lon_bins) - 1
        i_lat = np.digitize(lat, lat_bins) - 1
        i_lon = np.clip(i_lon, 0, res - 1)
        i_lat = np.clip(i_lat, 0, res - 1)
        box_id = i_lon * res + i_lat

        n_boundary = 0
        for b in range(res * res):
            in_box = box_id == b
            if in_box.sum() == 0:
                continue
            unique = np.unique(labels[in_box])
            if len(unique) > 1:
                n_boundary += 1
        N_boundary.append(n_boundary)

    return np.array(resolutions), np.array(N_boundary)


def fit_dimension(resolutions, N_boundary):
    """Fit box-counting dimension from log N vs log(1/eps)."""
    eps = 2 * np.pi / np.array(resolutions, dtype=float)
    valid = N_boundary > 0
    if valid.sum() < 3:
        return None, None
    log_inv_eps = np.log(1.0 / eps[valid])
    log_N = np.log(N_boundary[valid].astype(float))
    slope, intercept = np.polyfit(log_inv_eps, log_N, 1)
    resid = log_N - (slope * log_inv_eps + intercept)
    r2 = 1 - np.sum(resid**2) / np.sum((log_N - log_N.mean())**2)
    return slope, r2


def basin_entropy(lon, lat, labels, res=100):
    """Compute basin entropy S_b and boundary basin entropy S_bb."""
    lon_bins = np.linspace(-np.pi, np.pi, res + 1)
    lat_bins = np.linspace(-np.pi / 2, np.pi / 2, res + 1)
    i_lon = np.clip(np.digitize(lon, lon_bins) - 1, 0, res - 1)
    i_lat = np.clip(np.digitize(lat, lat_bins) - 1, 0, res - 1)
    box_id = i_lon * res + i_lat

    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    all_S = []
    boundary_S = []

    for b in range(res * res):
        in_box = box_id == b
        n = in_box.sum()
        if n == 0:
            continue
        box_labels = labels[in_box]
        counts = np.array([np.sum(box_labels == l) for l in unique_labels])
        probs = counts / n
        probs = probs[probs > 0]
        S = -np.sum(probs * np.log(probs))
        all_S.append(S)

        if len(np.unique(box_labels)) > 1:
            boundary_S.append(S)

    S_b = np.mean(all_S) if all_S else 0.0
    S_bb = np.mean(boundary_S) if boundary_S else 0.0
    return S_b, S_bb, len(all_S), len(boundary_S)


def main() -> int:
    apply_dirac_style()
    out_results = PROJECT_ROOT / "results" / "04_wada_basin_entropy"
    out_figs = PROJECT_ROOT / "figures"
    out_results.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    lon, lat, labels = load_clean_2d()
    print(f"loaded {len(labels)} clean 2D samples")

    escape_labels = np.unique(labels[labels > 0])
    print(f"escape classes: {escape_labels.tolist()}")

    # Full boundary dimension
    print("\n=== Full boundary (all 4 classes) ===")
    res_full, N_full = box_counting_boundary(lon, lat, labels, GRID_RESOLUTIONS)
    D_full, r2_full = fit_dimension(res_full, N_full)
    print(f"  D_full = {D_full:.4f}, R² = {r2_full:.4f}")

    # Merged boundary dimensions (Wada test)
    # Merge (i,j) into one class, measure boundary between {i,j} and k
    merge_configs = [
        ("merge_12", {1: 99, 2: 99}, "escape 1+2 vs escape 3 + bound"),
        ("merge_13", {1: 99, 3: 99}, "escape 1+3 vs escape 2 + bound"),
        ("merge_23", {2: 99, 3: 99}, "escape 2+3 vs escape 1 + bound"),
        ("merge_bound_1", {0: 99, 1: 99}, "bound+1 vs escape 2 + escape 3"),
        ("merge_bound_2", {0: 99, 2: 99}, "bound+2 vs escape 1 + escape 3"),
        ("merge_bound_3", {0: 99, 3: 99}, "bound+3 vs escape 1 + escape 2"),
    ]

    wada_results = {}
    print("\n=== Wada merging test ===")
    for name, merge_map, desc in merge_configs:
        res_m, N_m = box_counting_boundary(lon, lat, labels, GRID_RESOLUTIONS, merge_map)
        D_m, r2_m = fit_dimension(res_m, N_m)
        wada_results[name] = {"D": float(D_m) if D_m else None,
                               "r2": float(r2_m) if r2_m else None,
                               "description": desc}
        print(f"  {name:20s}: D = {D_m:.4f}, R² = {r2_m:.4f}  ({desc})")

    # Wada criterion: all merged dimensions should equal D_full
    merged_Ds = [v["D"] for v in wada_results.values() if v["D"] is not None]
    if merged_Ds:
        D_spread = max(merged_Ds) - min(merged_Ds)
        D_mean = np.mean(merged_Ds)
        D_diff_from_full = abs(D_mean - D_full)
        is_wada = D_diff_from_full < 0.1 and D_spread < 0.1
        wada_verdict = "YES" if is_wada else ("MARGINAL" if D_diff_from_full < 0.2 else "NO")
    else:
        wada_verdict = "UNKNOWN"

    print(f"\n  Full D = {D_full:.4f}")
    print(f"  Merged D mean = {D_mean:.4f}, spread = {D_spread:.4f}")
    print(f"  |D_merged - D_full| = {D_diff_from_full:.4f}")
    print(f"  Wada verdict: {wada_verdict}")

    # Basin entropy
    print("\n=== Basin entropy ===")
    S_b, S_bb, n_boxes, n_boundary = basin_entropy(lon, lat, labels, res=100)
    print(f"  S_b  = {S_b:.4f} (mean over {n_boxes} boxes)")
    print(f"  S_bb = {S_bb:.4f} (mean over {n_boundary} boundary boxes)")
    S_max = np.log(len(np.unique(labels)))
    print(f"  S_max = ln({len(np.unique(labels))}) = {S_max:.4f}")
    print(f"  S_b/S_max = {S_b/S_max:.4f}")

    results = {
        "D_full": float(D_full),
        "D_full_r2": float(r2_full),
        "wada_test": wada_results,
        "wada_verdict": wada_verdict,
        "D_merged_mean": float(D_mean),
        "D_merged_spread": float(D_spread),
        "basin_entropy_S_b": float(S_b),
        "boundary_basin_entropy_S_bb": float(S_bb),
        "S_max": float(S_max),
        "n_boxes_total": n_boxes,
        "n_boxes_boundary": n_boundary,
        "grid_resolution_for_entropy": 100,
    }

    with open(out_results / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved results.json")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)
    fig.patch.set_facecolor(PALETTE["bg_deep"])

    # Panel 1: Box-counting scaling (full + merged)
    ax = axes[0]
    eps_full = 2 * np.pi / res_full.astype(float)
    ax.plot(1/eps_full, N_full, "o-", color=PALETTE["cyan"], ms=6, lw=2,
            label=f"full: D={D_full:.3f}")
    for name, merge_map, desc in merge_configs[:3]:
        res_m, N_m = box_counting_boundary(lon, lat, labels, GRID_RESOLUTIONS, merge_map)
        D_m = wada_results[name]["D"]
        ax.plot(1/eps_full, N_m, "s--", ms=4, lw=1.2, alpha=0.7,
                label=f"{name}: D={D_m:.3f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$1/\varepsilon$"); ax.set_ylabel(r"$N(\varepsilon)$")
    ax.set_title("Wada merging test", color=PALETTE["text"])
    ax.legend(fontsize=8)

    # Panel 2: Merged dimensions comparison
    ax = axes[1]
    names = list(wada_results.keys())
    Ds = [wada_results[n]["D"] for n in names]
    colors = [PALETTE["lime"] if abs(d - D_full) < 0.1 else PALETTE["coral"] for d in Ds]
    ax.barh(range(len(names)), Ds, color=colors, alpha=0.8)
    ax.axvline(D_full, color=PALETTE["cyan"], ls="--", lw=2, label=f"D_full={D_full:.3f}")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("box-counting dimension")
    ax.set_title(f"Wada: {wada_verdict}", color=PALETTE["text"])
    ax.legend(fontsize=9)

    # Panel 3: Basin entropy text summary
    ax = axes[2]
    ax.axis("off")
    lines = [
        f"Basin entropy diagnostics",
        f"",
        f"S_b   = {S_b:.4f}",
        f"S_bb  = {S_bb:.4f}",
        f"S_max = {S_max:.4f}",
        f"S_b / S_max = {S_b/S_max:.4f}",
        f"",
        f"Boundary boxes: {n_boundary}/{n_boxes}",
        f"  = {100*n_boundary/n_boxes:.1f}%",
        f"",
        f"Full boundary D = {D_full:.4f}",
        f"Merged D mean   = {D_mean:.4f}",
        f"Wada verdict    = {wada_verdict}",
    ]
    for i, line in enumerate(lines):
        ax.text(0.05, 0.95 - i * 0.07, line, transform=ax.transAxes,
                fontsize=11, color=PALETTE["text"], family="monospace", va="top")

    fig.suptitle("Wada test & basin entropy (2D at-rest)",
                 color=PALETTE["text"], fontsize=14, y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_figs / f"04_basin_entropy.{ext}",
                    facecolor=PALETTE["bg_deep"], bbox_inches="tight")
    plt.close(fig)
    print("saved figures/04_basin_entropy.png/pdf")

    return 0


if __name__ == "__main__":
    sys.exit(main())
