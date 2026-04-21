"""Select ≤ K seed ICs for Newton refinement, drawn from periodic-labelled
cells clustered by (u_x, u_y) proximity.

Strategy:
1. From basin_map: filter cells with label==0 AND E in a physical range
   [-3, -0.5]. Drop symplectic-drift outliers (|E| > 100 from chaotic
   close-approaches that didn't escape within T_max).
2. Cluster the surviving cells in (u_x, u_y) space with k-means, K clusters.
3. For each cluster: pick the cell with the LOWEST LD (most confidently
   periodic), use as seed.
4. De-dup clusters under S3 cyclic rotation.
5. Append the known figure-8 hit from figure_8_sanity.json as candidate #1
   (guarantees at least one convergence since the Chenciner–Montgomery
   orbit lives on this section at T ≈ 6.326).
"""
import json
from pathlib import Path
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.ndimage import map_coordinates

HERE = Path(__file__).resolve().parent
K = 20
E_LO, E_HI = -3.0, -0.5   # bounded, reasonable range
OUTLIER_E = 100.0


def _lookup_ld(ld_field, u_grid_fine, u_x, u_y):
    """Interpolate LD from the 512² fine grid at coordinate (u_x, u_y)."""
    ni = len(u_grid_fine)
    du = u_grid_fine[1] - u_grid_fine[0]
    i = (u_x - u_grid_fine[0]) / du
    j = (u_y - u_grid_fine[0]) / du
    return float(map_coordinates(ld_field, [[i], [j]], order=1)[0])


def s3_canonical_form(u_x, u_y):
    cands = []
    for theta in (0, 2*np.pi/3, 4*np.pi/3):
        c, s = np.cos(theta), np.sin(theta)
        cands.append((c*u_x - s*u_y, s*u_x + c*u_y))
    return min(cands)


def select():
    basin = np.load(HERE / "basin_map.npz")
    labels = basin["labels"]
    E = basin["energies"].astype(np.float64)
    us = basin["u_x_grid"]  # same as u_y_grid

    ld_data = np.load(HERE / "ld_field.npz")
    ld = ld_data["ld"]
    us_fine = ld_data["u_grid"]

    # Filter periodic + bounded + not-outlier
    mask = (labels == 0) & (E > E_LO) & (E < E_HI) & (np.abs(E) < OUTLIER_E)
    ii, jj = np.where(mask)
    pts = np.stack([us[ii], us[jj]], axis=1)
    pt_E = E[ii, jj]
    print(f"periodic + bounded cells: {len(pts)}")
    if len(pts) == 0:
        raise RuntimeError("No periodic cells survived filtering — pipeline bug")

    # K-means cluster
    k_use = min(K - 1, max(4, len(pts) // 30))  # leave slot for figure-8
    centres, cluster_idx = kmeans2(pts, k_use, seed=42, minit="++")
    print(f"k-means into {k_use} clusters")

    cands = []
    for c_id in range(k_use):
        in_cluster = cluster_idx == c_id
        if not in_cluster.any():
            continue
        pts_c = pts[in_cluster]
        E_c = pt_E[in_cluster]
        # Score: pick the point with the SMALLEST LD within this cluster
        ld_vals = np.array([_lookup_ld(ld, us_fine, ux, uy)
                            for ux, uy in pts_c])
        best = int(np.argmin(ld_vals))
        cands.append({
            "cluster_id": int(c_id),
            "u_x": float(pts_c[best, 0]),
            "u_y": float(pts_c[best, 1]),
            "E": float(E_c[best]),
            "ld": float(ld_vals[best]),
            "cluster_size": int(in_cluster.sum()),
            "source": "kmeans-cluster-min-LD",
        })

    # Prepend figure-8 hit
    f8 = json.loads((HERE / "figure_8_sanity.json").read_text())
    if f8.get("top_10_by_delta_E"):
        f8_best = f8["top_10_by_delta_E"][0]
        cands.insert(0, {
            "cluster_id": -1,
            "u_x": f8_best["u_x"],
            "u_y": f8_best["u_y"],
            "E": f8_best["E"],
            "ld": _lookup_ld(ld, us_fine, f8_best["u_x"], f8_best["u_y"]),
            "cluster_size": 1,
            "source": "figure_8_sanity",
        })

    # S3 cyclic dedup
    seen = set()
    unique = []
    for c in cands:
        key = tuple(round(x, 2) for x in s3_canonical_form(c["u_x"], c["u_y"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
        if len(unique) >= K:
            break

    (HERE / "candidates_ICs.json").write_text(json.dumps(unique, indent=2))
    print(f"\nselected {len(unique)} unique candidates (target ≤ {K}):")
    for i, c in enumerate(unique):
        print(f"  #{i+1:2d}: u=({c['u_x']:+.3f}, {c['u_y']:+.3f}), "
              f"E={c['E']:+.3f}, LD={c['ld']:.3f}, src={c['source']}")
    return unique


if __name__ == "__main__":
    select()
