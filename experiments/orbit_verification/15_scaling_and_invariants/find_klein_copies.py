"""
Locate the 4 Klein copies of each novel orbit (A, B) in:
  (a) the ML basin label map, reading off the label at each copy,
  (b) the ML-verified candidate list from velocity_space_results.json,
  (c) the Hristov 2024 catalog (sweep results).

For each copy, integrate forward one period with scipy DOP853 and plot
all 4 trajectories side by side to visualise the D_2 symmetry.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

HERE = Path(__file__).parent
HP_JSON   = HERE.parent / "06_high_precision" / "hp_heyoka_newton_results.json"
BASIN     = HERE.parent.parent / "orbit_discovery" / "basin_map.npz"
VERIFIED  = HERE.parent.parent / "orbit_discovery" / "velocity_space_results.json"

KLEIN = {
    "I":        lambda v1, v2: ( v1,  v2),
    "sigma_x":  lambda v1, v2: (-v1,  v2),
    "sigma_y":  lambda v1, v2: ( v1, -v2),
    "sigma_xy": lambda v1, v2: (-v1, -v2),
}


def rhs_3body(t, s):
    r = s[[0, 4, 8]]
    y = s[[1, 5, 9]]
    vx = s[[2, 6, 10]]
    vy = s[[3, 7, 11]]
    ax = np.zeros(3); ay = np.zeros(3)
    for i in range(3):
        for j in range(3):
            if i == j: continue
            dx = r[j] - r[i]; dy = y[j] - y[i]
            r3 = (dx*dx + dy*dy) ** 1.5
            ax[i] += dx / r3; ay[i] += dy / r3
    out = np.empty(12)
    for i, idx in enumerate([0, 4, 8]):
        out[idx]   = vx[i]
        out[idx+1] = vy[i]
        out[idx+2] = ax[i]
        out[idx+3] = ay[i]
    return out


def pack(v1, v2):
    return np.array([
        -1.0, 0.0, v1, v2,
        +1.0, 0.0, v1, v2,
        0.0, 0.0, -2*v1, -2*v2,
    ])


def trajectory(v1, v2, T, n=2000):
    ts = np.linspace(0.0, T, n)
    sol = solve_ivp(rhs_3body, (0.0, T), pack(v1, v2),
                    t_eval=ts, method="DOP853", rtol=1e-12, atol=1e-13)
    return sol.y  # shape (12, n)


def basin_lookup(v1, v2, basin):
    v1_grid = basin["v1_grid"]
    v2_grid = basin["v2_grid"]
    labels = basin["labels"]
    # nearest cell
    i = int(np.argmin(np.abs(v1_grid - v1)))
    j = int(np.argmin(np.abs(v2_grid - v2)))
    return i, j, int(labels[i, j]), float(v1_grid[i]), float(v2_grid[j])


def find_in_verified(v1p, v2p, verified, tol=0.02):
    hits = []
    for k, row in enumerate(verified):
        dv = ((row["v1"] - v1p) ** 2 + (row["v2"] - v2p) ** 2) ** 0.5
        if dv < tol:
            hits.append({
                "index": k,
                "v1": row["v1"], "v2": row["v2"], "T": row["T"],
                "residual": row.get("residual"),
                "dv": dv,
            })
    return hits


def main():
    orbits = {o["name"]: o for o in json.loads(HP_JSON.read_text())}
    basin = np.load(BASIN, allow_pickle=True)
    verified = json.loads(VERIFIED.read_text())["verified"]
    print(f"Verified candidates in velocity_space_results.json: {len(verified)}")
    print(f"  ICs span v1 in [{min(r['v1'] for r in verified):+.3f}, "
          f"{max(r['v1'] for r in verified):+.3f}], "
          f"v2 in [{min(r['v2'] for r in verified):+.3f}, "
          f"{max(r['v2'] for r in verified):+.3f}]")
    print()

    out = {}
    for name in ["A", "B"]:
        o = orbits[name]
        v1 = float(o["v1_HP"]); v2 = float(o["v2_HP"]); T = float(o["T_HP"])
        print(f"## Orbit {name} (v1, v2) = ({v1:+.6f}, {v2:+.6f}), T = {T:.4f}")
        copies = []
        for img_name, g in KLEIN.items():
            v1p, v2p = g(v1, v2)
            i, j, lab, v1_cell, v2_cell = basin_lookup(v1p, v2p, basin)
            hits = find_in_verified(v1p, v2p, verified, tol=0.02)
            print(f"  {img_name:<10} (v1,v2) = ({v1p:+.6f}, {v2p:+.6f})")
            print(f"      basin cell ({i:3d}, {j:3d}) -> label {lab}  "
                  f"(cell center {v1_cell:+.4f}, {v2_cell:+.4f})")
            if hits:
                print(f"      ML-verified match(es): {len(hits)}")
                for h in hits:
                    print(f"        idx={h['index']}  dv={h['dv']:.4f}  "
                          f"T={h['T']:.4f}  residual={h['residual']}")
            else:
                print(f"      ML-verified match: none within tol=0.02")
            copies.append({
                "image": img_name, "v1": v1p, "v2": v2p,
                "basin_cell": (i, j), "basin_label": lab,
                "ml_matches": hits,
            })
        out[name] = copies
        print()

    # Plot the 4 trajectories side by side for A and B
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.5))
    body_colors = ["tab:blue", "tab:orange", "tab:green"]
    for row, name in enumerate(["A", "B"]):
        o = orbits[name]
        v1 = float(o["v1_HP"]); v2 = float(o["v2_HP"]); T = float(o["T_HP"])
        for col, (img_name, g) in enumerate(KLEIN.items()):
            v1p, v2p = g(v1, v2)
            traj = trajectory(v1p, v2p, T)
            ax = axes[row, col]
            for b, idx in enumerate([0, 4, 8]):
                ax.plot(traj[idx], traj[idx+1], color=body_colors[b], lw=0.9)
                ax.scatter(traj[idx, 0], traj[idx+1, 0],
                           color=body_colors[b], s=15, zorder=3)
            ax.set_aspect("equal")
            ax.set_title(f"{name} / {img_name}\n({v1p:+.4f}, {v2p:+.4f})",
                         fontsize=9)
            ax.grid(alpha=0.3)
    fig.suptitle("The four Klein-group (D_2) copies of orbits A and B", y=1.00)
    fig.tight_layout()
    fig_path = HERE / "klein_copies.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure: {fig_path}")

    (HERE / "klein_copies.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
