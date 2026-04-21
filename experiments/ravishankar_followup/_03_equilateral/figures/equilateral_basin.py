"""4-panel overview of the equilateral-section search."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from experiments.ravishankar_followup._00_style import (
    make_fig, save_both, add_orbit_marker, seq_cmap, BODY, ORBIT_EXT,
)
from experiments.ravishankar_followup._00_style.figure_helpers import (
    PAPER_STYLE, BLOG_STYLE,
)
from experiments.ravishankar_followup._03_equilateral.section import section_ic
from experiments.ravishankar_followup._03_equilateral.eom import accelerations

HERE = Path(__file__).resolve().parent.parent


def _integrate_period(u_x, u_y, T, n=800):
    r0, v0 = section_ic((u_x, u_y))
    y0 = np.concatenate([r0.ravel(), v0.ravel()])
    def ode(t, y):
        r = y[:6].reshape(3, 2); v = y[6:].reshape(3, 2)
        return np.concatenate([v.ravel(), accelerations(r).ravel()])
    sol = solve_ivp(ode, (0, T), y0, method="DOP853", rtol=1e-10, atol=1e-12,
                    dense_output=True)
    ts = np.linspace(0, T, n)
    y = sol.sol(ts)
    return y[:6].T.reshape(-1, 3, 2)


def render(kind="paper"):
    basin = np.load(HERE / "basin_map.npz")
    ld_data = np.load(HERE / "ld_field.npz")
    hp_cands = json.loads((HERE / "hp_verified_candidates.json").read_text())

    plt.style.use(str(PAPER_STYLE if kind == "paper" else BLOG_STYLE))
    fig, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(10, 10) if kind == "blog" else (7.5, 7.5)
    )
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # (a) raw labels
    labels = basin["labels"]
    # Use 4 distinct colors for 4 labels (0=periodic, 1/2/3 = escape by body)
    label_cmap = plt.cm.colors.ListedColormap(["#F1C40F", "#3498DB", "#E74C3C", "#27AE60"])
    im_a = ax_a.imshow(labels.T, origin="lower", cmap=label_cmap, aspect="equal",
                       extent=(-2, 2, -2, 2), interpolation="nearest")
    ax_a.set_title(f"(a) basin labels\nperiodic (yellow): {int((labels==0).sum())}/65536")
    ax_a.set_xlabel(r"$u_x$"); ax_a.set_ylabel(r"$u_y$")

    # (b) LD field
    ld = ld_data["ld"]
    im_b = ax_b.imshow(ld.T, origin="lower", cmap=seq_cmap(), aspect="equal",
                       extent=(-2, 2, -2, 2), vmin=0, vmax=np.log(4))
    ax_b.set_title(f"(b) LD entropy field\nmax = {ld.max():.2f} / {np.log(4):.2f}")
    ax_b.set_xlabel(r"$u_x$"); ax_b.set_ylabel(r"$u_y$")
    fig.colorbar(im_b, ax=ax_b, shrink=0.7, label="entropy")

    # (c) LD + HP candidates
    ax_c.imshow(ld.T, origin="lower", cmap=seq_cmap(), aspect="equal",
                extent=(-2, 2, -2, 2), vmin=0, vmax=np.log(4))
    for i, c in enumerate(hp_cands):
        name = c.get("name", f"EQ{i+1}")
        u_x, u_y = c["u"]
        color = ORBIT_EXT[i % len(ORBIT_EXT)]
        ax_c.scatter([u_x], [u_y], s=180, c=color, edgecolors="black",
                     linewidths=1.0, zorder=10)
        ax_c.annotate(name, (u_x, u_y), xytext=(8, 8), textcoords="offset points",
                      fontsize=12, fontweight="bold", color="white",
                      bbox=dict(facecolor=color, alpha=0.9, edgecolor="none"))
    ax_c.set_title(f"(c) HP-verified candidates (n={len(hp_cands)})")
    ax_c.set_xlabel(r"$u_x$"); ax_c.set_ylabel(r"$u_y$")

    # (d) sample orbit (EQ1)
    if hp_cands:
        c = hp_cands[0]
        traj = _integrate_period(c["u"][0], c["u"][1], c["T"])
        for i in range(3):
            ax_d.plot(traj[:, i, 0], traj[:, i, 1], color=BODY[i+1],
                      linewidth=1.5, label=f"body {i+1}")
        # Mark initial positions
        for i in range(3):
            ax_d.scatter(traj[0, i, 0], traj[0, i, 1], s=80, c=BODY[i+1],
                         edgecolors="black", linewidths=0.7, zorder=10)
        name = c.get("name", "EQ1")
        ax_d.set_title(f"(d) {name} trajectory (T={c['T']:.3f})")
        ax_d.set_aspect("equal")
        ax_d.set_xlabel(r"$x$"); ax_d.set_ylabel(r"$y$")
        ax_d.legend(loc="best", fontsize=8)

    fig.suptitle("Equilateral-section search: basin + LD + HP candidates + sample orbit",
                 fontsize=13 if kind == "paper" else 15)
    fig.tight_layout()
    save_both(fig, "equilateral_basin", HERE / "figures")
    plt.close(fig)


if __name__ == "__main__":
    for kind in ("paper", "blog"):
        render(kind)
    print("equilateral_basin_ok")
