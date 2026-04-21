"""2x2 panel: α-family of A, B, C, D in configuration space.

Each panel shows the orbit at α=0.5, 1, 2 overlaid. The α=1 trajectory is
integrated once with scipy.integrate.solve_ivp (adequate for a schematic).
α-rescaling is applied analytically to produce the other two copies.
"""
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from experiments.ravishankar_followup._00_style import BODY, ORBIT
from experiments.ravishankar_followup._00_style.figure_helpers import (
    PAPER_STYLE, BLOG_STYLE, save_both,
)

HERE = Path(__file__).resolve().parent.parent

# 50-digit values from experiments/orbit_verification/15_scaling_and_invariants/invariants.json
ICS = {
    "A": {"v1": 0.18900489195, "v2": -0.53976245280, "T": 19.73539189, "E": -1.51880093609},
    "B": {"v1": -0.20349168745, "v2": 0.51811286074, "T": 32.84907117, "E": -1.57045059002},
    "C": {"v1": 0.25543093565, "v2": -0.51638583901, "T": 35.04308702, "E": -1.50430210713},
    "D": {"v1": 0.55393899048, "v2": 0.46193410064, "T": 81.08361217, "E": -0.93930544448},
}


def eom(t, y):
    """dy/dt for equal-mass 3-body, m=G=1."""
    r = y[:6].reshape(3, 2)
    v = y[6:].reshape(3, 2)
    acc = np.zeros((3, 2))
    for i in range(3):
        for j in range(3):
            if i == j: continue
            d = r[j] - r[i]
            dist3 = (d @ d) ** 1.5
            acc[i] += d / dist3
    return np.concatenate([v.flatten(), acc.flatten()])


def integrate_one_period(orbit_name):
    ic = ICS[orbit_name]
    v1, v2, T = ic["v1"], ic["v2"], ic["T"]
    # Euler velocity section: r = [(-1,0),(1,0),(0,0)], p1=p2=(v1,v2), p3=-2(v1,v2)
    y0 = np.array([-1, 0, 1, 0, 0, 0, v1, v2, v1, v2, -2*v1, -2*v2], dtype=float)
    sol = solve_ivp(eom, (0, T), y0, method="DOP853", rtol=1e-10, atol=1e-12,
                    dense_output=True)
    t_dense = np.linspace(0, T, 500)
    y = sol.sol(t_dense)  # shape (12, 500)
    traj = y[:6].T.reshape(-1, 3, 2)
    return traj


def render(kind="paper"):
    plt.style.use(str(PAPER_STYLE if kind == "paper" else BLOG_STYLE))
    fig, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(8, 8) if kind == "blog" else (6.5, 6.5),
    )
    for ax, name in zip(axes.flat, "ABCD"):
        traj = integrate_one_period(name)  # (500, 3, 2)
        # α = {0.5, 1.0, 2.0} overlays; α=1 most opaque
        for alpha, alpha_alpha in [(0.5, 0.35), (1.0, 1.0), (2.0, 0.4)]:
            for i in range(3):
                ax.plot(alpha * traj[:, i, 0], alpha * traj[:, i, 1],
                        color=BODY[i+1], alpha=alpha_alpha, linewidth=1.2)
        T = ICS[name]["T"]; E = ICS[name]["E"]; Ts = T * abs(E)**1.5
        ax.text(0.02, 0.98,
                f"{name}\n$T={T:.2f}$, $E={E:.3f}$\n$T^\\star={Ts:.2f}$",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                color=ORBIT[name], fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        # Trim spines
        for spine in ax.spines.values(): spine.set_visible(False)

    fig.suptitle(r"$\alpha$-scaling families (overlays at $\alpha=0.5, 1, 2$)",
                 fontsize=12 if kind == "paper" else 14)
    fig.tight_layout()
    save_both(fig, "scaling_families", HERE / "figures")
    plt.close(fig)


if __name__ == "__main__":
    for kind in ("paper", "blog"):
        render(kind)
    print("scaling_families_ok")
