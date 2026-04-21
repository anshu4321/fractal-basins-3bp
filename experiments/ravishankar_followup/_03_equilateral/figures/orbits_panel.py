"""Panel of HP-verified orbits in configuration space."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from experiments.ravishankar_followup._00_style import BODY, ORBIT_EXT
from experiments.ravishankar_followup._00_style.figure_helpers import (
    PAPER_STYLE, BLOG_STYLE, save_both,
)
from experiments.ravishankar_followup._03_equilateral.section import section_ic
from experiments.ravishankar_followup._03_equilateral.eom import accelerations

HERE = Path(__file__).resolve().parent.parent


def _integrate(u_x, u_y, T, n=800):
    r0, v0 = section_ic((u_x, u_y))
    y0 = np.concatenate([r0.ravel(), v0.ravel()])
    def ode(t, y):
        r = y[:6].reshape(3, 2); v = y[6:].reshape(3, 2)
        return np.concatenate([v.ravel(), accelerations(r).ravel()])
    sol = solve_ivp(ode, (0, T), y0, method="DOP853", rtol=1e-10, atol=1e-12,
                    dense_output=True)
    ts = np.linspace(0, T, n)
    return sol.sol(ts)[:6].T.reshape(-1, 3, 2)


def render(kind="paper"):
    hp = json.loads((HERE / "hp_verified_candidates.json").read_text())
    topo = json.loads((HERE / "topology_match_equilateral.json").read_text())
    n = len(hp)
    ncols = min(2, n); nrows = (n + ncols - 1) // ncols

    plt.style.use(str(PAPER_STYLE if kind == "paper" else BLOG_STYLE))
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(6*ncols, 5*nrows) if kind == "blog" else (4.2*ncols, 3.5*nrows),
        squeeze=False,
    )
    for ax, c in zip(axes.flat, hp):
        name = c.get("name", "EQ?")
        traj = _integrate(c["u"][0], c["u"][1], c["T"])
        for i in range(3):
            ax.plot(traj[:, i, 0], traj[:, i, 1], color=BODY[i+1], linewidth=1.4)
        for i in range(3):
            ax.scatter(traj[0, i, 0], traj[0, i, 1], s=60, c=BODY[i+1],
                       edgecolors="black", linewidths=0.7, zorder=10)
        vt = topo.get(name, {})
        word = vt.get("word", "")
        verdict = vt.get("verdict", "")
        word_disp = f"word: '{word}' (len {len(word)})" if word else "non-syzygy (empty word)"
        ax.set_title(f"{name}: T={c['T']:.3f}\n{word_disp}\n{verdict}",
                     fontsize=9 if kind == "paper" else 11)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.flat[n:]:
        ax.set_visible(False)

    fig.suptitle(f"HP-verified equilateral-section orbits (n={n})",
                 fontsize=13 if kind == "paper" else 15)
    fig.tight_layout()
    save_both(fig, "orbits_panel_equilateral", HERE / "figures")
    plt.close(fig)


if __name__ == "__main__":
    for kind in ("paper", "blog"):
        render(kind)
    print("orbits_panel_equilateral_ok")
