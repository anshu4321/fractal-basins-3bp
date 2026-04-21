"""One 8s comet-tail GIF per HP-verified equilateral-section orbit."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

from experiments.ravishankar_followup._00_style import (
    save_gif_mp4, update_orbit_trail,
)
from experiments.ravishankar_followup._03_equilateral.section import section_ic
from experiments.ravishankar_followup._03_equilateral.eom import accelerations

HERE = Path(__file__).resolve().parent.parent


def _integrate(u_x, u_y, T, n=500):
    r0, v0 = section_ic((u_x, u_y))
    y0 = np.concatenate([r0.ravel(), v0.ravel()])
    def ode(t, y):
        r = y[:6].reshape(3, 2); v = y[6:].reshape(3, 2)
        return np.concatenate([v.ravel(), accelerations(r).ravel()])
    sol = solve_ivp(ode, (0, T), y0, method="DOP853", rtol=1e-10, atol=1e-12,
                    dense_output=True)
    ts = np.linspace(0, T, n)
    return sol.sol(ts)[:6].T.reshape(-1, 3, 2)


def build_one(name, u_x, u_y, T, fps=30, duration_s=8):
    traj = _integrate(u_x, u_y, T)
    n_frames = fps * duration_s
    tail_len = max(5, traj.shape[0] // 8)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    max_r = np.abs(traj).max() * 1.2
    xlim = (-max_r, max_r); ylim = (-max_r, max_r)

    def update(f):
        t_idx = int(f / (n_frames - 1) * (traj.shape[0] - 1))
        update_orbit_trail(ax, traj, t_idx, tail_len, T, xlim=xlim, ylim=ylim)
        ax.set_title(f"{name}: T={T:.3f} (equilateral-section orbit)",
                     fontsize=13, fontweight="bold")

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False,
                         interval=1000 / fps)
    return fig, anim


if __name__ == "__main__":
    hp = json.loads((HERE / "hp_verified_candidates.json").read_text())
    for c in hp:
        name = c.get("name", "EQ?")
        print(f"rendering new_orbit_{name} ...")
        fig, anim = build_one(name, c["u"][0], c["u"][1], c["T"])
        save_gif_mp4(anim, f"new_orbit_{name}", HERE / "animations", fps=30)
        plt.close(fig)
    print(f"new_orbit_trails_ok ({len(hp)} orbits)")
