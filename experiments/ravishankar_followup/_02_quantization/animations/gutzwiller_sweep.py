"""10s animation: ρ_p^osc(E) draws itself for A and B side by side."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from experiments.ravishankar_followup._00_style import save_gif_mp4, ORBIT

HERE = Path(__file__).resolve().parent.parent


def build():
    Es = np.linspace(0.3, 3.0, 600)
    curves = {}
    for name in "AB":
        g = json.loads((HERE / f"gutzwiller_{name}.json").read_text())
        S0 = g["S"]; T = g["T"]; nu2 = g["nu_mod_2"]
        amp = g["gutzwiller_contributions_per_repetition"][0]["amplitude_1_over_sqrt_det"]
        S_E = S0 * Es ** 0.5
        phase = S_E - nu2 * np.pi / 2
        rho = (T / np.pi) * amp * np.cos(phase)
        curves[name] = (Es, rho)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    ax.set_xlim(Es.min(), Es.max())
    ymax = max(np.max(np.abs(rho)) for _, rho in curves.values()) * 1.15
    ax.set_ylim(-ymax, ymax)
    ax.axhline(0, color="#AAAAAA", linewidth=0.6)
    ax.set_xlabel(r"$|E/E_0|$"); ax.set_ylabel(r"$\rho_p^{\mathrm{osc}}(E)$")
    ax.set_title("Gutzwiller single-primitive oscillation (A vs B)")

    lines = {name: ax.plot([], [], color=ORBIT[name], linewidth=2.0,
                           label=f"Orbit {name}")[0]
             for name in "AB"}
    ax.legend(loc="upper right", fontsize=11)

    n_frames = 300  # 10s at 30 fps

    def update(f):
        progress = (f + 1) / n_frames
        keep = int(progress * len(Es))
        for name in "AB":
            Es_, rho = curves[name]
            lines[name].set_data(Es_[:keep], rho[:keep])
        return list(lines.values())

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False, interval=1000/30)
    return fig, anim


if __name__ == "__main__":
    fig, anim = build()
    save_gif_mp4(anim, "gutzwiller_sweep", HERE / "animations", fps=30)
    plt.close(fig)
    print("gutzwiller_sweep_ok")
