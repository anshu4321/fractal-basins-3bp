"""Gutzwiller oscillation ρ_p^osc(E) for A and B over a scaling range of E."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from experiments.ravishankar_followup._00_style import (
    make_fig, save_both, ORBIT,
)

HERE = Path(__file__).resolve().parent.parent


def render(kind="paper"):
    fig, ax = make_fig(kind=kind,
                        figsize=(8, 4.5) if kind == "blog" else (5.5, 3.0))
    Es = np.linspace(0.3, 3.0, 800)  # E/E_0 range
    for name in "AB":
        g = json.loads((HERE / f"gutzwiller_{name}.json").read_text())
        E0 = g["E0"]; T = g["T"]; S0 = g["S"]
        nu2 = g["nu_mod_2"]
        # Amplitude and phase per scaling: S(E) = S0 (|E/E0|)^{-1/2}
        # Primitive r=1 contribution:
        amp_r1 = g["gutzwiller_contributions_per_repetition"][0]["amplitude_1_over_sqrt_det"]
        # At each E, S/hbar - nu*pi/2
        # Under E -> alpha*E (alpha < 0 for scaling, but use |alpha|), S -> alpha^{1/2} S_0.
        # So for Es = E/|E0| in [0.3, 3]:
        S_E = S0 * Es ** 0.5
        phase = S_E - nu2 * np.pi / 2
        rho = (T / np.pi) * amp_r1 * np.cos(phase)
        ax.plot(Es, rho, color=ORBIT[name], linewidth=1.8,
                label=fr"Orbit {name}, $|\lambda|={g['lambda_max_magnitude']:.2f}$")

    ax.set_xlabel(r"$|E/E_0|$")
    ax.set_ylabel(r"$\rho_p^{\mathrm{osc}}(E)$")
    ax.set_title("Gutzwiller single-primitive oscillation (A vs B)")
    ax.axhline(0, color="#AAAAAA", linewidth=0.6)
    ax.legend(loc="upper right", fontsize=9 if kind == "paper" else 11)
    save_both(fig, "gutzwiller_ab", HERE / "figures")
    plt.close(fig)


if __name__ == "__main__":
    for kind in ("paper", "blog"):
        render(kind)
    print("gutzwiller_ab_ok")
