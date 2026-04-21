"""Log-log plot of action S vs |E| showing scaling-invariance S ∝ |E|^{-1/2}."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from experiments.ravishankar_followup._00_style import (
    make_fig, save_both, add_orbit_marker,
)

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.resolve().parents[2]


def render(kind="paper"):
    inv = json.loads((ROOT / "experiments/orbit_verification/15_scaling_and_invariants/invariants.json").read_text())
    action = json.loads((HERE / "action_ABCD.json").read_text())

    fig, ax = make_fig(kind=kind,
                        figsize=(7, 5) if kind == "blog" else (5.0, 3.5))

    pts = []
    for o in inv["orbits"]:
        name = o["name"]
        E = abs(float(o["E"]))
        S = action[name]["S"]
        pts.append((E, S, name))

    # Theory line: S = S_A * (|E_A|/|E|)^{1/2}  through A's point
    E_A, S_A, _ = pts[0]
    E_range = np.logspace(-0.5, 0.5, 100) * E_A
    S_theory = S_A * (E_A / E_range) ** 0.5
    ax.plot(E_range, S_theory, color="#566573", linestyle="--", linewidth=1.4,
            label=r"theory: $S \propto |E|^{-1/2}$")

    for E, S, name in pts:
        add_orbit_marker(ax, E, S, name, size=14)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$|E|$"); ax.set_ylabel(r"$S_p$ (action)")
    ax.set_title("Action vs. energy: scaling-invariance check")
    ax.legend(loc="upper right", fontsize=9 if kind == "paper" else 11)

    save_both(fig, "action_vs_energy", HERE / "figures")
    plt.close(fig)


if __name__ == "__main__":
    for kind in ("paper", "blog"):
        render(kind)
    print("action_vs_energy_ok")
