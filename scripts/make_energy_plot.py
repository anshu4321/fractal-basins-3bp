"""Generate the figure-eight energy conservation plot.

Re-runs the 100-period integration and saves both the raw energy trace as
.npy and a publication-quality log-scale plot showing the bounded envelope
of |dE/E| sitting well below the 1e-10 gate.

Outputs:
    data/figure_eight_energy.npy
    figures/figure_eight_energy.png
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mega3bp.dynamics import total_energy
from mega3bp.integrators import yoshida6_integrate_energy_only
from mega3bp.orbits import FIGURE_EIGHT_PERIOD, figure_eight_state
from mega3bp.style import PALETTE, apply_dirac_style

GATE = 1e-10


def main() -> int:
    apply_dirac_style()
    print(f"jax backend : {jax.default_backend()}")
    print(f"devices     : {jax.devices()}")
    print()

    q0, p0 = figure_eight_state()
    E0 = float(total_energy(q0, p0))
    print(f"E(0) = {E0:.15e}")

    h = 0.005
    n_periods = 100
    n_steps = int(round(n_periods * FIGURE_EIGHT_PERIOD / h))
    print(f"step h = {h}, n_steps = {n_steps}, t_final = {n_steps * h:.4f}")

    print("integrating...")
    t0 = time.perf_counter()
    _, _, energies = yoshida6_integrate_energy_only(q0, p0, h, n_steps)
    energies.block_until_ready()
    print(f"  done in {time.perf_counter() - t0:.2f}s")

    energies_np = np.asarray(energies, dtype=np.float64)
    rel_err = (energies_np - E0) / E0
    abs_rel = np.abs(rel_err)
    t = np.arange(len(energies_np)) * h
    t_in_periods = t / FIGURE_EIGHT_PERIOD

    out_data = PROJECT_ROOT / "data"
    out_fig = PROJECT_ROOT / "figures"
    out_data.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)

    np.save(out_data / "figure_eight_energy.npy", energies_np)
    print(f"saved {out_data / 'figure_eight_energy.npy'}  ({energies_np.nbytes / 1024:.0f} kB)")

    # Replace zeros with floor for log plot
    floor = 1e-17
    plot_y = np.where(abs_rel > floor, abs_rel, floor)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    ax.semilogy(t_in_periods, plot_y, lw=0.55, color=PALETTE["cyan"], alpha=0.92)
    ax.axhline(GATE, color=PALETTE["coral"], ls="--", lw=1.3, label=f"gate = {GATE:.0e}")
    ax.set_xlabel("time / figure-eight period $T$")
    ax.set_ylabel(r"$|\Delta E / E|$")
    ax.set_title(
        "Chenciner-Montgomery figure-eight — energy conservation\n"
        rf"Yoshida 6th-order symplectic,  $h = {h}$,  $N = {n_steps:,}$ steps"
    )
    ax.set_xlim(0, n_periods)
    ax.set_ylim(1e-17, 1e-8)
    ax.grid(True, which="both", alpha=0.6)

    leg = ax.legend(loc="upper right")

    max_rel = float(abs_rel.max())
    ax.text(
        0.02,
        0.97,
        rf"max $|\Delta E/E|$ = {max_rel:.2e}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        color=PALETTE["text"],
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor=PALETTE["bg_elevate"],
            edgecolor=PALETTE["spine"],
        ),
    )

    fig.tight_layout()
    fig.savefig(out_fig / "figure_eight_energy.png", facecolor=PALETTE["bg_deep"])
    print(f"saved {out_fig / 'figure_eight_energy.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
