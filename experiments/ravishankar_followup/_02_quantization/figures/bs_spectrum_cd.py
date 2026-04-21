"""Two-panel BS spectra for C and D."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from experiments.ravishankar_followup._00_style import (
    make_fig, save_both, ORBIT,
)
from experiments.ravishankar_followup._00_style.figure_helpers import (
    PAPER_STYLE, BLOG_STYLE,
)

HERE = Path(__file__).resolve().parent.parent


def render(kind="paper"):
    plt.style.use(str(PAPER_STYLE if kind == "paper" else BLOG_STYLE))
    fig, axes = plt.subplots(
        nrows=1, ncols=2,
        figsize=(10, 5) if kind == "blog" else (7.0, 3.5),
    )
    for ax, name in zip(axes, "CD"):
        spec = json.loads((HERE / f"bs_spectrum_{name}.json").read_text())
        lvls = spec["levels"]
        n_vals = sorted(set(l["n"] for l in lvls))
        for n in n_vals:
            E_long = next(l["E_long"] for l in lvls if l["n"] == n)
            E_full = [l["E"] for l in lvls if l["n"] == n]
            ax.scatter([n] * len(E_full), E_full, s=14, c=ORBIT[name],
                       alpha=0.5, edgecolors="none")
            # Ground transverse level highlighted
            ax.scatter([n], [E_long], s=50, c=ORBIT[name],
                       edgecolors="black", linewidths=0.7, zorder=10)
        ax.set_xlabel(r"longitudinal quantum $n$")
        ax.set_ylabel(r"$E_{n, m_1, m_2}$")
        ax.set_title(fr"Orbit {name} (elliptic), $\nu_{{\mathrm{{mod}}\,2}} = {spec['nu_mod_2']}$")
        ax.axhline(spec["E0_reference"], color="#888888", linestyle=":",
                   linewidth=0.8, alpha=0.7, label=r"$E_0$ (classical)")
        ax.legend(loc="lower right", fontsize=8 if kind == "paper" else 10)

    fig.suptitle("Bohr–Sommerfeld spectra (marker = ground; transparent = transverse levels)",
                 fontsize=12 if kind == "paper" else 14)
    fig.tight_layout()
    save_both(fig, "bs_spectrum_cd", HERE / "figures")
    plt.close(fig)


if __name__ == "__main__":
    for kind in ("paper", "blog"):
        render(kind)
    print("bs_spectrum_cd_ok")
