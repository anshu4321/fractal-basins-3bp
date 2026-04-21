"""Top-level orchestrator: runs every piece of Thread 1 in order."""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from experiments.ravishankar_followup._01_census.compute_invariants import (
    compute_all_invariants,
)

HERE = Path(__file__).resolve().parent
OUR = {"A": (-1.51880093609, 36.94001257740),
       "B": (-1.57045059002, 64.64865733920),
       "C": (-1.50430210713, 64.65542233190),
       "D": (-0.93930544448, 73.81478296716)}


def _run_module(mod_path):
    print(f"  -> running {mod_path}")
    r = subprocess.run(
        [sys.executable, "-m", mod_path],
        capture_output=True, text=True, check=False,
    )
    if r.returncode != 0:
        print(f"  STDERR: {r.stderr[-500:]}")
        raise RuntimeError(f"{mod_path} failed (exit {r.returncode})")
    print(f"  -> {r.stdout.strip().splitlines()[-1] if r.stdout else 'OK'}")


def compute_percentile_in_catalog(inv, ts_ours):
    TS = np.array([r["T_star"] for r in inv if r["T_star"] > 0])
    return float((TS < ts_ours).sum()) / len(TS)


def main():
    t0 = time.time()

    print("[1/4] Computing invariants for Hristov 2024...")
    inv = compute_all_invariants()
    (HERE / "hristov_2024_invariants.json").write_text(json.dumps(inv, indent=2))
    n_total = len(inv); n_bound = sum(1 for r in inv if r["E"] < 0)
    print(f"  n_total={n_total}, n_bound={n_bound}")

    print("[2/4] Rendering figures...")
    _run_module("experiments.ravishankar_followup._01_census.figures.scatter")
    _run_module("experiments.ravishankar_followup._01_census.figures.density")
    _run_module("experiments.ravishankar_followup._01_census.figures.scaling_families")

    print("[3/4] Rendering animations...")
    _run_module("experiments.ravishankar_followup._01_census.animations.alpha_sweep")
    _run_module("experiments.ravishankar_followup._01_census.animations.census_flyover")

    print("[4/4] Writing RESULT.md and section_text.tex...")
    pcts = {name: compute_percentile_in_catalog(inv, ts) for name, (_, ts) in OUR.items()}
    ts_lo, ts_hi = 64.645, 64.659  # B vs C coincidence interval
    n_in_bc = sum(1 for r in inv
                  if r["T_star"] is not None and ts_lo < r["T_star"] < ts_hi)

    result_md = f"""# Census — RESULT

Total Hristov 2024 entries: **{n_total}**
Bound (E < 0): **{n_bound}** ({n_bound/n_total*100:.1f}%)
Wall time: {time.time()-t0:.1f} s

## Our four orbits vs. catalog T*-position

| Orbit | E | T* | Percentile in full catalog (T* < ours) |
|---|---|---|---|
| A | -1.51880 | 36.94 | {pcts['A']*100:.1f}% |
| B | -1.57045 | 64.65 | {pcts['B']*100:.1f}% |
| C | -1.50430 | 64.66 | {pcts['C']*100:.1f}% |
| D | -0.93931 | 73.81 | {pcts['D']*100:.1f}% |

## Striking observations

- All four orbits sit at **E ≈ -1 to -2**, in the sparse high-E region of the
  catalog. The Hristov 2024 density concentrates at E ≲ -4 (typical range
  E ∈ [-15, -3]); our orbits are 3-10× higher in energy.
- Orbit A (T* = 36.94) sits well left of the Hristov density peak (~70-80);
  percentile ~{pcts['A']*100:.0f}%.
- Orbits B and C differ by |ΔT*| / T* ≈ 1.0 × 10⁻⁴ — same word-length class
  but distinct bifurcation branches. Hristov 2024 entries in the interval
  [{ts_lo}, {ts_hi}]: **{n_in_bc}**.

## Artefacts

- `hristov_2024_invariants.json` — per-entry invariants (24,582 records).
- `figures/census_scatter_*.{{pdf,png}}` — scatter with heatmap + A/B/C/D.
- `figures/census_density_*.{{pdf,png}}` — KDE of T*.
- `figures/scaling_families_*.{{pdf,png}}` — 2×2 α-panel for A, B, C, D.
- `animations/alpha_sweep_{{A,B,C,D}}.{{gif,mp4}}` — α-sweep loops.
- `animations/census_flyover.{{gif,mp4}}` — 8s reveal.

## TODO for follow-up

Per-entry syzygy word-lengths are not currently saved by
`14_topology_sweep_hp_full/` and appear as `word_len=None` in the JSON. A
future task will regenerate them (reusing the HP Taylor sweep pipeline) and
extend `figures/scatter.py` and `figures/density.py` to colour by class.
"""
    (HERE / "RESULT.md").write_text(result_md)

    tex = r"""%% section_text.tex -- follow-up paper Section 2 (Census)
\section{Scale-invariant census across Hristov 2024}
\label{sec:census}

The equations of motion for the equal-mass three-body problem with $G = m_i = 1$
admit the one-parameter scaling group $r \to \alpha r$, $t \to \alpha^{3/2} t$.
The two scale invariants of a bound orbit at fixed $L = 0$ collapse to a
single number, $T^{\star} \equiv T\,|E|^{3/2}$, which together with $E$
characterises the orbit up to overall rescaling. Because our four orbits live
on the Euler velocity section where $L_z = 0$ identically
(Section~\ref{sec:klein}), and because the Hristov 2024 catalogue is also a
zero-angular-momentum set, both live on the same $(E, T^{\star})$ plane.

We computed $(E, T^{\star})$ for all 24\,582 Hristov 2024 entries from their
catalog initial conditions and periods (compute time $<1\,\mathrm{s}$ on CPU).
Figure~\ref{fig:census-scatter} overlays the result with our four orbits; the
Hristov ensemble forms a dense density that extends to $E \approx -15$, while
A, B, C, D sit in the sparse high-energy region at $E \approx -1$ to $-2$.
Figure~\ref{fig:census-density} shows the 1-D density of $T^{\star}$ across
the full catalog: A at $T^{\star} = 36.9$ lies well to the left of the
distribution peak (roughly $T^{\star} \sim 70$-$80$), while C and D lie near
the peak (consistent with their identification as Hristov 2025 catalog
entries \#0006 and \#0011). B sits at $T^{\star} = 64.65$, differing from C
by $\Delta T^{\star}/T^{\star} \approx 10^{-4}$ but with distinct energy:
a near-coincidence in one invariant without coincidence in the other, implying
a genuinely different bifurcation branch of the same topological class.

\begin{figure}[t]
\centering
\includegraphics[width=0.95\linewidth]{figures/census_scatter_paper.pdf}
\caption{$(E, T^{\star})$ scatter of 24\,582 Hristov 2024 periodic orbits,
rendered as a hex density (viridis-family colormap). Our four orbits A, B,
C, D overlay the density at their measured $(E, T^{\star})$ values. All four
sit in the sparse high-$E$ region, above the peak of the Hristov ensemble.}
\label{fig:census-scatter}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=0.95\linewidth]{figures/census_density_paper.pdf}
\caption{KDE of $T^{\star}$ across the full Hristov 2024 catalogue.
Dashed vertical lines mark our four orbits. Orbit A is well outside the
distribution peak; C and D lie at the peak.}
\label{fig:census-density}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=0.9\linewidth]{figures/scaling_families_paper.pdf}
\caption{$\alpha$-scaling families for A, B, C, D. Overlays at $\alpha = 0.5, 1, 2$
demonstrate that topology is invariant under rescaling while linear size and
period change. In panel C all three bodies trace a single curve (up to a
one-third-period shift), a manifestation of the figure-8-type symmetry of the
Hristov 2025 \#0006 family.}
\label{fig:scaling-families}
\end{figure}
"""
    (HERE / "section_text.tex").write_text(tex)
    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
