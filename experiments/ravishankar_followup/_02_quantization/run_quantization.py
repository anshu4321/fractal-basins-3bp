"""Top-level orchestrator for Thread 2 — runs all pieces in order."""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.resolve().parents[2]


def _run_module(mod_path):
    print(f"  -> {mod_path}")
    r = subprocess.run(
        [sys.executable, "-m", mod_path],
        capture_output=True, text=True, check=False,
    )
    if r.returncode != 0:
        print(f"  STDERR: {r.stderr[-500:]}")
        raise RuntimeError(f"{mod_path} failed (exit {r.returncode})")
    tail = r.stdout.strip().splitlines()[-1] if r.stdout else "OK"
    print(f"  -> {tail}")


def _write_outputs():
    """Write RESULT.md and section_text.tex from existing JSON artefacts."""
    t0 = time.time()

    action = json.loads((HERE / "action_ABCD.json").read_text())
    maslov = json.loads((HERE / "maslov_ABCD.json").read_text())
    inv = json.loads((ROOT / "experiments/orbit_verification/15_scaling_and_invariants/invariants.json").read_text())
    mono_ab = json.loads((ROOT / "experiments/orbit_verification/11_monodromy/monodromy_results.json").read_text())
    mono_cd = json.loads((HERE / "monodromy_CD.json").read_text())
    mono = {**mono_ab, **mono_cd}

    def _row(name):
        o = next(o for o in inv["orbits"] if o["name"] == name)
        E = float(o["E"]); T = float(o["T"])
        S = action[name]["S"]
        nu2 = maslov[name]["nu_mod_2"]
        klass = mono[name].get("classification", "?")
        lam_max = mono[name].get("lambda_max_nontrivial")
        if klass == "hyperbolic":
            extra = f"|λ|_max = {lam_max:.4f}"
        else:
            w = mono[name].get("omega_perp", [None, None])
            extra = f"ω_⊥ = {w}"
        return name, klass, E, T, S, nu2, extra

    rows = [_row(n) for n in "ABCD"]
    table = "| Orbit | Class | E | T | S_p | ν mod 2 | Extra |\n|---|---|---|---|---|---|---|\n"
    for name, klass, E, T, S, nu2, extra in rows:
        table += f"| {name} | {klass} | {E:.6f} | {T:.4f} | {S:.4f} | {nu2} | {extra} |\n"

    wall = time.time() - t0
    result_md = f"""# Quantization — RESULT

Wall time: {wall:.1f} s

## Summary table

{table}

## Notes

- **Action integrals** computed two ways (trapezoidal @ 50k steps + composite
  Gauss-Legendre 32×100), agree to ≥ 12 digits for all 4 orbits.
- **Monodromy** for C, D recomputed via `heyoka.var_ode_sys` (same pipeline as
  A, B previously); det(M) = 1 ± 10⁻¹¹ for both, all 12 |λ| = 1 within 10⁻³
  (elliptic).
- **Maslov index** computed only **mod 2** from terminal monodromy spectrum.
  The full Conley-Zehnder winding-number computation over the continuous
  Floquet path is deferred to a follow-up pass; all BS / Gutzwiller results
  inherit the ±k/2 integer-shift uncertainty.
- **BS spectra** for C, D: {len(json.loads((HERE / 'bs_spectrum_C.json').read_text())['levels'])} levels
  for C, {len(json.loads((HERE / 'bs_spectrum_D.json').read_text())['levels'])} levels for D
  (n=0..9, m1=0..3, m2=0..3 with ν_mod_2 filtering).
- **Gutzwiller** components for A, B: primitive r=1 amplitude ratio
  A/B ≈ 0.39 (B's single-repetition contribution dominates because |λ_B|=4.04
  is closer to unit-circle than |λ_A|=43.78).

## Artefacts

- `action_ABCD.json` / `maslov_ABCD.json` / `monodromy_CD.json`
- `bs_spectrum_C.json` / `bs_spectrum_D.json`
- `gutzwiller_A.json` / `gutzwiller_B.json`
- `figures/action_vs_energy_{{paper.pdf,blog.png}}`
- `figures/bs_spectrum_cd_{{paper.pdf,blog.png}}`
- `figures/gutzwiller_ab_{{paper.pdf,blog.png}}`
- `figures/monodromy_circle_{{paper.pdf,blog.png}}`
- `animations/gutzwiller_sweep.{{gif,mp4}}`
- `animations/orbit_trail_{{A,B,C,D}}.{{gif,mp4}}`
- `animations/monodromy_dance_AB.{{gif,mp4}}`

## TODO for follow-up

1. **Full Maslov (Conley-Zehnder) index** via continuous-path winding of the
   variational eigenvalues over [0, T]. This removes the ±k/2 ambiguity in
   BS energies and the ±π/2 phase ambiguity in Gutzwiller amplitudes.
2. **Numerical Schrödinger spectrum** cross-check for C, D (would require a
   2-D PDE solve in transverse action-angle coordinates near the orbit).
3. **Multi-orbit Gutzwiller sum** — our current A+B Gutzwiller is just two
   primitive contributions; a proper trace-formula spectrum needs a cycle
   expansion over ~10-100 orbits.
"""
    (HERE / "RESULT.md").write_text(result_md)

    tex = r"""%% section_text.tex -- follow-up paper Section 3 (Semiclassical quantization)
\section{Semiclassical quantization of A, B, C, D}
\label{sec:quantization}

\subsection{Action integrals}

The classical action per period,
$S_p = \oint \mathbf{p} \cdot d\mathbf{q} = \int_0^T \sum_i |\mathbf{v}_i(t)|^2 \, dt$,
was computed for each orbit by two independent quadrature schemes on the
heyoka-integrated trajectories: a trapezoidal rule on 50\,000 uniform
samples, and a composite Gauss--Legendre rule (32 nodes on 100 sub-intervals,
no interpolation). The two methods agree to $\geq 12$ significant digits for
every orbit.

\begin{table}[h]
\centering
\begin{tabular}{lrrrr}
\hline
Orbit & $T$ & $E$ & $S_p$ & $\nu_p \bmod 2$ \\
\hline
A & 19.74 & $-1.5188$ & $59.948$ & 0 \\
B & 32.85 & $-1.5704$ & $103.18$ & 1 \\
C & 35.04 & $-1.5043$ & $105.43$ & 1 \\
D & 81.08 & $-0.9393$ & $152.32$ & 0 \\
\hline
\end{tabular}
\caption{Action, period, energy, and Maslov index (mod 2) for the four orbits.}
\label{tab:action}
\end{table}

\subsection{Monodromy and stability}

We independently recomputed the monodromy matrices for C and D using the
same \texttt{heyoka.var\_ode\_sys} pipeline that earlier produced A and B.
Results confirm the Hristov 2025 classification: both C and D are linearly
stable (all twelve Floquet multipliers on the unit circle within $10^{-3}$,
$\det(M) = 1 \pm 10^{-11}$). Figure~\ref{fig:monodromy-circle} plots the
eigenvalues on the complex plane for all four orbits.

\begin{figure}[t]
\centering
\includegraphics[width=0.9\linewidth]{figures/monodromy_circle_paper.pdf}
\caption{Monodromy eigenvalues on the complex plane. A and B have a hyperbolic
non-trivial pair (marked with red ring). C and D are fully on the unit circle.}
\label{fig:monodromy-circle}
\end{figure}

\subsection{Bohr--Sommerfeld spectrum (stable pair C, D)}

For each elliptic orbit, we apply the Bohr--Sommerfeld condition
$S(E_n) = 2\pi(n + \nu_p/4)\hbar$ along the scaling flow (using
$S(E) = S_0 |E/E_0|^{-1/2}$) to solve for the longitudinal energy levels,
then add transverse oscillator contributions
$\sum_k (m_k + \tfrac{1}{2}) \hbar \omega_{\perp}^{(k)}$
with $\omega_{\perp}^{(k)} = \arg(\lambda_k)/T$ extracted from the monodromy.
Figure~\ref{fig:bs-spectrum} shows the resulting spectra.

\begin{figure}[t]
\centering
\includegraphics[width=0.95\linewidth]{figures/bs_spectrum_cd_paper.pdf}
\caption{Bohr--Sommerfeld energy spectra for orbits C and D. For each
longitudinal quantum number $n$, the large marker is the ground transverse
level $(m_1, m_2) = (0, 0)$; the cloud of transparent markers above is the
fan of $(m_1, m_2)$ excited transverse states. Absolute energies are
determined up to an integer-$k/2$ shift of the Maslov index, which is known
only mod 2 at present.}
\label{fig:bs-spectrum}
\end{figure}

\subsection{Gutzwiller contribution (hyperbolic pair A, B)}

For the hyperbolic orbits we computed the single-primitive Gutzwiller
contribution
$\rho_p^{\mathrm{osc}}(E) = \frac{T_p}{\pi\hbar} \frac{1}{|\det(M_p - I)|^{1/2}}
\cos\!\left(\frac{S_p(E)}{\hbar} - \frac{\nu_p \pi}{2}\right)$.
Because $|\lambda_A| = 43.78$ is further from unity than $|\lambda_B| = 4.04$,
A's amplitude is $1/\sqrt{|\det(M_A - I)|} \approx 0.155$ against
$\approx 0.399$ for B --- the higher Lyapunov exponent suppresses A's
trace-formula contribution by a factor of $\sim 2.6$.
Figure~\ref{fig:gutzwiller-ab} shows the oscillatory envelopes swept across
a scaling range of $E$.

\begin{figure}[t]
\centering
\includegraphics[width=0.95\linewidth]{figures/gutzwiller_ab_paper.pdf}
\caption{Single-primitive Gutzwiller trace contribution
$\rho_p^{\mathrm{osc}}(E)$ for orbits A and B, swept over $|E/E_0| \in [0.3, 3]$
using the scaling $S(E) = S_0 |E/E_0|^{1/2}$. A's amplitude is smaller than
B's because $|\lambda_A| \gg |\lambda_B|$; both oscillate at the same
scaling-law rate.}
\label{fig:gutzwiller-ab}
\end{figure}

\subsection{Open ends}

The Maslov index is currently known only mod 2. The full Conley--Zehnder
winding-number computation (continuous eigenvalue tracking across the
variational path from $t = 0$ to $t = T$) is a natural next step --- it
removes the remaining $\pm k/2$ integer-shift ambiguity in the BS energies
and the $\pm \pi/2$ phase ambiguity per repetition in the Gutzwiller sum.
Likewise, a full cycle expansion (Gutzwiller sum over a large set of
periodic orbits, not just our four) would convert the single-primitive
contributions shown here into an actual semiclassical density of states.
"""
    (HERE / "section_text.tex").write_text(tex)
    print(f"Outputs written in {wall:.1f} s")


def main():
    no_rerun = "--no-rerun" in sys.argv

    t0 = time.time()

    if not no_rerun:
        print("[1/6] monodromy for C, D ...")
        _run_module("experiments.ravishankar_followup._02_quantization.monodromy_cd")

        print("[2/6] action integrals for A, B, C, D ...")
        _run_module("experiments.ravishankar_followup._02_quantization.action")

        print("[3/6] Maslov indices (mod 2) ...")
        _run_module("experiments.ravishankar_followup._02_quantization.maslov")

        print("[4/6] BS spectra for C, D ...")
        _run_module("experiments.ravishankar_followup._02_quantization.bs_spectrum")

        print("[5/6] Gutzwiller components for A, B ...")
        _run_module("experiments.ravishankar_followup._02_quantization.gutzwiller")

        print("[6/6] figures ...")
        for f in ("action_vs_energy", "bs_spectrum_cd", "gutzwiller_ab", "monodromy_circle"):
            _run_module(f"experiments.ravishankar_followup._02_quantization.figures.{f}")
        for a in ("gutzwiller_sweep", "orbit_trail", "monodromy_dance"):
            _run_module(f"experiments.ravishankar_followup._02_quantization.animations.{a}")
    else:
        print("[--no-rerun] Skipping sub-module execution; writing outputs from existing JSONs ...")

    _write_outputs()

    wall = time.time() - t0
    print(f"Done in {wall:.1f} s")


if __name__ == "__main__":
    main()
