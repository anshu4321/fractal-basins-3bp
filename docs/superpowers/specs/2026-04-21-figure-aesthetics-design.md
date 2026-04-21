# Figure aesthetics (shared style spec for the follow-up paper)

**Date:** 2026-04-21
**Applies to:** all three Ravishankar-follow-up threads (census, quantization, equilateral).

---

## Goal

Every figure, table, and animation across the three threads shares one visual language, so the follow-up paper and any blog / deck derivatives feel like a single production. Publication-grade. No default-matplotlib-grey-dashed-grid artifacts. No unlabelled axes. No clipped legend text. Every deliverable is simultaneously **beautiful and technically precise**.

## Visual language (hard constraints)

### Color palette — body and orbit

All three bodies get fixed colors throughout every figure and every animation. Never recolor.

| Body | Hex     | Notes |
|---|---|---|
| 1 | `#1B4F72` | deep blue |
| 2 | `#C0392B` | vermilion |
| 3 | `#229954` | emerald |
| Center-of-mass / reference | `#566573` | slate grey |

Per-orbit color (for scatter plots where orbits are datapoints):

| Orbit | Hex     | Notes |
|---|---|---|
| A | `#8E44AD` | indigo |
| B | `#F39C12` | amber |
| C | `#16A085` | teal |
| D | `#E74C3C` | coral |

Any "new orbit" discovered in Thread 3 gets the next slot from a fixed extension palette: `#2874A6`, `#D68910`, `#117A65`, `#922B21`.

### Categorical palette for topology / classes

Use **Paul Tol's "bright" qualitative palette** (colorblind-safe) for the Hristov 2024 word-length bins in Thread 1:
- `#4477AA`, `#EE6677`, `#228833`, `#CCBB44`, `#66CCEE`, `#AA3377`, `#BBBBBB`.

For sequential / heat fields (label-disagreement in Thread 3): **`cmcrameri.batlow`** (perceptually uniform, colorblind-safe, print-safe). Fallback: `viridis`.

For diverging (signed quantities, Gutzwiller oscillations in Thread 2): **`cmcrameri.vik`**. Fallback: `RdBu_r`.

### Typography

Paper-grade PDFs: match the conference paper's main font — Computer Modern Serif for math and body, sans-serif for axis labels (CM Sans). Use LaTeX rendering when the figure ships to the paper.

Blog / web PNGs: Inter (sans) for body, JetBrains Mono for code / numeric tables, Computer Modern for math snippets (KaTeX-rendered or pre-baked PNG overlay).

Axis tick labels: 10pt. Axis labels: 12pt. Title: 14pt. Legend: 10pt. Never smaller than 8pt anywhere.

### Line / marker styling

- Line width default: 1.6 pt. Bold line: 2.4 pt. Axis spine: 0.8 pt.
- Never use dashed grid lines. If a grid is needed, use a light horizontal-only grid at `#EAEAEA`.
- Marker edge: always 0.7 pt black outline on filled markers (makes them readable on busy backgrounds).
- Marker size: scatter default 6 pt, highlight 14 pt (for A, B, C, D overlays).

### Canvas

- Paper PDF: a2paper 10pt, 5.0" × 3.5" single-column default, 5.0" × 7.0" for 2×2 panel, 10.0" × 3.5" for full-width single-row.
- Blog PNG: 1200 × 800 px base, 2400 × 1600 for retina.
- DPI: 600 for PDF (TeX-embedded vector where possible); 150 for PNG base, 300 for retina.
- Tight layout with `bbox_inches='tight'`, 0.1" padding, no exterior margin colour.

## Central style file

Produce once, reuse everywhere:

```
experiments/ravishankar_followup/00_style/
├── 3bp_paper.mplstyle       # matplotlib style sheet (paper)
├── 3bp_blog.mplstyle        # matplotlib style sheet (web)
├── palette.py               # Python module exporting body/orbit/class colors
├── figure_helpers.py        # make_fig(kind="paper"|"blog"), save_both(fig, stem), add_orbit_marker(...)
└── test_style.py            # renders a sanity panel to verify palette + typography
```

Every thread's figure script begins with:

```python
from experiments.ravishankar_followup._00_style.figure_helpers import make_fig, save_both, palette
```

and uses `save_both(fig, "figure_stem")` which emits `figure_stem_paper.pdf` and `figure_stem_blog.png` together. No thread ever hand-rolls its own style.

## Animation (GIF / MP4) standard

All animations delivered two ways:

1. **`.gif`** — for web and deck embed. Palette optimised with `gifsicle -O3 --colors 128`. Max 8 MB, aim for 2-4 MB. Loop infinite. 30 FPS playback.
2. **`.mp4`** — H.264, CRF 18, 60 FPS for smooth orbit visualization. For paper-grade static figures derived from animations, also emit a contact-sheet PDF with 6-8 frames at key times.

Animation canvas: 1200 × 800 (same as blog PNG). Production DPI 100 (smaller than static to keep file sizes manageable).

Every animation has:
- Wall-clock title text (orbit name, period T)
- Small time indicator (progress bar or t/T label) lower-right
- Bodies as filled circles with their fixed colors + black 0.7 pt outline
- Trailing fade (alpha gradient over the last 1/8 period) — the classic 3BP "comet tail"
- For scale-aware animations, add a small length-scale bar

## Per-thread figure inventory (with GIFs)

### Thread 1 — Scale-invariant census

Static figures:
- `census_scatter` — Hristov 2024 scatter in (E, T⋆), colored by word-length bin, A/B/C/D overlaid.
- `census_density` — per-class KDE along T⋆ axis, with A/B/C/D reference lines.
- `scaling_families` — 2×2 panel, α-family per orbit (α ∈ {0.5, 1, 2}) in configuration space.

Animations:
- `alpha_sweep_A.gif`, `_B.gif`, `_C.gif`, `_D.gif` — 4 GIFs, each showing one orbit smoothly rescaling over α ∈ [0.3, 3] in a 6-second loop. Zoom-level adjusts so the orbit fills the frame; this makes the *invariance of the shape under scaling* visually obvious. Shows T(α), E(α), T⋆ (constant!) as a readout that updates with α.
- `census_flyover.gif` — 8-second animation: start with an empty (E, T⋆) plane, points fade in bin by bin (length 20 first, then 30, ...), with the final frame highlighting A/B/C/D with marker-grow animation. Paper-embed as static final frame; web-embed as GIF.

### Thread 2 — Semiclassical quantization

Static figures:
- `action_vs_energy` — S_p(E) along scaling direction, log-log, showing the S ∝ |E|^{-1/2} relation; data points at (E, S) for A, B, C, D plus theory line.
- `bs_spectrum_CD` — quantum-number grid E_{n, m₁, m₂} for C and D, side by side. Annotate degeneracies.
- `gutzwiller_AB` — ρ^osc(E) over a scaling range for A and B, with amplitude envelopes showing |λ|^{-1/2} scaling difference.
- `monodromy_circle` — unit-circle plot showing the non-trivial eigenvalues of A, B, C, D with Krein signatures. Single figure, 4 panels.

Animations:
- `gutzwiller_sweep.gif` — 10-second animation: horizontal axis is E, the ρ^osc(E) curves for A and B draw themselves in time, with the amplitude ratio (43.78 vs 4.04 in |λ|) visible as a pulsation difference. Paper gets the static end-frame.
- `monodromy_dance_AB.gif` — 6-second loop: as t goes from 0 to T, the Floquet-propagator eigenvalues migrate along their paths in the complex plane, and at t=T they land at their monodromy positions. Visually demonstrates Maslov winding number. Useful for the Maslov-index exposition.
- `orbit_trail_A.gif`, `_B.gif`, `_C.gif`, `_D.gif` — 4 animations of each orbit trajectory (one full period, bodies traced with fading comet-tails). 8-second loop. These double as a general-purpose asset: they also illustrate the conference paper's orbits and can be re-used in the deck.

### Thread 3 — Equilateral-triangle section search

Static figures:
- `equilateral_basin` — 4-panel: (a) 4-class basin map on 256² grid, (b) LD field, (c) candidate overlay on LD field, (d) sample orbit trajectory per surviving candidate.
- `candidates_ld_scan` — LD vs. candidate-index bar chart, with survival-status color coding (survived Newton / HP / monodromy / novelty).
- `orbits_panel_equilateral` — 2×N panel of new orbits discovered (N = surviving count), each in configuration space with its syzygy word annotated.
- `section_comparison` — side-by-side schematic: Euler section vs. equilateral section, same style.

Animations:
- `basin_reveal.gif` — 10-second animation: the 256² basin paints itself in column by column (simulating the compute sweep); final frame static. Good for the deck.
- `ld_growth.gif` — 8-second loop: LD field evolves as the classifier trains (show 5 snapshots across epochs 1, 10, 25, 40, 50). Demonstrates the ML prior emerging.
- `new_orbit_X.gif` — one GIF per surviving HP-verified new orbit, same style as `orbit_trail_*` from Thread 2 (comet-tail, 8-second loop). These will become the "new orbit" assets for the paper and the deck.
- `equilateral_vs_euler.gif` — 8-second morphing animation: start with r₁=(−1,0), r₂=(+1,0), r₃=(0,0) (Euler); smoothly morph r_k to the equilateral vertices. Illustrates why this is a genuinely different section.

## Assembly and QC

Once all per-thread figures are generated, run a sanity pass:

1. Render all figures to a single "sheet" PDF (`followup_paper_figures_qc.pdf`) with 1 figure per page + metadata. Eyeball the full set for visual consistency — all orbit A's should be indigo, all LD fields should be batlow, etc.
2. For each animation, verify: frames are the right size; no jittering from matplotlib redraw artifacts; palette is optimised; file size ≤ 8 MB for GIFs.
3. Spot-check accessibility: render a greyscale version of every figure and verify that class distinctions remain legible (Paul Tol's bright palette passes this test; verify anyway).

## Acceptance criteria (for the aesthetics work itself)

1. `00_style/` module exists; all three threads' figure scripts import from it.
2. Running `test_style.py` produces a sanity panel showing the color palettes, typography samples, a body-triad scatter, and a sample orbit trail — all in the final paper style.
3. QC sheet (`followup_paper_figures_qc.pdf`) renders without errors and visually validates color consistency.
4. Every animation is produced in both `.gif` and `.mp4` forms and sits at the expected resolution / FPS / file size.
5. The conference paper's PDF, if re-compiled with the new style sheet, remains visually unchanged (we do not retroactively restyle the conference paper — the style sheet is forward-looking for the follow-up paper only).

## Dependencies

- None from other threads (this spec is a precondition for aesthetics in all three thread specs).
- **Consumed by**: every figure and animation script across all three threads.
- External: matplotlib ≥ 3.8, cmcrameri, imageio-ffmpeg (for mp4), gifsicle (for gif optimization), pillow.

## Estimated wall time

~90 minutes: palette setup ~15 min, style sheets + helpers ~30 min, test sanity panel ~15 min, animation helper (trail + fade + indicators) ~30 min.

The per-thread animations themselves are budgeted inside each thread's estimate, not here.
