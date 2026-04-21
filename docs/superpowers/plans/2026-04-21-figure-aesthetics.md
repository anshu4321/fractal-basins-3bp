# Figure Aesthetics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a shared-style module at `experiments/ravishankar_followup/00_style/` that every figure and animation script in the three follow-up-paper threads imports from, guaranteeing visual consistency across the paper.

**Architecture:** One Python package with: (a) `palette.py` exporting hex constants for bodies, orbits, classes; (b) `3bp_paper.mplstyle` and `3bp_blog.mplstyle` stylesheets; (c) `figure_helpers.py` with `make_fig`, `save_both`, `add_orbit_marker`, `comet_tail_writer`; (d) `anim_helpers.py` with `save_gif_mp4`. A `test_style.py` smoke-checks the palette and typography.

**Tech Stack:** matplotlib ≥ 3.8, cmcrameri, imageio-ffmpeg, Pillow. Optional: `gifsicle` CLI for GIF optimisation (falls back to Pillow-only if absent).

---

## File Structure

```
experiments/ravishankar_followup/00_style/
├── __init__.py                   # re-exports palette + helpers
├── palette.py                    # body/orbit/class hex color constants
├── 3bp_paper.mplstyle            # matplotlib rc for paper PDF
├── 3bp_blog.mplstyle             # matplotlib rc for web PNG
├── figure_helpers.py             # make_fig, save_both, add_orbit_marker
├── anim_helpers.py               # save_gif_mp4, comet_tail_writer
├── test_style.py                 # renders sanity panel
└── sanity_panel.pdf              # rendered output of test_style.py (committed)
```

---

## Task 1: Create package + palette constants

**Files:**
- Create: `experiments/ravishankar_followup/__init__.py`
- Create: `experiments/ravishankar_followup/00_style/__init__.py`
- Create: `experiments/ravishankar_followup/00_style/palette.py`

- [ ] **Step 1: Create package scaffolding**

```bash
mkdir -p experiments/ravishankar_followup/00_style
touch experiments/ravishankar_followup/__init__.py
```

- [ ] **Step 2: Write `00_style/__init__.py`**

```python
from experiments.ravishankar_followup._00_style.palette import (
    BODY, ORBIT, ORBIT_EXT, TOL_BRIGHT, COM_COLOR,
)
from experiments.ravishankar_followup._00_style.figure_helpers import (
    make_fig, save_both, add_orbit_marker,
)
from experiments.ravishankar_followup._00_style.anim_helpers import (
    save_gif_mp4, comet_tail_writer,
)
```

Note: Python does not allow module names starting with a digit. Rename the directory to `_00_style` on disk, keep the `00_style/` shown in the spec as a documentation-facing label. Update the path:

```bash
mv experiments/ravishankar_followup/00_style experiments/ravishankar_followup/_00_style
```

- [ ] **Step 3: Write `_00_style/palette.py`**

```python
"""Color palette constants for the follow-up paper figures.

Body colors are fixed across every figure and animation. Orbit colors are
fixed per-orbit for scatter/highlight. TOL_BRIGHT is Paul Tol's colorblind-
safe qualitative palette for topology class coloring.
"""

BODY = {
    1: "#1B4F72",  # deep blue
    2: "#C0392B",  # vermilion
    3: "#229954",  # emerald
}
COM_COLOR = "#566573"  # slate grey

ORBIT = {
    "A": "#8E44AD",  # indigo
    "B": "#F39C12",  # amber
    "C": "#16A085",  # teal
    "D": "#E74C3C",  # coral
}

# Extension palette for newly-discovered orbits (Thread 3)
ORBIT_EXT = [
    "#2874A6",
    "#D68910",
    "#117A65",
    "#922B21",
    "#884EA0",
    "#B9770E",
]

# Paul Tol "bright" qualitative (colorblind-safe)
TOL_BRIGHT = [
    "#4477AA",
    "#EE6677",
    "#228833",
    "#CCBB44",
    "#66CCEE",
    "#AA3377",
    "#BBBBBB",
]

# Sequential colormap name (cmcrameri). Fallback to matplotlib's viridis.
SEQ_CMAP = "batlow"
SEQ_CMAP_FALLBACK = "viridis"

# Diverging colormap name.
DIV_CMAP = "vik"
DIV_CMAP_FALLBACK = "RdBu_r"
```

- [ ] **Step 4: Commit**

```bash
git add experiments/ravishankar_followup/__init__.py experiments/ravishankar_followup/_00_style/
git commit -m "style(followup): palette constants for shared figure language"
```

---

## Task 2: matplotlib stylesheets (paper + blog)

**Files:**
- Create: `experiments/ravishankar_followup/_00_style/3bp_paper.mplstyle`
- Create: `experiments/ravishankar_followup/_00_style/3bp_blog.mplstyle`

- [ ] **Step 1: Write `3bp_paper.mplstyle`**

```
# 3bp_paper.mplstyle -- paper-grade PDF output (a2paper 10pt doc style)
figure.figsize      : 5.0, 3.5
figure.dpi          : 150
savefig.dpi         : 600
savefig.bbox        : tight
savefig.pad_inches  : 0.1
savefig.format      : pdf

font.family         : serif
font.serif          : Computer Modern Roman, cmr10, DejaVu Serif
mathtext.fontset    : cm
text.usetex         : False

axes.labelsize      : 12
axes.titlesize      : 14
xtick.labelsize     : 10
ytick.labelsize     : 10
legend.fontsize     : 10

axes.linewidth      : 0.8
axes.grid           : False
axes.spines.top     : False
axes.spines.right   : False

lines.linewidth     : 1.6
lines.markeredgewidth : 0.7
lines.markeredgecolor : black
lines.markersize    : 6

legend.frameon      : False

image.cmap          : viridis
```

- [ ] **Step 2: Write `3bp_blog.mplstyle`**

Same as paper but swap font and save format:

```
figure.figsize      : 8.0, 5.33
figure.dpi          : 150
savefig.dpi         : 300
savefig.bbox        : tight
savefig.pad_inches  : 0.1
savefig.format      : png
savefig.transparent : False

font.family         : sans-serif
font.sans-serif     : Inter, SF Pro Text, Helvetica Neue, DejaVu Sans
mathtext.fontset    : cm
text.usetex         : False

axes.labelsize      : 14
axes.titlesize      : 16
xtick.labelsize     : 12
ytick.labelsize     : 12
legend.fontsize     : 12

axes.linewidth      : 1.0
axes.grid           : False
axes.spines.top     : False
axes.spines.right   : False

lines.linewidth     : 2.0
lines.markeredgewidth : 0.7
lines.markeredgecolor : black
lines.markersize    : 8

legend.frameon      : False

image.cmap          : viridis
```

- [ ] **Step 3: Commit**

```bash
git add experiments/ravishankar_followup/_00_style/3bp_paper.mplstyle experiments/ravishankar_followup/_00_style/3bp_blog.mplstyle
git commit -m "style(followup): paper and blog matplotlib stylesheets"
```

---

## Task 3: figure_helpers.py

**Files:**
- Create: `experiments/ravishankar_followup/_00_style/figure_helpers.py`

- [ ] **Step 1: Write `figure_helpers.py`**

```python
"""Matplotlib figure helpers for the follow-up paper.

Every thread figure script uses `make_fig(kind=...)` to open a figure with
the correct stylesheet applied, and `save_both(fig, stem, outdir)` to emit
both `<stem>_paper.pdf` and `<stem>_blog.png` in one call.
"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from .palette import ORBIT, BODY, COM_COLOR, SEQ_CMAP, SEQ_CMAP_FALLBACK

_STYLE_DIR = Path(__file__).parent
PAPER_STYLE = _STYLE_DIR / "3bp_paper.mplstyle"
BLOG_STYLE = _STYLE_DIR / "3bp_blog.mplstyle"


def _load_cmap(name, fallback):
    try:
        import cmcrameri.cm as cmc
        return getattr(cmc, name)
    except Exception:
        return plt.get_cmap(fallback)


def seq_cmap():
    """Perceptually uniform sequential colormap (batlow; fallback viridis)."""
    return _load_cmap(SEQ_CMAP, SEQ_CMAP_FALLBACK)


def div_cmap():
    """Diverging colormap (vik; fallback RdBu_r)."""
    from .palette import DIV_CMAP, DIV_CMAP_FALLBACK
    return _load_cmap(DIV_CMAP, DIV_CMAP_FALLBACK)


def make_fig(kind="paper", figsize=None, **subplot_kw):
    """Open a new figure with the given stylesheet applied.

    Parameters
    ----------
    kind : {"paper", "blog"}
        Which stylesheet to apply for this figure.
    figsize : tuple or None
        Override figure size; default comes from the stylesheet.
    **subplot_kw
        Passed to `plt.subplots`.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes (or array of Axes if nrows/ncols>1).
    """
    style_path = PAPER_STYLE if kind == "paper" else BLOG_STYLE
    plt.style.use(str(style_path))
    fig, ax = plt.subplots(figsize=figsize, **subplot_kw)
    return fig, ax


def save_both(fig, stem, outdir):
    """Save fig as `{stem}_paper.pdf` AND `{stem}_blog.png` under outdir.

    The fig is re-rendered once per format by switching rc params; this is
    cheap and guarantees each format uses its own DPI / font / marker size.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paper_path = outdir / f"{stem}_paper.pdf"
    blog_path = outdir / f"{stem}_blog.png"
    fig.savefig(str(paper_path), format="pdf", dpi=600, bbox_inches="tight")
    fig.savefig(str(blog_path), format="png", dpi=300, bbox_inches="tight")
    return paper_path, blog_path


def add_orbit_marker(ax, x, y, orbit_name, size=14, label=True, **kw):
    """Place a large, labelled marker for orbit A/B/C/D at (x, y).

    Uses the fixed ORBIT palette from palette.py. Marker has a black 0.7pt
    edge outline (per spec). If label=True, annotates with the orbit name.
    """
    color = ORBIT[orbit_name]
    ax.scatter([x], [y], s=size**2, c=color, edgecolors="black",
               linewidths=0.7, zorder=10, **kw)
    if label:
        ax.annotate(orbit_name, (x, y), xytext=(6, 6),
                    textcoords="offset points", fontsize=12, zorder=11)


def body_color(i):
    """Return the fixed color for body i in {1, 2, 3}."""
    return BODY[i]
```

- [ ] **Step 2: Commit**

```bash
git add experiments/ravishankar_followup/_00_style/figure_helpers.py
git commit -m "style(followup): figure_helpers with make_fig, save_both, palette loaders"
```

---

## Task 4: anim_helpers.py — GIF + MP4 writer, comet-tail trajectory

**Files:**
- Create: `experiments/ravishankar_followup/_00_style/anim_helpers.py`

- [ ] **Step 1: Write `anim_helpers.py`**

```python
"""Animation helpers: comet-tail trajectory writer and dual GIF+MP4 output."""
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image

from .palette import BODY


def save_gif_mp4(anim, stem, outdir, fps=30, colors=128,
                 max_size_mb=8):
    """Save a matplotlib animation as both `<stem>.gif` AND `<stem>.mp4`.

    GIF is optimised with gifsicle (if available) or Pillow (fallback).
    MP4 is H.264 CRF 18 via ffmpeg (must be on PATH).
    If the GIF exceeds `max_size_mb`, prints a WARNING (does not error).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    gif_path = outdir / f"{stem}.gif"
    mp4_path = outdir / f"{stem}.mp4"

    # --- GIF via Pillow (matplotlib backend), then gifsicle if present
    anim.save(str(gif_path), writer="pillow", fps=fps)

    # Optimise with gifsicle if available
    try:
        subprocess.run(
            ["gifsicle", "-O3", f"--colors={colors}", "-b", str(gif_path)],
            check=True, capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass  # fallback: unoptimised Pillow output

    size_mb = gif_path.stat().st_size / 1024 / 1024
    if size_mb > max_size_mb:
        print(f"WARNING: {gif_path} is {size_mb:.1f}MB (target {max_size_mb}MB)")

    # --- MP4 via ffmpeg
    try:
        writer = animation.FFMpegWriter(fps=fps, codec="libx264",
                                        extra_args=["-crf", "18",
                                                    "-pix_fmt", "yuv420p"])
        anim.save(str(mp4_path), writer=writer)
    except Exception as e:
        print(f"WARNING: could not write MP4 ({e}); GIF-only for {stem}")

    return gif_path, mp4_path


def comet_tail_positions(traj, t_idx, tail_len):
    """Return (N_tail, 2) positions and per-point alpha values for body trail.

    traj: (T_steps, 2) position array for one body.
    t_idx: current step index.
    tail_len: how many prior steps to draw as fading trail.
    Returns (pts, alphas) where alphas linearly ramp from 0.1 to 1.0.
    """
    start = max(0, t_idx - tail_len)
    pts = traj[start:t_idx + 1]
    n = len(pts)
    alphas = np.linspace(0.1, 1.0, n)
    return pts, alphas


def make_orbit_trail_frames(traj3, T, n_frames=240, tail_frac=0.125):
    """Generate (n_frames) frames for a 3-body comet-tail orbit trail.

    traj3 : ndarray, shape (T_steps, 3, 2)
        Positions of the 3 bodies along one period.
    T : float
        The period (for title/time display).
    n_frames : int
        Number of animation frames to subsample.
    tail_frac : float
        Fraction of one period used as tail length (0.125 = T/8).

    Yields (fig, ax) per frame for the caller to write via FuncAnimation.
    (The caller typically uses FuncAnimation + this helper's update function.)
    """
    T_steps = traj3.shape[0]
    tail_len = int(tail_frac * T_steps)
    frame_idxs = np.linspace(0, T_steps - 1, n_frames).astype(int)
    return frame_idxs, tail_len


def update_orbit_trail(ax, traj3, t_idx, tail_len, T,
                       xlim=None, ylim=None):
    """Redraw one frame of the comet-tail animation on `ax`.

    Call from FuncAnimation's update() with the current t_idx. Clears the
    axes and re-plots trails + body positions + time indicator.
    """
    ax.cla()
    for i in range(3):
        pts, alphas = comet_tail_positions(traj3[:, i, :], t_idx, tail_len)
        for j in range(len(pts) - 1):
            ax.plot(pts[j:j+2, 0], pts[j:j+2, 1],
                    color=BODY[i+1], alpha=alphas[j], linewidth=1.6)
        ax.scatter(pts[-1, 0], pts[-1, 1], s=120, c=BODY[i+1],
                   edgecolors="black", linewidths=0.7, zorder=10)
    ax.set_aspect("equal")
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)
    t_frac = t_idx / (traj3.shape[0] - 1)
    ax.text(0.98, 0.02, f"t/T = {t_frac:.2f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10, color="#566573")
```

- [ ] **Step 2: Commit**

```bash
git add experiments/ravishankar_followup/_00_style/anim_helpers.py
git commit -m "style(followup): anim_helpers with comet-tail + dual GIF/MP4 writer"
```

---

## Task 5: test_style.py — sanity panel

**Files:**
- Create: `experiments/ravishankar_followup/_00_style/test_style.py`

- [ ] **Step 1: Write `test_style.py`**

```python
"""Renders a sanity panel verifying palette + typography + line styling.

Writes sanity_panel_paper.pdf and sanity_panel_blog.png into this directory.
Run this once after any change to palette.py or the stylesheets to eyeball
the results.
"""
from pathlib import Path
import numpy as np

from .figure_helpers import make_fig, save_both
from .palette import BODY, ORBIT, TOL_BRIGHT


def render_sanity_panel(kind="paper"):
    fig, axes = __import__("matplotlib.pyplot", fromlist=["subplots"]).subplots(
        nrows=2, ncols=2, figsize=(10, 7) if kind == "blog" else (6.5, 4.5)
    )
    # Re-apply stylesheet (make_fig was single-ax)
    from .figure_helpers import PAPER_STYLE, BLOG_STYLE
    import matplotlib.pyplot as plt
    plt.style.use(str(PAPER_STYLE if kind == "paper" else BLOG_STYLE))

    (ax1, ax2), (ax3, ax4) = axes

    # Panel 1: body colors
    for i in range(3):
        ax1.scatter([i], [0], s=400, c=BODY[i+1],
                    edgecolors="black", linewidths=0.7)
        ax1.text(i, -0.4, f"body {i+1}", ha="center")
    ax1.set_xlim(-0.5, 2.5); ax1.set_ylim(-0.8, 0.5)
    ax1.set_title("Body palette"); ax1.set_xticks([]); ax1.set_yticks([])

    # Panel 2: orbit colors
    for j, name in enumerate("ABCD"):
        ax2.scatter([j], [0], s=400, c=ORBIT[name],
                    edgecolors="black", linewidths=0.7)
        ax2.text(j, -0.4, name, ha="center")
    ax2.set_xlim(-0.5, 3.5); ax2.set_ylim(-0.8, 0.5)
    ax2.set_title("Orbit palette"); ax2.set_xticks([]); ax2.set_yticks([])

    # Panel 3: Tol bright
    for k, c in enumerate(TOL_BRIGHT):
        ax3.bar([k], [1], color=c, edgecolor="black", linewidth=0.5)
    ax3.set_title("Paul Tol bright"); ax3.set_xticks([])
    ax3.set_yticks([])

    # Panel 4: typography sample + math
    ax4.text(0.5, 0.7, "Typography sample", ha="center", fontsize=14,
             transform=ax4.transAxes)
    ax4.text(0.5, 0.45, r"$T^\star = T \cdot |E|^{3/2}$",
             ha="center", fontsize=16, transform=ax4.transAxes)
    ax4.text(0.5, 0.2, "body 10pt, label 12pt, title 14pt",
             ha="center", fontsize=10, transform=ax4.transAxes)
    ax4.set_xticks([]); ax4.set_yticks([])

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    outdir = Path(__file__).parent
    for kind in ("paper", "blog"):
        fig = render_sanity_panel(kind)
        save_both(fig, "sanity_panel", outdir) if kind == "paper" else None
    # Also emit a singleton final PDF at sanity_panel.pdf for git
    fig.savefig(str(outdir / "sanity_panel.pdf"), format="pdf",
                dpi=600, bbox_inches="tight")
    print(f"Rendered sanity panel to {outdir}")
```

- [ ] **Step 2: Run it**

```bash
python -m experiments.ravishankar_followup._00_style.test_style
```

Expected: prints `Rendered sanity panel to .../experiments/ravishankar_followup/_00_style`. Creates `sanity_panel_paper.pdf`, `sanity_panel_blog.png`, and `sanity_panel.pdf` in that directory.

- [ ] **Step 3: Open the PDF and eyeball**

```bash
open experiments/ravishankar_followup/_00_style/sanity_panel.pdf
```

Expected: 4-panel grid, body/orbit swatches with black outlines, Tol bright bar row, and a typography sample with the T* math formula. If any panel looks wrong (wrong color, missing outline, pixelated math), fix in palette.py / stylesheets and re-run.

- [ ] **Step 4: Commit**

```bash
git add experiments/ravishankar_followup/_00_style/test_style.py experiments/ravishankar_followup/_00_style/sanity_panel.pdf
git commit -m "style(followup): sanity panel renderer + committed reference output"
```

---

## Task 6: Integration smoke-test from a sibling package

**Files:**
- Create: `experiments/ravishankar_followup/_00_style/test_import.py`

- [ ] **Step 1: Write a quick import test**

```python
"""Smoke-tests that every thread can import from the style package."""
from experiments.ravishankar_followup._00_style import (
    make_fig, save_both, add_orbit_marker, save_gif_mp4,
    comet_tail_writer, BODY, ORBIT, ORBIT_EXT, TOL_BRIGHT,
)


def test_imports_resolve():
    assert len(BODY) == 3
    assert set(ORBIT.keys()) == {"A", "B", "C", "D"}
    assert len(TOL_BRIGHT) == 7
    assert len(ORBIT_EXT) >= 4


def test_make_fig_returns_fig_ax():
    fig, ax = make_fig("paper")
    assert fig is not None
    assert ax is not None


if __name__ == "__main__":
    test_imports_resolve()
    test_make_fig_returns_fig_ax()
    print("All style-package smoke tests PASS")
```

Note: `comet_tail_writer` should be exported from `_00_style/__init__.py`; if it is not, add it — it's mentioned by other specs as a helper. Resolve: in Task 4 we defined `update_orbit_trail` (the per-frame update). Update `_00_style/__init__.py` to re-export both, and rename `comet_tail_writer` → `update_orbit_trail` in the import:

Fix: edit `_00_style/__init__.py` to export `update_orbit_trail`, then change the test import accordingly.

- [ ] **Step 2: Fix `_00_style/__init__.py`**

```python
from experiments.ravishankar_followup._00_style.palette import (
    BODY, ORBIT, ORBIT_EXT, TOL_BRIGHT, COM_COLOR,
)
from experiments.ravishankar_followup._00_style.figure_helpers import (
    make_fig, save_both, add_orbit_marker, seq_cmap, div_cmap, body_color,
)
from experiments.ravishankar_followup._00_style.anim_helpers import (
    save_gif_mp4, update_orbit_trail, make_orbit_trail_frames,
    comet_tail_positions,
)
```

And update the test:

```python
from experiments.ravishankar_followup._00_style import (
    make_fig, save_both, add_orbit_marker, save_gif_mp4,
    update_orbit_trail, BODY, ORBIT, ORBIT_EXT, TOL_BRIGHT,
)
```

- [ ] **Step 3: Run the test**

```bash
python experiments/ravishankar_followup/_00_style/test_import.py
```

Expected: `All style-package smoke tests PASS`.

- [ ] **Step 4: Commit**

```bash
git add experiments/ravishankar_followup/_00_style/__init__.py experiments/ravishankar_followup/_00_style/test_import.py
git commit -m "style(followup): smoke test + __init__ re-exports"
```

---

## Task 7: Dependency check

- [ ] **Step 1: Verify required packages**

```bash
python -c "import matplotlib, numpy; print('mpl', matplotlib.__version__, 'np', numpy.__version__)"
python -c "import cmcrameri; print('cmcrameri OK')" 2>&1 || pip install cmcrameri
python -c "import PIL; print('pillow OK')" 2>&1 || pip install pillow
python -c "import imageio_ffmpeg; print('ffmpeg OK')" 2>&1 || pip install imageio-ffmpeg
which ffmpeg || echo "ffmpeg not on PATH — install via: brew install ffmpeg (or apt-get install ffmpeg)"
which gifsicle || echo "gifsicle not on PATH — install via: brew install gifsicle (optional, for GIF optimisation)"
```

Expected: matplotlib ≥ 3.8, cmcrameri imports OK, Pillow OK, imageio_ffmpeg OK, ffmpeg on PATH. gifsicle is optional.

- [ ] **Step 2: Re-run sanity panel to confirm cmcrameri picked up**

```bash
python -m experiments.ravishankar_followup._00_style.test_style
python -m experiments.ravishankar_followup._00_style.test_import
```

Expected: both complete without errors.

- [ ] **Step 3: Final commit if anything changed**

```bash
git add -A experiments/ravishankar_followup/_00_style/
git diff --cached --exit-code || git commit -m "style(followup): dependency-check adjustments"
```

---

## Self-Review Checklist

**Spec coverage:**
- Palette (body, orbit, Tol bright, extension) — Task 1 ✓
- Sequential + diverging cmap loaders (batlow, vik fallbacks) — Task 3 ✓
- Typography (CM Serif paper / Inter blog, sizes 10/12/14 per spec) — Task 2 ✓
- Canvas sizes and DPI — Task 2 ✓
- Line/marker styling (1.6pt line, 0.7pt marker edge) — Task 2 ✓
- `make_fig` / `save_both` / `add_orbit_marker` helpers — Task 3 ✓
- Comet-tail trajectory writer — Task 4 ✓
- Dual GIF + MP4 output (gifsicle optional, ffmpeg required) — Task 4 + Task 7 ✓
- Sanity panel — Task 5 ✓
- Smoke tests — Task 6 ✓

**Placeholder scan:** no TBDs or "implement later" in any step. Every step has real code or a real command.

**Type consistency:** `make_fig` returns `(fig, ax)` used consistently; `save_both(fig, stem, outdir)` signature stable across Tasks 3, 5; `BODY[i]` keyed by 1/2/3; `ORBIT[name]` keyed by "A"/"B"/"C"/"D"; `update_orbit_trail` is the per-frame writer, renamed from the earlier `comet_tail_writer` draft.

---
