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
