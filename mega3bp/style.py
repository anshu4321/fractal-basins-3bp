"""Dirac Labs-inspired matplotlib theme for the 3BP basin project.

Palette distilled from the published colors on diraclabs.com and refined
for scientific visualization: dark navy background, ice-blue text, cyan +
lime + lavender accents for data. The three accent colors correspond to
the three escape outcomes in the 3BP, mirroring the S_3 symmetry of the
equal-mass problem.

Usage:
    from mega3bp.style import apply_dirac_style, PALETTE, BASIN_COLORS
    apply_dirac_style()
    fig, ax = plt.subplots(...)

Avoid importing matplotlib at module top-level so this module can be
safely imported from code that does not need plotting.
"""
from __future__ import annotations

PALETTE: dict[str, str] = {
    # Surfaces
    "bg_deep":    "#0a0e1a",  # figure background (softer than pure #000)
    "bg_panel":   "#151a2b",  # axes / panel background
    "bg_elevate": "#1b2236",  # elevated surfaces, legends
    "grid":       "#1e2638",  # grid lines (very subtle)
    "spine":      "#2a3146",  # axes spines
    "divider":    "#39425a",  # figure dividers
    # Text
    "text":       "#e8f1fa",  # primary text — ice blue, not harsh white
    "text_mute":  "#8a95a8",  # secondary/tick labels
    "text_dim":   "#5c6477",  # captions, annotations
    # Accents (from Dirac palette, refined)
    "cyan":       "#00aeff",  # electric cyan — primary accent
    "cyan_glow":  "#abf1ff",  # ice blue — soft glow
    "lime":       "#b8e838",  # neon lime — desaturated for readability
    "lime_glow":  "#e0ff7a",  # lime glow
    "lavender":   "#b17ed9",  # soft lavender
    "lav_glow":   "#d8bcf2",  # lavender glow
    "coral":      "#ff6b6b",  # warning / bound basin
    "coral_glow": "#ffb3b3",
    "teal":       "#40d09b",  # mint teal — secondary
    "amber":      "#ffc857",  # amber — additional accent
}

# Basin outcome colors — the S_3 triple + bound + ambiguous
BASIN_COLORS: dict[str, str] = {
    "body1_escape": PALETTE["cyan"],
    "body2_escape": PALETTE["lime"],
    "body3_escape": PALETTE["lavender"],
    "bound":        PALETTE["coral"],
    "ambiguous":    "#4a5266",   # muted slate
}

# Matching glow/halo colors for animations
BASIN_GLOW: dict[str, str] = {
    "body1_escape": PALETTE["cyan_glow"],
    "body2_escape": PALETTE["lime_glow"],
    "body3_escape": PALETTE["lav_glow"],
    "bound":        PALETTE["coral_glow"],
    "ambiguous":    "#6b7488",
}

# Per-body colors (used when drawing trajectories with body-indexed colors,
# independent of escape outcome). These stay consistent across all figures.
BODY_COLORS: tuple[str, str, str] = (
    PALETTE["cyan"],
    PALETTE["lime"],
    PALETTE["lavender"],
)
BODY_GLOW: tuple[str, str, str] = (
    PALETTE["cyan_glow"],
    PALETTE["lime_glow"],
    PALETTE["lav_glow"],
)

# Cycle for default line color when not otherwise specified
_ACCENT_CYCLE = [
    PALETTE["cyan"],
    PALETTE["lime"],
    PALETTE["lavender"],
    PALETTE["coral"],
    PALETTE["teal"],
    PALETTE["amber"],
    PALETTE["cyan_glow"],
]


def apply_dirac_style() -> None:
    """Register the Dirac-inspired dark theme as the current matplotlib rcParams."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            # Backgrounds
            "figure.facecolor":  PALETTE["bg_deep"],
            "axes.facecolor":    PALETTE["bg_panel"],
            "savefig.facecolor": PALETTE["bg_deep"],
            "savefig.edgecolor": PALETTE["bg_deep"],
            # Text
            "text.color":      PALETTE["text"],
            "axes.labelcolor": PALETTE["text"],
            "axes.titlecolor": PALETTE["text"],
            "xtick.color":     PALETTE["text_mute"],
            "ytick.color":     PALETTE["text_mute"],
            # Spines and grid
            "axes.edgecolor": PALETTE["spine"],
            "axes.linewidth": 0.9,
            "axes.grid":      True,
            "grid.color":     PALETTE["grid"],
            "grid.linewidth": 0.5,
            "grid.alpha":     0.7,
            "grid.linestyle": "-",
            # Default color cycle
            "axes.prop_cycle": mpl.cycler("color", _ACCENT_CYCLE),
            # Fonts
            "font.family":     "sans-serif",
            "font.sans-serif": ["Inter", "SF Pro Display", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size":       10,
            "axes.titlesize":  13,
            "axes.titleweight":"normal",
            "axes.labelsize":  11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            # Legend
            "legend.frameon":    True,
            "legend.facecolor":  PALETTE["bg_elevate"],
            "legend.edgecolor":  PALETTE["spine"],
            "legend.framealpha": 0.92,
            # Quality
            "figure.dpi":     120,
            "savefig.dpi":    160,
            "savefig.bbox":   "tight",
            "lines.linewidth": 1.6,
            "lines.antialiased": True,
            "animation.html": "html5",
            "patch.edgecolor": PALETTE["spine"],
        }
    )


def dirac_gradient(n: int = 256) -> list[tuple[float, str]]:
    """Return a smooth cyan->lime gradient as a matplotlib-compatible LinearSegmentedColormap input.

    Intended use: continuous colormaps for heatmaps (e.g. density on shape sphere).
    """
    stops = [
        (0.00, PALETTE["bg_deep"]),
        (0.25, "#0e2a46"),
        (0.50, PALETTE["cyan"]),
        (0.75, PALETTE["lime"]),
        (1.00, PALETTE["lime_glow"]),
    ]
    return stops


def make_dirac_cmap(name: str = "dirac"):
    """Build and return a matplotlib LinearSegmentedColormap in Dirac aesthetic."""
    from matplotlib.colors import LinearSegmentedColormap

    stops = dirac_gradient()
    return LinearSegmentedColormap.from_list(name, stops)
