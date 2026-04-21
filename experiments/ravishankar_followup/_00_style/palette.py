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
