"""Insightful visualizations that teach physics.

1. Mollweide shape-sphere projection of all 10 Suvakov orbits
2. Pairwise distance "rhythm" (r12, r13, r23 vs time)
3. Energy breathing (KE, |PE| vs time — virial theorem)
4. Body-in-body frame (body 1 relative to body 2)
5. Velocity-space constellation (all orbits in v-space, colored by period)
6. Close-encounter heatmap
7. Moment of inertia breathing
"""
from __future__ import annotations

import sys
from pathlib import Path
from functools import partial

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

from mega3bp.integrators import yoshida6_step
from mega3bp.dynamics import total_energy, kinetic_energy, potential_energy
from mega3bp.shape_sphere import config_to_shape, moment_of_inertia, pairwise_distances

OUT_DIR = Path(__file__).parent
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

Q_COLLINEAR = jnp.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 0.0]], dtype=jnp.float64)

SUVAKOV = [
    ("Figure-eight",   0.3471168881, 0.5327249454, 6.3259139829),
    ("Butterfly I",    0.3068934205, 0.1255065670, 6.2346748391),
    ("Butterfly II",   0.392955223941802, 0.0975792352080344, 7.003707),
    ("Butterfly III",  0.4059155671, 0.2301631260, 13.8671234361),
    ("Moth I",         0.4644451728, 0.3960600146, 14.8943051743),
    ("Moth II",        0.4391659182, 0.4529676431, 28.6692709402),
    ("Moth III",       0.3834435199, 0.3773636946, 25.8392363356),
    ("Goggles",        0.0833000718, 0.1278892555, 10.4648495256),
    ("Dragonfly",      0.0805842255, 0.5888360898, 21.2723373956),
    ("Yin-Yang Ib",    0.2826986823, 0.3272087861, 10.9633031497),
    ("Yarn",           0.559064247131347, 0.349191558837891, 14.894307),
]

# Our discovered orbits
OURS = [
    ("Discovered 1",   0.559519, 0.431263, 79.532),
    ("Discovered 2",   0.556313, 0.434469, 79.068),
]


# ============================================================
# Integration helper
# ============================================================

@partial(jax.jit, static_argnames=("n_steps",))
def trace_orbit(v1, v2, T, n_steps=3000):
    """Returns (q_trace, p_trace)."""
    q0 = Q_COLLINEAR
    p0 = jnp.array([[v1, v2], [v1, v2], [-2*v1, -2*v2]], dtype=jnp.float64)
    h = T / n_steps

    def step(carry, _):
        q, p = carry
        q, p = yoshida6_step(q, p, jnp.float64(h))
        return (q, p), (q, p)

    (_, _), (q_tr, p_tr) = jax.lax.scan(step, (q0, p0), None, length=n_steps)
    q_tr = jnp.concatenate([q0[None], q_tr], axis=0)
    p_tr = jnp.concatenate([p0[None], p_tr], axis=0)
    return q_tr, p_tr


# ============================================================
# Styles
# ============================================================

def paper():
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm",
        "axes.labelsize": 10, "axes.titlesize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "black", "axes.linewidth": 0.6,
    })


def blog():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.weight": "300",
        "axes.labelsize": 11, "axes.titlesize": 13,
        "axes.labelcolor": "#b0b0c0", "axes.titlecolor": "#f0f0ff",
        "xtick.color": "#707090", "ytick.color": "#707090",
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
        "savefig.facecolor": "#080810", "axes.facecolor": "#080810",
        "axes.edgecolor": "#303048", "axes.linewidth": 0.4,
    })


# ============================================================
# 1. Shape-sphere Mollweide projection of all orbits
# ============================================================

def plot_shape_sphere_mollweide(mode="blog"):
    """All 10 Suvakov orbits on a Mollweide projection of the shape sphere.

    Each orbit starts at the same point (the collinear configuration at n = (1,0,0))
    but traces a different path. This is what the classifier sees as a 2D projection.
    """
    (blog if mode == "blog" else paper)()

    fig, ax = plt.subplots(figsize=(12, 6.5), subplot_kw={"projection": "mollweide"})

    # Color each orbit by its period
    periods = np.array([o[3] for o in SUVAKOV])
    norm = Normalize(vmin=periods.min(), vmax=periods.max())
    if mode == "blog":
        cmap = LinearSegmentedColormap.from_list("hot", ["#40a0ff", "#ff40a0", "#ffa040"])
    else:
        cmap = plt.cm.viridis

    for name, v1, v2, T in SUVAKOV:
        q_tr, _ = trace_orbit(v1, v2, T, n_steps=2000)
        n_tr = np.array(jax.vmap(config_to_shape)(q_tr))

        # Convert (n1, n2, n3) on unit sphere to (lon, lat) for Mollweide
        # lat = arcsin(n3), lon = arctan2(n2, n1)
        lat = np.arcsin(np.clip(n_tr[:, 2], -1, 1))
        lon = np.arctan2(n_tr[:, 1], n_tr[:, 0])
        # Unwrap longitude to avoid jumps
        color = cmap(norm(T))
        ax.plot(lon, lat, color=color, lw=0.8 if mode == "paper" else 1.2,
                 alpha=0.75, label=f"{name} (T={T:.2f})")

    # Mark special points
    # Three collision points on equator (n3=0) at 120 degree spacing
    # (2,3) at lon=60, (1,2) at lon=180, (1,3) at lon=-60
    # Lagrange points at poles
    specials = [
        (np.radians(60), 0, "(2,3) coll", "o"),
        (np.pi, 0, "(1,2) coll", "o"),
        (-np.radians(60), 0, "(1,3) coll", "o"),
        (0, np.radians(90), "Lagrange N", "^"),
        (0, -np.radians(90), "Lagrange S", "v"),
    ]
    marker_color = "#ffff00" if mode == "blog" else "red"
    for lon, lat, lbl, m in specials:
        ax.plot(lon, lat, m, color=marker_color, ms=10, mec="black", mew=0.5, zorder=20)

    ax.grid(True, alpha=0.25)
    ax.set_title("Shape-sphere paths of all 11 Suvakov orbits\n"
                  "All start at (2,3)-collision vicinity; trace distinct closed curves",
                  fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left", fontsize=7, framealpha=0.9)

    fname = f"shape_sphere_mollweide_{mode}.pdf" if mode == "paper" else f"shape_sphere_mollweide_{mode}.png"
    plt.savefig(FIG_DIR / fname)
    plt.close()
    print(f"  Saved {fname}")


# ============================================================
# 2. Pairwise distance rhythm
# ============================================================

def plot_pairwise_rhythm(mode="blog"):
    """r12, r13, r23 vs time for 6 famous orbits.

    Reveals the "rhythm" of close approaches -- which pair comes close when.
    Close approaches are where the action happens in the 3BP.
    """
    (blog if mode == "blog" else paper)()

    orbits = SUVAKOV[:6]
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=False)
    axes = axes.flatten()

    if mode == "blog":
        colors = ["#ff4080", "#40ff90", "#60b0ff"]
    else:
        colors = ["#1a1a1a", "#505050", "#909090"]

    for i, (name, v1, v2, T) in enumerate(orbits):
        ax = axes[i]
        q_tr, _ = trace_orbit(v1, v2, T, n_steps=2000)
        q_np = np.array(q_tr)
        t = np.linspace(0, T, len(q_np))

        dists = np.array([
            np.linalg.norm(q_np[:, 1] - q_np[:, 0], axis=1),  # r12
            np.linalg.norm(q_np[:, 2] - q_np[:, 0], axis=1),  # r13
            np.linalg.norm(q_np[:, 2] - q_np[:, 1], axis=1),  # r23
        ])
        labels = [r"$r_{12}$", r"$r_{13}$", r"$r_{23}$"]
        for j in range(3):
            ax.plot(t, dists[j], color=colors[j], lw=1.2, alpha=0.85, label=labels[j])

        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"separation")
        ax.set_title(f"{name} (T={T:.3f})")
        if i == 0:
            ax.legend(loc="upper right", framealpha=0.8, fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Pairwise distance rhythm — the 'music' of each orbit",
                  fontsize=13, color="white" if mode == "blog" else "black", y=0.995)
    plt.tight_layout()

    fname = f"pairwise_rhythm_{mode}.pdf" if mode == "paper" else f"pairwise_rhythm_{mode}.png"
    plt.savefig(FIG_DIR / fname)
    plt.close()
    print(f"  Saved {fname}")


# ============================================================
# 3. Energy breathing (virial theorem)
# ============================================================

def plot_energy_breathing(mode="blog"):
    """KE(t) and |PE|(t) vs time for 4 orbits.

    Shows the virial theorem: for bound orbits, time-averaged <KE> = -<PE>/2.
    You can see kinetic and potential energy trading off rhythmically.
    """
    (blog if mode == "blog" else paper)()

    orbits = SUVAKOV[:4]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    if mode == "blog":
        ke_color = "#ff4080"
        pe_color = "#60b0ff"
        total_color = "#ffd040"
    else:
        ke_color = "#1a1a1a"
        pe_color = "#707070"
        total_color = "#b0b0b0"

    for i, (name, v1, v2, T) in enumerate(orbits):
        ax = axes[i]
        q_tr, p_tr = trace_orbit(v1, v2, T, n_steps=2000)
        KE = np.array(jax.vmap(kinetic_energy)(p_tr))
        PE = np.array(jax.vmap(potential_energy)(q_tr))
        E_total = KE + PE
        t = np.linspace(0, T, len(KE))

        ax.plot(t, KE, color=ke_color, lw=1.2, label=r"$T$ (kinetic)", alpha=0.9)
        ax.plot(t, -PE, color=pe_color, lw=1.2, label=r"$-V$ (|potential|)", alpha=0.9)
        ax.plot(t, E_total, color=total_color, lw=1.5, ls="-", label=r"$H = T + V$", alpha=0.95)

        # Time-averaged virial: <T> should equal -<V>/2
        T_avg = np.mean(KE)
        V_avg = np.mean(PE)
        virial_ratio = T_avg / (-V_avg / 2)

        ax.set_xlabel(r"$t$")
        ax.set_ylabel("energy")
        ax.set_title(f"{name}  —  virial ratio $\\langle T\\rangle/(-\\langle V\\rangle/2) = {virial_ratio:.4f}$")
        if i == 0:
            ax.legend(loc="lower right", framealpha=0.8, fontsize=9)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Energy breathing — kinetic and potential energy trade off; total conserved",
                  fontsize=13, color="white" if mode == "blog" else "black", y=0.995)
    plt.tight_layout()

    fname = f"energy_breathing_{mode}.pdf" if mode == "paper" else f"energy_breathing_{mode}.png"
    plt.savefig(FIG_DIR / fname)
    plt.close()
    print(f"  Saved {fname}")


# ============================================================
# 4. Body-in-body frame
# ============================================================

def plot_body_in_body_frame(mode="blog"):
    """Body 1's trajectory as seen from body 2's rest frame.

    Reveals the relative motion structure — for some orbits this becomes
    a near-closed curve, for others it's highly complex.
    """
    (blog if mode == "blog" else paper)()

    orbits = SUVAKOV[:6]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.flatten()

    if mode == "blog":
        color12 = "#ff6090"
        color13 = "#60ffa0"
    else:
        color12 = "#1a1a1a"
        color13 = "#707070"

    for i, (name, v1, v2, T) in enumerate(orbits):
        ax = axes[i]
        q_tr, _ = trace_orbit(v1, v2, T, n_steps=3000)
        q_np = np.array(q_tr)

        # body 1 relative to body 2
        rel12 = q_np[:, 0] - q_np[:, 1]
        # body 3 relative to body 2
        rel32 = q_np[:, 2] - q_np[:, 1]

        ax.plot(rel12[:, 0], rel12[:, 1], color=color12, lw=1.0, alpha=0.8,
                 label=r"body 1 from body 2's frame")
        ax.plot(rel32[:, 0], rel32[:, 1], color=color13, lw=1.0, alpha=0.8,
                 label=r"body 3 from body 2's frame")
        ax.plot(0, 0, "o", ms=10, color="yellow" if mode == "blog" else "black",
                 mec="black", mew=1, zorder=10, label="body 2 (center)")

        ax.set_aspect("equal")
        ax.set_title(name)
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend(loc="upper right", fontsize=7, framealpha=0.8)

    fig.suptitle("Orbit shape in body 2's rest frame — reveals hidden symmetries",
                  fontsize=13, color="white" if mode == "blog" else "black", y=0.995)
    plt.tight_layout()

    fname = f"body_in_body_{mode}.pdf" if mode == "paper" else f"body_in_body_{mode}.png"
    plt.savefig(FIG_DIR / fname)
    plt.close()
    print(f"  Saved {fname}")


# ============================================================
# 5. Velocity-space constellation
# ============================================================

def plot_velocity_constellation(mode="blog"):
    """All Suvakov orbits + our discovered orbits in (v1, v2) space, colored by period.

    Reveals where periodic orbits concentrate in the parameter space.
    """
    (blog if mode == "blog" else paper)()

    fig, ax = plt.subplots(figsize=(10, 9))

    periods = np.array([o[3] for o in SUVAKOV])
    norm = Normalize(vmin=periods.min(), vmax=max(periods.max(), 80))
    if mode == "blog":
        cmap = LinearSegmentedColormap.from_list(
            "warm", ["#40a0ff", "#20ffc0", "#ffd040", "#ff4080"])
    else:
        cmap = plt.cm.viridis

    # Plot Suvakov orbits
    for name, v1, v2, T in SUVAKOV:
        color = cmap(norm(T))
        ax.scatter(v1, v2, s=150, c=[color], marker="o",
                    edgecolors="white" if mode == "blog" else "black", linewidths=1.2, zorder=5)
        # Mirror images (symmetries)
        for sx, sy in [(-1, 1), (1, -1), (-1, -1)]:
            ax.scatter(sx * v1, sy * v2, s=60, c=[color], marker="o",
                        edgecolors="white" if mode == "blog" else "black",
                        linewidths=0.5, alpha=0.5, zorder=4)
        ax.annotate(name, (v1, v2), xytext=(8, 8), textcoords="offset points",
                     fontsize=7, color="white" if mode == "blog" else "black",
                     bbox=dict(boxstyle="round,pad=0.2",
                              fc="#0a0a18" if mode == "blog" else "white",
                              ec=color, lw=0.5, alpha=0.85))

    # Plot our discovered orbits with a special marker
    for name, v1, v2, T in OURS:
        color = cmap(norm(T))
        ax.scatter(v1, v2, s=250, c=[color], marker="*",
                    edgecolors="white" if mode == "blog" else "black", linewidths=2, zorder=10)
        for sx, sy in [(-1, 1), (1, -1), (-1, -1)]:
            ax.scatter(sx * v1, sy * v2, s=100, c=[color], marker="*",
                        edgecolors="white" if mode == "blog" else "black",
                        linewidths=1, alpha=0.5, zorder=9)

    ax.axhline(0, color="#606070" if mode == "blog" else "grey", lw=0.5, alpha=0.5)
    ax.axvline(0, color="#606070" if mode == "blog" else "grey", lw=0.5, alpha=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$v_1$ (first velocity component)")
    ax.set_ylabel(r"$v_2$ (second velocity component)")
    ax.set_title("Periodic orbit constellation in velocity space\n"
                  "Circles: Suvakov catalog. Stars: our discoveries.")
    ax.grid(True, alpha=0.2)

    # Colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(sm, ax=ax, label="Period T", shrink=0.6)

    fname = f"velocity_constellation_{mode}.pdf" if mode == "paper" else f"velocity_constellation_{mode}.png"
    plt.savefig(FIG_DIR / fname)
    plt.close()
    print(f"  Saved {fname}")


# ============================================================
# 6. Close-encounter heatmap
# ============================================================

def plot_close_encounter_heatmap(mode="blog"):
    """For each orbit, plot min pairwise distance over time as a heatmap row.

    Shows when the bodies come close (bright) vs when they're spread out (dark).
    A periodic orbit has a regular pattern; chaotic orbits have irregular patterns.
    """
    (blog if mode == "blog" else paper)()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Compute min distance over time for each orbit, normalized to its period
    n_samples = 400
    orbit_data = []
    for name, v1, v2, T in SUVAKOV:
        q_tr, _ = trace_orbit(v1, v2, T, n_steps=n_samples)
        dists = np.array(jax.vmap(pairwise_distances)(q_tr))  # (n_samples+1, 3)
        min_dist = np.min(dists, axis=1)  # (n_samples+1,)
        orbit_data.append((name, min_dist))

    # Build 2D heatmap: rows = orbits, columns = time fraction
    data = np.array([d[1] for d in orbit_data])
    names = [d[0] for d in orbit_data]

    if mode == "blog":
        cmap = LinearSegmentedColormap.from_list(
            "encounter", ["#ff2040", "#ffa040", "#40a0ff", "#202030"])
    else:
        cmap = "viridis_r"

    im = ax.imshow(data, aspect="auto", cmap=cmap,
                    extent=[0, 1, len(names), 0],
                    interpolation="bilinear")

    ax.set_yticks(np.arange(len(names)) + 0.5)
    ax.set_yticklabels(names)
    ax.set_xlabel("Time (normalized to period)")
    ax.set_title("Close-encounter map — when do the bodies get close?\n"
                  "Bright = close approach, dark = widely separated")

    cbar = plt.colorbar(im, ax=ax, label=r"min pairwise distance $\min(r_{12},r_{13},r_{23})$")

    plt.tight_layout()
    fname = f"close_encounter_{mode}.pdf" if mode == "paper" else f"close_encounter_{mode}.png"
    plt.savefig(FIG_DIR / fname)
    plt.close()
    print(f"  Saved {fname}")


# ============================================================
# 7. Moment of inertia breathing
# ============================================================

def plot_inertia_breathing(mode="blog"):
    """Moment of inertia I(t) for multiple orbits.

    I = sum_i |q_i - COM|^2. For periodic orbits, I breathes periodically.
    Shows how compact vs spread out the system is at each moment.
    """
    (blog if mode == "blog" else paper)()

    fig, ax = plt.subplots(figsize=(11, 6))

    orbits = SUVAKOV[:8]
    if mode == "blog":
        cmap = LinearSegmentedColormap.from_list(
            "warm", ["#40a0ff", "#20ffc0", "#ffd040", "#ff4080"])
    else:
        cmap = plt.cm.viridis

    periods = [o[3] for o in orbits]
    norm = Normalize(vmin=min(periods), vmax=max(periods))

    for name, v1, v2, T in orbits:
        q_tr, _ = trace_orbit(v1, v2, T, n_steps=2000)
        I = np.array(jax.vmap(moment_of_inertia)(q_tr))
        t_norm = np.linspace(0, 1, len(I))
        color = cmap(norm(T))
        ax.plot(t_norm, I, color=color, lw=1.3, alpha=0.85,
                 label=f"{name} (T={T:.2f})")

    ax.set_xlabel("Time (normalized to period)")
    ax.set_ylabel(r"Moment of inertia $I = \sum_i |q_i - Q|^2$")
    ax.set_title("Breathing pattern: how compact the three-body system is\n"
                  "All orbits start collinear (I=2); expand and contract differently")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85, ncol=2)
    ax.grid(True, alpha=0.2)

    fname = f"inertia_breathing_{mode}.pdf" if mode == "paper" else f"inertia_breathing_{mode}.png"
    plt.savefig(FIG_DIR / fname)
    plt.close()
    print(f"  Saved {fname}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("INSIGHTFUL VISUALIZATIONS")
    print(f"Output: {FIG_DIR}")
    print("=" * 60)

    plots = [
        ("Shape-sphere Mollweide projection", plot_shape_sphere_mollweide),
        ("Pairwise distance rhythm", plot_pairwise_rhythm),
        ("Energy breathing (virial)", plot_energy_breathing),
        ("Body-in-body frame", plot_body_in_body_frame),
        ("Velocity constellation", plot_velocity_constellation),
        ("Close-encounter heatmap", plot_close_encounter_heatmap),
        ("Moment of inertia breathing", plot_inertia_breathing),
    ]

    for name, fn in plots:
        print(f"\n{name}:")
        try:
            fn(mode="paper")
            fn(mode="blog")
        except Exception as e:
            print(f"  FAILED: {e}")

    print(f"\nDone. See {FIG_DIR}")


if __name__ == "__main__":
    main()
