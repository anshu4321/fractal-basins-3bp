"""Generate publication-grade figures summarizing the 3BP verification pipeline.

Actions #1–#5: catalog scan (Hristov 2024 + 2025), syzygy words, dual-tool HP
verification, and the tier-1 ML-prior ablation study.

Outputs (paper PDF + blog PNG via existing save_both):
  figures/catalog_scan_map_{paper,blog}.{pdf,png}
  figures/orbit_novelty_matrix_{paper,blog}.{pdf,png}
  figures/syzygy_word_comparison_{paper,blog}.{pdf,png}
  figures/hp_dual_tool_residuals_{paper,blog}.{pdf,png}
  figures/ml_prior_ci_{paper,blog}.{pdf,png}
  figures/summary_verdict_card_{paper,blog}.{pdf,png}
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

# Project imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mega3bp.integrators import yoshida6_integrate

# Reuse existing style + save helpers + shape-sphere math
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_novel_figures import (  # type: ignore
    style,
    save_both,
    shape_sphere_hopf,
    mollweide_project,
)

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Data locations
# -----------------------------------------------------------------------------
VERIF_DIR = ROOT / "experiments" / "orbit_verification"
DISC_DIR = ROOT / "experiments" / "orbit_discovery"
H2025_STABLE_TXT = VERIF_DIR / "00_catalogs" / "hristov_2025_stable.txt"
H2024_SOL_TXT = VERIF_DIR / "00_catalogs" / "hristov_2024_sol_80.txt"
H2025_SYZ_TXT = VERIF_DIR / "00_catalogs" / "hristov_2025_syzygies.txt"
HP_JSON = VERIF_DIR / "06_high_precision" / "hp_heyoka_newton_results.json"
H2025_VERDICT = VERIF_DIR / "07_hristov_2025_definitive" / "verdict.json"
H2024_SWEEP = VERIF_DIR / "08_hristov_2024_sweep" / "sweep_results.json"
SYZ_VERDICT = VERIF_DIR / "09_syzygy_words" / "verdict.json"
SYZ_WORDS = VERIF_DIR / "09_syzygy_words" / "words.json"
HP_CROSS = VERIF_DIR / "10_hp_cross_check" / "verdict.json"
TIER1_AGG = DISC_DIR / "tier1_ablation" / "aggregated.json"

# Body-color / novel-color palette via make_novel_figures
ORBIT_ORDER = ["A", "B", "C", "D"]

# Novelty-verdict colors (shared logic)
#   NEW family  -> coral (A)
#   NEW orbit   -> amber (B)
#   REDISCOVERY -> teal  (C, D)
VERDICT_COLORS_BLOG = {
    "new_family":  "#ff6b6b",
    "new_orbit":   "#ffc857",
    "rediscovery": "#40d09b",
}
VERDICT_COLORS_PAPER = {
    "new_family":  "#c2185b",
    "new_orbit":   "#e07b00",
    "rediscovery": "#2e8b57",
}

VERDICT = {
    "A": "new_family",
    "B": "new_orbit",
    "C": "rediscovery",
    "D": "rediscovery",
}
REDISCOVERY_OF = {"C": "#0006", "D": "#0011"}  # Hristov 2025 ids


def verdict_color(mode, key):
    table = VERDICT_COLORS_PAPER if mode == "paper" else VERDICT_COLORS_BLOG
    return table[VERDICT[key]]


# -----------------------------------------------------------------------------
# Load HP data once (per-orbit ICs for re-integration and residuals)
# -----------------------------------------------------------------------------
with open(HP_JSON) as f:
    HP_RESULTS = {r["name"]: r for r in json.load(f)}

with open(HP_CROSS) as f:
    HP_CROSS_DATA = json.load(f)

with open(SYZ_VERDICT) as f:
    SYZ_VERDICT_DATA = json.load(f)

with open(SYZ_WORDS) as f:
    SYZ_WORDS_DATA = json.load(f)


def ic_double(key):
    """Return (v1, v2, T) as float64 from HP results (ic_*_double values)."""
    r = HP_RESULTS[key]
    return (
        float(r["ic_v1_double"]),
        float(r["ic_v2_double"]),
        float(r["ic_T_double"]),
    )


def integrate_one_period(v1, v2, T, n_steps=5000):
    """Yoshida6 float64 integration; returns q(t), p(t), t."""
    Q0 = jnp.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 0.0]], dtype=jnp.float64)
    h = T / n_steps
    p0 = jnp.array([[v1, v2], [v1, v2], [-2 * v1, -2 * v2]], dtype=jnp.float64)
    q_tr, p_tr, _ = yoshida6_integrate(Q0, p0, jnp.float64(h), n_steps)
    stride = max(1, n_steps // 3000)
    return (
        np.asarray(q_tr[::stride]),
        np.asarray(p_tr[::stride]),
        np.arange(len(q_tr[::stride])) * h * stride,
    )


# =============================================================================
# Fig 1 — catalog_scan_map (2 panels)
# =============================================================================
def fig_catalog_scan_map(mode):
    s = style(mode)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8) if mode == "paper" else (13, 5.6))
    fig.patch.set_facecolor(s["bg"])
    ax1, ax2 = axes

    # ---------- Panel 1: A, B, C, D vs Hristov 2025 stable catalog ----------
    ax1.set_facecolor(s["bg"] if mode == "blog" else "white")
    v1_cat, v2_cat = [], []
    with open(H2025_STABLE_TXT) as f:
        for line in f:
            toks = line.strip().split()
            if len(toks) < 2:
                continue
            try:
                v1_cat.append(float(toks[0]))
                v2_cat.append(float(toks[1]))
            except Exception:
                pass
    v1_cat = np.array(v1_cat)
    v2_cat = np.array(v2_cat)
    # The catalog values in hristov_2025_stable.txt are (|v1|, |v2|, T, T*).
    # Our orbits' (v1, v2) can have signs; match is within canonical equiv.
    # Plot catalog as both +/- to make scatter visually symmetric, like Hristov's figures.
    ax1.scatter(
        v1_cat,
        v2_cat,
        s=6,
        color=s["mute"],
        alpha=0.55,
        edgecolor="none",
        zorder=1,
        label="Hristov 2025 stable (971)",
    )
    ax1.scatter(
        -v1_cat,
        v2_cat,
        s=6,
        color=s["mute"],
        alpha=0.25,
        edgecolor="none",
        zorder=1,
    )

    # Overlay A, B, C, D as colored stars at their HP (v1, v2).
    for k in ORBIT_ORDER:
        v1, v2, _ = ic_double(k)
        col = verdict_color(mode, k)
        ax1.scatter(
            [v1],
            [v2],
            marker="*",
            s=340,
            color=col,
            zorder=5,
            edgecolor="white" if mode == "paper" else s["text"],
            linewidth=1.2,
        )
        # Text label offset per orbit
        off = {"A": (0.03, 0.04), "B": (-0.09, 0.05), "C": (0.03, -0.06), "D": (0.03, 0.04)}[k]
        verdict_txt = {
            "A": "A (NEW family)",
            "B": "B (NEW orbit)",
            "C": "C = #0006",
            "D": "D = #0011",
        }[k]
        ax1.text(
            v1 + off[0],
            v2 + off[1],
            verdict_txt,
            fontsize=8 if mode == "paper" else 10,
            color=col,
            fontweight="bold",
            zorder=6,
        )

    ax1.set_xlim(-0.75, 0.75)
    ax1.set_ylim(-0.75, 0.75)
    ax1.set_aspect("equal")
    ax1.set_xlabel("$v_1$", color=s["text"] if mode == "blog" else "#222")
    ax1.set_ylabel("$v_2$", color=s["text"] if mode == "blog" else "#222")
    ax1.set_title(
        "Panel 1 — against Hristov 2025 stable catalog (971 orbits)",
        color=s["text"] if mode == "blog" else "#111",
        fontweight="bold",
        pad=8,
        fontsize=11,
    )
    ax1.legend(
        frameon=False,
        loc="upper right",
        fontsize=8,
        labelcolor=s["text"] if mode == "blog" else "#222",
    )
    ax1.tick_params(colors=s["mute"])
    ax1.axhline(0, color=s["mute"], lw=0.4, alpha=0.4)
    ax1.axvline(0, color=s["mute"], lw=0.4, alpha=0.4)

    # ---------- Panel 2: min-dv histogram vs Hristov 2024 (24,582) ----------
    ax2.set_facecolor(s["bg"] if mode == "blog" else "white")
    with open(H2024_SWEEP) as f:
        sweep = json.load(f)
    a_dv = np.array(
        [r["best"]["A"]["dv_min"] for r in sweep["results"] if r["best"]["A"]["dv_min"] is not None]
    )
    b_dv = np.array(
        [r["best"]["B"]["dv_min"] for r in sweep["results"] if r["best"]["B"]["dv_min"] is not None]
    )
    log_a = np.log10(a_dv)
    log_b = np.log10(b_dv)
    bins = np.linspace(min(log_a.min(), log_b.min()) - 0.2, max(log_a.max(), log_b.max()) + 0.2, 70)
    col_a = verdict_color(mode, "A")
    col_b = verdict_color(mode, "B")
    ax2.hist(
        log_a,
        bins=bins,
        color=col_a,
        alpha=0.55,
        label=f"A vs H2024 (n={len(log_a)})",
        edgecolor="white" if mode == "paper" else s["text"],
        linewidth=0.3,
    )
    ax2.hist(
        log_b,
        bins=bins,
        color=col_b,
        alpha=0.55,
        label=f"B vs H2024 (n={len(log_b)})",
        edgecolor="white" if mode == "paper" else s["text"],
        linewidth=0.3,
    )
    # Strong-match threshold
    ax2.axvline(
        np.log10(1e-4),
        color=s["highlight"],
        lw=1.2,
        linestyle="--",
        label=r"strong-match: $10^{-4}$",
    )
    # Medians
    ax2.axvline(np.median(log_a), color=col_a, lw=0.8, linestyle=":")
    ax2.axvline(np.median(log_b), color=col_b, lw=0.8, linestyle=":")

    ax2.set_xlabel(r"$\log_{10}(\min\ dv)$ across 24,582 entries",
                   color=s["text"] if mode == "blog" else "#222")
    ax2.set_ylabel("count", color=s["text"] if mode == "blog" else "#222")
    ax2.set_title(
        "Panel 2 — min-$dv$ distribution vs Hristov 2024 free-fall (24,582)",
        color=s["text"] if mode == "blog" else "#111",
        fontweight="bold",
        pad=8,
        fontsize=11,
    )
    ax2.legend(
        frameon=False,
        loc="upper right",
        fontsize=8,
        labelcolor=s["text"] if mode == "blog" else "#222",
    )
    ax2.tick_params(colors=s["mute"])
    ax2.grid(True, axis="y", alpha=0.25, linewidth=0.4)

    # Annotation: min dv ≈ 0.6, orders of magnitude above the 1e-4 strong-match line
    ax2.text(
        0.02,
        0.98,
        (
            f"A: best $dv={a_dv.min():.3f}$\n"
            f"B: best $dv={b_dv.min():.3f}$\n"
            f"0 strong matches\n"
            f"({len(a_dv)}/24582 entries\nproduced Euler crossings)"
        ),
        transform=ax2.transAxes,
        fontsize=7.5 if mode == "paper" else 8.5,
        color=s["mute"],
        va="top",
        ha="left",
        bbox=dict(facecolor=s["bg"], edgecolor="none", alpha=0.6),
    )

    fig.suptitle(
        "Catalog scan: Actions #1 (H2025) + #2 (H2024)",
        color=s["text"] if mode == "blog" else "#111",
        fontweight="bold",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    save_both(fig, "catalog_scan_map", mode)
    plt.close(fig)


# =============================================================================
# Fig 2 — orbit_novelty_matrix (4 rows × 2 cols)
# =============================================================================
def fig_orbit_novelty_matrix(mode):
    s = style(mode)
    fig, axes = plt.subplots(4, 2, figsize=(9, 12) if mode == "paper" else (11, 14))
    fig.patch.set_facecolor(s["bg"])

    # Re-integrate per orbit once
    trajectories = {}
    for k in ORBIT_ORDER:
        v1, v2, T = ic_double(k)
        q, p, t = integrate_one_period(v1, v2, T, n_steps=6000)
        trajectories[k] = {"q": q, "p": p, "t": t, "T": T, "v1": v1, "v2": v2}

    # Novelty verdicts per orbit
    labels = {
        "A": "NEW family (topology (132)^8 length 24)",
        "B": "NEW orbit / KNOWN family (topology (312)^14, matches H2025 rows 6, 7)",
        "C": "REDISCOVERY of Hristov 2025 #0006 (dv=4.65e-51)",
        "D": "REDISCOVERY of Hristov 2025 #0011 (dv=4.64e-51)",
    }

    for row, k in enumerate(ORBIT_ORDER):
        tr = trajectories[k]
        col = verdict_color(mode, k)

        # --- left: (x, y) configuration trajectories ---
        ax = axes[row, 0]
        ax.set_facecolor(s["bg"] if mode == "blog" else "white")
        q = tr["q"]
        axis_half = 3.5
        for b in range(3):
            ax.plot(
                q[:, b, 0],
                q[:, b, 1],
                lw=0.9,
                alpha=0.8,
                color=s["body_colors"][b],
                label=f"Body {b+1}",
            )
            ax.scatter(
                [q[0, b, 0]],
                [q[0, b, 1]],
                s=22,
                color=s["body_colors"][b],
                zorder=5,
                edgecolor="white" if mode == "paper" else s["text"],
                linewidth=0.6,
            )
        ax.set_xlim(-axis_half, axis_half)
        ax.set_ylim(-axis_half, axis_half)
        ax.set_aspect("equal")
        ax.tick_params(colors=s["mute"])
        ax.set_title(
            f"Orbit {k}  $T={tr['T']:.3f}$  ·  {labels[k]}",
            color=col,
            fontweight="bold",
            fontsize=10 if mode == "paper" else 11,
            pad=6,
        )
        if row == 3:
            ax.set_xlabel("$x$", color=s["text"] if mode == "blog" else "#222")
        ax.set_ylabel("$y$", color=s["text"] if mode == "blog" else "#222")
        if row == 0:
            ax.legend(
                frameon=False,
                loc="upper right",
                fontsize=7,
                labelcolor=s["text"] if mode == "blog" else "#222",
            )

        # --- right: shape-sphere Mollweide projection ---
        ax2 = axes[row, 1]
        ax2.set_facecolor(s["bg"] if mode == "blog" else "white")
        # Mollweide grid
        for lat in np.linspace(-np.pi / 2, np.pi / 2, 7):
            lons = np.linspace(-np.pi, np.pi, 200)
            lats = np.full_like(lons, lat)
            w = np.stack(
                [np.cos(lats) * np.cos(lons), np.cos(lats) * np.sin(lons), np.sin(lats)],
                axis=-1,
            )
            x, y = mollweide_project(w)
            ax2.plot(x, y, color=s["mute"], alpha=0.15, lw=0.3)
        for lon in np.linspace(-np.pi, np.pi, 7):
            lats = np.linspace(-np.pi / 2, np.pi / 2, 200)
            lons = np.full_like(lats, lon)
            w = np.stack(
                [np.cos(lats) * np.cos(lons), np.cos(lats) * np.sin(lons), np.sin(lats)],
                axis=-1,
            )
            x, y = mollweide_project(w)
            ax2.plot(x, y, color=s["mute"], alpha=0.15, lw=0.3)
        # Boundary
        ts = np.linspace(0, 2 * np.pi, 360)
        ax2.plot(2 * np.sqrt(2) * np.cos(ts), np.sqrt(2) * np.sin(ts),
                 color=s["mute"], lw=0.8)

        # Trace
        w = shape_sphere_hopf(q)
        x, y = mollweide_project(w)
        dx = np.diff(x)
        breaks = np.where(np.abs(dx) > 2.0)[0] + 1
        segments = np.split(np.arange(len(x)), breaks)
        for seg in segments:
            if len(seg) > 1:
                ax2.plot(x[seg], y[seg], color=col, lw=1.1, alpha=0.95)
        # Lagrange poles
        for yp, lbl in [(np.sqrt(2), "$L_+$"), (-np.sqrt(2), "$L_-$")]:
            ax2.scatter(
                [0], [yp], marker="*", s=80, color=s["highlight"], zorder=6,
                edgecolor="white" if mode == "paper" else s["text"], linewidth=0.6,
            )
            ax2.text(0.08, yp, lbl, color=s["highlight"], fontsize=8, fontweight="bold")

        ax2.set_xlim(-2 * np.sqrt(2) - 0.2, 2 * np.sqrt(2) + 0.2)
        ax2.set_ylim(-np.sqrt(2) - 0.3, np.sqrt(2) + 0.3)
        ax2.set_aspect("equal")
        ax2.axis("off")
        ax2.set_title(
            f"Shape-sphere (Mollweide)",
            color=s["text"] if mode == "blog" else "#111",
            fontweight="bold",
            fontsize=10,
            pad=4,
        )

    fig.suptitle(
        "Orbit novelty matrix: A (new family), B (new orbit), C & D (rediscoveries)",
        color=s["text"] if mode == "blog" else "#111",
        fontweight="bold",
        fontsize=13,
        y=1.00,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    save_both(fig, "orbit_novelty_matrix", mode)
    plt.close(fig)


# =============================================================================
# Fig 3 — syzygy_word_comparison
# =============================================================================
def _digit_colors(mode):
    if mode == "paper":
        return {"1": "#d62728", "2": "#2ca02c", "3": "#1f77b4"}
    return {"1": "#ff6b6b", "2": "#b8e838", "3": "#00aeff"}


def _draw_word_strip(ax, word, y, col_map, height=0.8, text_label=None,
                     text_color="#111", cell_lw=0.3):
    """Draw a word as horizontal cells at vertical position y, using col_map."""
    for i, d in enumerate(word):
        rect = Rectangle(
            (i, y - height / 2),
            1,
            height,
            facecolor=col_map.get(d, "#888888"),
            edgecolor="white",
            linewidth=cell_lw,
        )
        ax.add_patch(rect)
    if text_label is not None:
        ax.text(
            -1.0,
            y,
            text_label,
            ha="right",
            va="center",
            fontsize=10,
            color=text_color,
            fontweight="bold",
        )


def fig_syzygy_word_comparison(mode):
    s = style(mode)
    fig, axes = plt.subplots(
        2, 1,
        figsize=(11, 8) if mode == "paper" else (13, 9),
        gridspec_kw=dict(height_ratios=[1.1, 1.3]),
    )
    fig.patch.set_facecolor(s["bg"])
    col_map = _digit_colors(mode)

    words = SYZ_WORDS_DATA["orbits"]
    matches = SYZ_VERDICT_DATA["matches"]

    # -------- Top panel: word strips for A, B, C, D --------
    ax = axes[0]
    ax.set_facecolor(s["bg"] if mode == "blog" else "white")

    ys = [4, 3, 2, 1]  # rows for A, B, C, D
    max_len = max(len(words[k]["word"]) for k in ORBIT_ORDER)

    for k, y in zip(ORBIT_ORDER, ys):
        w = words[k]["word"]
        verdict_col = verdict_color(mode, k)
        label_text = f"{k}  ({len(w)})"
        _draw_word_strip(
            ax, w, y, col_map,
            text_label=label_text,
            text_color=verdict_col,
            height=0.75,
        )
        # right-side annotation (canonical form)
        canon = words[k]["equivalence_canonical"]
        # Abbreviate canonical if too long
        canon_short = canon[:12] + "..." if len(canon) > 12 else canon
        ax.text(
            len(w) + 1.0, y,
            f"canon: {canon_short}",
            fontsize=8 if mode == "paper" else 9,
            color=s["mute"],
            va="center",
        )

    ax.set_xlim(-6, max_len + 12)
    ax.set_ylim(0.2, 4.8)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(
        "Syzygy words for orbits A, B, C, D (colored 1/2/3)",
        color=s["text"] if mode == "blog" else "#111",
        fontweight="bold",
        fontsize=11,
        pad=10,
    )
    ax.axis("off")

    # Digit legend
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=col_map[str(i)], label=f"close encounter: body {i}")
                  for i in (1, 2, 3)]
    ax.legend(
        handles=legend_els,
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        fontsize=8,
        labelcolor=s["text"] if mode == "blog" else "#222",
    )

    # -------- Bottom panel: A (no match) and B (matches rows 6, 7) --------
    ax2 = axes[1]
    ax2.set_facecolor(s["bg"] if mode == "blog" else "white")

    # Parse Hristov 2025 syzygies file
    hristov_labels = {}
    with open(H2025_SYZ_TXT) as f:
        for line in f:
            # Format: "6)   213213...  42"
            line = line.strip()
            if not line:
                continue
            toks = line.replace(")", "").split()
            try:
                row = int(toks[0])
                word = toks[1]
                hristov_labels[row] = word
            except Exception:
                continue

    # A: show A's word and "NO MATCH" annotation
    y_a = 5
    _draw_word_strip(
        ax2, words["A"]["word"], y_a, col_map,
        text_label=f"A ({len(words['A']['word'])})",
        text_color=verdict_color(mode, "A"),
        height=0.7,
    )
    ax2.annotate(
        "NO MATCH in any of 971 Hristov 2025 topologies",
        xy=(len(words["A"]["word"]) / 2, y_a),
        xytext=(len(words["A"]["word"]) + 3, y_a + 0.9),
        fontsize=10,
        color=verdict_color(mode, "A"),
        fontweight="bold",
        arrowprops=dict(
            arrowstyle="->",
            color=verdict_color(mode, "A"),
            lw=1.0,
        ),
    )

    # B: show B's word and Hristov rows 6 & 7 (both identical label)
    y_b = 3.2
    _draw_word_strip(
        ax2, words["B"]["word"], y_b, col_map,
        text_label=f"B ({len(words['B']['word'])})",
        text_color=verdict_color(mode, "B"),
        height=0.7,
    )
    # Hristov row 6
    y_r6 = 2.0
    if 6 in hristov_labels:
        _draw_word_strip(
            ax2, hristov_labels[6], y_r6, col_map,
            text_label=f"H2025 row 6 ({len(hristov_labels[6])})",
            text_color=s["mute"],
            height=0.6,
        )
    # Hristov row 7
    y_r7 = 0.9
    if 7 in hristov_labels:
        _draw_word_strip(
            ax2, hristov_labels[7], y_r7, col_map,
            text_label=f"H2025 row 7 ({len(hristov_labels[7])})",
            text_color=s["mute"],
            height=0.6,
        )

    # Connector line B <-> H2025 rows 6, 7 (right edge to right edge)
    b_width = len(words["B"]["word"])
    ax2.plot(
        [b_width + 0.2, b_width + 0.2],
        [y_b, y_r7],
        color=verdict_color(mode, "B"),
        lw=1.2,
        linestyle=":",
        alpha=0.8,
    )
    ax2.text(
        b_width + 1.2,
        (y_b + y_r7) / 2,
        "B matches rows\n6 & 7 (same\nfamily as C)\ncyclic/permuted",
        fontsize=9,
        color=verdict_color(mode, "B"),
        fontweight="bold",
        va="center",
        ha="left",
    )

    ax2.set_xlim(-10, max(b_width, len(hristov_labels.get(6, ""))) + 18)
    ax2.set_ylim(-0.2, 7.5)
    ax2.axis("off")
    ax2.set_title(
        "Syzygy comparison: A (no catalog match) vs B (matches H2025 rows 6 & 7)",
        color=s["text"] if mode == "blog" else "#111",
        fontweight="bold",
        fontsize=11,
        pad=8,
    )

    fig.suptitle(
        "Action #3 — syzygy-word topology comparison",
        color=s["text"] if mode == "blog" else "#111",
        fontweight="bold",
        fontsize=13,
        y=1.00,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    save_both(fig, "syzygy_word_comparison", mode)
    plt.close(fig)


# =============================================================================
# Fig 4 — hp_dual_tool_residuals
# =============================================================================
def fig_hp_dual_tool_residuals(mode):
    s = style(mode)
    fig, ax = plt.subplots(
        figsize=(7.8, 4.6) if mode == "paper" else (9.6, 5.6)
    )
    fig.patch.set_facecolor(s["bg"])
    ax.set_facecolor(s["bg"] if mode == "blog" else "white")

    hey = HP_CROSS_DATA["heyoka_200bit"]
    mp = HP_CROSS_DATA["mpmath_taylor_30dps"]

    names = ORBIT_ORDER
    x = np.arange(len(names))
    w = 0.38
    hey_res = np.array([hey[k]["residual"] for k in names])
    mp_res = np.array([mp[k]["residual"] for k in names])

    # Muted professional palette: blue for heyoka, orange for mpmath.
    if mode == "paper":
        hey_color = "#3b6fa8"  # muted blue
        mp_color = "#d08a3a"   # muted orange
        text_color = "#222"
    else:
        hey_color = "#5ea8de"
        mp_color = "#ecb46a"
        text_color = s["text"]

    log_hey = np.log10(hey_res)
    log_mp = np.log10(mp_res)

    bars1 = ax.bar(
        x - w / 2, log_hey, w,
        color=hey_color, alpha=0.92,
        edgecolor="white" if mode == "paper" else s["text"],
        linewidth=0.6,
        label="heyoka (200-bit MPFR Taylor)",
        zorder=3,
    )
    bars2 = ax.bar(
        x + w / 2, log_mp, w,
        color=mp_color, alpha=0.92,
        edgecolor="white" if mode == "paper" else s["text"],
        linewidth=0.6,
        label="mpmath (Taylor, 30 dps)",
        zorder=3,
    )

    # Axis range: cover values (all between 10^-50 and 10^-48)
    # and leave room above for the gate line + labels.
    y_min = -52.0
    y_max = -25.0
    ax.set_ylim(y_min, y_max)

    # Major ticks at nice log-scale positions.
    major_ticks = [-50, -45, -40, -35, -30, -25]
    ax.set_yticks(major_ticks)
    ax.set_yticklabels([rf"$10^{{{t}}}$" for t in major_ticks])
    ax.tick_params(axis="y", which="major", length=4, width=0.7)
    ax.tick_params(axis="y", which="minor", length=0)  # minor ticks off
    ax.tick_params(axis="x", length=0)

    # Annotate each bar with its numerical value, above the bar
    # (remember: bars grow negative, so "above" means toward less-negative y).
    label_fs = 8 if mode == "paper" else 9
    for b, v in zip(bars1, hey_res):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.6,
            f"{v:.2e}",
            ha="center", va="bottom",
            fontsize=label_fs,
            color=text_color,
        )
    for b, v in zip(bars2, mp_res):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.6,
            f"{v:.2e}",
            ha="center", va="bottom",
            fontsize=label_fs,
            color=text_color,
        )

    # Gate line at 1e-30
    gate = np.log10(1e-30)
    ax.axhline(
        gate,
        color="#9b1b1b" if mode == "paper" else s["highlight"],
        lw=1.3, linestyle="--", alpha=0.9, zorder=2,
    )
    ax.text(
        len(names) - 0.45, gate + 0.35,
        r"gate ($10^{-30}$)",
        ha="right", va="bottom",
        fontsize=10 if mode == "paper" else 11,
        color="#9b1b1b" if mode == "paper" else s["highlight"],
        fontweight="bold",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"Orbit {n}" for n in names],
        fontsize=12 if mode == "paper" else 13,
        color=text_color,
    )
    ax.set_ylabel(
        r"closure residual $\|R\|$",
        color=text_color,
        fontsize=12 if mode == "paper" else 13,
    )
    ax.set_title(
        "High-precision closure residual, two independent integrators",
        color="#111" if mode == "paper" else s["text"],
        fontweight="bold",
        fontsize=12 if mode == "paper" else 13,
        pad=10,
    )
    leg = ax.legend(
        frameon=False,
        loc="lower left",
        fontsize=10 if mode == "paper" else 11,
        labelcolor=text_color,
    )

    ax.grid(True, axis="y", alpha=0.3, linewidth=0.4, zorder=1)
    ax.tick_params(colors=s["mute"] if mode == "blog" else "#444")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    save_both(fig, "hp_dual_tool_residuals", mode)
    plt.close(fig)


# =============================================================================
# Fig 5 — ml_prior_ci
# =============================================================================
def fig_ml_prior_ci(mode):
    s = style(mode)
    with open(TIER1_AGG) as f:
        agg = json.load(f)

    ld = agg["per_method"]["label_disagreement"]["n_verified"]
    ce = agg["per_method"]["classifier_entropy"]["n_verified"]
    paired = agg["paired_n_verified_ld_vs_ce"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5) if mode == "paper" else (13, 6))
    fig.patch.set_facecolor(s["bg"])

    col_ld = "#40d09b" if mode == "blog" else "#2e8b57"
    col_ce = "#ff6b6b" if mode == "blog" else "#c2185b"

    # --- Panel 1: bar chart with CI and per-seed dots ---
    ax = axes[0]
    ax.set_facecolor(s["bg"] if mode == "blog" else "white")
    means = [ld["mean"], ce["mean"]]
    ci_lo = [ld["ci_lo"], ce["ci_lo"]]
    ci_hi = [ld["ci_hi"], ce["ci_hi"]]
    err = np.array([[m - lo, hi - m] for m, lo, hi in zip(means, ci_lo, ci_hi)]).T
    x = [0, 1]
    colors = [col_ld, col_ce]

    bars = ax.bar(
        x, means, width=0.55, color=colors, alpha=0.8,
        edgecolor="white" if mode == "paper" else s["text"], linewidth=0.6,
    )
    ax.errorbar(
        x, means, yerr=err, fmt="none", ecolor=s["text"] if mode == "blog" else "#222",
        capsize=6, lw=1.2, capthick=1.2,
    )

    # per-seed dots jittered around bar centers
    rng = np.random.default_rng(0)
    for xi, vals, c in [(0, ld["values"], col_ld), (1, ce["values"], col_ce)]:
        jx = xi + rng.uniform(-0.18, 0.18, len(vals))
        ax.scatter(
            jx, vals, s=40, color=c,
            edgecolor="white" if mode == "paper" else s["text"], linewidth=0.6,
            zorder=6, alpha=0.95,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["$k$-NN label\ndisagreement", "Classifier\nentropy"],
        fontsize=10 if mode == "paper" else 11,
        color=s["text"] if mode == "blog" else "#222",
    )
    ax.set_ylabel(
        "Verified periodic orbits / shot  (out of 50)",
        color=s["text"] if mode == "blog" else "#222",
    )
    ax.set_ylim(0, max(max(ld["values"]), max(ce["values"])) + 8)
    ax.set_title(
        "10-seed means with 95% CIs",
        color=s["text"] if mode == "blog" else "#111",
        fontweight="bold", pad=8,
    )
    # Headline stat
    ax.text(
        0.5, 0.96,
        f"$\\Delta = +{paired['mean_diff']:.1f}$   "
        f"$p = {paired['p_two_sided']:.1e}$   "
        f"Cohen's $d = {paired['cohens_d_paired']:.1f}$",
        transform=ax.transAxes,
        fontsize=10 if mode == "paper" else 11,
        color=s["text"] if mode == "blog" else "#111",
        ha="center", va="top", fontweight="bold",
        bbox=dict(
            facecolor=s["bg"] if mode == "blog" else "#f5f5f5",
            edgecolor=s["mute"], alpha=0.8,
        ),
    )
    ax.tick_params(colors=s["mute"])
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.4)

    # Annotate means above bars
    for xi, m in zip(x, means):
        ax.text(
            xi, m + 1.1, f"{m:.1f}",
            ha="center", va="bottom",
            fontsize=10 if mode == "paper" else 11,
            color=s["text"] if mode == "blog" else "#111",
            fontweight="bold",
        )

    # --- Panel 2: paired-line plot ---
    ax2 = axes[1]
    ax2.set_facecolor(s["bg"] if mode == "blog" else "white")

    ld_vals = ld["values"]
    ce_vals = ce["values"]
    for i, (a, b) in enumerate(zip(ld_vals, ce_vals)):
        ax2.plot(
            [0, 1], [a, b],
            color=s["mute"], lw=1.0, alpha=0.7, zorder=2,
        )
    ax2.scatter(
        [0] * len(ld_vals), ld_vals, s=80, color=col_ld, alpha=0.92, zorder=5,
        edgecolor="white" if mode == "paper" else s["text"], linewidth=0.6,
        label="Label disagreement",
    )
    ax2.scatter(
        [1] * len(ce_vals), ce_vals, s=80, color=col_ce, alpha=0.92, zorder=5,
        edgecolor="white" if mode == "paper" else s["text"], linewidth=0.6,
        label="Classifier entropy",
    )

    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(
        ["$k$-NN label\ndisagreement", "Classifier\nentropy"],
        fontsize=10 if mode == "paper" else 11,
        color=s["text"] if mode == "blog" else "#222",
    )
    ax2.set_ylabel(
        "Verified orbits (per seed)",
        color=s["text"] if mode == "blog" else "#222",
    )
    ax2.set_title(
        f"Per-seed paired comparison ({paired['n']} seeds; LD wins {paired['n']}/10)",
        color=s["text"] if mode == "blog" else "#111",
        fontweight="bold", pad=8,
    )
    ax2.tick_params(colors=s["mute"])
    ax2.grid(True, axis="y", alpha=0.3, linewidth=0.4)
    ax2.set_xlim(-0.4, 1.4)

    fig.suptitle(
        "Action #5 — Tier-1 ML-prior ablation (10 seeds, paired)",
        color=s["text"] if mode == "blog" else "#111",
        fontweight="bold", fontsize=13, y=1.02,
    )
    plt.tight_layout()
    save_both(fig, "ml_prior_ci", mode)
    plt.close(fig)


# =============================================================================
# Fig 6 — summary_verdict_card
# =============================================================================
def _card(ax, key, mode, s, trajectory):
    """Draw one card for orbit `key` with thumbnail and verdict checklist."""
    col = verdict_color(mode, key)
    bg = s["bg"] if mode == "blog" else "white"
    mute = s["mute"]
    fg = s["text"] if mode == "blog" else "#111"

    ax.set_facecolor(bg)
    ax.set_xticks([])
    ax.set_yticks([])
    # border
    for spine in ax.spines.values():
        spine.set_edgecolor(col)
        spine.set_linewidth(1.6)

    # Layout: top half = title + trajectory thumbnail; bottom half = checklist
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Header strip
    ax.add_patch(
        Rectangle((0, 0.88), 1, 0.12, facecolor=col, edgecolor="none", alpha=0.85)
    )
    v1, v2, T = ic_double(key)
    ax.text(
        0.05, 0.94,
        f"Orbit {key}",
        fontsize=14, color="white", fontweight="bold", va="center",
    )
    ax.text(
        0.98, 0.94,
        f"$T={T:.3f}$",
        fontsize=10, color="white", va="center", ha="right",
    )

    # Trajectory thumbnail as inset axis
    inset = ax.inset_axes([0.06, 0.40, 0.88, 0.44])
    inset.set_facecolor(bg)
    for sp in inset.spines.values():
        sp.set_visible(False)
    inset.set_xticks([])
    inset.set_yticks([])
    q = trajectory["q"]
    for b in range(3):
        inset.plot(
            q[:, b, 0], q[:, b, 1],
            color=s["body_colors"][b], lw=0.8, alpha=0.85,
        )
        inset.scatter(
            [q[0, b, 0]], [q[0, b, 1]],
            s=16, color=s["body_colors"][b], zorder=5,
            edgecolor="white" if mode == "paper" else s["text"], linewidth=0.4,
        )
    inset.set_xlim(-3.5, 3.5)
    inset.set_ylim(-3.5, 3.5)
    inset.set_aspect("equal")

    # Checklist below (use latex checkmark/X for serif PDF compatibility)
    if mode == "paper":
        PASS = r"$\checkmark$"
        FAIL = r"$\times$"
    else:
        PASS = "\u2713"  # ✓
        FAIL = "\u2717"  # ✗
    ok_col = "#2e8b57" if mode == "paper" else "#40d09b"
    bad_col = "#c2185b" if mode == "paper" else "#ff6b6b"

    # Per-orbit checklist
    if key == "A":
        lines = [
            (f"{PASS}", "Numerical novelty (H2024 + H2025)", ok_col),
            (f"{PASS}", "Topology: NEW (no match in 971)", ok_col),
            (f"{PASS}", "HP closure < 1e-30 (dual-tool)", ok_col),
        ]
        verdict_txt = "NEW orbit + NEW family"
    elif key == "B":
        lines = [
            (f"{PASS}", "Numerical novelty (H2024 + H2025)", ok_col),
            (f"{FAIL}", "Topology: family match (H2025 rows 6, 7)", bad_col),
            (f"{PASS}", "HP closure < 1e-30 (dual-tool)", ok_col),
        ]
        verdict_txt = "NEW orbit + KNOWN family"
    elif key == "C":
        lines = [
            (f"{FAIL}", "Matches Hristov 2025 #0006", bad_col),
            (f"{FAIL}", "Topology = H2025 row 6", bad_col),
            (f"{PASS}", "HP closure < 1e-30 (dual-tool)", ok_col),
        ]
        verdict_txt = "REDISCOVERY of H2025 #0006"
    else:  # D
        lines = [
            (f"{FAIL}", "Matches Hristov 2025 #0011", bad_col),
            (f"{FAIL}", "Topology = H2025 row 11", bad_col),
            (f"{PASS}", "HP closure < 1e-30 (dual-tool)", ok_col),
        ]
        verdict_txt = "REDISCOVERY of H2025 #0011"

    # Draw checklist lines
    y0 = 0.34
    dy = 0.07
    for i, (sym, txt, c) in enumerate(lines):
        yy = y0 - i * dy
        ax.text(0.06, yy, sym, fontsize=13, color=c, fontweight="bold", va="center")
        ax.text(0.14, yy, txt, fontsize=9 if mode == "paper" else 10,
                color=fg, va="center")

    # Final verdict at bottom
    verdict_col = col
    ax.text(
        0.5, 0.06,
        verdict_txt,
        fontsize=11, color=verdict_col, fontweight="bold",
        ha="center", va="center",
    )


def fig_summary_verdict_card(mode):
    s = style(mode)
    fig, axes = plt.subplots(2, 2, figsize=(10, 11) if mode == "paper" else (12, 13))
    fig.patch.set_facecolor(s["bg"])

    # Pre-integrate trajectories
    trajectories = {}
    for k in ORBIT_ORDER:
        v1, v2, T = ic_double(k)
        q, p, t = integrate_one_period(v1, v2, T, n_steps=5000)
        trajectories[k] = {"q": q, "p": p, "t": t, "T": T}

    for ax, k in zip(axes.ravel(), ORBIT_ORDER):
        _card(ax, k, mode, s, trajectories[k])

    fig.suptitle(
        "Verification report card — A, B, C, D",
        color=s["text"] if mode == "blog" else "#111",
        fontweight="bold", fontsize=14, y=0.995,
    )

    # Caption / footnote
    fig.text(
        0.5, 0.01,
        "Check rules: numerical novelty = no match at dv<1e-4 across H2024 (24,582) + H2025 (971).  "
        "Topology via Euler-crossing syzygy word (Action #3). "
        "HP closure: heyoka 200-bit & mpmath Taylor 30 dps both < 1e-30 (Action #4).",
        ha="center", fontsize=8 if mode == "paper" else 9,
        color=s["mute"], style="italic",
    )
    plt.tight_layout(rect=[0, 0.025, 1, 0.98])
    save_both(fig, "summary_verdict_card", mode)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    for mode in ["paper", "blog"]:
        print(f"\n===== {mode.upper()} MODE =====")
        print("Fig 1 — catalog_scan_map ...")
        fig_catalog_scan_map(mode)
        print("Fig 2 — orbit_novelty_matrix ...")
        fig_orbit_novelty_matrix(mode)
        print("Fig 3 — syzygy_word_comparison ...")
        fig_syzygy_word_comparison(mode)
        print("Fig 4 — hp_dual_tool_residuals ...")
        fig_hp_dual_tool_residuals(mode)
        print("Fig 5 — ml_prior_ci ...")
        fig_ml_prior_ci(mode)
        print("Fig 6 — summary_verdict_card ...")
        fig_summary_verdict_card(mode)
    print("\nAll 12 files written to paper2/figures/")
