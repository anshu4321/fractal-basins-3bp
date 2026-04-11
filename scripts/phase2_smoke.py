"""Phase 2 smoke test: batch integrator end-to-end on 1024 shape-sphere ICs.

Samples 1024 points uniformly on the Montgomery shape sphere, builds the
corresponding planar configurations with zero initial momentum, integrates
them in a single vmapped lax.scan to t_max, and labels each trajectory with
its escape outcome. Produces the first aesthetic visualizations of the
basin partition.

Outputs:
    figures/phase2_basin_shape.png      - shape sphere points colored by outcome
    figures/phase2_trajectories.png     - ~24 sample trajectories colored by outcome
    figures/phase2_escape_histogram.png - escape-time histogram by outcome
    figures/phase2_rmin.png             - r_min_ever distribution (close-encounter check)
    figures/phase2_dashboard.png        - combined multi-panel figure

All plots use the Dirac palette.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb

from mega3bp.batch import integrate_batch
from mega3bp.escape import LABEL_NAMES
from mega3bp.shape_sphere import sample_shape_sphere, shape_to_config
from mega3bp.style import (
    BASIN_COLORS,
    BASIN_GLOW,
    PALETTE,
    apply_dirac_style,
)

N_ICS = 1024
RNG_SEED = 0xC0FFEE

H = 0.01
T_MAX = 200.0            # 200 time units ~ 32 figure-eight periods
N_STEPS = int(round(T_MAX / H))
R_ESCAPE = 5.0
BINARY_FACTOR = 2.0
R_CLOSE = 1e-3

# Small number of trajectories to record for the overlay plot
N_OVERLAY = 24


def label_to_name(label: int) -> str:
    return LABEL_NAMES.get(int(label), "unknown")


def color_for_label(label: int) -> str:
    name = label_to_name(label)
    return BASIN_COLORS.get(name, PALETTE["text_dim"])


def main() -> int:
    apply_dirac_style()
    print(f"jax backend: {jax.default_backend()}")
    print(f"devices    : {jax.devices()}")
    print()
    print("=== sampling ICs ===")
    key = jax.random.PRNGKey(RNG_SEED)
    shape_pts = sample_shape_sphere(key, N_ICS)
    configs = shape_to_config(shape_pts, inertia=1.0)
    p0 = jnp.zeros_like(configs)
    print(f"  {N_ICS} shape-sphere points -> planar configs (zero initial momentum)")

    # ---- Full batch integration (no recording, fast path) ------------------
    print()
    print("=== integrating full batch ===")
    print(f"  h={H}  T={T_MAX}  N_STEPS={N_STEPS}")
    t0 = time.perf_counter()
    result = integrate_batch(
        configs, p0, H, N_STEPS,
        r_escape=R_ESCAPE,
        binary_factor=BINARY_FACTOR,
        r_close=R_CLOSE,
        record_every=0,
    )
    jax.block_until_ready(result["label"])
    elapsed = time.perf_counter() - t0
    print(f"  done in {elapsed:.2f}s ({N_ICS * N_STEPS / elapsed:,.0f} traj-steps/sec)")

    labels = np.asarray(result["label"])
    escape_times = np.asarray(result["escape_time"])
    r_min_ever = np.asarray(result["r_min_ever"])
    ambiguous = np.asarray(result["ambiguous"])
    shape_np = np.asarray(shape_pts)

    # ---- Summary -----------------------------------------------------------
    print()
    print("=== outcome summary ===")
    unique, counts = np.unique(labels, return_counts=True)
    total = labels.size
    for lab, cnt in zip(unique, counts):
        name = label_to_name(lab)
        pct = 100 * cnt / total
        print(f"  {name:16s} ({int(lab):+d})  : {cnt:5d}  ({pct:5.1f}%)")
    amb_frac = float(ambiguous.mean())
    print(f"  ambiguous fraction  : {amb_frac:.3%}")
    print(f"  min r_min_ever      : {float(r_min_ever.min()):.3e}")
    print(f"  max r_min_ever      : {float(r_min_ever.max()):.3e}")

    # ---- Small batch re-run with trajectory recording for overlay plot -----
    print()
    print(f"=== re-running {N_OVERLAY} representative trajectories (with trace) ===")

    # Pick trajectories: emphasize bound (the interesting chaotic dance)
    # while guaranteeing representative escapers.
    def pick_examples() -> np.ndarray:
        sel: list[int] = []
        bound_idx = np.where(labels == 0)[0]
        sel.extend(bound_idx[:10].tolist())
        for lab in [1, 2, 3]:
            idx = np.where(labels == lab)[0]
            sel.extend(idx[:4].tolist())
        amb = np.where(labels == -1)[0]
        sel.extend(amb[:2].tolist())
        return np.array(sel, dtype=np.int32)

    example_idx = pick_examples()
    q0_sub = configs[example_idx]
    p0_sub = p0[example_idx]
    t0 = time.perf_counter()
    sub_result = integrate_batch(
        q0_sub, p0_sub, H, N_STEPS,
        r_escape=R_ESCAPE,
        binary_factor=BINARY_FACTOR,
        r_close=R_CLOSE,
        record_every=20,
    )
    jax.block_until_ready(sub_result["q_trace"])
    print(f"  recorded traces in {time.perf_counter() - t0:.2f}s")

    q_trace = np.asarray(sub_result["q_trace"])  # shape (T_rec, B, 3, 2)
    sub_labels = np.asarray(sub_result["label"])

    # --------------------------------------------------------------------
    out_fig = PROJECT_ROOT / "figures"
    out_fig.mkdir(exist_ok=True)

    # ---- Plot 1: shape sphere scatter colored by outcome ------------------
    print()
    print("=== plotting ===")
    fig = plt.figure(figsize=(8, 7.5), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(PALETTE["bg_deep"])
    ax.xaxis.set_pane_color((0, 0, 0, 0))
    ax.yaxis.set_pane_color((0, 0, 0, 0))
    ax.zaxis.set_pane_color((0, 0, 0, 0))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_color(PALETTE["spine"])
        axis.set_tick_params(colors=PALETTE["text_mute"])
        axis.label.set_color(PALETTE["text"])

    # Wireframe sphere
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    sx = np.outer(np.cos(u), np.sin(v))
    sy = np.outer(np.sin(u), np.sin(v))
    sz = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(sx, sy, sz, color=PALETTE["spine"], lw=0.3, alpha=0.35)

    # Draw bound first (the majority) with small faint markers so they recede.
    # Escape categories go on top with solid prominent markers. Ambiguous on very top.
    bound_mask = labels == 0
    if bound_mask.any():
        ax.scatter(
            shape_np[bound_mask, 0], shape_np[bound_mask, 1], shape_np[bound_mask, 2],
            s=6, c=BASIN_COLORS["bound"], alpha=0.25,
            edgecolors="none", depthshade=False,
            label=f"bound ({bound_mask.sum()})",
        )
    for lab in (1, 2, 3):
        mask = labels == lab
        if not mask.any():
            continue
        name = label_to_name(lab)
        ax.scatter(
            shape_np[mask, 0], shape_np[mask, 1], shape_np[mask, 2],
            s=60, c=BASIN_COLORS[name], alpha=1.0,
            edgecolors=PALETTE["bg_deep"], linewidths=0.7,
            depthshade=False,
            label=f"{name} ({mask.sum()})",
        )
    amb_mask = labels == -1
    if amb_mask.any():
        ax.scatter(
            shape_np[amb_mask, 0], shape_np[amb_mask, 1], shape_np[amb_mask, 2],
            s=55, c=BASIN_COLORS["ambiguous"], alpha=0.9,
            edgecolors=PALETTE["text"], linewidths=0.8,
            marker="X", depthshade=False,
            label=f"ambiguous ({amb_mask.sum()})",
        )
    ax.set_xlabel(r"$n_1$")
    ax.set_ylabel(r"$n_2$")
    ax.set_zlabel(r"$n_3$")
    ax.set_title(
        f"Phase 2 basin preview — {N_ICS} shape-sphere ICs\nat rest, integrated to $t = {T_MAX}$",
        color=PALETTE["text"], pad=18,
    )
    leg = ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=8)
    leg.get_frame().set_facecolor(PALETTE["bg_elevate"])
    leg.get_frame().set_edgecolor(PALETTE["spine"])
    for text in leg.get_texts():
        text.set_color(PALETTE["text"])
    fig.tight_layout()
    fig.savefig(out_fig / "phase2_basin_shape.png")
    print(f"  saved {out_fig / 'phase2_basin_shape.png'}")
    plt.close(fig)

    # ---- Plot 2: trajectory overlay --------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.set_facecolor(PALETTE["bg_deep"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    # Clamp viewing region to the initial-triangle scale. Escapers will leave
    # the frame — this is intentional so the bound chaotic dynamics stay visible.
    TRAJ_LIM = 2.5

    def draw_traj(k: int, lw_mult: float = 1.0, alpha_mult: float = 1.0, order: int = 5):
        lab = int(sub_labels[k])
        name = label_to_name(lab)
        col = BASIN_COLORS[name]
        for body in range(3):
            xs = q_trace[:, k, body, 0]
            ys = q_trace[:, k, body, 1]
            segs = np.stack(
                [np.column_stack([xs[:-1], ys[:-1]]),
                 np.column_stack([xs[1:], ys[1:]])],
                axis=1,
            )
            alphas = np.linspace(0.25, 0.85, len(segs)) * alpha_mult
            rgba = np.zeros((len(segs), 4))
            rgba[:, :3] = to_rgb(col)
            rgba[:, 3] = alphas
            lc = LineCollection(segs, colors=rgba, linewidths=1.1 * lw_mult, zorder=order)
            ax.add_collection(lc)
            ax.scatter(xs[0], ys[0], s=18, c=col, alpha=0.95, edgecolors=PALETTE["bg_deep"],
                       linewidths=0.4, zorder=order + 1)

    # Draw bound under, escapes over (so escape tangents are visible)
    for k in range(q_trace.shape[1]):
        if int(sub_labels[k]) == 0:
            draw_traj(k, lw_mult=0.9, alpha_mult=0.75, order=4)
    for k in range(q_trace.shape[1]):
        if int(sub_labels[k]) not in (0, -1):
            draw_traj(k, lw_mult=1.3, alpha_mult=1.0, order=6)
    for k in range(q_trace.shape[1]):
        if int(sub_labels[k]) == -1:
            draw_traj(k, lw_mult=1.1, alpha_mult=0.9, order=7)

    # Legend with dummy proxy artists
    handles = []
    for name, col in BASIN_COLORS.items():
        handles.append(plt.Line2D([0], [0], color=col, lw=2.2, label=name))
    leg = ax.legend(handles=handles, loc="upper right", fontsize=8)
    leg.get_frame().set_facecolor(PALETTE["bg_elevate"])
    leg.get_frame().set_edgecolor(PALETTE["spine"])
    for text in leg.get_texts():
        text.set_color(PALETTE["text"])

    ax.text(
        0.5, 0.98,
        f"{q_trace.shape[1]} sample trajectories  ·  colored by escape outcome",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=12, color=PALETTE["text"],
    )
    ax.text(
        0.5, 0.955,
        f"viewport clamped to $\\pm{TRAJ_LIM}$; escapers leave the frame",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=9, color=PALETTE["text_dim"],
    )

    ax.set_xlim(-TRAJ_LIM, TRAJ_LIM)
    ax.set_ylim(-TRAJ_LIM, TRAJ_LIM)

    fig.tight_layout()
    fig.savefig(out_fig / "phase2_trajectories.png")
    print(f"  saved {out_fig / 'phase2_trajectories.png'}")
    plt.close(fig)

    # ---- Plot 3: escape time histogram -----------------------------------
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    finite = np.isfinite(escape_times)
    # At-rest ICs trigger escape very early (t < 30 typically); use log bins to
    # spread the distribution out instead of a single spike at t≈0.
    finite_times = escape_times[finite & (escape_times > 0)]
    if finite_times.size > 0:
        t_lo = max(float(finite_times.min()) * 0.5, 0.5)
        t_hi = max(float(finite_times.max()) * 1.2, 10.0)
    else:
        t_lo, t_hi = 0.5, 50.0
    bins = np.logspace(np.log10(t_lo), np.log10(t_hi), 30)

    for lab in [1, 2, 3]:
        mask = (labels == lab) & finite
        if mask.sum() == 0:
            continue
        name = label_to_name(lab)
        ax.hist(
            escape_times[mask],
            bins=bins,
            color=BASIN_COLORS[name],
            alpha=0.72,
            label=f"{name} ({mask.sum()})",
            edgecolor=PALETTE["bg_deep"],
            linewidth=0.5,
        )
    ax.set_xlabel(r"escape time $t_{\mathrm{esc}}$  (log)")
    ax.set_ylabel("count")
    ax.set_xscale("log")
    ax.set_xlim(t_lo, t_hi)
    ax.set_title(
        f"Escape time distribution — {N_ICS} ICs from shape sphere at rest",
        color=PALETTE["text"],
    )
    leg = ax.legend(loc="upper right")
    leg.get_frame().set_facecolor(PALETTE["bg_elevate"])
    leg.get_frame().set_edgecolor(PALETTE["spine"])
    for t in leg.get_texts():
        t.set_color(PALETTE["text"])
    fig.tight_layout()
    fig.savefig(out_fig / "phase2_escape_histogram.png")
    print(f"  saved {out_fig / 'phase2_escape_histogram.png'}")
    plt.close(fig)

    # ---- Plot 4: r_min distribution --------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    log_rmin = np.log10(np.maximum(r_min_ever, 1e-6))
    ax.hist(
        log_rmin,
        bins=50,
        color=PALETTE["cyan"],
        alpha=0.85,
        edgecolor=PALETTE["bg_deep"],
        linewidth=0.5,
    )
    ax.axvline(
        np.log10(R_CLOSE),
        color=PALETTE["coral"],
        ls="--",
        lw=1.3,
        label=f"ambiguous threshold ($r_{{\\mathrm{{close}}}} = {R_CLOSE:g}$)",
    )
    ax.set_xlabel(r"$\log_{10} r_{\min}$ over trajectory")
    ax.set_ylabel("count")
    ax.set_title(
        f"Minimum pair distance distribution — {amb_frac:.2%} ambiguous",
        color=PALETTE["text"],
    )
    leg = ax.legend(loc="upper left")
    leg.get_frame().set_facecolor(PALETTE["bg_elevate"])
    leg.get_frame().set_edgecolor(PALETTE["spine"])
    for t in leg.get_texts():
        t.set_color(PALETTE["text"])
    fig.tight_layout()
    fig.savefig(out_fig / "phase2_rmin.png")
    print(f"  saved {out_fig / 'phase2_rmin.png'}")
    plt.close(fig)

    # ---- Dashboard (combined) -------------------------------------------
    fig = plt.figure(figsize=(16, 11), dpi=150, constrained_layout=True)
    fig.patch.set_facecolor(PALETTE["bg_deep"])
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1])

    # Panel A: shape sphere 3D
    axA = fig.add_subplot(gs[0, 0], projection="3d")
    axA.set_facecolor(PALETTE["bg_deep"])
    for axis in (axA.xaxis, axA.yaxis, axA.zaxis):
        axis.line.set_color(PALETTE["spine"])
        axis.set_pane_color((0, 0, 0, 0))
        axis.set_tick_params(colors=PALETTE["text_mute"])
        axis.label.set_color(PALETTE["text"])
    axA.plot_wireframe(sx, sy, sz, color=PALETTE["spine"], lw=0.25, alpha=0.35)
    # Bound underneath (faint), escapes on top (solid, prominent)
    if bound_mask.any():
        axA.scatter(
            shape_np[bound_mask, 0], shape_np[bound_mask, 1], shape_np[bound_mask, 2],
            s=4, c=BASIN_COLORS["bound"], alpha=0.22,
            edgecolors="none", depthshade=False,
        )
    for lab in (1, 2, 3):
        mask = labels == lab
        if not mask.any():
            continue
        axA.scatter(
            shape_np[mask, 0], shape_np[mask, 1], shape_np[mask, 2],
            s=42, c=BASIN_COLORS[label_to_name(lab)], alpha=1.0,
            edgecolors=PALETTE["bg_deep"], linewidths=0.6,
            depthshade=False,
        )
    if amb_mask.any():
        axA.scatter(
            shape_np[amb_mask, 0], shape_np[amb_mask, 1], shape_np[amb_mask, 2],
            s=40, c=BASIN_COLORS["ambiguous"], alpha=0.9,
            edgecolors=PALETTE["text"], linewidths=0.6,
            marker="X", depthshade=False,
        )
    axA.set_title("basin map on shape sphere", color=PALETTE["text"], pad=10)
    axA.set_xlabel(r"$n_1$")
    axA.set_ylabel(r"$n_2$")
    axA.set_zlabel(r"$n_3$")

    # Panel B: trajectory overlay (same clamping as standalone plot)
    axB = fig.add_subplot(gs[0, 1])
    axB.set_facecolor(PALETTE["bg_deep"])
    for spine in axB.spines.values():
        spine.set_visible(False)
    axB.set_xticks([]); axB.set_yticks([])
    axB.set_aspect("equal")

    def draw_traj_on(axB, k: int, lw_mult: float = 1.0, alpha_mult: float = 1.0, order: int = 5):
        lab = int(sub_labels[k])
        col = BASIN_COLORS[label_to_name(lab)]
        for body in range(3):
            xs = q_trace[:, k, body, 0]
            ys = q_trace[:, k, body, 1]
            segs = np.stack(
                [np.column_stack([xs[:-1], ys[:-1]]),
                 np.column_stack([xs[1:], ys[1:]])],
                axis=1,
            )
            alphas = np.linspace(0.2, 0.8, len(segs)) * alpha_mult
            rgba = np.zeros((len(segs), 4))
            rgba[:, :3] = to_rgb(col)
            rgba[:, 3] = alphas
            lc = LineCollection(segs, colors=rgba, linewidths=0.9 * lw_mult, zorder=order)
            axB.add_collection(lc)

    for k in range(q_trace.shape[1]):
        if int(sub_labels[k]) == 0:
            draw_traj_on(axB, k, lw_mult=0.8, alpha_mult=0.7, order=4)
    for k in range(q_trace.shape[1]):
        if int(sub_labels[k]) not in (0, -1):
            draw_traj_on(axB, k, lw_mult=1.25, alpha_mult=1.0, order=6)
    for k in range(q_trace.shape[1]):
        if int(sub_labels[k]) == -1:
            draw_traj_on(axB, k, lw_mult=1.0, alpha_mult=0.9, order=7)

    axB.set_xlim(-TRAJ_LIM, TRAJ_LIM)
    axB.set_ylim(-TRAJ_LIM, TRAJ_LIM)
    axB.set_title("sample trajectories", color=PALETTE["text"])

    # Panel C: outcome counts as a horizontal bar chart
    axC = fig.add_subplot(gs[0, 2])
    axC.set_facecolor(PALETTE["bg_panel"])
    names_ord = ["body1_escape", "body2_escape", "body3_escape", "bound", "ambiguous"]
    codes = {-1: "ambiguous", 0: "bound", 1: "body1_escape", 2: "body2_escape", 3: "body3_escape"}
    count_map = {name: 0 for name in names_ord}
    for lab, cnt in zip(unique, counts):
        count_map[codes[int(lab)]] = int(cnt)
    vals = [count_map[n] for n in names_ord]
    cols = [BASIN_COLORS[n] for n in names_ord]
    ypos = np.arange(len(names_ord))
    bars = axC.barh(ypos, vals, color=cols, alpha=0.9, edgecolor=PALETTE["bg_deep"], linewidth=0.8)
    axC.set_yticks(ypos)
    axC.set_yticklabels(names_ord, color=PALETTE["text"])
    axC.set_xlabel("count")
    axC.set_title("outcome counts", color=PALETTE["text"])
    axC.invert_yaxis()
    for bar, val in zip(bars, vals):
        if val > 0:
            axC.text(val, bar.get_y() + bar.get_height() / 2,
                     f" {val}", va="center", color=PALETTE["text"], fontsize=9)

    # Panel D: escape time histogram (log x, same bins as standalone)
    axD = fig.add_subplot(gs[1, 0])
    for lab in [1, 2, 3]:
        mask = (labels == lab) & finite
        if mask.sum() == 0:
            continue
        axD.hist(
            escape_times[mask], bins=bins,
            color=BASIN_COLORS[label_to_name(lab)], alpha=0.72,
            edgecolor=PALETTE["bg_deep"], linewidth=0.4,
            label=label_to_name(lab),
        )
    axD.set_xlabel(r"escape time $t_{\mathrm{esc}}$  (log)")
    axD.set_ylabel("count")
    axD.set_xscale("log")
    axD.set_xlim(t_lo, t_hi)
    axD.set_title("escape time distribution")

    # Panel E: r_min histogram
    axE = fig.add_subplot(gs[1, 1])
    axE.hist(
        log_rmin, bins=50, color=PALETTE["cyan"], alpha=0.85,
        edgecolor=PALETTE["bg_deep"], linewidth=0.4,
    )
    axE.axvline(np.log10(R_CLOSE), color=PALETTE["coral"], ls="--", lw=1.2)
    axE.set_xlabel(r"$\log_{10} r_{\min}$")
    axE.set_ylabel("count")
    axE.set_title(rf"$r_{{\min}}$ distribution  ·  ambig {amb_frac:.2%}")

    # Panel F: summary text
    axF = fig.add_subplot(gs[1, 2])
    axF.axis("off")
    axF.set_facecolor(PALETTE["bg_deep"])
    lines = [
        r"$\bf{Phase\ 2\ smoke\ test}$",
        "",
        f"N ICs          : {N_ICS}",
        f"h              : {H}",
        f"T_max          : {T_MAX}",
        f"n_steps        : {N_STEPS}",
        f"r_escape       : {R_ESCAPE}",
        f"binary_factor  : {BINARY_FACTOR}",
        f"r_close        : {R_CLOSE}",
        "",
        f"GPU            : H100 80GB",
        f"integration    : {elapsed:.2f}s",
        f"throughput     : {N_ICS * N_STEPS / elapsed / 1e6:.2f} Mstep/s",
        "",
        f"bound          : {count_map['bound']} ({100*count_map['bound']/total:.1f}%)",
        f"body1 escapes  : {count_map['body1_escape']} ({100*count_map['body1_escape']/total:.1f}%)",
        f"body2 escapes  : {count_map['body2_escape']} ({100*count_map['body2_escape']/total:.1f}%)",
        f"body3 escapes  : {count_map['body3_escape']} ({100*count_map['body3_escape']/total:.1f}%)",
        f"ambiguous      : {count_map['ambiguous']} ({100*count_map['ambiguous']/total:.1f}%)",
    ]
    for i, line in enumerate(lines):
        axF.text(
            0.02, 0.97 - i * 0.055, line,
            transform=axF.transAxes, ha="left", va="top",
            fontsize=10, color=PALETTE["text"],
            family="monospace",
        )

    fig.suptitle(
        "Phase 2 smoke test — planar 3BP basin preview",
        color=PALETTE["text"], fontsize=16, y=0.995,
    )
    fig.savefig(out_fig / "phase2_dashboard.png", facecolor=PALETTE["bg_deep"])
    print(f"  saved {out_fig / 'phase2_dashboard.png'}")
    plt.close(fig)

    print()
    print("=== phase 2 smoke test complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
