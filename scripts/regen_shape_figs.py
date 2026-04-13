"""Regenerate shape sphere figures in Dirac palette."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mega3bp.shape_sphere import (
    sample_shape_sphere, shape_to_config,
    COLLISION_12, COLLISION_13, COLLISION_23, LAGRANGE_NORTH, LAGRANGE_SOUTH,
)
from mega3bp.style import PALETTE, apply_dirac_style

apply_dirac_style()

key = jax.random.PRNGKey(0)
pts = np.asarray(sample_shape_sphere(key, 10000))
configs = np.asarray(shape_to_config(jnp.asarray(pts)))

# ---- Figure 1a: 3D shape sphere ----
fig = plt.figure(figsize=(10, 10), dpi=180)
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor(PALETTE["bg_deep"])
fig.patch.set_facecolor(PALETTE["bg_deep"])

ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.5, c=PALETTE["cyan"], alpha=0.15,
           edgecolors="none", rasterized=True)

landmarks = {
    r"$C_{12}$": (np.asarray(COLLISION_12), PALETTE["coral"]),
    r"$C_{13}$": (np.asarray(COLLISION_13), PALETTE["coral"]),
    r"$C_{23}$": (np.asarray(COLLISION_23), PALETTE["coral"]),
    r"$L_+$": (np.asarray(LAGRANGE_NORTH), PALETTE["lime"]),
    r"$L_-$": (np.asarray(LAGRANGE_SOUTH), PALETTE["lime"]),
}
for label, (pt, color) in landmarks.items():
    ax.scatter(*pt, s=120, c=color, edgecolors="white", linewidths=0.8, zorder=10)
    ax.text(pt[0]*1.15, pt[1]*1.15, pt[2]*1.15, label, color=color,
            fontsize=13, fontweight="bold", ha="center")

u = np.linspace(0, 2*np.pi, 100)
eq_x = np.cos(u)
eq_y = np.sin(u)
eq_z = np.zeros_like(u)
ax.plot(eq_x, eq_y, eq_z, color=PALETTE["text_mute"], lw=0.8, alpha=0.5)

ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_zlim(-1.2, 1.2)
ax.set_xlabel(r"$n_1$", color=PALETTE["text"], fontsize=12)
ax.set_ylabel(r"$n_2$", color=PALETTE["text"], fontsize=12)
ax.set_zlabel(r"$n_3$", color=PALETTE["text"], fontsize=12)
ax.tick_params(colors=PALETTE["text_mute"], labelsize=8)
ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor(PALETTE["spine"]); ax.yaxis.pane.set_edgecolor(PALETTE["spine"])
ax.zaxis.pane.set_edgecolor(PALETTE["spine"])
ax.grid(color=PALETTE["grid"], alpha=0.2)
ax.view_init(elev=25, azim=135)

fig.tight_layout()
fig.savefig(PROJECT_ROOT / "figures" / "shape_sphere_3d.png",
            facecolor=PALETTE["bg_deep"], bbox_inches="tight")
plt.close(fig)
print("saved shape_sphere_3d.png")

# ---- Figure 1b: configurations ----
fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=180)
fig.patch.set_facecolor(PALETTE["bg_deep"])

indices = [0, 1250, 2500, 3750, 5000, 6250, 7500, 8750]
colors_body = [PALETTE["cyan"], PALETTE["lime"], PALETTE["lavender"]]

for idx, ax in zip(indices, axes.flat):
    ax.set_facecolor(PALETTE["bg_panel"])
    cfg = configs[idx]
    for b in range(3):
        ax.plot(cfg[b, 0], cfg[b, 1], "o", color=colors_body[b], ms=10,
                markeredgecolor="white", markeredgewidth=0.5, zorder=5)
    for i in range(3):
        for j in range(i+1, 3):
            ax.plot([cfg[i,0], cfg[j,0]], [cfg[i,1], cfg[j,1]],
                    "-", color=PALETTE["text_mute"], lw=0.8, alpha=0.5)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    n = pts[idx]
    ax.set_title(f"({n[0]:.2f}, {n[1]:.2f}, {n[2]:.2f})",
                 color=PALETTE["text_mute"], fontsize=9, family="monospace")
    ax.tick_params(colors=PALETTE["text_mute"], labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["spine"])

fig.suptitle("Triangle configurations at sampled shape-sphere points",
             color=PALETTE["text"], fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(PROJECT_ROOT / "figures" / "shape_sphere_configs.png",
            facecolor=PALETTE["bg_deep"], bbox_inches="tight")
plt.close(fig)
print("saved shape_sphere_configs.png")
