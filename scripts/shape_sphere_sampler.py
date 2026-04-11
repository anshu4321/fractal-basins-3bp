"""Phase 1 deliverable: shape sphere sampler with 10k visualization.

Verifies the Montgomery shape sphere parameterization by:
  1. Sampling 10k points uniformly on S^2 with the natural measure.
  2. Mapping each to a planar 3-body configuration with unit moment of inertia.
  3. Confirming round-trip invariance (config -> shape -> config) for 100 samples.
  4. Confirming the three binary-collision points and the two Lagrange points
     map exactly to their analytic predictions.
  5. Producing two figures:
       figures/shape_sphere_3d.png  - 10k points on the unit sphere with
         collision and Lagrange landmarks marked.
       figures/shape_sphere_configs.png - 2D scatter of body positions for
         all 10k sampled configurations, showing how Cartesian space gets
         covered.
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

from mega3bp.shape_sphere import (
    COLLISION_12,
    COLLISION_13,
    COLLISION_23,
    LAGRANGE_NORTH,
    LAGRANGE_SOUTH,
    config_to_shape,
    moment_of_inertia,
    pairwise_distances,
    sample_shape_sphere,
    shape_to_config,
)

N_SAMPLES = 10_000
RNG_SEED = 0
TARGET_INERTIA = 1.0


def assert_close(name: str, actual: jnp.ndarray, expected: jnp.ndarray, atol: float = 1e-12) -> None:
    diff = float(jnp.max(jnp.abs(actual - expected)))
    if diff > atol:
        raise SystemExit(f"FAIL {name}: max diff {diff:.3e} > {atol:.0e}")
    print(f"  {name}: max diff {diff:.3e}")


def main() -> int:
    print(f"jax backend: {jax.default_backend()}")
    print(f"devices    : {jax.devices()}")
    print()

    # ---- Landmark verification --------------------------------------------------
    print("=== landmark check ===")
    print("Mapping shape -> config -> shape for collision and Lagrange points...")

    landmarks = jnp.stack(
        [COLLISION_12, COLLISION_13, COLLISION_23, LAGRANGE_NORTH, LAGRANGE_SOUTH]
    )
    landmark_configs = shape_to_config(landmarks, inertia=TARGET_INERTIA)

    names = ["coll(1,2)", "coll(1,3)", "coll(2,3)", "Lagrange N", "Lagrange S"]
    for name, lm, q in zip(names, landmarks, landmark_configs):
        d12, d13, d23 = pairwise_distances(q)
        I = float(moment_of_inertia(q))
        print(
            f"  {name}  shape={tuple(round(float(x), 4) for x in lm)}  "
            f"r12={float(d12):.4f}  r13={float(d13):.4f}  r23={float(d23):.4f}  "
            f"I={I:.4f}"
        )

    # Collision (1,2): r12 should be 0
    assert float(pairwise_distances(landmark_configs[0])[0]) < 1e-12, "coll(1,2) failed"
    # Collision (1,3): r13 should be 0
    assert float(pairwise_distances(landmark_configs[1])[1]) < 1e-12, "coll(1,3) failed"
    # Collision (2,3): r23 should be 0
    assert float(pairwise_distances(landmark_configs[2])[2]) < 1e-12, "coll(2,3) failed"

    # Lagrange points: equilateral triangle, so all three distances equal
    for i in (3, 4):
        d = pairwise_distances(landmark_configs[i])
        assert (
            float(jnp.max(jnp.abs(d - d.mean()))) < 1e-12
        ), f"Lagrange landmark {names[i]} not equilateral"
    print("  landmarks verified")
    print()

    # ---- Sample 10k points and validate -----------------------------------------
    print(f"=== sampling {N_SAMPLES} points ===")
    key = jax.random.PRNGKey(RNG_SEED)
    t0 = time.perf_counter()
    pts = sample_shape_sphere(key, N_SAMPLES)
    pts.block_until_ready()
    print(f"  sampled in {time.perf_counter() - t0:.3f}s")

    # On the unit sphere
    norms = jnp.linalg.norm(pts, axis=-1)
    print(f"  ||n|| range: [{float(jnp.min(norms)):.15f}, {float(jnp.max(norms)):.15f}]")

    # Map to configs
    t0 = time.perf_counter()
    configs = shape_to_config(pts, inertia=TARGET_INERTIA)
    configs.block_until_ready()
    print(f"  shape -> config in {time.perf_counter() - t0:.3f}s")

    # Validate moment of inertia and CM
    I_vals = moment_of_inertia(configs)
    cms = jnp.mean(configs, axis=-2)
    print(f"  moment of inertia: mean={float(jnp.mean(I_vals)):.12f}  std={float(jnp.std(I_vals)):.3e}")
    print(f"  |CM|^2 max: {float(jnp.max(jnp.sum(cms * cms, axis=-1))):.3e}")

    # Round-trip check on a sub-sample
    print()
    print("=== round-trip config -> shape ===")
    pts_back = config_to_shape(configs)
    diffs = jnp.linalg.norm(pts_back - pts, axis=-1)
    max_rt = float(jnp.max(diffs))
    mean_rt = float(jnp.mean(diffs))
    print(f"  max round-trip distance: {max_rt:.3e}")
    print(f"  mean round-trip distance: {mean_rt:.3e}")
    if max_rt > 1e-12:
        raise SystemExit(
            f"FAIL: round-trip distance {max_rt:.3e} > 1e-12 — Hopf map sign or "
            "Jacobi convention bug between shape_to_config and config_to_shape"
        )

    # ---- Uniformity check (mean point should be near origin) --------------------
    print()
    print("=== uniformity check ===")
    mean_n = jnp.mean(pts, axis=0)
    print(f"  mean of {N_SAMPLES} samples: {tuple(round(float(x), 4) for x in mean_n)}  (expect ~0)")
    print(f"  ||mean||: {float(jnp.linalg.norm(mean_n)):.4e}  (Monte-Carlo error ~ 1/sqrt(N) ~ 1e-2)")

    # ---- Plot 1: 3D scatter on the shape sphere ---------------------------------
    out_fig = PROJECT_ROOT / "figures"
    out_fig.mkdir(parents=True, exist_ok=True)

    pts_np = np.asarray(pts)
    print()
    print("=== plotting ===")
    fig = plt.figure(figsize=(8, 7), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        pts_np[:, 0], pts_np[:, 1], pts_np[:, 2],
        s=1.5, c="C0", alpha=0.35, depthshade=False,
    )
    # Wireframe sphere as visual aid
    u = np.linspace(0, 2 * np.pi, 32)
    v = np.linspace(0, np.pi, 16)
    sx = np.outer(np.cos(u), np.sin(v))
    sy = np.outer(np.sin(u), np.sin(v))
    sz = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(sx, sy, sz, color="0.7", lw=0.3, alpha=0.4)

    landmark_pts = np.asarray(landmarks)
    ax.scatter(
        landmark_pts[:3, 0], landmark_pts[:3, 1], landmark_pts[:3, 2],
        s=80, c="C3", marker="X", edgecolors="black", linewidth=0.5,
        label="binary collisions",
    )
    ax.scatter(
        landmark_pts[3:, 0], landmark_pts[3:, 1], landmark_pts[3:, 2],
        s=80, c="C2", marker="*", edgecolors="black", linewidth=0.5,
        label="Lagrange equilateral",
    )
    ax.set_xlabel(r"$n_1$")
    ax.set_ylabel(r"$n_2$")
    ax.set_zlabel(r"$n_3$")
    ax.set_title(
        f"Montgomery shape sphere: {N_SAMPLES:,} uniform samples (natural measure)\n"
        "binary collisions on equator, Lagrange points at poles"
    )
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_fig / "shape_sphere_3d.png")
    print(f"  saved {out_fig / 'shape_sphere_3d.png'}")
    plt.close(fig)

    # ---- Plot 2: 2D scatter of body positions for the 10k configurations -------
    configs_np = np.asarray(configs)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    colors = ["C0", "C1", "C2"]
    labels = ["body 1", "body 2", "body 3"]
    for i in range(3):
        ax.scatter(
            configs_np[:, i, 0], configs_np[:, i, 1],
            s=1, c=colors[i], alpha=0.25, label=labels[i],
        )
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"{N_SAMPLES:,} planar configurations sampled from shape sphere\n"
        "gauge: body-1-to-2 along $+x$, COM at origin, $I = 1$"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", markerscale=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_fig / "shape_sphere_configs.png")
    print(f"  saved {out_fig / 'shape_sphere_configs.png'}")
    plt.close(fig)

    return 0


if __name__ == "__main__":
    sys.exit(main())
