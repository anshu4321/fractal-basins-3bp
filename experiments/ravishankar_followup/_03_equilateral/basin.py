"""Run 256x256 basin map on the equilateral section.

For each (u_x, u_y) in [-2, 2]^2 grid point:
1. Build IC via section.section_ic (equilateral triangle, edge 1; velocity
   field v_k = R(2*pi*k/3) @ u, which is S3-invariant and has sum(v_k) = 0).
2. Integrate with JAX Yoshida-6 (6th-order symplectic) on GPU via vmap.
3. Classify:
   - label 1/2/3 if body i's distance from COM exceeds R_ESCAPE at any time
     during the run (first-to-escape wins);
   - label 0 (periodic/bounded) otherwise.
4. Record final energy (=initial energy for an exact symplectic; we save the
   final integrator-level E so drift acts as a sanity check).

Uses mega3bp.integrators.yoshida6_step (JAX) which matches
experiments/orbit_discovery/extract_basin_map.py.
"""
from __future__ import annotations

import sys
import time
from functools import partial
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from mega3bp.integrators import yoshida6_step
from mega3bp.dynamics import total_energy

HERE = Path(__file__).resolve().parent

# Grid / integrator parameters.
GRID_N = 256
U_LO, U_HI = -2.0, 2.0
T_MAX = 80.0
DT = 0.01
N_STEPS = int(T_MAX / DT)   # 8000
R_ESCAPE = 50.0
BATCH_SIZE = 1024           # tune for A100 VRAM; 256^2 = 65536 total

R_TRI = 1.0 / np.sqrt(3.0)
_VERT = jnp.array(
    [
        [jnp.cos(0.0),            jnp.sin(0.0)],
        [jnp.cos(2 * jnp.pi / 3), jnp.sin(2 * jnp.pi / 3)],
        [jnp.cos(4 * jnp.pi / 3), jnp.sin(4 * jnp.pi / 3)],
    ],
    dtype=jnp.float64,
) * R_TRI


def _rot2d(theta):
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([[c, -s], [s, c]], dtype=jnp.float64)


_R0 = _rot2d(jnp.float64(0.0))
_R1 = _rot2d(jnp.float64(2 * jnp.pi / 3))
_R2 = _rot2d(jnp.float64(4 * jnp.pi / 3))


def _section_ic_jax(u):
    """u: (2,) jax array -> (q, p) each (3, 2). Equal unit masses, so p = v."""
    v0 = _R0 @ u
    v1 = _R1 @ u
    v2 = _R2 @ u
    v = jnp.stack([v0, v1, v2], axis=0)
    return _VERT, v


@partial(jax.jit, static_argnames=("n_steps",))
def classify_one(u_x, u_y, n_steps):
    """Integrate one IC, return (label, E_final)."""
    u = jnp.array([u_x, u_y], dtype=jnp.float64)
    q0, p0 = _section_ic_jax(u)
    h = jnp.float64(DT)

    def step_fn(carry, _):
        q, p, label, escaped = carry
        q, p = yoshida6_step(q, p, h)
        com = jnp.mean(q, axis=0)
        body_r = jnp.linalg.norm(q - com, axis=1)   # (3,)
        max_r = jnp.max(body_r)
        new_escape = (~escaped) & (max_r > R_ESCAPE)
        escaper = jnp.int8(jnp.argmax(body_r) + 1)
        label = jnp.where(new_escape, escaper, label)
        escaped = escaped | new_escape
        return (q, p, label, escaped), None

    init = (q0, p0, jnp.int8(0), jnp.bool_(False))
    (q_f, p_f, label, _), _ = jax.lax.scan(
        step_fn, init, jnp.arange(n_steps, dtype=jnp.int32)
    )
    E_final = total_energy(q_f, p_f)
    return label, jnp.float32(E_final)


@partial(jax.jit, static_argnames=("n_steps",))
def batch_classify(ux_b, uy_b, n_steps):
    return jax.vmap(lambda a, b: classify_one(a, b, n_steps))(ux_b, uy_b)


def run_basin():
    print("=" * 60)
    print("EQUILATERAL BASIN MAP (256x256, JAX GPU Yoshida-6)")
    print(f"grid:    {GRID_N} x {GRID_N} = {GRID_N ** 2} cells")
    print(f"u:       [{U_LO}, {U_HI}]^2")
    print(f"T_max:   {T_MAX}  (dt={DT}, n_steps={N_STEPS})")
    print(f"R_esc:   {R_ESCAPE}")
    print(f"backend: {jax.default_backend()}  devices={jax.devices()}")
    print("=" * 60)

    us = np.linspace(U_LO, U_HI, GRID_N)
    U_x, U_y = np.meshgrid(us, us, indexing="ij")
    ux_flat = U_x.ravel().astype(np.float64)
    uy_flat = U_y.ravel().astype(np.float64)
    N = ux_flat.size

    labels_all = np.zeros(N, dtype=np.int8)
    energies_all = np.zeros(N, dtype=np.float32)

    t0 = time.perf_counter()
    for i in range(0, N, BATCH_SIZE):
        xb = jnp.asarray(ux_flat[i:i + BATCH_SIZE])
        yb = jnp.asarray(uy_flat[i:i + BATCH_SIZE])
        lab_b, e_b = batch_classify(xb, yb, N_STEPS)
        labels_all[i:i + BATCH_SIZE] = np.asarray(lab_b)
        energies_all[i:i + BATCH_SIZE] = np.asarray(e_b)
        n_done = min(i + BATCH_SIZE, N)
        dt_elapsed = time.perf_counter() - t0
        eta = dt_elapsed * (N - n_done) / max(n_done, 1)
        print(f"  {n_done}/{N}  [{dt_elapsed:.1f}s, ETA {eta:.1f}s]", flush=True)

    wall = time.perf_counter() - t0

    labels = labels_all.reshape(GRID_N, GRID_N)
    energies = energies_all.reshape(GRID_N, GRID_N)

    out = HERE / "basin_map.npz"
    np.savez(
        out,
        labels=labels,
        energies=energies,
        u_x_grid=us,
        u_y_grid=us,
        grid_n=GRID_N,
        T_max=T_MAX,
        dt=DT,
        R_escape=R_ESCAPE,
        wall_s=wall,
    )

    unique, counts = np.unique(labels, return_counts=True)
    print("\nLabel distribution:")
    for u, c in zip(unique, counts):
        kind = {0: "periodic/bounded", 1: "body-1-esc",
                2: "body-2-esc", 3: "body-3-esc"}.get(int(u), f"label={u}")
        print(f"  {kind:20s}  {c}  ({100 * c / N:.2f}%)")

    n_periodic = int((labels == 0).sum())
    print(
        f"\nwrote {out}  grid={GRID_N}^2={N}  n_periodic={n_periodic}  "
        f"wall={wall:.1f}s"
    )
    return labels, energies


if __name__ == "__main__":
    run_basin()
