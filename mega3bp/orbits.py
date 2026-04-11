"""Reference periodic orbits for testing the integrator.

Chenciner-Montgomery figure-eight orbit: the first explicitly constructed
planar three-body choreography with the three bodies chasing each other
around a lemniscate shape. Used here as the correctness gate for the
integrator (Phase 1, from the project scoping contract).

Reference:
    Chenciner & Montgomery, "A remarkable periodic solution of the three-body
    problem in the case of equal masses," Annals of Math. 152 (2000) 881-901.
"""
from __future__ import annotations

import jax.numpy as jnp

Array = jnp.ndarray

FIGURE_EIGHT_PERIOD = 6.32591398


def figure_eight_state() -> tuple[Array, Array]:
    """Return (q, p) for the figure-eight orbit in COM frame.

    Body 1, 2 are placed symmetrically about the origin; body 3 at origin.
    Velocities chosen so total linear momentum is exactly zero.
    Masses are 1 (natural units), so p = m*v = v.
    """
    q = jnp.array(
        [
            [-0.97000436, 0.24308753],
            [0.97000436, -0.24308753],
            [0.0, 0.0],
        ],
        dtype=jnp.float64,
    )
    v3 = jnp.array([-0.93240737, -0.86473146], dtype=jnp.float64)
    v12 = -0.5 * v3
    p = jnp.stack([v12, v12, v3], axis=0)
    return q, p
