"""Montgomery shape sphere parameterization for the planar equal-mass 3BP.

The configuration space of three planar bodies in the COM frame, modulo
overall rotation and scaling, is the complex projective line CP^1 = S^2.
The Hopf map sends mass-weighted Jacobi coordinates (zeta_1, zeta_2) in C^2
to a point (n_1, n_2, n_3) on the unit sphere. The natural Fubini-Study
metric descends to the round metric on S^2, so the natural measure for
uniform sampling is just uniform-on-S^2.

Three binary-collision singularities sit on the n_3 = 0 equator, 120 deg apart:
    (1,2) collision: (-1,            0,   0)
    (1,3) collision: ( 1/2, -sqrt(3)/2,   0)
    (2,3) collision: ( 1/2,  sqrt(3)/2,   0)
Two equilateral (Lagrange) configurations sit at the poles (0, 0, +-1).

Conventions used here:
    rho_1 = (q_2 - q_1) / sqrt(2)
    rho_2 = sqrt(2/3) * (q_3 - (q_1 + q_2) / 2)
    zeta_i = rho_i_x + i * rho_i_y
    I = |rho_1|^2 + |rho_2|^2 = sum_i |q_i - Q|^2  (m = 1)

Inverse map shape -> config: choose the U(1) gauge where zeta_1 is real and
non-negative. This makes the body-1-to-body-2 separation point along the
positive x-axis.

Reference: Montgomery, Nonlinearity 11 (1998) 363-376.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jnp.ndarray

SQRT2 = jnp.sqrt(jnp.asarray(2.0, dtype=jnp.float64))
SQRT_2_OVER_3 = jnp.sqrt(jnp.asarray(2.0 / 3.0, dtype=jnp.float64))

# Reference points on the shape sphere (for plotting and tests).
COLLISION_12 = jnp.array([-1.0, 0.0, 0.0], dtype=jnp.float64)
COLLISION_13 = jnp.array([0.5, -jnp.sqrt(jnp.asarray(3.0, dtype=jnp.float64)) / 2.0, 0.0], dtype=jnp.float64)
COLLISION_23 = jnp.array([0.5, jnp.sqrt(jnp.asarray(3.0, dtype=jnp.float64)) / 2.0, 0.0], dtype=jnp.float64)
LAGRANGE_NORTH = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64)
LAGRANGE_SOUTH = jnp.array([0.0, 0.0, -1.0], dtype=jnp.float64)


def sample_shape_sphere(key: jax.Array, n: int) -> Array:
    """Sample n points uniformly on the shape sphere (round measure on S^2)."""
    g = jax.random.normal(key, (n, 3), dtype=jnp.float64)
    return g / jnp.linalg.norm(g, axis=-1, keepdims=True)


def shape_to_config(n: Array, inertia: float = 1.0) -> Array:
    """Map a shape-sphere point (..., 3) to a planar config (..., 3, 2).

    Gauge: zeta_1 is real and non-negative -> body-1-to-2 vector along +x.
    Scale: total moment of inertia equals `inertia`.

    Returns positions in the COM frame, shape (..., 3, 2).
    """
    n1, n2, n3 = n[..., 0], n[..., 1], n[..., 2]
    s = jnp.sqrt(jnp.asarray(inertia, dtype=jnp.float64))

    a2 = jnp.maximum((1.0 + n1) / 2.0, 0.0)  # |zeta_1|^2 (normalized)
    zeta1_re = jnp.sqrt(a2)
    # At binary-(1,2) collision n1 = -1 -> zeta1_re = 0; assign zeta_2 along
    # the imaginary axis so the inverse stays well-defined and bodies 1,2
    # coincide on the x-axis, but zeta_2 is fully imaginary on a continuous limit.
    safe = zeta1_re > 1e-30
    denom = jnp.where(safe, 2.0 * zeta1_re, jnp.ones_like(zeta1_re))
    fallback_re = jnp.zeros_like(n2)
    fallback_im = jnp.sqrt(jnp.maximum((1.0 - n1) / 2.0, 0.0))
    # Hopf inverse: 2 * zeta_1 * conj(zeta_2) = n_2 + i n_3
    # With zeta_1 = a (real, positive), zeta_2 = c + i d:
    #   2 a (c - i d) = n_2 + i n_3   =>   c = n_2 / (2a),  d = -n_3 / (2a)
    zeta2_re = jnp.where(safe, n2 / denom, fallback_re)
    zeta2_im = jnp.where(safe, -n3 / denom, fallback_im)

    # Apply overall scale
    zeta1_re = s * zeta1_re
    zeta2_re = s * zeta2_re
    zeta2_im = s * zeta2_im

    rho1 = jnp.stack([zeta1_re, jnp.zeros_like(zeta1_re)], axis=-1)
    rho2 = jnp.stack([zeta2_re, zeta2_im], axis=-1)

    # Inverse Jacobi:
    #   q_3 = sqrt(2/3) * rho_2
    #   q_1 = -q_3/2 - (sqrt(2)/2) * rho_1
    #   q_2 = -q_3/2 + (sqrt(2)/2) * rho_1
    q3 = SQRT_2_OVER_3 * rho2
    q1 = -0.5 * q3 - 0.5 * SQRT2 * rho1
    q2 = -0.5 * q3 + 0.5 * SQRT2 * rho1
    return jnp.stack([q1, q2, q3], axis=-2)


def config_to_shape(q: Array) -> Array:
    """Map a planar config (..., 3, 2) -> shape sphere point (..., 3).

    Translation, rotation, and scaling invariant. The CM is removed first.
    """
    q = q - jnp.mean(q, axis=-2, keepdims=True)
    rho1 = (q[..., 1, :] - q[..., 0, :]) / SQRT2
    rho2 = SQRT_2_OVER_3 * (q[..., 2, :] - 0.5 * (q[..., 0, :] + q[..., 1, :]))
    a2 = jnp.sum(rho1 * rho1, axis=-1)
    b2 = jnp.sum(rho2 * rho2, axis=-1)
    inertia = a2 + b2
    n1 = (a2 - b2) / inertia
    # zeta_1 * conj(zeta_2) = (rho1.x + i rho1.y)(rho2.x - i rho2.y)
    #                       = (rho1 . rho2) + i (rho1.y rho2.x - rho1.x rho2.y)
    dot = rho1[..., 0] * rho2[..., 0] + rho1[..., 1] * rho2[..., 1]
    cross = rho1[..., 1] * rho2[..., 0] - rho1[..., 0] * rho2[..., 1]
    n2 = 2.0 * dot / inertia
    n3 = 2.0 * cross / inertia
    return jnp.stack([n1, n2, n3], axis=-1)


def moment_of_inertia(q: Array) -> Array:
    """Total moment of inertia about the CM. m=1, so I = sum_i |q_i - Q|^2."""
    q = q - jnp.mean(q, axis=-2, keepdims=True)
    return jnp.sum(q * q, axis=(-2, -1))


def pairwise_distances(q: Array) -> Array:
    """Returns the three pairwise distances [r12, r13, r23] for each config.

    Last axis has length 3.
    """
    d12 = jnp.linalg.norm(q[..., 1, :] - q[..., 0, :], axis=-1)
    d13 = jnp.linalg.norm(q[..., 2, :] - q[..., 0, :], axis=-1)
    d23 = jnp.linalg.norm(q[..., 2, :] - q[..., 1, :], axis=-1)
    return jnp.stack([d12, d13, d23], axis=-1)
