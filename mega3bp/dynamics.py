"""Hamiltonian and forces for the planar equal-mass three-body problem.

H(q, p) = (1/2) sum_i |p_i|^2 - sum_{i<j} 1 / |q_i - q_j|

State layout: q shape (3, 2), p shape (3, 2). Batched shapes (..., 3, 2) also work.
"""
from __future__ import annotations

import jax.numpy as jnp

Array = jnp.ndarray


def kinetic_energy(p: Array) -> Array:
    return 0.5 * jnp.sum(p * p, axis=(-2, -1))


def potential_energy(q: Array) -> Array:
    d12 = q[..., 1, :] - q[..., 0, :]
    d13 = q[..., 2, :] - q[..., 0, :]
    d23 = q[..., 2, :] - q[..., 1, :]
    r12 = jnp.sqrt(jnp.sum(d12 * d12, axis=-1))
    r13 = jnp.sqrt(jnp.sum(d13 * d13, axis=-1))
    r23 = jnp.sqrt(jnp.sum(d23 * d23, axis=-1))
    return -(1.0 / r12 + 1.0 / r13 + 1.0 / r23)


def total_energy(q: Array, p: Array) -> Array:
    return kinetic_energy(p) + potential_energy(q)


def force(q: Array) -> Array:
    """F_i = -dV/dq_i on each body. Returns shape matching q.

    F_i = sum_{j != i} (q_j - q_i) / |q_j - q_i|^3
    """
    q1 = q[..., 0, :]
    q2 = q[..., 1, :]
    q3 = q[..., 2, :]
    d12 = q2 - q1
    d13 = q3 - q1
    d23 = q3 - q2
    inv12_3 = jnp.power(jnp.sum(d12 * d12, axis=-1), -1.5)[..., None]
    inv13_3 = jnp.power(jnp.sum(d13 * d13, axis=-1), -1.5)[..., None]
    inv23_3 = jnp.power(jnp.sum(d23 * d23, axis=-1), -1.5)[..., None]
    F1 = d12 * inv12_3 + d13 * inv13_3
    F2 = -d12 * inv12_3 + d23 * inv23_3
    F3 = -d13 * inv13_3 - d23 * inv23_3
    return jnp.stack([F1, F2, F3], axis=-2)


def angular_momentum(q: Array, p: Array) -> Array:
    """Total angular momentum L_z = sum_i (q_ix * p_iy - q_iy * p_ix)."""
    return jnp.sum(q[..., 0] * p[..., 1] - q[..., 1] * p[..., 0], axis=-1)
