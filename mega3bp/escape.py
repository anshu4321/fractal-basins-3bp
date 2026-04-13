"""Escape criterion for the planar 3-body problem.

At each integrator step we ask: has one body separated from the other two
far enough, and consistently enough, that it is "escaping"? We use a
simplified Standish-style test:

A body i is flagged as escaping iff:
  1. The OTHER pair (jk) is the current closest pair (r_jk <= r_ij and r_jk <= r_ik).
     This identifies bodies j, k as the "binary" and i as the candidate escapee.
  2. Both r_ij and r_ik exceed r_escape (body i is far from both).
  3. r_ij > binary_factor * r_jk AND r_ik > binary_factor * r_jk (body i is
     meaningfully farther than the binary is tight).
  4. The radial velocities for both separating pairs are outward:
     rdot_ij > 0 AND rdot_ik > 0 (body i is pulling away from both).

The criterion is latched: once any step sees a body escaping, that label is
frozen and escape_time records the first time of detection. The integrator
still runs to t_max for all trajectories (no early exit under jax.lax.scan).

A separate ambiguous flag latches if any pair distance drops below r_close
at any step. The trajectory is integrated through the close encounter but
labeled unreliable; the contract's 5% ambiguous threshold triggers a
stop-and-rethink (switch to KS regularization) if exceeded on the full sweep.

Label encoding (int8):
    -1 = ambiguous (close encounter occurred at some step)
     0 = bound (no escape triggered before t_max)
     1 = body 1 escapes
     2 = body 2 escapes
     3 = body 3 escapes
"""
from __future__ import annotations

import jax.numpy as jnp

Array = jnp.ndarray

BOUND = jnp.int8(0)
ESCAPE_1 = jnp.int8(1)
ESCAPE_2 = jnp.int8(2)
ESCAPE_3 = jnp.int8(3)
AMBIGUOUS = jnp.int8(-1)

LABEL_NAMES: dict[int, str] = {
    -1: "ambiguous",
    0: "bound",
    1: "body1_escape",
    2: "body2_escape",
    3: "body3_escape",
}

# Sensible defaults in natural units (I_0 = 1, typical triangle ~ O(1))
DEFAULT_R_ESCAPE = 5.0
DEFAULT_BINARY_FACTOR = 2.0
DEFAULT_R_CLOSE = 1e-3


def pair_distances(q: Array) -> tuple[Array, Array, Array]:
    """Returns (r12, r13, r23) — broadcasts over any leading batch dims."""
    r12 = jnp.linalg.norm(q[..., 1, :] - q[..., 0, :], axis=-1)
    r13 = jnp.linalg.norm(q[..., 2, :] - q[..., 0, :], axis=-1)
    r23 = jnp.linalg.norm(q[..., 2, :] - q[..., 1, :], axis=-1)
    return r12, r13, r23


def radial_velocities(q: Array, p: Array) -> tuple[Array, Array, Array]:
    """Returns (rdot12, rdot13, rdot23). Since m=1, velocities equal momenta.

    rdot_ij = d/dt |q_i - q_j| = (q_i - q_j) . (p_i - p_j) / |q_i - q_j|
    Positive = bodies i, j separating.
    """
    d12 = q[..., 1, :] - q[..., 0, :]
    d13 = q[..., 2, :] - q[..., 0, :]
    d23 = q[..., 2, :] - q[..., 1, :]
    r12 = jnp.linalg.norm(d12, axis=-1)
    r13 = jnp.linalg.norm(d13, axis=-1)
    r23 = jnp.linalg.norm(d23, axis=-1)
    v12 = p[..., 1, :] - p[..., 0, :]
    v13 = p[..., 2, :] - p[..., 0, :]
    v23 = p[..., 2, :] - p[..., 1, :]
    rdot12 = jnp.sum(d12 * v12, axis=-1) / r12
    rdot13 = jnp.sum(d13 * v13, axis=-1) / r13
    rdot23 = jnp.sum(d23 * v23, axis=-1) / r23
    return rdot12, rdot13, rdot23


def instant_escape_label(
    q: Array,
    p: Array,
    r_escape: float = DEFAULT_R_ESCAPE,
    binary_factor: float = DEFAULT_BINARY_FACTOR,
) -> Array:
    """Compute the escape label that THIS STATE ALONE would trigger.

    Returns an int8 scalar or batched array: 0 (not escaping), or 1/2/3 if
    body 1/2/3 is identified as escaping under the Standish-style criterion.
    """
    r12, r13, r23 = pair_distances(q)
    rdot12, rdot13, rdot23 = radial_velocities(q, p)

    # Body 1 escaping: bodies 2,3 form the tightest pair (binary), body 1 is far
    # and moving outward from both.
    esc1 = (
        (r23 <= r12)
        & (r23 <= r13)
        & (r12 > r_escape)
        & (r13 > r_escape)
        & (r12 > binary_factor * r23)
        & (r13 > binary_factor * r23)
        & (rdot12 > 0.0)
        & (rdot13 > 0.0)
    )
    # Body 2 escaping: bodies 1,3 form the binary.
    esc2 = (
        (r13 <= r12)
        & (r13 <= r23)
        & (r12 > r_escape)
        & (r23 > r_escape)
        & (r12 > binary_factor * r13)
        & (r23 > binary_factor * r13)
        & (rdot12 > 0.0)
        & (rdot23 > 0.0)
    )
    # Body 3 escaping: bodies 1,2 form the binary.
    esc3 = (
        (r12 <= r13)
        & (r12 <= r23)
        & (r13 > r_escape)
        & (r23 > r_escape)
        & (r13 > binary_factor * r12)
        & (r23 > binary_factor * r12)
        & (rdot13 > 0.0)
        & (rdot23 > 0.0)
    )

    label = jnp.where(
        esc1, ESCAPE_1,
        jnp.where(esc2, ESCAPE_2,
        jnp.where(esc3, ESCAPE_3, BOUND)),
    )
    return label


def two_body_energy(q: Array, p: Array, i: int, j: int, k: int) -> Array:
    """Two-body energy of body i relative to the (j,k) binary center of mass.

    E_2b = (1/2) mu |v_rel|^2 - (m_i * m_binary) / r
    For equal masses m=1: m_binary = 2, mu = 2/3, v_rel = v_i - v_cm_jk.
    Positive means unbound from the binary.
    """
    q_cm = (q[..., j, :] + q[..., k, :]) / 2.0
    p_cm = (p[..., j, :] + p[..., k, :]) / 2.0
    v_cm = p_cm  # m=1
    r_vec = q[..., i, :] - q_cm
    v_vec = p[..., i, :] - v_cm
    r = jnp.linalg.norm(r_vec, axis=-1)
    v2 = jnp.sum(v_vec * v_vec, axis=-1)
    mu = 2.0 / 3.0
    m_total = 3.0  # m_i * (m_j + m_k) for gravitational parameter
    return 0.5 * mu * v2 - m_total / r


def instant_escape_label_strict(
    q: Array,
    p: Array,
    r_escape: float = DEFAULT_R_ESCAPE,
    binary_factor: float = DEFAULT_BINARY_FACTOR,
) -> Array:
    """Strict escape criterion: Standish geometry + positive two-body energy.

    Body i is escaping iff:
      1. Standish geometric conditions (binary identified, body far and receding).
      2. Two-body energy of body i w.r.t. the binary CM is positive (unbound).
    """
    base_label = instant_escape_label(q, p, r_escape, binary_factor)

    e2b_1 = two_body_energy(q, p, 0, 1, 2)
    e2b_2 = two_body_energy(q, p, 1, 0, 2)
    e2b_3 = two_body_energy(q, p, 2, 0, 1)

    strict_1 = (base_label == ESCAPE_1) & (e2b_1 > 0.0)
    strict_2 = (base_label == ESCAPE_2) & (e2b_2 > 0.0)
    strict_3 = (base_label == ESCAPE_3) & (e2b_3 > 0.0)

    return jnp.where(
        strict_1, ESCAPE_1,
        jnp.where(strict_2, ESCAPE_2,
        jnp.where(strict_3, ESCAPE_3, BOUND)),
    )


def min_pair_distance(q: Array) -> Array:
    r12, r13, r23 = pair_distances(q)
    return jnp.minimum(jnp.minimum(r12, r13), r23)
