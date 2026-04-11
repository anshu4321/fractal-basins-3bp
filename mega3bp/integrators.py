"""Symplectic integrators for H = T(p) + V(q) separable Hamiltonians.

Stormer-Verlet is the 2nd-order symmetric base method. Yoshida compositions
promote it to higher even order via carefully chosen sub-step coefficients.

Reference: Yoshida, "Construction of higher order symplectic integrators,"
Physics Letters A 150 (1990) 262-268. We use Solution A for the 6th-order
composition (7 stages per step).
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .dynamics import force, total_energy

Array = jnp.ndarray


# Yoshida 1990 Solution A, 6th order, symmetric composition of Verlet.
# Seven base-method applications per composed step with weights
# [w3, w2, w1, w0, w1, w2, w3].
_W1 = -1.17767998417887
_W2 = 0.235573213359357
_W3 = 0.784513610477560
_W0 = 1.0 - 2.0 * (_W1 + _W2 + _W3)

YOSHIDA6_WEIGHTS = jnp.array(
    [_W3, _W2, _W1, _W0, _W1, _W2, _W3],
    dtype=jnp.float64,
)


def verlet_substep(q: Array, p: Array, dt: Array) -> tuple[Array, Array]:
    """Drift-Kick-Drift Stormer-Verlet. dt may be a scalar jax Array."""
    q_half = q + 0.5 * dt * p
    p_new = p + dt * force(q_half)
    q_new = q_half + 0.5 * dt * p_new
    return q_new, p_new


def yoshida6_step(q: Array, p: Array, h: Array) -> tuple[Array, Array]:
    """One 6th-order composed step of size h (= 7 Verlet sub-steps)."""
    sub_dts = YOSHIDA6_WEIGHTS * h

    def inner(state: tuple[Array, Array], dt: Array) -> tuple[tuple[Array, Array], None]:
        q, p = state
        q, p = verlet_substep(q, p, dt)
        return (q, p), None

    (q_out, p_out), _ = jax.lax.scan(inner, (q, p), sub_dts)
    return q_out, p_out


@partial(jax.jit, static_argnames=("n_steps",))
def yoshida6_integrate(
    q0: Array,
    p0: Array,
    h: float,
    n_steps: int,
) -> tuple[Array, Array, Array]:
    """Integrate for n_steps of size h. Records (q, p, energy) every step.

    Returns:
        q_trace: shape (n_steps+1, 3, 2) including the initial state
        p_trace: shape (n_steps+1, 3, 2)
        energy: shape (n_steps+1,)
    """
    h_arr = jnp.asarray(h, dtype=jnp.float64)

    def step(carry: tuple[Array, Array], _: None) -> tuple[tuple[Array, Array], tuple[Array, Array, Array]]:
        q, p = carry
        q, p = yoshida6_step(q, p, h_arr)
        E = total_energy(q, p)
        return (q, p), (q, p, E)

    E0 = total_energy(q0, p0)
    (q_final, p_final), (q_trace, p_trace, energies) = jax.lax.scan(
        step, (q0, p0), None, length=n_steps
    )

    # Prepend initial state so trace has length n_steps + 1
    q_trace = jnp.concatenate([q0[None], q_trace], axis=0)
    p_trace = jnp.concatenate([p0[None], p_trace], axis=0)
    energies = jnp.concatenate([E0[None], energies], axis=0)
    return q_trace, p_trace, energies


@partial(jax.jit, static_argnames=("n_steps",))
def yoshida6_integrate_energy_only(
    q0: Array,
    p0: Array,
    h: float,
    n_steps: int,
) -> tuple[Array, Array, Array]:
    """Like yoshida6_integrate but records only the energy trace (cheap memory)."""
    h_arr = jnp.asarray(h, dtype=jnp.float64)

    def step(carry: tuple[Array, Array], _: None) -> tuple[tuple[Array, Array], Array]:
        q, p = carry
        q, p = yoshida6_step(q, p, h_arr)
        E = total_energy(q, p)
        return (q, p), E

    E0 = total_energy(q0, p0)
    (q_final, p_final), energies = jax.lax.scan(step, (q0, p0), None, length=n_steps)
    energies = jnp.concatenate([E0[None], energies], axis=0)
    return q_final, p_final, energies
