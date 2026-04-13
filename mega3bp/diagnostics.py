"""Diagnostics for periodic orbit verification.

Provides conservation monitoring (energy, angular momentum), minimum
pairwise distance tracking, and multi-period stability measurement.
All functions are JAX-compatible and work on single orbits.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .dynamics import total_energy, angular_momentum
from .integrators import yoshida6_step
from .shape_sphere import pairwise_distances

Array = jnp.ndarray


@partial(jax.jit, static_argnames=("n_steps", "n_samples"))
def conservation_check(
    q0: Array, p0: Array, T: float, n_steps: int = 10_000, n_samples: int = 1000,
) -> dict:
    """Integrate for time T, sampling E(t) and L(t) at n_samples intermediate times.

    Returns dict with:
        dE_rel_max: max |Delta E / E| over the integration
        dL_max: max |Delta L| over the integration
        E_trace: energy at sampled times, shape (n_samples,)
        L_trace: angular momentum at sampled times, shape (n_samples,)
    """
    h = T / n_steps
    E0 = total_energy(q0, p0)
    L0 = angular_momentum(q0, p0)
    sample_interval = max(1, n_steps // n_samples)

    def step_fn(carry, step_idx):
        q, p = carry
        q, p = yoshida6_step(q, p, jnp.float64(h))
        E = total_energy(q, p)
        L = angular_momentum(q, p)
        is_sample = (step_idx % sample_interval == 0)
        return (q, p), (E, L, is_sample)

    (qf, pf), (Es, Ls, masks) = jax.lax.scan(
        step_fn, (q0, p0), jnp.arange(n_steps))

    dE_rel = jnp.abs((Es - E0) / E0)
    dL = jnp.abs(Ls - L0)

    return {
        "dE_rel_max": jnp.max(dE_rel),
        "dL_max": jnp.max(dL),
        "E0": E0,
        "L0": L0,
        "E_trace": Es[::sample_interval][:n_samples],
        "L_trace": Ls[::sample_interval][:n_samples],
    }


@partial(jax.jit, static_argnames=("n_steps",))
def min_distance_check(
    q0: Array, p0: Array, T: float, n_steps: int = 10_000,
) -> dict:
    """Track per-pair minimum distances over one period.

    Returns dict with:
        r_min: global minimum of min-over-pairs distance
        r_min_per_pair: (3,) minimum distance per pair [r12, r13, r23]
        r_init_max: max pairwise distance at t=0
        r_min_ratio: r_min / r_init_max
    """
    h = T / n_steps
    r_init = pairwise_distances(q0)
    r_init_max = jnp.max(r_init)

    def step_fn(carry, _):
        q, p, r_min_pairs = carry
        q, p = yoshida6_step(q, p, jnp.float64(h))
        r_pairs = pairwise_distances(q)  # (3,)
        r_min_pairs = jnp.minimum(r_min_pairs, r_pairs)
        return (q, p, r_min_pairs), None

    init_mins = jnp.full(3, jnp.inf)
    (_, _, r_min_pairs), _ = jax.lax.scan(
        step_fn, (q0, p0, init_mins), None, length=n_steps)

    return {
        "r_min": jnp.min(r_min_pairs),
        "r_min_per_pair": r_min_pairs,
        "r_init_max": r_init_max,
        "r_min_ratio": jnp.min(r_min_pairs) / r_init_max,
    }


@partial(jax.jit, static_argnames=("n_steps", "n_periods"))
def multi_period_stability(
    q0: Array, p0: Array, T: float,
    n_steps: int = 10_000, n_periods: int = 10,
) -> dict:
    """Integrate for n_periods * T, measuring state residual at each period boundary.

    Returns dict with:
        residuals: (n_periods,) array of ||state(k*T) - state(0)|| for k=1..n_periods
        residual_10: last residual value
        diverged: bool, True if last residual > 1e-4
    """
    h = T / n_steps
    state0 = jnp.concatenate([q0.ravel(), p0.ravel()])

    def one_period(carry, _):
        q, p = carry
        def step_fn(qp, _):
            q, p = qp
            q, p = yoshida6_step(q, p, jnp.float64(h))
            return (q, p), None
        (q, p), _ = jax.lax.scan(step_fn, (q, p), None, length=n_steps)
        state_k = jnp.concatenate([q.ravel(), p.ravel()])
        residual = jnp.linalg.norm(state_k - state0)
        return (q, p), residual

    (_, _), residuals = jax.lax.scan(one_period, (q0, p0), None, length=n_periods)

    return {
        "residuals": residuals,
        "residual_10": residuals[-1],
        "diverged": residuals[-1] > 1e-4,
    }
