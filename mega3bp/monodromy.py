"""Monodromy matrix computation via JAX automatic differentiation.

The monodromy matrix M = d(Phi_T)/d(x0) is the Jacobian of the full-period
flow map at the periodic orbit's initial condition. For a 12D phase space
(3 bodies x 2D positions + momenta), M is 12x12.

Hamiltonian structure guarantees:
  - det(M) = 1 (symplecticity)
  - Eigenvalues come in reciprocal pairs (lambda, 1/lambda)
  - Four trivial eigenvalues at +1 (time translation + scale symmetry in free-fall)
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .integrators import yoshida6_step

Array = jnp.ndarray


@partial(jax.jit, static_argnames=("n_steps",))
def _flow_map(x0_flat: Array, h: Array, n_steps: int) -> Array:
    """Full-period flow: x0_flat (12,) -> xT_flat (12,).

    x0_flat = [q1x, q1y, q2x, q2y, q3x, q3y, p1x, p1y, p2x, p2y, p3x, p3y]
    """
    q0 = x0_flat[:6].reshape(3, 2)
    p0 = x0_flat[6:].reshape(3, 2)

    def step_fn(carry, _):
        q, p = carry
        q, p = yoshida6_step(q, p, h)
        return (q, p), None

    (qf, pf), _ = jax.lax.scan(step_fn, (q0, p0), None, length=n_steps)
    return jnp.concatenate([qf.ravel(), pf.ravel()])


@partial(jax.jit, static_argnames=("n_steps",))
def monodromy_matrix(
    q0: Array, p0: Array, T: float, n_steps: int = 10_000,
) -> Array:
    """Compute the 12x12 monodromy matrix via forward-mode autodiff.

    Args:
        q0: initial positions, shape (3, 2)
        p0: initial momenta, shape (3, 2)
        T: orbital period
        n_steps: integration steps

    Returns:
        M: monodromy matrix, shape (12, 12)
    """
    h = jnp.float64(T / n_steps)
    x0 = jnp.concatenate([q0.ravel(), p0.ravel()])
    M = jax.jacfwd(lambda x: _flow_map(x, h, n_steps))(x0)
    return M


def analyze_monodromy(M: np.ndarray, tol: float = 1e-4) -> dict:
    """Analyze eigenvalue spectrum of the monodromy matrix.

    Returns dict with:
        eigenvalues: full spectrum (12 complex values)
        trivial_eigenvalues: those within tol of +1
        nontrivial_eigenvalues: the rest
        n_trivial: count of trivial eigenvalues
        stability_class: 'stable' / 'unstable' / 'parabolic'
        stability_index: sum of |lambda + 1/lambda| / 2 over nontrivial pairs
        reciprocal_check: max |lambda_i * lambda_j - 1| over identified pairs
        det_M: determinant (should be 1)
    """
    M = np.asarray(M)
    eigvals = np.linalg.eigvals(M)

    # Sort by distance from +1 to identify trivial eigenvalues
    dist_from_1 = np.abs(eigvals - 1.0)
    order = np.argsort(dist_from_1)
    eigvals_sorted = eigvals[order]

    # Identify trivial eigenvalues (should be 4 at +1)
    trivial_mask = dist_from_1 < tol
    n_trivial = int(np.sum(trivial_mask))
    trivial = eigvals[trivial_mask]
    nontrivial = eigvals[~trivial_mask]

    # Stability classification based on nontrivial eigenvalues
    if len(nontrivial) == 0:
        stability_class = "stable"
        stability_index = 0.0
    else:
        max_abs = np.max(np.abs(nontrivial))
        if max_abs < 1.0 + tol:
            stability_class = "stable"
        elif np.any(np.abs(np.abs(nontrivial) - 1.0) < tol):
            stability_class = "parabolic"
        else:
            stability_class = "unstable"

        # Stability index: for each reciprocal pair, |lambda + 1/lambda| / 2
        stability_index = float(np.sum(np.abs(nontrivial + 1.0 / nontrivial)) / (2.0 * len(nontrivial)))

    # Reciprocal pair check
    reciprocal_errors = []
    used = set()
    for i, ev in enumerate(nontrivial):
        if i in used:
            continue
        target = 1.0 / ev
        dists = np.abs(nontrivial - target)
        dists[list(used)] = np.inf
        dists[i] = np.inf
        j = np.argmin(dists)
        if dists[j] < 0.1:
            reciprocal_errors.append(float(np.abs(ev * nontrivial[j] - 1.0)))
            used.add(j)

    return {
        "eigenvalues": eigvals.tolist(),
        "trivial_eigenvalues": trivial.tolist(),
        "nontrivial_eigenvalues": nontrivial.tolist(),
        "n_trivial": n_trivial,
        "stability_class": stability_class,
        "stability_index": stability_index,
        "reciprocal_check": float(max(reciprocal_errors)) if reciprocal_errors else 0.0,
        "det_M": float(np.real(np.linalg.det(M))),
    }
