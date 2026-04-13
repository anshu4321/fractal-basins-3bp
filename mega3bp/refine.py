"""Full-phase-space Gauss-Newton refiner for periodic orbit candidates.

For free-fall orbits on the shape sphere (p=0 at t=0), the initial condition
is determined by 2 parameters (theta, phi) on S^2 plus the period T.
The shooting residual F(theta, phi, T) = state(T) - state(0) lives in R^12.

Gauss-Newton minimizes ||F||^2 via the update:
    dp = -(J^T J)^{-1} J^T F
where J = dF/d(theta, phi, T) is 12x3, computed by JAX forward-mode autodiff.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .shape_sphere import shape_to_config, config_to_shape
from .integrators import yoshida6_step
from .dynamics import total_energy

Array = jnp.ndarray

# Default integration resolution for the refiner.
# h = T / REFINE_N_STEPS, so accuracy scales with T but step count is fixed.
REFINE_N_STEPS = 10_000


def _angles_to_state(theta: Array, phi: Array) -> tuple[Array, Array]:
    """Convert shape-sphere angles to (q0, p0) for free-fall launch."""
    n = jnp.array([jnp.sin(theta) * jnp.cos(phi),
                    jnp.sin(theta) * jnp.sin(phi),
                    jnp.cos(theta)])
    q0 = shape_to_config(n, inertia=1.0)
    p0 = jnp.zeros_like(q0)
    return q0, p0


@partial(jax.jit, static_argnames=("n_steps",))
def _integrate_for_T(theta: Array, phi: Array, T: Array,
                     n_steps: int = REFINE_N_STEPS) -> tuple[Array, Array, Array, Array]:
    """Integrate free-fall orbit from (theta, phi) for time T.

    Uses n_steps Yoshida-6 steps with h = T / n_steps.
    Returns (q0, p0, qf, pf).
    """
    q0, p0 = _angles_to_state(theta, phi)
    h = T / n_steps

    def step_fn(carry, _):
        q, p = carry
        q, p = yoshida6_step(q, p, h)
        return (q, p), None

    (qf, pf), _ = jax.lax.scan(step_fn, (q0, p0), None, length=n_steps)
    return q0, p0, qf, pf


@partial(jax.jit, static_argnames=("n_steps",))
def shooting_residual(params: Array, n_steps: int = REFINE_N_STEPS) -> Array:
    """F(theta, phi, T) = state(T) - state(0), flattened to R^12.

    params: array of shape (3,) containing [theta, phi, T].
    Returns: array of shape (12,).
    """
    theta, phi, T = params[0], params[1], params[2]
    q0, p0, qf, pf = _integrate_for_T(theta, phi, T, n_steps)
    dq = (qf - q0).ravel()   # (6,)
    dp = (pf - p0).ravel()   # (6,)
    return jnp.concatenate([dq, dp])


@partial(jax.jit, static_argnames=("n_steps",))
def _lm_step(params: Array, lam: Array, n_steps: int = REFINE_N_STEPS) -> tuple[Array, Array, Array]:
    """One Levenberg-Marquardt step. Returns (candidate_params, residual_norm, step_norm)."""
    def F(p):
        return shooting_residual(p, n_steps)

    residual = F(params)
    J = jax.jacfwd(F)(params)          # (12, 3)
    JtJ = J.T @ J                      # (3, 3)
    JtF = J.T @ residual               # (3,)
    # Adaptive LM damping: lam * diag(J^T J) for scale-invariance
    damping = lam * jnp.diag(jnp.diag(JtJ) + 1e-12)
    dp = jnp.linalg.solve(JtJ + damping, JtF)
    return params - dp, jnp.linalg.norm(residual), jnp.linalg.norm(dp)


def gauss_newton_refine(
    theta0: float,
    phi0: float,
    T0: float,
    n_steps: int = REFINE_N_STEPS,
    max_iter: int = 50,
    tol: float = 1e-10,
    verbose: bool = True,
) -> dict:
    """Refine a periodic orbit candidate via Levenberg-Marquardt.

    Adaptive damping: starts at lam=1e-2, increases by 10x on divergence,
    decreases by 3x on improvement. Only accepts steps that reduce residual.

    Args:
        theta0, phi0: initial shape-sphere angles
        T0: approximate period
        n_steps: integration steps (h = T / n_steps)
        max_iter: maximum iterations
        tol: convergence tolerance on ||F||
        verbose: print iteration progress

    Returns:
        dict with keys: theta, phi, T, residual_norm, n_iter, converged,
                        q0, p0, qf, pf, closure_12d
    """
    params = jnp.array([theta0, phi0, T0], dtype=jnp.float64)
    lam = jnp.float64(1e-2)  # initial damping

    # Compute initial residual
    res0 = shooting_residual(params, n_steps)
    best_residual = float(jnp.linalg.norm(res0))
    best_params = params
    current_residual = best_residual

    if verbose:
        print(f"  LM iter   0: ||F|| = {best_residual:.2e}, lam = {float(lam):.1e}")

    stall_count = 0
    for i in range(max_iter):
        candidate, _, step_norm = _lm_step(params, lam, n_steps)

        # Enforce constraints on candidate
        theta_val = float(candidate[0])
        theta_val = max(0.01, min(float(jnp.pi) - 0.01, theta_val))
        candidate = candidate.at[0].set(theta_val)
        candidate = candidate.at[2].set(jnp.maximum(candidate[2], 0.01))

        # Evaluate candidate
        new_res = shooting_residual(candidate, n_steps)
        new_residual = float(jnp.linalg.norm(new_res))

        if new_residual < current_residual:
            # Accept step, reduce damping
            rel_improvement = (current_residual - new_residual) / max(current_residual, 1e-30)
            params = candidate
            current_residual = new_residual
            lam = jnp.maximum(lam / 3.0, jnp.float64(1e-12))

            if new_residual < best_residual:
                best_residual = new_residual
                best_params = params

            # Only reset stall counter on meaningful improvement
            if rel_improvement > 1e-4:
                stall_count = 0
            else:
                stall_count += 1
        else:
            # Reject step, increase damping
            lam = jnp.minimum(lam * 10.0, jnp.float64(1e6))
            stall_count += 1

        if verbose and (i + 1) % 5 == 0:
            print(f"  LM iter {i+1:3d}: ||F|| = {current_residual:.2e}, "
                  f"lam = {float(lam):.1e}, best = {best_residual:.2e}")

        if current_residual < tol:
            if verbose:
                print(f"  Converged at iter {i+1}: ||F|| = {current_residual:.2e}")
            break

        # Give up if stalled for too long
        if stall_count > 10:
            if verbose:
                print(f"  Stalled at iter {i+1}: ||F|| = {current_residual:.2e} "
                      f"(best = {best_residual:.2e})")
            break

        # Early exit: if residual is still large after 15 iterations, hopeless
        if (i + 1) >= 15 and best_residual > 1e-2:
            if verbose:
                print(f"  Hopeless at iter {i+1}: best ||F|| = {best_residual:.2e} "
                      f">> 1e-8 threshold, giving up")
            break

    # Extract final state for diagnostics
    theta, phi, T = float(best_params[0]), float(best_params[1]), float(best_params[2])
    q0, p0, qf, pf = _integrate_for_T(
        jnp.float64(theta), jnp.float64(phi), jnp.float64(T), n_steps)

    residual_vec = jnp.concatenate([(qf - q0).ravel(), (pf - p0).ravel()])
    closure_norm = float(jnp.linalg.norm(residual_vec))

    return {
        "theta": theta,
        "phi": phi,
        "T": T,
        "residual_norm": closure_norm,
        "n_iter": min(i + 1, max_iter),
        "converged": closure_norm < tol,
        "q0": q0,
        "p0": p0,
        "qf": qf,
        "pf": pf,
        "closure_12d": closure_norm,
    }


def refine_from_shape_point(
    n0,
    T_approx: float,
    n_steps: int = REFINE_N_STEPS,
    max_iter: int = 50,
    tol: float = 1e-10,
    verbose: bool = True,
) -> dict:
    """Convenience wrapper: refine from a shape-sphere 3-vector + approximate period.

    Converts n0 = (n1, n2, n3) to (theta, phi) and calls gauss_newton_refine.
    """
    n0 = np.asarray(n0, dtype=np.float64)
    theta0 = float(np.arccos(np.clip(n0[2], -1, 1)))
    phi0 = float(np.arctan2(n0[1], n0[0]))
    return gauss_newton_refine(theta0, phi0, T_approx, n_steps, max_iter, tol, verbose)
