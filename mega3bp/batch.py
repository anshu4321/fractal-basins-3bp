"""Vmapped batch integration for the planar 3BP with escape labeling.

All mega3bp.dynamics and mega3bp.integrators functions broadcast naturally
over leading batch dimensions because they use (..., 3, 2) indexing. So the
batch integrator does not need explicit jax.vmap — we just pass batched
(q, p) tensors of shape (B, 3, 2) into the existing Yoshida step.

The scan carries per-trajectory latched state:
  - label (B,)         int8, 0 while not escaping, 1/2/3 once first trigger fires
  - escape_time (B,)   float64, inf while not escaping, first trigger time once latched
  - ambiguous (B,)     bool, True if r_min ever dropped below r_close
  - r_min_ever (B,)    float64, min pairwise distance ever seen (diagnostic)
  - max_sep (B,)       float64, max pairwise distance ever seen (diagnostic)

After the scan completes, label is overwritten to -1 on any ambiguous
trajectory, matching the contract's encoding.

Also supports optional trajectory recording via `record_every` for
visualization: every N-th step is stashed in the scan output. For 1M
trajectories this should stay zero (record_every=0) to avoid memory blowup.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .dynamics import total_energy
from .escape import (
    AMBIGUOUS,
    BOUND,
    DEFAULT_BINARY_FACTOR,
    DEFAULT_R_CLOSE,
    DEFAULT_R_ESCAPE,
    instant_escape_label,
    instant_escape_label_strict,
    min_pair_distance,
    pair_distances,
)
from .integrators import yoshida6_step

Array = jnp.ndarray


@partial(
    jax.jit,
    static_argnames=("n_steps", "record_every"),
)
def integrate_batch(
    q0: Array,
    p0: Array,
    h: float,
    n_steps: int,
    r_escape: float = DEFAULT_R_ESCAPE,
    binary_factor: float = DEFAULT_BINARY_FACTOR,
    r_close: float = DEFAULT_R_CLOSE,
    record_every: int = 0,
) -> dict[str, Array]:
    """Integrate a batch of trajectories with escape labeling.

    Args:
        q0, p0: shape (B, 3, 2)
        h: integrator step size
        n_steps: number of Yoshida steps
        r_escape: minimum pair distance for "far enough" escape check
        binary_factor: factor by which the escapee must be farther than the binary pair
        r_close: pair distance below which a trajectory is flagged ambiguous
        record_every: if > 0, record (q, p) every N steps into the return dict

    Returns dict with:
        q_final, p_final:  (B, 3, 2) final state
        label:             (B,)    int8, -1/0/1/2/3 per the encoding in mega3bp.escape
        escape_time:       (B,)    float64, first detection time (inf if bound)
        ambiguous:         (B,)    bool, True if close encounter ever occurred
        r_min_ever:        (B,)    float64, minimum pair distance over whole integration
        max_sep:           (B,)    float64, maximum pair distance over whole integration
        q_trace (optional): (n_recorded, B, 3, 2) if record_every > 0, else absent
        p_trace (optional): (n_recorded, B, 3, 2) if record_every > 0, else absent
    """
    B = q0.shape[0]
    h_arr = jnp.asarray(h, dtype=jnp.float64)

    init_label = jnp.zeros(B, dtype=jnp.int8)
    init_time = jnp.full(B, jnp.inf, dtype=jnp.float64)
    init_amb = jnp.zeros(B, dtype=jnp.bool_)
    # Initial diagnostics at t=0
    init_r_min = min_pair_distance(q0)
    r12_0, r13_0, r23_0 = pair_distances(q0)
    init_max_sep = jnp.maximum(jnp.maximum(r12_0, r13_0), r23_0)

    record = record_every > 0

    def step_body(carry, i):
        q, p, label, etime, amb, rmin_ever, maxsep_ever = carry
        q_new, p_new = yoshida6_step(q, p, h_arr)

        # Diagnostics
        r12, r13, r23 = pair_distances(q_new)
        r_min_step = jnp.minimum(jnp.minimum(r12, r13), r23)
        r_max_step = jnp.maximum(jnp.maximum(r12, r13), r23)
        rmin_next = jnp.minimum(rmin_ever, r_min_step)
        maxsep_next = jnp.maximum(maxsep_ever, r_max_step)

        # Ambiguous latch
        amb_next = amb | (r_min_step < r_close)

        # Escape latch — only update if label is still 0 (BOUND)
        new_label_step = instant_escape_label(q_new, p_new, r_escape, binary_factor)
        first_hit = (label == BOUND) & (new_label_step != BOUND)
        label_next = jnp.where(first_hit, new_label_step, label)
        t_now = (i.astype(jnp.float64) + 1.0) * h_arr
        etime_next = jnp.where(first_hit, t_now, etime)

        carry_next = (q_new, p_new, label_next, etime_next, amb_next, rmin_next, maxsep_next)
        if record:
            # Conditionally stash state (for recording we use a host-selected stride mask
            # applied after the scan; here we just stash every step's q, p and downsample).
            return carry_next, (q_new, p_new)
        return carry_next, None

    init_carry = (q0, p0, init_label, init_time, init_amb, init_r_min, init_max_sep)
    indices = jnp.arange(n_steps)

    if record:
        # Recording is wasteful for large B: the scan output holds every step
        # in DRAM, and we slice at the end. Intended only for small batches
        # (trajectory overlays, animations). For 1M ICs use record_every=0.
        final, (q_trace_all, p_trace_all) = jax.lax.scan(step_body, init_carry, indices)
        idx = jnp.arange(0, n_steps, record_every)
        q_trace = q_trace_all[idx]
        p_trace = p_trace_all[idx]
    else:
        final, _ = jax.lax.scan(step_body, init_carry, indices)
        q_trace = None
        p_trace = None

    q_f, p_f, label, etime, amb, rmin_ever, maxsep_ever = final

    # Ambiguous trumps escape label
    label_final = jnp.where(amb, AMBIGUOUS, label)

    out = {
        "q_final": q_f,
        "p_final": p_f,
        "label": label_final,
        "escape_time": etime,
        "ambiguous": amb,
        "r_min_ever": rmin_ever,
        "max_sep": maxsep_ever,
    }
    if record:
        out["q_trace"] = q_trace
        out["p_trace"] = p_trace
    return out


@partial(
    jax.jit,
    static_argnames=("n_steps", "strict_escape"),
)
def integrate_batch_diag(
    q0: Array,
    p0: Array,
    h: float,
    n_steps: int,
    r_escape: float = DEFAULT_R_ESCAPE,
    binary_factor: float = DEFAULT_BINARY_FACTOR,
    r_close: float = DEFAULT_R_CLOSE,
    strict_escape: bool = True,
) -> dict[str, Array]:
    """Batch integrator with per-trajectory energy-error tracking.

    Like integrate_batch but additionally tracks max |ΔE/E| over each
    trajectory, and optionally uses the strict escape criterion (Standish
    geometry + positive two-body energy).

    Returns dict with all fields from integrate_batch plus:
        max_dE_rel:  (B,) float64, max |ΔE/E| over the trajectory
        E_initial:   (B,) float64, initial total energy
    """
    B = q0.shape[0]
    h_arr = jnp.asarray(h, dtype=jnp.float64)

    E0 = total_energy(q0, p0)

    init_label = jnp.zeros(B, dtype=jnp.int8)
    init_time = jnp.full(B, jnp.inf, dtype=jnp.float64)
    init_amb = jnp.zeros(B, dtype=jnp.bool_)
    init_r_min = min_pair_distance(q0)
    r12_0, r13_0, r23_0 = pair_distances(q0)
    init_max_sep = jnp.maximum(jnp.maximum(r12_0, r13_0), r23_0)
    init_max_dE = jnp.zeros(B, dtype=jnp.float64)

    escape_fn = instant_escape_label_strict if strict_escape else instant_escape_label

    def step_body(carry, i):
        q, p, label, etime, amb, rmin_ever, maxsep_ever, max_dE = carry
        q_new, p_new = yoshida6_step(q, p, h_arr)

        r12, r13, r23 = pair_distances(q_new)
        r_min_step = jnp.minimum(jnp.minimum(r12, r13), r23)
        r_max_step = jnp.maximum(jnp.maximum(r12, r13), r23)
        rmin_next = jnp.minimum(rmin_ever, r_min_step)
        maxsep_next = jnp.maximum(maxsep_ever, r_max_step)

        amb_next = amb | (r_min_step < r_close)

        E_now = total_energy(q_new, p_new)
        dE_rel = jnp.abs((E_now - E0) / jnp.where(jnp.abs(E0) > 1e-30, E0, 1.0))
        max_dE_next = jnp.maximum(max_dE, dE_rel)

        new_label_step = escape_fn(q_new, p_new, r_escape, binary_factor)
        first_hit = (label == BOUND) & (new_label_step != BOUND)
        label_next = jnp.where(first_hit, new_label_step, label)
        t_now = (i.astype(jnp.float64) + 1.0) * h_arr
        etime_next = jnp.where(first_hit, t_now, etime)

        carry_next = (q_new, p_new, label_next, etime_next, amb_next,
                      rmin_next, maxsep_next, max_dE_next)
        return carry_next, None

    init_carry = (q0, p0, init_label, init_time, init_amb,
                  init_r_min, init_max_sep, init_max_dE)
    final, _ = jax.lax.scan(step_body, init_carry, jnp.arange(n_steps))

    q_f, p_f, label, etime, amb, rmin_ever, maxsep_ever, max_dE = final
    label_final = jnp.where(amb, AMBIGUOUS, label)

    return {
        "q_final": q_f,
        "p_final": p_f,
        "label": label_final,
        "escape_time": etime,
        "ambiguous": amb,
        "r_min_ever": rmin_ever,
        "max_sep": maxsep_ever,
        "max_dE_rel": max_dE,
        "E_initial": E0,
    }
