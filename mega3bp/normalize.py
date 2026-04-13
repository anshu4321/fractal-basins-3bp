"""Canonical normalization for orbit initial conditions.

Converts to the standard free-fall form used by Hristov et al.:
  - Total energy E = -1
  - Total mass M = 3 (unit masses, already satisfied)
  - Center of mass at origin
  - Body 1 on positive x-axis at t=0
  - Bodies in canonical order (sorted by x-coordinate)

Under the scaling q -> s*q, p -> p/sqrt(s), t -> s^(3/2)*t:
  E -> E/s, so s = |E| gives E' = -1 for bound orbits.
  T* = |E|^(3/2) * T is the scale-invariant period.
"""
from __future__ import annotations

import numpy as np


def normalize_free_fall(q0: np.ndarray, T: float) -> dict:
    """Normalize a free-fall (p=0) initial condition to E=-1, canonical form.

    Args:
        q0: body positions, shape (3, 2), in COM frame
        T: orbital period

    Returns:
        dict with: q0_canonical (3,2), T_star (scale-invariant period),
                   scale_factor, rotation_angle, body_permutation, E_original
    """
    q0 = np.asarray(q0, dtype=np.float64)

    # 1. Center at COM (should already be, but enforce)
    q0 = q0 - q0.mean(axis=0, keepdims=True)

    # 2. Compute energy (free-fall: E = V(q) since p=0)
    d12 = np.linalg.norm(q0[1] - q0[0])
    d13 = np.linalg.norm(q0[2] - q0[0])
    d23 = np.linalg.norm(q0[2] - q0[1])
    E = -(1.0/d12 + 1.0/d13 + 1.0/d23)  # V < 0 always

    # 3. Scale so E' = -1:  q' = s*q, s = |E|
    s = abs(E)
    q_scaled = q0 * s
    T_star = (abs(E) ** 1.5) * T

    # 4. Rotate so body 1 is on positive x-axis
    angle = np.arctan2(q_scaled[0, 1], q_scaled[0, 0])
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    q_rotated = (R @ q_scaled.T).T  # (3, 2)

    # 5. Permute bodies so x-coordinates are in canonical order (ascending)
    perm = np.argsort(q_rotated[:, 0])
    q_canonical = q_rotated[perm]

    return {
        "q0_canonical": q_canonical,
        "T_star": float(T_star),
        "scale_factor": float(s),
        "rotation_angle": float(-angle),
        "body_permutation": perm.tolist(),
        "E_original": float(E),
    }


def reduced_ic_4d(q0_canonical: np.ndarray) -> np.ndarray:
    """Extract the 4D reduced initial condition vector for catalog matching.

    For free-fall equal-mass 3BP with COM=0 and M=3:
      q3 = -(q1 + q2) by COM constraint
    So the full IC is determined by (q1x, q1y, q2x, q2y).

    Returns: array of shape (4,).
    """
    q = np.asarray(q0_canonical)
    return np.array([q[0, 0], q[0, 1], q[1, 0], q[1, 1]])


def verify_normalization_consistency(q0: np.ndarray, T: float, n_trials: int = 3) -> float:
    """Run normalization n_trials times with different intermediate scalings.

    Returns max discrepancy between trials. Should be < 1e-10.
    """
    results = []
    for trial in range(n_trials):
        # Perturb by a random scale factor, then normalize
        rng = np.random.default_rng(trial)
        scale = rng.uniform(0.5, 2.0)
        q_perturbed = q0 * scale
        T_perturbed = T * (scale ** 1.5)
        result = normalize_free_fall(q_perturbed, T_perturbed)
        results.append(result["q0_canonical"])

    max_diff = 0.0
    for i in range(1, len(results)):
        diff = np.max(np.abs(results[i] - results[0]))
        max_diff = max(max_diff, diff)
    return max_diff
