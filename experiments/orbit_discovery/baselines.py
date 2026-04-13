"""Four baseline sampling strategies for orbit discovery comparison.

All methods select N candidate initial conditions on the shape sphere.
The verification pipeline (Group 1 tests) is applied identically to all.

Methods:
  1. Uniform random: sample uniformly on S^2
  2. Physics-informed grid: collinear + isosceles grid (Li-Liao style)
  3. Label disagreement (no classifier): k-NN on ground-truth basin labels
  4. Classifier entropy: prediction entropy from trained BasinMLP
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def uniform_random_candidates(n_candidates: int, seed: int = 0) -> np.ndarray:
    """Sample n_candidates points uniformly on S^2.

    Returns: (n_candidates, 3) array of shape-sphere points.
    """
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((n_candidates, 3))
    return g / np.linalg.norm(g, axis=1, keepdims=True)


def physics_grid_candidates(n_candidates: int) -> np.ndarray:
    """Li-Liao/Hristov-style physics-informed grid.

    Concentrates samples in two known productive regions:
    1. Collinear configurations (equator of shape sphere, n3=0)
    2. Isosceles configurations (great circles through poles and collision points)

    Grid spacing chosen so total count ~ n_candidates.
    """
    candidates = []

    # Collinear: n3 = 0, vary angle phi along equator
    n_collinear = n_candidates // 2
    phis = np.linspace(0, 2 * np.pi, n_collinear, endpoint=False)
    for phi in phis:
        candidates.append([np.cos(phi), np.sin(phi), 0.0])

    # Isosceles: great circles through poles and each collision point
    n_iso = n_candidates - n_collinear
    n_per_circle = n_iso // 3
    collision_phis = [np.pi, -np.pi/3, np.pi/3]
    for cp in collision_phis:
        thetas = np.linspace(0.05, np.pi - 0.05, n_per_circle)
        for theta in thetas:
            candidates.append([
                np.sin(theta) * np.cos(cp),
                np.sin(theta) * np.sin(cp),
                np.cos(theta),
            ])

    candidates = np.array(candidates[:n_candidates])
    return candidates / np.linalg.norm(candidates, axis=1, keepdims=True)


def label_disagreement_candidates(
    shape_pts: np.ndarray,
    labels: np.ndarray,
    n_candidates: int,
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Select candidates by k-NN label disagreement on ground-truth labels.

    Returns: (candidates (n_candidates, 3), disagreement_scores (n_candidates,))
    """
    tree = cKDTree(shape_pts)
    _, idx = tree.query(shape_pts, k=k + 1)
    neighbor_labels = labels[idx[:, 1:]]
    own_labels = labels[:, None]
    disagree_frac = np.mean(neighbor_labels != own_labels, axis=1)

    top = np.argsort(-disagree_frac)[:n_candidates]
    return shape_pts[top], disagree_frac[top]


def classifier_entropy_candidates(
    model,
    shape_pts: np.ndarray,
    n_candidates: int,
    batch_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    """Select candidates by classifier prediction entropy.

    Uses BasinMLP softmax to compute Shannon entropy H(x) = -sum p_i log p_i.

    Returns: (candidates (n_candidates, 3), entropy_scores (n_candidates,))
    """
    import jax.numpy as jnp
    from mega3bp.ml import predict_proba

    all_ent = []
    for start in range(0, len(shape_pts), batch_size):
        batch = jnp.array(shape_pts[start:start + batch_size], dtype=jnp.float64)
        probs = predict_proba(model, batch)
        ent = -jnp.sum(probs * jnp.log(probs + 1e-10), axis=-1)
        all_ent.append(np.asarray(ent))
    entropies = np.concatenate(all_ent)

    top = np.argsort(-entropies)[:n_candidates]
    return shape_pts[top], entropies[top]
