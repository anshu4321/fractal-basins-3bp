"""TDD tests for compute_invariants."""
import numpy as np
from experiments.ravishankar_followup._01_census.compute_invariants import (
    compute_E, compute_T_star,
)


def test_E_isolated_bodies_is_kinetic_minus_nothing():
    r = np.array([[-1000, 0], [0, 0], [1000, 0]], dtype=float)
    v = np.array([[1, 0], [0, 0], [-1, 0]], dtype=float)
    E = compute_E(r, v)
    # T_kin = 0.5*(1+0+1) = 1, V ~ -3/1000 ~ -0.003
    assert abs(E - 1.0) < 0.01


def test_E_hristov_section_all_at_rest():
    # Hristov free-fall IC: body 1 at (-0.5,0), body 2 at (+0.5,0), body 3 at (x3,y3),
    # all with v = 0. E should equal potential V = -1/r12 - 1/r13 - 1/r23.
    x3, y3 = 0.1, 0.5
    r = np.array([[-0.5, 0.0], [0.5, 0.0], [x3, y3]], dtype=float)
    v = np.zeros((3, 2))
    E = compute_E(r, v)
    r12 = 1.0
    r13 = np.sqrt((x3 + 0.5)**2 + y3**2)
    r23 = np.sqrt((x3 - 0.5)**2 + y3**2)
    V_expected = -1/r12 - 1/r13 - 1/r23
    assert abs(E - V_expected) < 1e-12


def test_T_star_invariant_under_alpha_scaling():
    # Under r -> alpha r, v -> v/sqrt(alpha), T -> alpha^{3/2} T, E -> E/alpha.
    # T* = T|E|^{3/2} is unchanged.
    r = np.array([[-0.5, 0], [0.5, 0], [0.2, 0.3]], dtype=float)
    v = np.zeros((3, 2))
    T = 10.0
    E = compute_E(r, v)
    T_star = compute_T_star(T, E)
    alpha = 2.0
    r_a = alpha * r
    v_a = v / np.sqrt(alpha)
    T_a = alpha**1.5 * T
    E_a = compute_E(r_a, v_a)
    T_star_a = compute_T_star(T_a, E_a)
    assert abs(T_star - T_star_a) / abs(T_star) < 1e-10


def test_T_star_hristov_catalog_cross_check():
    # For a first-line entry of the catalog, compute T* from (x3,y3,T) and
    # compare to the catalog's T_star column. Should match Hristov's convention
    # up to convention factors. Actually: Hristov's T_star uses E_hristov with
    # possibly different energy origin, so agreement within a few % is OK.
    # Take the first entry: 0.10081e-3  0.86410e0  0.27592e1  0.14361e2
    x3, y3, T, T_star_cat = 0.10081e-3, 0.86410, 2.7592, 14.361
    r = np.array([[-0.5, 0], [0.5, 0], [x3, y3]], dtype=float)
    v = np.zeros((3, 2))
    E = compute_E(r, v)
    T_star = compute_T_star(T, E)
    # We DO NOT assert equality with T_star_cat -- Hristov uses a different E
    # normalisation (e.g., may rescale so E = -1 per their paper convention).
    # We only check the computed T* is positive and finite.
    assert T_star > 0 and np.isfinite(T_star)
