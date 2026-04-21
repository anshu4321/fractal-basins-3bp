import numpy as np

from experiments.ravishankar_followup._03_equilateral.section import (
    section_ic, verify_s3_symmetric, R_TRI
)


def test_equilateral_distances_all_one():
    r, v = section_ic((0.3, 0.5))
    d12 = np.linalg.norm(r[0] - r[1])
    d13 = np.linalg.norm(r[0] - r[2])
    d23 = np.linalg.norm(r[1] - r[2])
    assert abs(d12 - 1.0) < 1e-12
    assert abs(d13 - 1.0) < 1e-12
    assert abs(d23 - 1.0) < 1e-12


def test_Lz_formula():
    """L_z = 3 * R_TRI * u_y; zero iff u_y = 0."""
    for u_x, u_y in [(0.5, 0.5), (1.0, -0.3), (-0.2, 0.8), (2, 2), (-1, -1)]:
        info = verify_s3_symmetric((u_x, u_y))
        expected_Lz = 3.0 * R_TRI * u_y
        assert abs(info["L_z"] - expected_Lz) < 1e-12, (
            f"Lz mismatch for u=({u_x},{u_y}): got {info['L_z']}, expected {expected_Lz}"
        )


def test_Lz_zero_when_uy_zero():
    """Restricting to u_y=0 gives the L_z=0 subspace."""
    for u_x in [0.5, 1.0, -0.2, 0.0, 2.0, -1.0]:
        info = verify_s3_symmetric((u_x, 0.0))
        assert abs(info["L_z"]) < 1e-12, f"Lz != 0 for u=({u_x},0): {info['L_z']}"


def test_total_momentum_identically_zero():
    for u in [(0.5, 0.5), (1.0, -0.3), (1.5, 2.0)]:
        info = verify_s3_symmetric(u)
        assert max(abs(x) for x in info["p_total"]) < 1e-12
