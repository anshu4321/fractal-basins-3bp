"""S3-symmetric equilateral-triangle section.

Bodies at vertices of an equilateral triangle with pairwise distance 1
(i.e., vertex distance from centre = 1/sqrt(3)).

Velocity field: S3-invariant form v_k = R(2*pi*k/3) . u for a 2D vector
u = (u_x, u_y). This yields sum(v_k) = 0 automatically (the sum of three
rotation matrices at 0, 2pi/3, 4pi/3 is the zero matrix). Angular momentum
is L_z = 3 * R_TRI * u_y; set u_y = 0 to restrict to the L_z = 0 subspace.
"""
import numpy as np

R_TRI = 1.0 / np.sqrt(3.0)  # vertex distance from centre; edge length = 1
_VERT = np.array([
    [np.cos(0),                 np.sin(0)],
    [np.cos(2*np.pi/3),         np.sin(2*np.pi/3)],
    [np.cos(4*np.pi/3),         np.sin(4*np.pi/3)],
]) * R_TRI


def _rot2d(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def section_ic(u):
    """u: (2,) -> (r, v) at t=0 on the equilateral section."""
    u = np.asarray(u, dtype=float)
    r = _VERT.copy()
    v = np.array([_rot2d(2 * np.pi * k / 3) @ u for k in range(3)])
    return r, v


def verify_s3_symmetric(u):
    """Return diagnostics dict for a given u."""
    from .eom import energy, angular_momentum_z
    r, v = section_ic(u)
    p_tot = v.sum(axis=0)
    Lz = angular_momentum_z(r, v)
    d12 = np.linalg.norm(r[0] - r[1])
    d13 = np.linalg.norm(r[0] - r[2])
    d23 = np.linalg.norm(r[1] - r[2])
    return {"p_total": p_tot.tolist(), "L_z": Lz,
            "d12": float(d12), "d13": float(d13), "d23": float(d23),
            "E": energy(r, v)}
