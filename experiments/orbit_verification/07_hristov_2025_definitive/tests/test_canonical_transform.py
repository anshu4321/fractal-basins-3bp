"""Test the canonical Euler-section transform against a synthesized case."""
from __future__ import annotations

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from check_C_D_vs_hristov_2025 import canonical_euler_transform


def test_canonical_transform_identity():
    # Already-canonical state: body 1 at (-1,0), body 2 at (+1,0),
    # body 3 at origin, v1=0.3, v2=-0.5, unit mass.
    v1_true, v2_true = 0.3, -0.5
    q = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    p = np.array([[v1_true, v2_true],
                  [v1_true, v2_true],
                  [-2 * v1_true, -2 * v2_true]])
    v1_got, v2_got, aux = canonical_euler_transform(q, p)
    assert abs(v1_got - v1_true) < 1e-14
    assert abs(v2_got - v2_true) < 1e-14
    assert aux["midpoint_body"] == 2  # body 3 (index 2)


def test_canonical_transform_scaled_rotated():
    v1_true, v2_true = 0.2554309356506809, -0.516385839015133  # orbit C
    scale = 2.5
    theta = 0.37  # arbitrary rotation

    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])

    # Canonical positions
    q_can = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    p_can = np.array([[v1_true, v2_true],
                      [v1_true, v2_true],
                      [-2 * v1_true, -2 * v2_true]])

    # Inverse scale (to simulate a "Hristov-like" scale where bodies are closer)
    # r_canonical = r_raw / d  =>  r_raw = r_canonical * d
    # v_canonical = v_raw * sqrt(d)  =>  v_raw = v_canonical / sqrt(d)
    d = scale  # distance |body1 - midpoint| in raw frame
    q_raw = q_can * d
    p_raw = p_can / np.sqrt(d)

    # Apply rotation and translation
    shift = np.array([5.7, -2.1])
    q_raw = q_raw @ R.T + shift
    p_raw = p_raw @ R.T  # momenta rotate but do not translate

    v1_got, v2_got, aux = canonical_euler_transform(q_raw, p_raw)
    # Allow sign/reflection ambiguity: absolute agreement on (|v1|, |v2|)
    assert abs(v1_got - v1_true) < 1e-12 or abs(v1_got + v1_true) < 1e-12
    assert abs(v2_got - v2_true) < 1e-12 or abs(v2_got + v2_true) < 1e-12
