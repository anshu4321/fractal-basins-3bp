"""Scale-invariant compute: E, T* for an equal-mass 3BP IC."""
import numpy as np


def compute_E(r, v, masses=(1.0, 1.0, 1.0), G=1.0):
    """Total energy E = sum(0.5*m_i*|v_i|^2) - sum_{i<j} G*m_i*m_j/|r_i-r_j|.

    Parameters
    ----------
    r : ndarray (3, 2)
    v : ndarray (3, 2)
    masses : tuple of 3 floats
    G : float
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    masses = np.asarray(masses, dtype=float)
    T_kin = 0.5 * np.sum(masses[:, None] * v * v)
    V = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            d = np.linalg.norm(r[i] - r[j])
            if d < 1e-14:
                return float("inf")
            V -= G * masses[i] * masses[j] / d
    return float(T_kin + V)


def compute_T_star(T, E):
    """Scale-invariant period: T* = T * |E|^{3/2}."""
    return float(T * abs(E) ** 1.5)


import json
from pathlib import Path


def compute_all_invariants(max_entries=None):
    """Full Hristov 2024 -> list of {row, E, T_star, T_star_hristov, T, word_len}.

    `word_len` is None (placeholder) -- per-entry word-lengths are not currently
    saved by the topology sweep pipeline; a follow-up task will enrich this.
    """
    from .parse_hristov import load_hristov_2024
    records_in = load_hristov_2024()
    if max_entries is not None:
        records_in = records_in[:max_entries]
    records_out = []
    for rec in records_in:
        E = compute_E(rec["r"], rec["v"])
        T_star = compute_T_star(rec["T"], E)
        records_out.append({
            "row": rec["row"],
            "E": E,
            "T_star": T_star,
            "T_star_hristov": rec["T_star_hristov"],
            "T": rec["T"],
            "word_len": None,
        })
    return records_out
