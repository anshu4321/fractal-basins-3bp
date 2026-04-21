"""Loader for Hristov 2024 catalog (free-fall convention, 4 columns).

File format: one line per orbit, whitespace-separated columns
    x3  y3  T  T_star_hristov
Body 1 at (-0.5, 0), body 2 at (+0.5, 0), body 3 at (x3, y3), all at rest.
"""
from pathlib import Path
import numpy as np

_CATALOG_PATH = Path(__file__).resolve().parents[2] / \
    "orbit_verification/00_catalogs/hristov_2024_sol_80.txt"


def load_hristov_2024():
    """Yield one record per non-blank, non-comment line.

    Record dict: {"row": int (1-based), "r": (3,2) ndarray, "v": (3,2) zeros,
                  "T": float, "T_star_hristov": float}.
    """
    records = []
    with open(_CATALOG_PATH) as f:
        row = 0
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row += 1
            parts = [p for p in line.split() if p]
            if len(parts) < 4:
                raise ValueError(f"Unexpected column count on row {row}: {len(parts)}")
            x3, y3, T, T_star = map(float, parts[:4])
            r = np.array([[-0.5, 0.0], [0.5, 0.0], [x3, y3]])
            v = np.zeros((3, 2))
            records.append({"row": row, "r": r, "v": v,
                            "T": T, "T_star_hristov": T_star})
    return records
