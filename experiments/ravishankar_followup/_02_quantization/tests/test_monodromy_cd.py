"""Test that monodromy_CD.json passes elliptic classification checks."""
import json
from pathlib import Path


def test_CD_elliptic():
    p = Path(__file__).resolve().parents[1] / "monodromy_CD.json"
    assert p.exists(), f"monodromy_CD.json not found at {p}"
    d = json.loads(p.read_text())
    for name in ("C", "D"):
        assert name in d, f"Missing orbit {name} in monodromy_CD.json"
        rec = d[name]
        assert abs(rec["det_M"] - 1.0) < 1e-6, (
            f"{name} det(M)={rec['det_M']:.6e} deviates from 1 beyond 1e-6"
        )
        mags = rec["eigenvalue_magnitudes"]
        max_dev = max(abs(m - 1.0) for m in mags)
        assert max_dev < 1e-3, (
            f"{name} not elliptic: max |lambda|-1 = {max_dev:.3e} > 1e-3"
        )
        assert len(rec["omega_perp"]) == 2, (
            f"{name} omega_perp has {len(rec['omega_perp'])} entries, expected 2"
        )
        assert all(w > 0 for w in rec["omega_perp"]), (
            f"{name} omega_perp has non-positive entries: {rec['omega_perp']}"
        )
