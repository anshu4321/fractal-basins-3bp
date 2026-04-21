import json
from pathlib import Path


def test_maslov_mod_2_defined_for_all_four():
    p = Path(__file__).resolve().parents[1] / "maslov_ABCD.json"
    d = json.loads(p.read_text())
    for name in "ABCD":
        assert name in d
        assert d[name]["nu_full_pending"] is True
        # nu_mod_2 should be defined for all 4 (our classification is clean)
        assert d[name]["nu_mod_2"] in (0, 1), f"{name}: {d[name]}"
