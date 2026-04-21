"""Centralised loaders + heyoka 3-body Euler-section system factory."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def load_invariants():
    """Return the 15_scaling_and_invariants JSON (may not exist on all pods)."""
    p = ROOT / "experiments/orbit_verification/15_scaling_and_invariants/invariants.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_hp_ic(name):
    """Return (v1, v2, T) for orbit A/B/C/D.

    Source priority:
    1. 06_high_precision/hp_heyoka_newton_results.json  (most precise)
    2. 11_monodromy/monodromy_results.json              (always present)
    3. 15_scaling_and_invariants/invariants.json        (50-digit values)
    """
    # --- source 1: high-precision Newton results ---
    hp_path = ROOT / "experiments/orbit_verification/06_high_precision/hp_heyoka_newton_results.json"
    if hp_path.exists():
        hp = json.loads(hp_path.read_text())
        if isinstance(hp, dict):
            entry = hp.get(name)
        else:
            entry = next((r for r in hp if r.get("name") == name), None)
        if entry is not None:
            # HP file uses v1_HP/v2_HP/T_HP keys; fall back to v1/v2/T for other sources
            v1 = entry.get("v1_HP") or entry.get("ic_v1_double") or entry["v1"]
            v2 = entry.get("v2_HP") or entry.get("ic_v2_double") or entry["v2"]
            T  = entry.get("T_HP")  or entry.get("ic_T_double")  or entry["T"]
            return float(v1), float(v2), float(T)

    # --- source 2: monodromy results (always present, has v1/v2/T) ---
    mono_path = ROOT / "experiments/orbit_verification/11_monodromy/monodromy_results.json"
    if mono_path.exists():
        mono = json.loads(mono_path.read_text())
        if name in mono:
            e = mono[name]
            return float(e["v1"]), float(e["v2"]), float(e["T"])

    # --- source 3: invariants JSON ---
    inv = load_invariants()
    if inv is not None:
        entry = next((o for o in inv["orbits"] if o["name"] == name), None)
        if entry is not None:
            return float(entry["v1"]), float(entry["v2"]), float(entry["T"])

    raise FileNotFoundError(f"No IC data found for orbit {name!r} in any source.")


def load_ab_monodromy():
    return json.loads(
        (ROOT / "experiments/orbit_verification/11_monodromy/monodromy_results.json").read_text()
    )


def build_heyoka_system(fp_type=None):
    """Construct a heyoka taylor_adaptive factory for the Euler-section 3-body system.

    Returns a callable make(v1, v2) -> heyoka.taylor_adaptive instance at t=0 with IC:
        r1 = (-1, 0), r2 = (+1, 0), r3 = (0, 0)
        p1 = p2 = (v1, v2),  p3 = -2 (v1, v2)
    """
    import heyoka as hy

    r1x, r1y = hy.make_vars("r1x", "r1y")
    r2x, r2y = hy.make_vars("r2x", "r2y")
    r3x, r3y = hy.make_vars("r3x", "r3y")
    v1x, v1y = hy.make_vars("v1x", "v1y")
    v2x, v2y = hy.make_vars("v2x", "v2y")
    v3x, v3y = hy.make_vars("v3x", "v3y")

    def d(ax, ay, bx, by):
        return hy.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + 1e-300)

    d12 = d(r1x, r1y, r2x, r2y)
    d13 = d(r1x, r1y, r3x, r3y)
    d23 = d(r2x, r2y, r3x, r3y)

    a1x = (r2x - r1x) / d12 ** 3 + (r3x - r1x) / d13 ** 3
    a1y = (r2y - r1y) / d12 ** 3 + (r3y - r1y) / d13 ** 3
    a2x = (r1x - r2x) / d12 ** 3 + (r3x - r2x) / d23 ** 3
    a2y = (r1y - r2y) / d12 ** 3 + (r3y - r2y) / d23 ** 3
    a3x = (r1x - r3x) / d13 ** 3 + (r2x - r3x) / d23 ** 3
    a3y = (r1y - r3y) / d13 ** 3 + (r2y - r3y) / d23 ** 3

    sys_spec = [(r1x, v1x), (r1y, v1y),
                (r2x, v2x), (r2y, v2y),
                (r3x, v3x), (r3y, v3y),
                (v1x, a1x), (v1y, a1y),
                (v2x, a2x), (v2y, a2y),
                (v3x, a3x), (v3y, a3y)]

    def make(v1, v2):
        y0 = np.array([-1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                       v1, v2, v1, v2, -2 * v1, -2 * v2])
        return hy.taylor_adaptive(sys_spec, y0, tol=1e-15)

    return make
