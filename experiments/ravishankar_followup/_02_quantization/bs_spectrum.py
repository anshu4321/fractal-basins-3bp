"""Bohr-Sommerfeld spectrum for elliptic orbits C and D."""
import json
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.resolve().parents[2]


def _load_inputs():
    inv = json.loads((ROOT / "experiments/orbit_verification/15_scaling_and_invariants/invariants.json").read_text())
    action = json.loads((HERE / "action_ABCD.json").read_text())
    maslov = json.loads((HERE / "maslov_ABCD.json").read_text())
    mono = json.loads((HERE / "monodromy_CD.json").read_text())
    return inv, action, maslov, mono


def bs_spectrum(name, n_max=10, m_max=4):
    inv, action, maslov, mono = _load_inputs()
    orbit = next(o for o in inv["orbits"] if o["name"] == name)
    E0 = float(orbit["E"])
    S0 = action[name]["S"]
    nu_mod_2 = maslov[name]["nu_mod_2"]
    omegas = mono[name]["omega_perp"]

    nu_over_4 = nu_mod_2 / 4.0  # 0.0 or 0.25

    levels = []
    for n in range(n_max):
        arg = 2 * np.pi * (n + nu_over_4)
        if arg <= 0:
            continue
        # Longitudinal: E_n^long preserves sign of E0 (bound orbit, E < 0)
        # Using |E|-scaling: S(E) = S0 |E/E0|^{-1/2}, solve S(E) = 2π(n+ν/4)
        ratio = S0 / arg
        E_n_long = E0 * ratio ** 2 if E0 > 0 else -abs(E0) * ratio ** 2
        # Actually E < 0 throughout, so
        E_n_long = -abs(E0) * ratio ** 2 if False else E0 * ratio ** 2  # sign preservation
        # simpler: if E0 < 0, E_n < 0 (ratio^2 > 0, E0 < 0 → E_n < 0)
        # So:
        E_n_long = E0 * ratio ** 2
        # But that's positive × negative = negative — correct.
        for m1 in range(m_max):
            for m2 in range(m_max):
                E_full = E_n_long + (m1 + 0.5) * omegas[0] + (m2 + 0.5) * omegas[1]
                levels.append({
                    "n": n, "m1": m1, "m2": m2,
                    "E_long": E_n_long,
                    "E": E_full,
                })
    return {
        "name": name,
        "E0_reference": E0,
        "S0": S0,
        "nu_mod_2": nu_mod_2,
        "nu_over_4_used": nu_over_4,
        "nu_full_pending": True,
        "omega_perp": omegas,
        "levels": levels,
        "note": "ν_p is only known mod 2; a true Maslov shift of ±k/2 (unknown integer k) "
                "rescales all E_n by (1 + k/2/(n+ν_mod_2/4))² — absolute energies are "
                "therefore indicative. Relative level spacings within a fixed n-branch "
                "(i.e., varying m1, m2) are exact.",
    }


def main():
    for name in ("C", "D"):
        spec = bs_spectrum(name)
        (HERE / f"bs_spectrum_{name}.json").write_text(json.dumps(spec, indent=2))
        print(f"Orbit {name}: {len(spec['levels'])} levels, ν_mod_2 = {spec['nu_mod_2']}, "
              f"E_0^long = {spec['levels'][0]['E_long']:.4f}, E_0^ground_transverse = "
              f"{spec['levels'][0]['E']:.4f}")


if __name__ == "__main__":
    main()
