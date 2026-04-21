"""Gutzwiller trace-formula components for hyperbolic orbits A and B."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.resolve().parents[2]


def _load_inputs():
    inv = json.loads((ROOT / "experiments/orbit_verification/15_scaling_and_invariants/invariants.json").read_text())
    action = json.loads((HERE / "action_ABCD.json").read_text())
    maslov = json.loads((HERE / "maslov_ABCD.json").read_text())
    mono_ab = json.loads((ROOT / "experiments/orbit_verification/11_monodromy/monodromy_results.json").read_text())
    return inv, action, maslov, mono_ab


def gutzwiller_components(name):
    inv, action, maslov, mono = _load_inputs()
    orbit = next(o for o in inv["orbits"] if o["name"] == name)
    E0 = float(orbit["E"])
    T = float(orbit["T"])
    S = action[name]["S"]
    nu_mod_2 = maslov[name]["nu_mod_2"]
    # The dominant non-trivial eigenvalue (real, for hyperbolic)
    entry = mono[name]
    mags = np.asarray(entry["eigenvalue_magnitudes"])
    log_mags = np.log(np.maximum(mags, 1e-15))
    idx_dominant = int(np.argmax(np.abs(log_mags)))
    lam = complex(entry["eigenvalues_real"][idx_dominant],
                  entry["eigenvalues_imag"][idx_dominant])
    lam_abs = float(abs(lam))
    s_lyap = float(np.log(lam_abs))  # stability exponent per period

    # |det(M^r - I)|^{1/2} for r = 1..5 — pure-hyperbolic asymptotic:
    # det(M - I) ≈ (λ - 1)(1/λ - 1) = -(λ - 1)²/λ
    amplitudes = []
    for r in range(1, 6):
        lam_r = lam ** r
        # Generic hyperbolic-pair formula
        val = abs((lam_r - 1) * (1/lam_r - 1))
        # amp = 1 / sqrt(|det(M^r - I)|)
        amp = 1.0 / val ** 0.5 if val > 0 else float("inf")
        amplitudes.append({"r": r,
                           "det_M_minus_I": float(val),
                           "amplitude_1_over_sqrt_det": float(amp)})
    return {
        "name": name,
        "E0": E0,
        "T": T,
        "S": S,
        "nu_mod_2": nu_mod_2,
        "lambda_dominant_complex_real": float(lam.real),
        "lambda_dominant_complex_imag": float(lam.imag),
        "lambda_max_magnitude": lam_abs,
        "stability_exponent_s_per_period": s_lyap,
        "lyapunov_per_unit_time": s_lyap / T,
        "gutzwiller_contributions_per_repetition": amplitudes,
        "note": "Full Gutzwiller trace-formula amplitude for r-th repetition is "
                "T_p / (πℏ) × cos(r S_p/ℏ − r ν_p π/2) × amplitude_1_over_sqrt_det. "
                "With ν_p known only mod 2, phases have ±π/2 uncertainty per repetition.",
    }


def main():
    for name in "AB":
        g = gutzwiller_components(name)
        (HERE / f"gutzwiller_{name}.json").write_text(json.dumps(g, indent=2))
        print(f"Orbit {name}: S = {g['S']:.6g}, |λ| = {g['lambda_max_magnitude']:.4f}, "
              f"s = {g['stability_exponent_s_per_period']:.4f}, "
              f"ν_mod_2 = {g['nu_mod_2']}, "
              f"r=1 amp = {g['gutzwiller_contributions_per_repetition'][0]['amplitude_1_over_sqrt_det']:.4f}")


if __name__ == "__main__":
    main()
