"""Simplified Maslov (Conley-Zehnder) index via terminal eigenvalue inspection.

Computes ν mod 2 from the monodromy eigenvalue spectrum at t=T.
A full continuous-path winding computation is deferred to a follow-up pass.
"""
from pathlib import Path
import json
import numpy as np

HERE = Path(__file__).resolve().parent


def _load_all_monodromy():
    """Merge A/B (from 11_monodromy) with C/D (our local JSON) into one dict."""
    mono_ab = json.loads(
        (HERE.resolve().parents[2] /
         "experiments/orbit_verification/11_monodromy/monodromy_results.json").read_text()
    )
    mono_cd = json.loads((HERE / "monodromy_CD.json").read_text())
    out = {}
    for name in "AB":
        out[name] = mono_ab[name]
    for name in "CD":
        out[name] = mono_cd[name]
    return out


def _identify_nontrivial_eigs(real, imag, mag):
    """Return the 2 non-trivial eigenvalues for a 12x12 monodromy.

    For elliptic orbits, these are the 2 eigenvalues with largest |arg|.
    For hyperbolic orbits, these are the eigenvalues with largest |log|λ||.
    """
    eigs = np.asarray(real) + 1j * np.asarray(imag)
    mag = np.asarray(mag)
    log_mag = np.log(np.maximum(mag, 1e-15))
    args = np.angle(eigs)
    # Hyperbolic test: any |log|λ|| > 0.01 (|λ| > e^0.01 ≈ 1.01)
    has_hyperbolic = np.any(np.abs(log_mag) > 0.01)
    if has_hyperbolic:
        # Take the 2 with largest |log|λ||
        idx = np.argsort(-np.abs(log_mag))[:2]
    else:
        # Elliptic: take the 2 with largest |arg|, ignoring the 10 trivial-1 cluster
        # Trivial eigenvalues have arg ≈ 0 AND mag ≈ 1; non-trivial have |arg| > 1e-4
        sorted_idx = np.argsort(-np.abs(args))
        idx = []
        seen = []
        for i in sorted_idx:
            a = args[i]
            if abs(a) < 1e-4:
                continue
            # pair-skip: conjugate
            if any(abs(abs(a) - abs(s)) < 1e-6 for s in seen):
                continue
            seen.append(a)
            idx.append(i)
            if len(idx) == 2:
                break
    return idx, eigs, args


def maslov_index_mod_2(monodromy_entry):
    """Return {nu_mod_2, rationale}."""
    real = monodromy_entry["eigenvalues_real"]
    imag = monodromy_entry["eigenvalues_imag"]
    mag = monodromy_entry["eigenvalue_magnitudes"]
    idx, eigs, args = _identify_nontrivial_eigs(real, imag, mag)
    name = monodromy_entry.get("name", "?")
    class_ = monodromy_entry.get("classification", "?")

    if class_ == "hyperbolic":
        # ν mod 2 from sign of dominant real eigenvalue
        if len(idx) == 0:
            return {"nu_mod_2": None, "rationale": "no non-trivial eigenvalues"}
        lam_dom = complex(eigs[idx[0]])
        if abs(lam_dom.imag) > 1e-6:
            return {"nu_mod_2": None,
                    "rationale": "dominant non-trivial eigenvalue is complex; "
                                 "mod-2 Maslov from sign-of-real is not defined"}
        nu2 = 0 if lam_dom.real > 0 else 1
        return {"nu_mod_2": nu2,
                "rationale": f"hyperbolic: λ_dom = {lam_dom.real:+.6f} "
                             f"({'+' if lam_dom.real > 0 else '-'}), "
                             f"ν mod 2 = {nu2}",
                "lambda_dominant_real": float(lam_dom.real),
                "lambda_dominant_imag": float(lam_dom.imag),
                }
    elif class_ == "linearly stable":
        # ν mod 2 from sum of arg/π for 2 non-trivial pairs
        if len(idx) < 2:
            return {"nu_mod_2": None,
                    "rationale": "could not identify 2 non-trivial elliptic pairs"}
        a1 = abs(args[idx[0]])
        a2 = abs(args[idx[1]])
        nu_approx = (a1 + a2) / np.pi
        nu2_float = nu_approx % 2
        nu2 = int(round(nu2_float))
        return {"nu_mod_2": nu2,
                "rationale": f"elliptic: sum(|arg|)/π = {nu_approx:.4f}, "
                             f"ν mod 2 ≈ {nu2} (rounding)",
                "arg_nontrivial_1_pi": float(a1 / np.pi),
                "arg_nontrivial_2_pi": float(a2 / np.pi),
                "nu_approx_lower_bound": float(nu_approx),
                }
    return {"nu_mod_2": None, "rationale": f"unknown class {class_}"}


def main():
    mono = _load_all_monodromy()
    out = {}
    for name in "ABCD":
        if name in mono:
            r = maslov_index_mod_2(mono[name])
            out[name] = {**r,
                         "nu_full_pending": True,
                         "note": "Full Conley-Zehnder winding computation deferred "
                                 "to a follow-up task; this is the mod-2 answer only."}
            print(f"{name}: nu_mod_2 = {r.get('nu_mod_2')}, rationale = {r['rationale']}")
    (HERE / "maslov_ABCD.json").write_text(json.dumps(out, indent=2))
    print(f"wrote maslov_ABCD.json")


if __name__ == "__main__":
    main()
