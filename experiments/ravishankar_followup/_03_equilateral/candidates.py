"""Select top-K high-LD candidate ICs from ld_field.npz.

De-duplicates candidates that are equivalent under the S3 cyclic rotation
(u -> R(2pi/3) u). Returns canonical representatives.
"""
from pathlib import Path
import json
import numpy as np

HERE = Path(__file__).resolve().parent
K = 20


def s3_canonical_form(u_x, u_y):
    """Smallest-lex rotation of u under the cyclic S3 subgroup."""
    cands = []
    for theta in (0, 2*np.pi/3, 4*np.pi/3):
        c, s = np.cos(theta), np.sin(theta)
        cands.append((c*u_x - s*u_y, s*u_x + c*u_y))
    return min(cands)


def top_k():
    data = np.load(HERE / "ld_field.npz")
    ld = data["ld"]; us = data["u_grid"]
    E = data["E_field"]

    # Local-max via maximum-filter
    from scipy.ndimage import maximum_filter
    local_max = (ld == maximum_filter(ld, size=5))
    # Strip edge cells
    local_max[:5, :] = local_max[-5:, :] = local_max[:, :5] = local_max[:, -5:] = False
    # Require bounded E and reasonably high LD
    local_max = local_max & (E < 0) & (E > -5) & (ld > 0.5)

    raw = []
    for i, j in zip(*np.where(local_max)):
        raw.append({
            "i": int(i), "j": int(j),
            "u_x": float(us[i]), "u_y": float(us[j]),
            "E": float(E[i, j]),
            "ld": float(ld[i, j]),
        })
    raw.sort(key=lambda c: -c["ld"])

    # S3 cyclic dedup: canonicalise to (smallest lex) under 2pi/3 rotation
    seen = set()
    unique = []
    for c in raw:
        key = tuple(round(v, 3) for v in s3_canonical_form(c["u_x"], c["u_y"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
        if len(unique) >= K:
            break

    out = unique
    (HERE / "candidates_ICs.json").write_text(json.dumps(out, indent=2))
    print(f"selected {len(out)} unique candidates from {len(raw)} local maxima")
    for i, c in enumerate(out[:10]):
        print(f"  #{i+1}: u=({c['u_x']:.3f}, {c['u_y']:.3f}), E={c['E']:.3f}, LD={c['ld']:.3f}")
    return out


if __name__ == "__main__":
    top_k()
