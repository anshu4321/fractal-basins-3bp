"""Parse all acquired catalogs into a uniform JSON schema.

Schema per orbit:
  orbit_id: str               (e.g. "hristov2024_00001")
  x3: float                   third-body initial x (free-fall canonical form)
  y3: float                   third-body initial y
  T: float                    period
  T_star: float               scale-invariant period T |E|^(3/2)
  source: str                 catalog short-name
  stable: bool|null
  topology_label: str|null    if catalog provides one

The Hristov files use the standard free-fall convention (their text says):
  bodies 1, 2 at (-0.5, 0), (0.5, 0); body 3 at (x, y); zero velocities.
  Energy normalized so E ~ -1.
  Each line: x  y  T  T_star
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

CAT = Path(__file__).parent


def hash_file(p):
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


# ---------- Hristov 2024: full catalog ----------
def parse_hristov_2024():
    sol = CAT / "hristov_2024_sol_80.txt"
    eig = CAT / "hristov_2024_eigen.txt"

    # Eigenvalue file: header per orbit "i.c. N -->" then 4 numbers
    eig_blocks = {}
    if eig.exists():
        idx = None
        cur = []
        for line in eig.read_text().splitlines():
            if line.startswith("i.c."):
                if idx is not None:
                    eig_blocks[idx] = cur
                idx = int(line.split()[1])
                cur = []
            elif line.strip():
                try:
                    cur.append(float(line.replace("e", "E")))
                except ValueError:
                    pass
        if idx is not None:
            eig_blocks[idx] = cur

    orbits = []
    for i, line in enumerate(sol.read_text().splitlines(), start=1):
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            x = float(parts[0]); y = float(parts[1])
            T = float(parts[2]); Ts = float(parts[3])
        except ValueError:
            continue
        eigs = eig_blocks.get(i)
        # Stability heuristic from eigenvalue magnitudes (paper's convention)
        stable = None
        if eigs and len(eigs) >= 4:
            # Hristov uses the 4 nontrivial eigenvalues.
            # Stable iff all |λ| ≤ 1 + tol.
            stable = all(abs(e) <= 1.0 + 1e-3 for e in eigs[:4])
        orbits.append({
            "orbit_id": f"hristov2024_{i:05d}",
            "x3": x, "y3": y, "T": T, "T_star": Ts,
            "source": "hristov_2024",
            "stable": stable, "topology_label": None,
        })
    return orbits


# ---------- Hristov 2025: 971 stable ----------
def parse_hristov_2025_stable():
    sol = CAT / "hristov_2025_stable.txt"
    syz = CAT / "hristov_2025_syzygies.txt"
    syz_lines = syz.read_text().splitlines() if syz.exists() else []
    orbits = []
    for i, line in enumerate(sol.read_text().splitlines(), start=1):
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            x = float(parts[0]); y = float(parts[1])
            T = float(parts[2]); Ts = float(parts[3])
        except ValueError:
            continue
        topology = syz_lines[i-1].strip() if i-1 < len(syz_lines) else None
        orbits.append({
            "orbit_id": f"hristov2025_{i:04d}",
            "x3": x, "y3": y, "T": T, "T_star": Ts,
            "source": "hristov_2025_stable",
            "stable": True,  # by construction
            "topology_label": topology,
        })
    return orbits


# ---------- Li-Liao 2017: 695 (already parsed earlier) ----------
def parse_li_liao_2017():
    """Load the previously parsed 695 entries from paper2/li_liao_catalog.json.

    NOTE: Li-Liao 2017 uses our convention (Euler collinear, v-space), NOT
    the Hristov free-fall convention. We store the (v1, v2) directly as
    (x3, y3) is not the right field; instead, we record source flag so the
    matching code can apply the right transform later.
    """
    src = CAT / "li_liao_2017_695.json"
    if not src.exists():
        return []
    raw = json.loads(src.read_text())
    out = []
    for c in raw:
        out.append({
            "orbit_id": f"liliao2017_{c['family']}_{c['index']:03d}",
            # Li-Liao uses VELOCITY-SPACE convention; store as v1, v2:
            "v1": c["v1"], "v2": c["v2"],
            "T": c["T"], "T_star": c["T_star"],
            "source": "li_liao_2017",
            "stable": None, "topology_label": c["family"],
            "L_f": c.get("L_f"),
        })
    return out


def main():
    print("Parsing catalogs ...")
    h2024 = parse_hristov_2024()
    print(f"  hristov_2024: {len(h2024)} orbits")
    h2025 = parse_hristov_2025_stable()
    print(f"  hristov_2025_stable: {len(h2025)} orbits")
    ll17 = parse_li_liao_2017()
    print(f"  li_liao_2017: {len(ll17)} orbits")

    out = {
        "hristov_2024": h2024,
        "hristov_2025_stable": h2025,
        "li_liao_2017": ll17,
        "total": len(h2024) + len(h2025) + len(ll17),
    }
    (CAT / "all_catalogs.json").write_text(json.dumps(out, indent=2))
    print(f"\nTotal entries: {out['total']}")
    print(f"Saved to {CAT / 'all_catalogs.json'}")

    # Verify hashes match SHA256SUMS
    print("\nVerifying file hashes against SHA256SUMS.txt ...")
    sums = {}
    for line in (CAT / "SHA256SUMS.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            sums[parts[1]] = parts[0]
    for fn, expected in sums.items():
        p = CAT / fn
        if p.exists():
            actual = hash_file(p)
            ok = "OK" if actual == expected else "MISMATCH"
            print(f"  {fn}: {ok}")
        else:
            print(f"  {fn}: MISSING")


if __name__ == "__main__":
    main()
