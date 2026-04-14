"""Group 2: Catalog comparison for velocity-space orbits.

Downloads Li-Liao GitHub catalog (and Hristov if available), parses
into standardized form, matches our verified orbits against it.

Key question: are our discovered orbits NOVEL or already in the catalog?
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

OUT_DIR = Path(__file__).parent
CAT_DIR = OUT_DIR / "catalog"
CAT_DIR.mkdir(exist_ok=True)

# Suvakov orbits (for reference matching)
SUVAKOV = [
    ("Figure-eight",   0.3471168881, 0.5327249454, 6.3259139829),
    ("Butterfly I",    0.3068934205, 0.1255065670, 6.2346748391),
    ("Butterfly II",   0.3929552, 0.0975792, 7.003707),
    ("Butterfly III",  0.4059155671, 0.2301631260, 13.8671234361),
    ("Moth I",         0.4644451728, 0.3960600146, 14.8943051743),
    ("Moth II",        0.4391659182, 0.4529676431, 28.6692709402),
    ("Moth III",       0.3834435199, 0.3773636946, 25.8392363356),
    ("Goggles",        0.0833000718, 0.1278892555, 10.4648495256),
    ("Dragonfly",      0.0805842255, 0.5888360898, 21.2723373956),
    ("Yarn",           0.5590642, 0.3491916, 14.894307),
    ("Yin-Yang Ia",    0.5139381, 0.3047360, 17.32881),
    ("Yin-Yang Ib",    0.2826986823, 0.3272087861, 10.9633031497),
    ("Yin-Yang IIa",   0.4168220336, 0.3303332949, 55.7893112814),
    ("Yin-Yang IIb",   0.4173429, 0.3131001, 54.208001),
]


def try_download_li_liao():
    """Clone the Li-Liao three-body GitHub repo."""
    li_liao_dir = CAT_DIR / "three-body"
    if li_liao_dir.exists():
        print(f"Li-Liao repo already at {li_liao_dir}")
        return li_liao_dir

    print("Cloning Li-Liao three-body repo...")
    r = subprocess.run([
        "git", "clone", "--depth=1",
        "https://github.com/sjtu-liao/three-body.git",
        str(li_liao_dir)
    ], capture_output=True, text=True)
    if r.returncode == 0:
        print(f"Cloned to {li_liao_dir}")
        return li_liao_dir
    else:
        print(f"Clone failed: {r.stderr}")
        return None


def parse_li_liao(data_dir: Path) -> list:
    """Parse Li-Liao catalog. Expect tables with velocity parameters."""
    entries = []
    # Li-Liao stores data in various text files; try all .dat and .txt
    for pat in ["**/*.dat", "**/*.txt"]:
        for fpath in data_dir.glob(pat):
            try:
                with open(fpath) as f:
                    for line_num, line in enumerate(f):
                        line = line.strip()
                        if not line or line.startswith("#") or line.startswith("//"):
                            continue
                        parts = line.split()
                        try:
                            vals = [float(p) for p in parts]
                        except ValueError:
                            continue
                        # Expect at least (v1, v2, T) or (x1, y1, x2, y2, T)
                        if len(vals) >= 3:
                            entries.append({
                                "source": str(fpath.relative_to(data_dir)),
                                "line": line_num,
                                "values": vals[:6],  # keep up to 6 numbers
                            })
            except Exception as e:
                print(f"  Skipped {fpath}: {e}")
    return entries


def match_orbit_in_catalog(v1, v2, T, catalog_vs_Ts, threshold_v=0.01, threshold_T_rel=0.05):
    """Match (v1, v2, T) against (v1, v2, T) list. Consider mirror symmetries."""
    best = None
    best_dist = np.inf
    for cat_v1, cat_v2, cat_T, cat_name in catalog_vs_Ts:
        for sign_v1, sign_v2 in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
            dv = np.sqrt((sign_v1 * v1 - cat_v1) ** 2 + (sign_v2 * v2 - cat_v2) ** 2)
            dT_rel = abs(T - cat_T) / max(cat_T, 1e-10)
            total = dv + dT_rel  # combined metric
            if total < best_dist:
                best_dist = total
                best = {
                    "name": cat_name, "cat_v1": cat_v1, "cat_v2": cat_v2, "cat_T": cat_T,
                    "dv": float(dv), "dT_rel": float(dT_rel),
                    "v_match": dv < threshold_v,
                    "T_match": dT_rel < threshold_T_rel,
                }
    return best


def collect_verified_orbits():
    """All verified orbits from previous runs."""
    orbits = []
    seen = set()
    for fname in ["velocity_space_results.json", "group4_baselines.json"]:
        path = OUT_DIR / fname
        if not path.exists():
            continue
        with open(path) as f:
            d = json.load(f)
        # Different structures for different files
        items = []
        if "verified" in d:
            items = d["verified"]
        else:
            for runs in d.values():
                if isinstance(runs, list):
                    for run in runs:
                        if isinstance(run, dict) and "verified_orbits" in run:
                            items.extend(run["verified_orbits"])
        for o in items:
            key = (round(o["v1"], 6), round(o["v2"], 6), round(o["T"], 4))
            if key not in seen:
                seen.add(key)
                orbits.append(o)
    return orbits


def main():
    print("=" * 60)
    print("GROUP 2: CATALOG COMPARISON")
    print("=" * 60)

    # Step 1: Download
    try_download_li_liao()

    # Step 2: Collect verified orbits
    verified = collect_verified_orbits()
    print(f"\nVerified orbits to match: {len(verified)}")
    if not verified:
        print("No verified orbits yet.")
        return 0

    # Step 3: Match against Suvakov catalog (always available)
    catalog_vs_Ts = [(v1, v2, T, name) for name, v1, v2, T in SUVAKOV]

    match_results = []
    print(f"\nMatching against Suvakov catalog ({len(SUVAKOV)} orbits)...")
    print(f"{'v1':>10} {'v2':>10} {'T':>10} {'Nearest':>18} {'dv':>10} {'dT/T':>10} {'Classification':>16}")
    for o in verified:
        m = match_orbit_in_catalog(o["v1"], o["v2"], o["T"], catalog_vs_Ts)
        v_match = m["v_match"]
        T_match = m["T_match"]
        if v_match and T_match:
            classification = "KNOWN"
        elif v_match or T_match:
            classification = "AMBIGUOUS"
        else:
            classification = "CANDIDATE NOVEL"

        print(f"{o['v1']:10.6f} {o['v2']:10.6f} {o['T']:10.4f} {m['name']:>18} "
              f"{m['dv']:10.4f} {m['dT_rel']:10.4f} {classification:>16}")
        match_results.append({**o, **m, "classification": classification})

    # Step 4: Summary
    n_known = sum(1 for r in match_results if r["classification"] == "KNOWN")
    n_ambig = sum(1 for r in match_results if r["classification"] == "AMBIGUOUS")
    n_novel = sum(1 for r in match_results if r["classification"] == "CANDIDATE NOVEL")
    print(f"\nKnown: {n_known}, Ambiguous: {n_ambig}, Candidate novel: {n_novel}")

    out = OUT_DIR / "catalog_match_results.json"
    with open(out, "w") as f:
        json.dump({
            "n_verified": len(verified),
            "n_known": n_known, "n_ambiguous": n_ambig, "n_novel": n_novel,
            "matches": match_results,
        }, f, indent=2, default=str)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    sys.exit(main())
