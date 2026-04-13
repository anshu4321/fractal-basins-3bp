"""Group 2: Catalog Comparison (Tests 2.1-2.4).

Downloads/parses catalogs, normalizes our orbits, matches them,
and classifies each as known/ambiguous/candidate-novel.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from mega3bp.normalize import normalize_free_fall, reduced_ic_4d, verify_normalization_consistency
from catalog import load_catalog, match_orbit, parse_li_liao_data, build_catalog

OUT_DIR = Path(__file__).parent


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def main():
    print("=" * 60)
    print("GROUP 2: CATALOG COMPARISON")
    print("=" * 60)

    # Test 2.1: Acquire catalog
    catalog_dir = OUT_DIR / "05_catalog"
    catalog_path = catalog_dir / "hristov_2024.parquet"
    json_fallback = catalog_path.with_suffix(".json")

    if not catalog_path.exists() and not json_fallback.exists():
        print("\nCatalog not found. Attempting to download Li-Liao data...")
        print("If this fails, manually download from:")
        print("  https://github.com/sjtu-liao/three-body")
        print("  or contact ivanh@fmi.uni-sofia.bg for Hristov 2024 catalog")
        li_liao_dir = catalog_dir / "three-body-raw"
        if not li_liao_dir.exists():
            subprocess.run([
                "git", "clone", "--depth=1",
                "https://github.com/sjtu-liao/three-body.git",
                str(li_liao_dir)
            ], check=False)

        if li_liao_dir.exists():
            entries = parse_li_liao_data(li_liao_dir)
            print(f"Parsed {len(entries)} catalog entries")
            out = build_catalog(entries, catalog_path)
            print(f"Catalog written to {out}")
        else:
            print("WARNING: Could not acquire catalog. Manual download required.")
            return 1

    # Load catalog
    catalog_file = catalog_path if catalog_path.exists() else json_fallback
    catalog_ics, catalog_T_stars, catalog_ids = load_catalog(catalog_file)
    print(f"Catalog loaded: {len(catalog_ids)} entries")

    # Test 2.2: Normalize our orbits
    verified_path = OUT_DIR / "group1_verified.json"
    if not verified_path.exists():
        print("ERROR: group1_verified.json not found. Run Group 1 first.")
        return 1

    with open(verified_path) as f:
        verified = json.load(f)
    print(f"Loaded {len(verified)} Group-1-verified orbits")

    normalized = []
    for v in verified:
        q0 = np.array(v["q0"]).reshape(3, 2)
        T = v["T_refined"]
        norm = normalize_free_fall(q0, T)
        max_diff = verify_normalization_consistency(q0, T)
        norm["consistency_check"] = float(max_diff)
        norm["candidate_idx"] = v["candidate_idx"]
        norm["q0_canonical"] = norm["q0_canonical"].tolist()
        normalized.append(norm)

    write_json(OUT_DIR / "06_normalized" / "your_orbits_canonical.json", normalized)

    # Test 2.3: Orbit matching
    match_results = []
    for norm in normalized:
        ic_4d = reduced_ic_4d(np.array(norm["q0_canonical"]))
        T_star = norm["T_star"]
        match = match_orbit(ic_4d, T_star, catalog_ics, catalog_T_stars, catalog_ids)
        match["candidate_idx"] = norm["candidate_idx"]
        match["T_star"] = T_star
        match_results.append(match)

    write_json(OUT_DIR / "07_matching" / "matching_table.json", match_results)

    # Write matching report
    report_lines = ["# Orbit Matching Report\n\n"]
    report_lines.append("| Orbit | Classification | Nearest Catalog ID | "
                        "IC Dist | dT*/T* | IC Match | T Match |\n")
    report_lines.append("|-------|---------------|-------------------|---------|--------|----------|--------|\n")
    for m in match_results:
        report_lines.append(
            f"| {m['candidate_idx']} | {m['classification']} | "
            f"{m['nearest_catalog_id']} | {m['min_dist_4d']:.2e} | "
            f"{m['dT_rel']:.2e} | {m['ic_match']} | {m['T_match']} |\n")

    (OUT_DIR / "07_matching" / "matching_report.md").write_text("".join(report_lines))

    # Identify candidate novel orbits for Test 2.4
    novel_candidates = [m for m in match_results if m["classification"] == "candidate_novel"]
    write_json(OUT_DIR / "08_high_precision" / "novel_orbits.json",
               novel_candidates if novel_candidates else [])

    # Write verdict
    n_known = sum(1 for m in match_results if m["classification"] == "known")
    n_ambiguous = sum(1 for m in match_results if m["classification"] == "ambiguous")
    n_novel = len(novel_candidates)

    verdict = [
        "# GROUP 2 VERDICT: Catalog Comparison\n\n",
        f"## Summary\n\n",
        f"Matched {len(match_results)} Group-1-verified orbits against catalog "
        f"({len(catalog_ids)} entries).\n\n",
        f"- Known: {n_known}\n",
        f"- Ambiguous: {n_ambiguous}\n",
        f"- Candidate novel: {n_novel}\n\n",
    ]

    if n_novel > 0:
        verdict.append(
            f"## Branch decision\n\n"
            f"**{n_novel} candidate novel orbit(s) found.** "
            f"Proceed to Test 2.4 (high-precision re-verification) before claiming novelty.\n"
            f"If any survive Test 2.4, headline becomes: "
            f"'first ML-discovered orbits in equal-mass planar free-fall 3BP.'\n")
    else:
        verdict.append(
            f"## Branch decision\n\n"
            f"**No novel orbits.** This is the expected outcome. "
            f"Headline becomes: 'ML-based sampling prior efficiently rediscovers "
            f"known orbits without orbit knowledge.'\n"
            f"Continue to Group 3.\n")

    (OUT_DIR / "GROUP2_VERDICT.md").write_text("".join(verdict))
    print(f"\nGroup 2 complete. Known: {n_known}, Ambiguous: {n_ambiguous}, Novel: {n_novel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
