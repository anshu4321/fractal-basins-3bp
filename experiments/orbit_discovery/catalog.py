"""Hristov et al. 2024 catalog parser and orbit matcher.

Sources:
  1. Li-Liao GitHub: https://github.com/sjtu-liao/three-body
     (initial conditions for 600+ families)
  2. Hristov et al. 2024 supplementary data from arXiv:2308.16159
     or direct request to ivanh@fmi.uni-sofia.bg

Catalog format: JSON or parquet with fields:
  orbit_id, q1x, q1y, q2x, q2y, T_star, source
All in canonical form: E=-1, M=3, COM=0.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False


def parse_li_liao_data(data_dir: Path) -> list[dict]:
    """Parse Li-Liao three-body data files.

    The Li-Liao GitHub repo stores initial conditions in text files.
    Format varies but typically: x1 y1 x2 y2 T [stability_index]
    """
    entries = []
    for fpath in sorted(data_dir.glob("*.txt")):
        with open(fpath) as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    vals = [float(p) for p in parts]
                except ValueError:
                    continue
                entry = {
                    "source": fpath.stem,
                    "orbit_id": f"{fpath.stem}_{line_num}",
                    "q1x": vals[0], "q1y": vals[1],
                    "q2x": vals[2], "q2y": vals[3],
                    "T_star": vals[4],
                }
                if len(vals) > 5:
                    entry["stability_index"] = vals[5]
                entries.append(entry)
    return entries


def build_catalog(entries: list[dict], out_path: Path) -> Path:
    """Write catalog entries to parquet (if available) or JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if HAS_PARQUET:
        table = pa.table({
            "orbit_id": [e["orbit_id"] for e in entries],
            "q1x": [e["q1x"] for e in entries],
            "q1y": [e["q1y"] for e in entries],
            "q2x": [e["q2x"] for e in entries],
            "q2y": [e["q2y"] for e in entries],
            "T_star": [e["T_star"] for e in entries],
            "source": [e.get("source", "") for e in entries],
        })
        pq.write_table(table, out_path)
        return out_path
    else:
        json_path = out_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(entries, f, indent=2)
        return json_path


def load_catalog(path: Path) -> tuple[np.ndarray, np.ndarray, list]:
    """Load catalog as (N, 4) IC array, T_star array, and orbit_id list."""
    if path.suffix == ".parquet" and HAS_PARQUET:
        table = pq.read_table(path)
        df = table.to_pydict()
    elif path.suffix == ".json":
        with open(path) as f:
            df_list = json.load(f)
        df = {k: [e[k] for e in df_list] for k in df_list[0].keys()}
    else:
        raise ValueError(f"Unknown catalog format: {path.suffix}")

    ics = np.column_stack([df["q1x"], df["q1y"], df["q2x"], df["q2y"]])
    T_stars = np.array(df["T_star"])
    orbit_ids = df["orbit_id"]
    return ics, T_stars, orbit_ids


def match_orbit(
    ic_4d: np.ndarray,
    T_star: float,
    catalog_ics: np.ndarray,
    catalog_T_stars: np.ndarray,
    catalog_ids: list,
    ic_threshold: float = 1e-3,
    T_threshold: float = 1e-3,
) -> dict:
    """Match a single orbit against the catalog.

    Classification per PROMPT Test 2.3:
    - known: min IC distance < 1e-3 AND |dT*/T*| < 1e-3
    - ambiguous: one criterion passes but not both
    - candidate_novel: both fail
    """
    dists = np.linalg.norm(catalog_ics - ic_4d[None, :], axis=1)
    min_idx = int(np.argmin(dists))
    min_dist = float(dists[min_idx])

    dT_rel = abs(T_star - catalog_T_stars[min_idx]) / max(abs(catalog_T_stars[min_idx]), 1e-30)

    ic_match = min_dist < ic_threshold
    T_match = dT_rel < T_threshold

    if ic_match and T_match:
        classification = "known"
    elif ic_match or T_match:
        classification = "ambiguous"
    else:
        classification = "candidate_novel"

    return {
        "classification": classification,
        "min_dist_4d": min_dist,
        "nearest_catalog_id": catalog_ids[min_idx],
        "nearest_T_star": float(catalog_T_stars[min_idx]),
        "dT_rel": float(dT_rel),
        "ic_match": bool(ic_match),
        "T_match": bool(T_match),
    }
