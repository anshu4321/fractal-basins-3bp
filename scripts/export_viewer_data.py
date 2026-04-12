"""Export a subsample of the basin dataset as JSON for the Three.js viewer.

The full 1M dataset is too large for browser rendering. This exports a
stratified subsample (default 100k) that preserves the class distribution,
keeping all escape/ambiguous points and subsampling bound.

Output: viewer/basin_data.json with {n1, n2, n3, label} arrays.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, required=True)
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--out", type=str, default=str(PROJECT_ROOT / "viewer" / "basin_data.json"))
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    print(f"loaded {len(df):,} rows")

    # Keep all escape + ambiguous, subsample bound
    escape = df[df.label != 0]
    bound = df[df.label == 0]
    n_bound = args.n - len(escape)
    if n_bound > 0 and n_bound < len(bound):
        bound = bound.sample(n=n_bound, random_state=42)
    sub = pd.concat([escape, bound], ignore_index=True).sample(frac=1, random_state=42)
    print(f"exported {len(sub):,} rows ({len(escape):,} escape/ambiguous + {len(bound):,} bound)")

    data = {
        "n1": sub.shape_n1.round(5).tolist(),
        "n2": sub.shape_n2.round(5).tolist(),
        "n3": sub.shape_n3.round(5).tolist(),
        "label": sub.label.tolist(),
    }
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f)
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
