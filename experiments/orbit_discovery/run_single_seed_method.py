"""Run one (seed, method) pair of the orbit-discovery baseline comparison.

Thin wrapper around run_baselines_velocity.py internals so Tier 1's
seed-sweep orchestrator can run workers in parallel.

CLI:
    python run_single_seed_method.py --seed N --method M --out-dir PATH \
        [--basin-map PATH] [--k K]

The basin_map.npz must already exist (train once via extract_basin_map.py).
Keeps the paper's defaults: k=6 for label_disagreement, N_CANDIDATES=5000,
N_REFINE_TOP=50, classifier hidden=128, n_layers=4 (but the classifier
is not re-trained here; entropy values come from the cached basin_map.npz).

Writes {out_dir}/seed{N}_{method}.json with a single dict containing the
run results (same schema as one element of a list in group4_baselines.json).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os

import numpy as np

# Import the run_baselines_velocity helpers.
# NOTE: run_baselines_velocity performs jax.config.update at import time, so
# we import it before anything else that touches JAX precision.
import run_baselines_velocity as rbv  # noqa: E402 -- sibling module

# Allow env-var overrides for smoke testing. The orchestrator never sets these;
# only a human running manually does. Keeps paper defaults otherwise.
if "SMOKE_N_CANDIDATES" in os.environ:
    rbv.N_CANDIDATES = int(os.environ["SMOKE_N_CANDIDATES"])
if "SMOKE_N_REFINE_TOP" in os.environ:
    rbv.N_REFINE_TOP = int(os.environ["SMOKE_N_REFINE_TOP"])
if "SMOKE_N_STEPS" in os.environ:
    rbv.N_STEPS = int(os.environ["SMOKE_N_STEPS"])


def label_disagreement_candidates_k(labels_grid, v1_grid, v2_grid, n, seed, k=6):
    """Same as rbv.label_disagreement_candidates but with configurable k."""
    from scipy.spatial import cKDTree

    V1, V2 = np.meshgrid(v1_grid, v2_grid)
    pts = np.column_stack([V1.ravel(), V2.ravel()])
    labels = labels_grid.ravel()

    valid = labels >= 0
    pts_v = pts[valid]
    lab_v = labels[valid]

    tree = cKDTree(pts_v)
    _, idx = tree.query(pts_v, k=k)
    neighbor_labels = lab_v[idx[:, 1:]]
    own_labels = lab_v[:, None]
    disagree = np.mean(neighbor_labels != own_labels, axis=1)

    rng = np.random.default_rng(seed)
    noisy = disagree + rng.normal(0, 0.01, size=len(disagree))
    top = np.argsort(-noisy)[:n]
    return pts_v[top]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--method", required=True,
                    choices=["label_disagreement", "classifier_entropy"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--basin-map", default=None,
                    help="Path to basin_map.npz (default: same dir as this script)")
    ap.add_argument("--k", type=int, default=6,
                    help="k for label_disagreement (default 6 = 5 neighbors + self)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"seed{args.seed}_{args.method}.json"

    basin_path = Path(args.basin_map) if args.basin_map \
        else Path(__file__).parent / "basin_map.npz"
    if not basin_path.exists():
        print(f"ERROR: basin_map not found at {basin_path}", file=sys.stderr)
        return 1

    print(f"Loading {basin_path}", flush=True)
    d = np.load(basin_path)
    labels_grid = d["labels"]
    entropy_grid = d["entropy"]
    v1_grid = d["v1_grid"]
    v2_grid = d["v2_grid"]

    t_start = time.perf_counter()
    if args.method == "label_disagreement":
        cands = label_disagreement_candidates_k(
            labels_grid, v1_grid, v2_grid,
            n=rbv.N_CANDIDATES, seed=args.seed, k=args.k)
    elif args.method == "classifier_entropy":
        cands = rbv.classifier_entropy_candidates(
            entropy_grid, v1_grid, v2_grid,
            n=rbv.N_CANDIDATES, seed=args.seed)
    else:
        raise ValueError(args.method)

    print(f"Generated {len(cands)} candidates for {args.method} seed={args.seed}",
          flush=True)

    result = rbv.run_one(args.method, cands, args.seed)
    result["wall_time_s"] = time.perf_counter() - t_start

    # Write atomically
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, default=str))
    tmp.rename(out_path)
    print(f"Wrote {out_path}", flush=True)
    print(f"  n_verified={result['n_verified']}, "
          f"hit_rate_verify={result['hit_rate_verify']:.5f}, "
          f"wall={result['wall_time_s']:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
