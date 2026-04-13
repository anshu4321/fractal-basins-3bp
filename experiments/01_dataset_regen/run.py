"""Experiment 01: Dataset regeneration with per-trajectory diagnostics.

Rebuilds the 10^6-sample at-rest and 6D datasets with strict escape
criterion (Standish geometry + positive two-body energy) and logs
per-trajectory: max|ΔE/E|, min_pairwise_distance, escape lifetime.

Outputs:
    results/01_dataset_regen/at_rest_with_diagnostics.npz
    results/01_dataset_regen/6d_with_diagnostics.npz
    results/01_dataset_regen/results.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from mega3bp.batch import integrate_batch_diag
from mega3bp.dynamics import angular_momentum, total_energy
from mega3bp.escape import LABEL_NAMES
from mega3bp.shape_sphere import (
    sample_phase_space_6d,
    sample_shape_sphere,
    shape_to_config,
)

N_ICS = 1_048_576
SEED_2D = 0xFEEDBEEF
SEED_6D = 0xDEAD6D01
H = 0.01
T_MAX = 500.0
N_STEPS = int(round(T_MAX / H))
R_ESCAPE = 5.0
BINARY_FACTOR = 2.0
R_CLOSE = 1e-3
CHUNK = 65536
P_MAX_6D = 3.0


def run_dataset(name, q_full, p_full, shape_np, extra_cols, n_ics, out_dir):
    """Integrate and collect full diagnostics."""
    n_chunks = (n_ics + CHUNK - 1) // CHUNK
    all_labels = np.empty(n_ics, dtype=np.int8)
    all_etimes = np.empty(n_ics, dtype=np.float64)
    all_amb = np.empty(n_ics, dtype=np.bool_)
    all_rmin = np.empty(n_ics, dtype=np.float64)
    all_maxsep = np.empty(n_ics, dtype=np.float64)
    all_max_dE = np.empty(n_ics, dtype=np.float64)
    all_E0 = np.empty(n_ics, dtype=np.float64)

    t_total = time.perf_counter()
    for i in range(n_chunks):
        lo = i * CHUNK
        hi = min(lo + CHUNK, n_ics)
        actual = hi - lo
        q_c = q_full[lo:hi]
        p_c = p_full[lo:hi]
        if actual < CHUNK:
            pad = CHUNK - actual
            q_c = jnp.concatenate([q_c, jnp.repeat(q_c[:1], pad, axis=0)])
            p_c = jnp.concatenate([p_c, jnp.repeat(p_c[:1], pad, axis=0)])

        out = integrate_batch_diag(
            q_c, p_c, H, N_STEPS,
            r_escape=R_ESCAPE, binary_factor=BINARY_FACTOR, r_close=R_CLOSE,
            strict_escape=True,
        )
        jax.block_until_ready(out["label"])

        for k, arr in [("label", all_labels), ("escape_time", all_etimes),
                        ("ambiguous", all_amb), ("r_min_ever", all_rmin),
                        ("max_sep", all_maxsep), ("max_dE_rel", all_max_dE),
                        ("E_initial", all_E0)]:
            arr[lo:hi] = np.asarray(out[k])[:actual]

        tput = actual * N_STEPS / (time.perf_counter() - t_total) if i == 0 else 0
        dt_chunk = time.perf_counter() - t_total
        print(f"  {name} chunk {i+1}/{n_chunks}  [{lo}:{hi}]")

    wall = time.perf_counter() - t_total
    tput = n_ics * N_STEPS / wall
    print(f"  {name} done: {wall:.1f}s ({tput/1e6:.1f} Mstep/s)")

    save_dict = {
        "shape_n": shape_np,
        "label": all_labels,
        "escape_time": all_etimes,
        "ambiguous": all_amb,
        "r_min_ever": all_rmin,
        "max_sep": all_maxsep,
        "max_dE_rel": all_max_dE,
        "E_initial": all_E0,
    }
    save_dict.update(extra_cols)

    out_path = out_dir / f"{name}_with_diagnostics.npz"
    np.savez_compressed(out_path, **save_dict)
    print(f"  saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    unique, counts = np.unique(all_labels, return_counts=True)
    label_dist = {LABEL_NAMES[int(l)]: int(c) for l, c in zip(unique, counts)}
    frac_bad_energy = float((all_max_dE > 1e-6).mean())
    frac_amb = float(all_amb.mean())

    dE_pcts = np.percentile(all_max_dE, [50, 90, 99, 99.9, 100])
    rmin_pcts = np.percentile(all_rmin, [0, 1, 5, 50])

    stats = {
        "name": name,
        "n_ics": n_ics,
        "wall_seconds": round(wall, 1),
        "throughput_Msteps": round(tput / 1e6, 1),
        "label_counts": label_dist,
        "frac_energy_error_gt_1e6": round(frac_bad_energy, 6),
        "frac_ambiguous": round(frac_amb, 6),
        "max_dE_percentiles": {
            "p50": float(dE_pcts[0]),
            "p90": float(dE_pcts[1]),
            "p99": float(dE_pcts[2]),
            "p99.9": float(dE_pcts[3]),
            "max": float(dE_pcts[4]),
        },
        "r_min_percentiles": {
            "min": float(rmin_pcts[0]),
            "p1": float(rmin_pcts[1]),
            "p5": float(rmin_pcts[2]),
            "p50": float(rmin_pcts[3]),
        },
    }

    print(f"  labels: {label_dist}")
    print(f"  frac |ΔE/E| > 1e-6: {frac_bad_energy:.4%}")
    print(f"  max |ΔE/E| percentiles: p50={dE_pcts[0]:.2e} p99={dE_pcts[2]:.2e} max={dE_pcts[4]:.2e}")
    print(f"  r_min percentiles: min={rmin_pcts[0]:.3e} p1={rmin_pcts[1]:.3e} p5={rmin_pcts[2]:.3e}")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_ICS)
    parser.add_argument("--dataset", choices=["at_rest", "6d", "both"], default="both")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / "results" / "01_dataset_regen"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"jax backend: {jax.default_backend()}")
    print(f"devices: {jax.devices()}")
    print(f"N ICs: {args.n:,}, h={H}, t_max={T_MAX}, n_steps={N_STEPS}")
    print(f"strict escape: True (Standish + two-body energy)")
    print()

    all_stats = {}

    if args.dataset in ("at_rest", "both"):
        print("=== At-rest (2D shape sphere, p=0) ===")
        key = jax.random.PRNGKey(SEED_2D)
        shape_pts = np.asarray(sample_shape_sphere(key, args.n))
        q = shape_to_config(jnp.asarray(shape_pts), inertia=1.0)
        p = jnp.zeros_like(q)
        stats = run_dataset("at_rest", q, p, shape_pts, {}, args.n, out_dir)
        all_stats["at_rest"] = stats
        print()

    if args.dataset in ("6d", "both"):
        print("=== 6D (non-zero momentum, p_max=3.0) ===")
        key = jax.random.PRNGKey(SEED_6D)
        shape_pts, pi_jacobi, q, p = sample_phase_space_6d(key, args.n, p_max=P_MAX_6D)
        shape_np = np.asarray(shape_pts)
        pi_np = np.asarray(pi_jacobi)
        E = np.asarray(total_energy(q, p))
        L = np.asarray(angular_momentum(q, p))
        extra = {"pi_jacobi": pi_np, "energy": E, "angular_momentum": L}
        stats = run_dataset("6d", q, p, shape_np, extra, args.n, out_dir)
        all_stats["6d"] = stats
        print()

    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"saved {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
