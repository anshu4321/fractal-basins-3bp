"""Phase 2 full dataset generation.

Samples N_ICS points uniformly on the Montgomery shape sphere, integrates
each from rest for t_max units, and saves the labeled dataset to parquet.

Default: 262,144 (2^18) ICs, t_max = 500, h = 0.01. On H100 fp64 this
takes roughly 15-20 minutes and produces a ~20 MB parquet file under
data/phase2_basin_262k.parquet.

Schema:
    shape_n1, shape_n2, shape_n3   float64  IC on the shape sphere
    label                           int8     -1 ambiguous / 0 bound / 1-3 escape
    escape_time                     float64  inf if bound
    ambiguous                       bool     close encounter flag
    r_min_ever                      float64  min pair distance ever
    max_sep                         float64  max pair distance ever

For 1M ICs, pass --n=1048576 (roughly 60-80 min run).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from mega3bp.batch import integrate_batch
from mega3bp.escape import LABEL_NAMES
from mega3bp.shape_sphere import sample_shape_sphere, shape_to_config

DEFAULT_N_ICS = 262144
DEFAULT_SEED = 0xFEEDBEEF

H = 0.01
T_MAX = 500.0
N_STEPS = int(round(T_MAX / H))
R_ESCAPE = 5.0
BINARY_FACTOR = 2.0
R_CLOSE = 1e-3

# Chunk size: the integrator state for 65k trajectories is ~50 MB which JIT
# compiles fine. Larger chunks get better throughput but eat more peak memory.
CHUNK = 65536


def run_chunk(configs_chunk, p0_chunk) -> dict[str, np.ndarray]:
    out = integrate_batch(
        configs_chunk,
        p0_chunk,
        H,
        N_STEPS,
        r_escape=R_ESCAPE,
        binary_factor=BINARY_FACTOR,
        r_close=R_CLOSE,
    )
    jax.block_until_ready(out["label"])
    return {
        "label": np.asarray(out["label"]),
        "escape_time": np.asarray(out["escape_time"]),
        "ambiguous": np.asarray(out["ambiguous"]),
        "r_min_ever": np.asarray(out["r_min_ever"]),
        "max_sep": np.asarray(out["max_sep"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=DEFAULT_N_ICS,
                        help=f"number of initial conditions to sample (default {DEFAULT_N_ICS})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"PRNG seed (default {DEFAULT_SEED:#x})")
    parser.add_argument("--chunk", type=int, default=CHUNK)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    print(f"jax backend : {jax.default_backend()}")
    print(f"devices     : {jax.devices()}")
    print(f"N ICs       : {args.n:,}")
    print(f"h           : {H}")
    print(f"t_max       : {T_MAX}")
    print(f"n_steps     : {N_STEPS}")
    print(f"chunk size  : {args.chunk}")
    print(f"r_escape    : {R_ESCAPE}")
    print(f"binary_fac  : {BINARY_FACTOR}")
    print(f"r_close     : {R_CLOSE}")
    print()

    key = jax.random.PRNGKey(args.seed)
    t_sample = time.perf_counter()
    shape_pts_full = sample_shape_sphere(key, args.n)
    configs_full = shape_to_config(shape_pts_full, inertia=1.0)
    shape_pts_full.block_until_ready()
    print(f"sampled {args.n:,} shape-sphere points in {time.perf_counter() - t_sample:.2f}s")

    p0_full = jnp.zeros_like(configs_full)

    # Process in chunks so the jitted kernel compiles once for a fixed shape.
    # Pad final chunk to a power-of-2 if needed so JIT does not recompile.
    n_chunks = (args.n + args.chunk - 1) // args.chunk
    print(f"running {n_chunks} chunks of up to {args.chunk}")
    print()

    all_labels = np.empty(args.n, dtype=np.int8)
    all_etimes = np.empty(args.n, dtype=np.float64)
    all_amb = np.empty(args.n, dtype=np.bool_)
    all_rmin = np.empty(args.n, dtype=np.float64)
    all_maxsep = np.empty(args.n, dtype=np.float64)

    t_total = time.perf_counter()
    for i in range(n_chunks):
        lo = i * args.chunk
        hi = min(lo + args.chunk, args.n)
        actual = hi - lo

        configs_chunk = configs_full[lo:hi]
        p0_chunk = p0_full[lo:hi]

        # Pad to args.chunk to keep jit static shape — pad with copies of first IC
        if actual < args.chunk:
            pad_n = args.chunk - actual
            configs_chunk = jnp.concatenate(
                [configs_chunk, jnp.repeat(configs_chunk[:1], pad_n, axis=0)], axis=0
            )
            p0_chunk = jnp.concatenate(
                [p0_chunk, jnp.repeat(p0_chunk[:1], pad_n, axis=0)], axis=0
            )

        t0 = time.perf_counter()
        chunk_out = run_chunk(configs_chunk, p0_chunk)
        dt = time.perf_counter() - t0

        # Trim padding
        for k in ("label", "escape_time", "ambiguous", "r_min_ever", "max_sep"):
            chunk_out[k] = chunk_out[k][:actual]

        all_labels[lo:hi] = chunk_out["label"]
        all_etimes[lo:hi] = chunk_out["escape_time"]
        all_amb[lo:hi] = chunk_out["ambiguous"]
        all_rmin[lo:hi] = chunk_out["r_min_ever"]
        all_maxsep[lo:hi] = chunk_out["max_sep"]

        throughput = actual * N_STEPS / dt
        print(
            f"  chunk {i+1:3d}/{n_chunks}  [{lo:>10,}:{hi:>10,}]  "
            f"{dt:6.2f}s  {throughput/1e6:6.2f} Mstep/s"
        )

    total_dt = time.perf_counter() - t_total
    total_throughput = args.n * N_STEPS / total_dt
    print()
    print(f"=== integration complete ===")
    print(f"  total time     : {total_dt:.2f}s ({total_dt/60:.2f} min)")
    print(f"  total throughput: {total_throughput/1e6:.2f} M traj-steps/sec")
    print()

    # Outcome summary
    print("=== outcome summary ===")
    unique, counts = np.unique(all_labels, return_counts=True)
    total = all_labels.size
    for lab, cnt in zip(unique, counts):
        name = LABEL_NAMES[int(lab)]
        pct = 100 * cnt / total
        print(f"  {name:16s} ({int(lab):+d})  : {cnt:10,}  ({pct:5.2f}%)")
    print(f"  ambiguous fraction : {all_amb.mean():.4%}")
    print(f"  min r_min_ever     : {all_rmin.min():.3e}")
    print(f"  max r_min_ever     : {all_rmin.max():.3e}")
    print()

    # Write parquet
    shape_np = np.asarray(shape_pts_full)
    df = pd.DataFrame({
        "shape_n1": shape_np[:, 0],
        "shape_n2": shape_np[:, 1],
        "shape_n3": shape_np[:, 2],
        "label": all_labels,
        "escape_time": all_etimes,
        "ambiguous": all_amb,
        "r_min_ever": all_rmin,
        "max_sep": all_maxsep,
    })

    out_dir = PROJECT_ROOT / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.out:
        out_path = Path(args.out)
    else:
        tag = f"{args.n}".replace("000000", "M").replace("000", "k")
        out_path = out_dir / f"phase2_basin_{tag}.parquet"

    table = pa.Table.from_pandas(df)
    pq.write_table(table, out_path, compression="snappy")
    print(f"saved {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
