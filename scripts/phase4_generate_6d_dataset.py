"""Phase 4: 6D dataset generation (non-zero momentum).

Samples N ICs in the full 6D phase space: shape sphere (theta, phi) for
positions + 4 Jacobi momentum components (pi1x, pi1y, pi2x, pi2y).
Momenta sampled uniformly in a 4-ball of radius p_max.  No angular-momentum
constraint — L is recorded as a feature.

Schema:
    shape_n1, shape_n2, shape_n3   float64  IC on the shape sphere
    pi1x, pi1y, pi2x, pi2y        float64  Jacobi momenta
    energy                         float64  total energy E = T + V
    angular_momentum               float64  L_z
    label                          int8     -1 ambiguous / 0 bound / 1-3 escape
    escape_time                    float64  inf if bound
    ambiguous                      bool
    r_min_ever                     float64
    max_sep                        float64
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
import pyarrow as pa
import pyarrow.parquet as pq

from mega3bp.batch import integrate_batch
from mega3bp.dynamics import angular_momentum, total_energy
from mega3bp.escape import LABEL_NAMES
from mega3bp.shape_sphere import sample_phase_space_6d

DEFAULT_N_ICS = 1_048_576
DEFAULT_SEED = 0xDEAD6D01

H = 0.01
T_MAX = 500.0
N_STEPS = int(round(T_MAX / H))
R_ESCAPE = 5.0
BINARY_FACTOR = 2.0
R_CLOSE = 1e-3
CHUNK = 65536


def run_chunk(q_chunk, p_chunk) -> dict[str, np.ndarray]:
    out = integrate_batch(
        q_chunk, p_chunk, H, N_STEPS,
        r_escape=R_ESCAPE, binary_factor=BINARY_FACTOR, r_close=R_CLOSE,
    )
    jax.block_until_ready(out["label"])
    return {k: np.asarray(out[k]) for k in
            ("label", "escape_time", "ambiguous", "r_min_ever", "max_sep")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=DEFAULT_N_ICS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--p_max", type=float, default=3.0,
                        help="radius of 4-ball for momentum sampling")
    parser.add_argument("--chunk", type=int, default=CHUNK)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    print(f"jax backend : {jax.default_backend()}")
    print(f"devices     : {jax.devices()}")
    print(f"N ICs       : {args.n:,}")
    print(f"p_max       : {args.p_max}")
    print(f"h / t_max   : {H} / {T_MAX}  ({N_STEPS} steps)")
    print(f"chunk size  : {args.chunk}")
    print()

    key = jax.random.PRNGKey(args.seed)
    t0 = time.perf_counter()
    shape_pts, pi_jacobi, q_full, p_full = sample_phase_space_6d(
        key, args.n, p_max=args.p_max,
    )
    jax.block_until_ready(p_full)
    print(f"sampled {args.n:,} 6D ICs in {time.perf_counter() - t0:.2f}s")

    E = np.asarray(total_energy(q_full, p_full))
    L = np.asarray(angular_momentum(q_full, p_full))
    print(f"energy   : mean={E.mean():.3f}  std={E.std():.3f}  "
          f"min={E.min():.3f}  max={E.max():.3f}")
    print(f"E < 0    : {(E < 0).mean():.2%}")
    print(f"|L| mean : {np.abs(L).mean():.3f}  std={L.std():.3f}")
    print()

    n_chunks = (args.n + args.chunk - 1) // args.chunk
    print(f"integrating {n_chunks} chunks ...")

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

        q_c = q_full[lo:hi]
        p_c = p_full[lo:hi]

        if actual < args.chunk:
            pad = args.chunk - actual
            q_c = jnp.concatenate([q_c, jnp.repeat(q_c[:1], pad, axis=0)])
            p_c = jnp.concatenate([p_c, jnp.repeat(p_c[:1], pad, axis=0)])

        tc = time.perf_counter()
        out = run_chunk(q_c, p_c)
        dt = time.perf_counter() - tc

        for k in ("label", "escape_time", "ambiguous", "r_min_ever", "max_sep"):
            out[k] = out[k][:actual]

        all_labels[lo:hi] = out["label"]
        all_etimes[lo:hi] = out["escape_time"]
        all_amb[lo:hi] = out["ambiguous"]
        all_rmin[lo:hi] = out["r_min_ever"]
        all_maxsep[lo:hi] = out["max_sep"]

        tput = actual * N_STEPS / dt
        print(f"  chunk {i+1:3d}/{n_chunks}  [{lo:>10,}:{hi:>10,}]  "
              f"{dt:6.2f}s  {tput/1e6:6.2f} Mstep/s")

    wall = time.perf_counter() - t_total
    print(f"\n=== integration complete ({wall:.1f}s, {wall/60:.1f} min) ===")
    print(f"  throughput: {args.n * N_STEPS / wall / 1e6:.2f} Mstep/s\n")

    print("=== outcome summary ===")
    unique, counts = np.unique(all_labels, return_counts=True)
    for lab, cnt in zip(unique, counts):
        print(f"  {LABEL_NAMES[int(lab)]:16s} ({int(lab):+d})  : "
              f"{cnt:10,}  ({100*cnt/args.n:5.2f}%)")
    print(f"  ambiguous fraction : {all_amb.mean():.4%}")
    print()

    shape_np = np.asarray(shape_pts)
    pi_np = np.asarray(pi_jacobi)

    table = pa.table({
        "shape_n1": shape_np[:, 0],
        "shape_n2": shape_np[:, 1],
        "shape_n3": shape_np[:, 2],
        "pi1x": pi_np[:, 0],
        "pi1y": pi_np[:, 1],
        "pi2x": pi_np[:, 2],
        "pi2y": pi_np[:, 3],
        "energy": E,
        "angular_momentum": L,
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
        out_path = out_dir / f"phase4_6d_basin_{tag}_pmax{args.p_max:.1f}.parquet"

    pq.write_table(table, out_path, compression="snappy")
    print(f"saved {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
