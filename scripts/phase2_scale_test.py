"""Phase 2 scaling benchmark.

Runs the batch integrator at increasing batch sizes and measures throughput
so we can verify (a) memory scaling is linear, (b) per-step cost decreases
as GPU-launch overhead amortizes over more parallel work, and (c) outcome
statistics stabilize as sample size grows.

Deterministic seeds so runs are comparable.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from mega3bp.batch import integrate_batch
from mega3bp.escape import LABEL_NAMES
from mega3bp.shape_sphere import sample_shape_sphere, shape_to_config

BATCH_SIZES = [1024, 8192, 65536]
H = 0.01
T_MAX = 500.0
N_STEPS = int(round(T_MAX / H))
R_ESCAPE = 5.0
BINARY_FACTOR = 2.0
R_CLOSE = 1e-3
SEED = 0xC0FFEE


def main() -> int:
    print(f"jax backend : {jax.default_backend()}")
    print(f"devices     : {jax.devices()}")
    print(f"n_steps     : {N_STEPS}  (h={H}, t_max={T_MAX})")
    print()

    results: list[dict] = []
    for B in BATCH_SIZES:
        print(f"=== B = {B} ===")
        key = jax.random.PRNGKey(SEED)
        shape_pts = sample_shape_sphere(key, B)
        configs = shape_to_config(shape_pts, inertia=1.0)
        p0 = jnp.zeros_like(configs)

        # Warm up (jit compile)
        print("  compiling...")
        t0 = time.perf_counter()
        warm = integrate_batch(
            configs[:128], p0[:128], H, 50,
            r_escape=R_ESCAPE, binary_factor=BINARY_FACTOR, r_close=R_CLOSE,
        )
        jax.block_until_ready(warm["label"])
        print(f"    {time.perf_counter() - t0:.2f}s")

        print("  running full batch...")
        t0 = time.perf_counter()
        out = integrate_batch(
            configs, p0, H, N_STEPS,
            r_escape=R_ESCAPE, binary_factor=BINARY_FACTOR, r_close=R_CLOSE,
        )
        jax.block_until_ready(out["label"])
        dt = time.perf_counter() - t0

        labels = np.asarray(out["label"])
        amb_frac = float(np.asarray(out["ambiguous"]).mean())
        throughput = B * N_STEPS / dt

        counts = {name: 0 for name in ("ambiguous", "bound", "body1_escape", "body2_escape", "body3_escape")}
        for lab, cnt in zip(*np.unique(labels, return_counts=True)):
            counts[LABEL_NAMES[int(lab)]] = int(cnt)
        total = labels.size

        print(f"  elapsed     : {dt:.2f}s")
        print(f"  throughput  : {throughput / 1e6:.2f} M traj-steps/sec")
        print(f"  amb fraction: {amb_frac:.3%}")
        for name, cnt in counts.items():
            if cnt:
                print(f"    {name:16s}: {cnt:7d}  ({100 * cnt / total:5.2f}%)")

        results.append({"B": B, "dt": dt, "throughput": throughput, "counts": counts, "amb": amb_frac})
        print()

    # Summary
    print("=== summary ===")
    print(f"{'B':>8}  {'dt[s]':>9}  {'Mstep/s':>9}  {'esc %':>7}  {'amb %':>6}")
    for r in results:
        esc_pct = 100 * (r["counts"]["body1_escape"] + r["counts"]["body2_escape"] + r["counts"]["body3_escape"]) / r["B"]
        print(
            f"{r['B']:>8}  {r['dt']:>9.2f}  {r['throughput']/1e6:>9.2f}  {esc_pct:>6.2f}%  {r['amb']*100:>5.2f}%"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
