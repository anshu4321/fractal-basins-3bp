"""Tier 1 orchestrator: 10-seed × 2-method ablation for the ML prior claim.

Spawns 4 concurrent subprocess workers that each run one (seed, method)
via run_single_seed_method.py. Reports live progress to
tier1_ablation/status.json using mega3bp.gcp_telemetry.

Usage:
    python tier1_orchestrator.py [--n-seeds 10] [--workers 4]

Subprocesses are launched with XLA_FLAGS+OMP_NUM_THREADS set so each worker
uses ~3 of the 12 CPU cores, keeping total utilization near 100%.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mega3bp.gcp_telemetry import Telemetry  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "tier1_ablation"
RESULTS_DIR = OUT_DIR / "results"
STATUS_PATH = OUT_DIR / "status.json"

METHODS = ["label_disagreement", "classifier_entropy"]


def run_worker(seed: int, method: str, out_dir: str,
               basin_map: str, python_exec: str,
               project_root: str, omp_threads: int) -> dict:
    """Run a single (seed, method) subprocess. Returns result summary.

    Executed in a worker process via ProcessPoolExecutor, but internally uses
    subprocess.run so JAX is fully isolated per job.
    """
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp_threads)
    env["MKL_NUM_THREADS"] = str(omp_threads)
    env["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
    # Be explicit about CPU-only to avoid GPU contention:
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"

    out_path = Path(out_dir) / f"seed{seed}_{method}.json"
    if out_path.exists():
        # Skip already-completed jobs (idempotent resume)
        try:
            return json.loads(out_path.read_text())
        except Exception:
            pass  # fall through and re-run

    script = str(Path(project_root) / "experiments" / "orbit_discovery"
                 / "run_single_seed_method.py")
    cmd = [python_exec, "-u", script,
           "--seed", str(seed),
           "--method", method,
           "--out-dir", out_dir,
           "--basin-map", basin_map]

    t0 = time.perf_counter()
    log_path = Path(out_dir) / f"seed{seed}_{method}.log"
    # Tee subprocess stdout+stderr into a per-job log for debugging
    with open(log_path, "w") as logf:
        logf.write(f"$ {' '.join(cmd)}\n")
        logf.flush()
        proc = subprocess.run(cmd, cwd=project_root, env=env,
                              stdout=logf, stderr=subprocess.STDOUT)
    dt = time.perf_counter() - t0

    if proc.returncode != 0:
        return {
            "seed": seed, "method": method,
            "_error": f"returncode={proc.returncode}",
            "_wall": dt, "_log": str(log_path),
        }

    if not out_path.exists():
        return {
            "seed": seed, "method": method,
            "_error": "worker returned 0 but produced no JSON",
            "_wall": dt, "_log": str(log_path),
        }
    try:
        return json.loads(out_path.read_text())
    except Exception as e:
        return {"seed": seed, "method": method, "_error": f"json parse: {e}",
                "_wall": dt, "_log": str(log_path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--omp-threads", type=int, default=3)
    ap.add_argument("--python-exec", default="python3.10")
    ap.add_argument("--basin-map",
                    default=str(HERE / "basin_map.npz"))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    jobs = [(seed, method)
            for seed in range(args.n_seeds)
            for method in METHODS]
    total = len(jobs)

    t = Telemetry(STATUS_PATH, total=total, stage="init")
    t.update(stage="tier1_running",
             n_seeds=args.n_seeds, methods=METHODS, workers=args.workers,
             omp_threads=args.omp_threads,
             basin_map=args.basin_map,
             label_disagreement_done=0,
             classifier_entropy_done=0,
             median_n_verified_ld=None,
             median_n_verified_ce=None,
             force=True)

    print(f"Tier 1 orchestrator: {total} jobs "
          f"({args.n_seeds} seeds × {len(METHODS)} methods), "
          f"{args.workers} parallel workers, omp_threads={args.omp_threads}",
          flush=True)

    done_count = 0
    per_method_results = {m: [] for m in METHODS}

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        future_to_job = {
            ex.submit(run_worker,
                      seed=seed, method=method,
                      out_dir=str(RESULTS_DIR),
                      basin_map=args.basin_map,
                      python_exec=args.python_exec,
                      project_root=str(PROJECT_ROOT),
                      omp_threads=args.omp_threads): (seed, method)
            for (seed, method) in jobs
        }

        for fut in as_completed(future_to_job):
            seed, method = future_to_job[fut]
            try:
                result = fut.result()
            except Exception as e:
                result = {"seed": seed, "method": method,
                          "_error": f"executor exception: {e}"}

            done_count += 1
            if "_error" in result:
                print(f"[FAIL] seed={seed} method={method}: "
                      f"{result['_error']}", flush=True)
            else:
                per_method_results[method].append(result)
                print(f"[OK {done_count}/{total}] seed={seed} {method}: "
                      f"n_verified={result.get('n_verified')}, "
                      f"hit_rate_verify={result.get('hit_rate_verify', 0):.5f}",
                      flush=True)

            # Median n_verified so far per method
            import numpy as np
            ld_nvs = [r.get("n_verified", 0)
                      for r in per_method_results["label_disagreement"]]
            ce_nvs = [r.get("n_verified", 0)
                      for r in per_method_results["classifier_entropy"]]
            t.update(
                progress=done_count,
                label_disagreement_done=len(ld_nvs),
                classifier_entropy_done=len(ce_nvs),
                median_n_verified_ld=(float(np.median(ld_nvs))
                                      if ld_nvs else None),
                median_n_verified_ce=(float(np.median(ce_nvs))
                                      if ce_nvs else None),
            )

    t.done(stage="complete",
           label_disagreement_done=len(per_method_results["label_disagreement"]),
           classifier_entropy_done=len(per_method_results["classifier_entropy"]))

    print(f"\nAll {total} jobs finished.", flush=True)
    print(f"  label_disagreement:  {len(per_method_results['label_disagreement'])}/{args.n_seeds}")
    print(f"  classifier_entropy:  {len(per_method_results['classifier_entropy'])}/{args.n_seeds}")
    print(f"Results in {RESULTS_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
