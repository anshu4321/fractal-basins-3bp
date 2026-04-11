"""Phase 1 correctness gate.

Integrate the Chenciner-Montgomery figure-eight orbit for 100 periods using
Yoshida 6th-order symplectic and assert that the relative energy error stays
below 1e-10 with no secular drift. If this fails, Phase 2 is blocked.

Usage:
    cd /workspace/3bp
    venv/bin/python scripts/run_figure_eight_gate.py
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

from mega3bp.dynamics import angular_momentum, total_energy
from mega3bp.integrators import yoshida6_integrate_energy_only
from mega3bp.orbits import FIGURE_EIGHT_PERIOD, figure_eight_state

GATE = 1e-10


def main() -> int:
    print(f"jax backend : {jax.default_backend()}")
    print(f"devices     : {jax.devices()}")
    print(f"fp64 enabled: {jax.config.read('jax_enable_x64')}")
    print()

    q0, p0 = figure_eight_state()
    E0 = float(total_energy(q0, p0))
    L0 = float(angular_momentum(q0, p0))
    print(f"E(0) = {E0:.15e}")
    print(f"L(0) = {L0:.15e}")
    print(f"COM position = {jnp.sum(q0, axis=0)}")
    print(f"total momentum = {jnp.sum(p0, axis=0)}")
    print()

    h = 0.005
    n_periods = 100
    t_final = n_periods * FIGURE_EIGHT_PERIOD
    n_steps = int(round(t_final / h))
    print(f"integrator  : Yoshida 6th-order (7-stage symmetric composition)")
    print(f"step size h : {h}")
    print(f"n_steps     : {n_steps}")
    print(f"t_final     : {n_steps * h:.6f}  ({n_periods} periods)")
    print()

    # Warm-up (compile + run with tiny n_steps) to separate compile time
    print("compiling...")
    t0 = time.perf_counter()
    q_warm, p_warm, e_warm = yoshida6_integrate_energy_only(q0, p0, h, 10)
    e_warm.block_until_ready()
    print(f"  compile + warmup: {time.perf_counter() - t0:.3f}s")

    # Main run
    print("integrating...")
    t0 = time.perf_counter()
    q_final, p_final, energies = yoshida6_integrate_energy_only(q0, p0, h, n_steps)
    energies.block_until_ready()
    elapsed = time.perf_counter() - t0
    print(f"  run time: {elapsed:.3f}s  ({n_steps / elapsed:,.0f} steps/sec)")
    print()

    # Energy diagnostics
    rel_err = (energies - E0) / E0
    max_abs = float(jnp.max(jnp.abs(rel_err)))
    final_abs = float(jnp.abs(rel_err[-1]))

    # Drift check: fit linear slope over second half of trace
    half = len(rel_err) // 2
    x = jnp.arange(half, len(rel_err), dtype=jnp.float64)
    y = rel_err[half:]
    slope, intercept = jnp.polyfit(x, y, 1)
    drift_over_100p = float(slope) * len(rel_err)
    print(f"initial E       : {E0:.15e}")
    print(f"final   E       : {float(energies[-1]):.15e}")
    print(f"max |dE/E|      : {max_abs:.3e}")
    print(f"final |dE/E|    : {final_abs:.3e}")
    print(f"envelope slope (2nd half, per step) : {float(slope):+.3e}")
    print(f"envelope drift projected to 100T    : {drift_over_100p:+.3e}")
    print()

    # Invariants should still be zero
    L_final = float(angular_momentum(q_final, p_final))
    p_tot_final = jnp.sum(p_final, axis=0)
    print(f"L(final) = {L_final:.3e}   (should be ~0)")
    print(f"p_tot(final) = {p_tot_final}")
    print()

    gate_pass = max_abs < GATE
    drift_pass = abs(drift_over_100p) < GATE

    if gate_pass and drift_pass:
        print(f"PASS: max |dE/E| = {max_abs:.3e}  <  gate {GATE:.0e}")
        print(f"PASS: no secular drift (projected |drift| = {abs(drift_over_100p):.3e} < {GATE:.0e})")
        return 0
    else:
        if not gate_pass:
            print(f"FAIL: max |dE/E| = {max_abs:.3e}  >=  gate {GATE:.0e}")
        if not drift_pass:
            print(f"FAIL: secular drift detected ({drift_over_100p:+.3e} over 100T, gate {GATE:.0e})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
