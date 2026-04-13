"""Validate pipeline with known Suvakov-Dmitrasinovic periodic orbits.

These 15 ICs correspond to 13 distinct orbit families.
Convention: isosceles collinear start.
  body1 = (-1, 0), body2 = (1, 0), body3 = (0, 0)
  body1 vel = body2 vel = (v1, v2)
  body3 vel = (-2*v1, -2*v2)

These are NOT free-fall (p != 0). We integrate with Yoshida-6
and check if ||state(T) - state(0)|| -> 0.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from functools import partial

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from mega3bp.integrators import yoshida6_step
from mega3bp.dynamics import total_energy, angular_momentum
from mega3bp.shape_sphere import config_to_shape

OUT_DIR = Path(__file__).parent

SUVAKOV_ORBITS = [
    {"name": "Figure-eight",
     "v1": 0.3471168881, "v2": 0.5327249454,
     "T": 6.3259139829, "E": -1.287141996},
    {"name": "Butterfly I",
     "v1": 0.3068934205, "v2": 0.1255065670,
     "T": 6.2346748391, "E": -2.170193590},
    {"name": "Butterfly II",
     "v1": 0.392955223941802, "v2": 0.0975792352080344,
     "T": 7.003707, "E": -2.008193454},
    {"name": "Butterfly III",
     "v1": 0.4059155671, "v2": 0.2301631260,
     "T": 13.8671234361, "E": -1.846772463},
    {"name": "Moth I",
     "v1": 0.4644451728, "v2": 0.3960600146,
     "T": 14.8943051743, "E": -1.382281439},
    {"name": "Moth II",
     "v1": 0.4391659182, "v2": 0.4529676431,
     "T": 28.6692709402, "E": -1.305860832},
    {"name": "Moth III",
     "v1": 0.3834435199, "v2": 0.3773636946,
     "T": 25.8392363356, "E": -1.631703127},
    {"name": "Bumblebee",
     "v1": 0.1842784887, "v2": 0.5871881740,
     "T": 63.5343529785, "E": -1.363754461},
    {"name": "Dragonfly",
     "v1": 0.0805842255, "v2": 0.5888360898,
     "T": 21.2723373956, "E": -1.440334726},
    {"name": "Goggles",
     "v1": 0.0833000718, "v2": 0.1278892555,
     "T": 10.4648495256, "E": -2.430116309},
    {"name": "Yarn",
     "v1": 0.559064247131347, "v2": 0.349191558837891,
     "T": 14.894307, "E": -1.196537268},
    {"name": "Yin-Yang Ia",
     "v1": 0.513938054919243, "v2": 0.304736003875733,
     "T": 17.32881, "E": -1.429010931},
    {"name": "Yin-Yang Ib",
     "v1": 0.2826986823, "v2": 0.3272087861,
     "T": 10.9633031497, "E": -1.939047596},
    {"name": "Yin-Yang IIa",
     "v1": 0.4168220336, "v2": 0.3303332949,
     "T": 55.7893112814, "E": -1.651417920},
    {"name": "Yin-Yang IIb",
     "v1": 0.417342877101898, "v2": 0.313100116109848,
     "T": 54.208001, "E": -1.683379721},
]


def make_ic(orbit):
    """Convert Suvakov convention to our (q, p) format."""
    v1, v2 = orbit["v1"], orbit["v2"]
    q = jnp.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 0.0]], dtype=jnp.float64)
    p = jnp.array([[v1, v2], [v1, v2], [-2*v1, -2*v2]], dtype=jnp.float64)
    return q, p


@partial(jax.jit, static_argnames=("n_steps",))
def integrate_and_check(q0, p0, T, n_steps):
    """Integrate for time T and return closure residual."""
    h = T / n_steps

    def step_fn(carry, _):
        q, p = carry
        q, p = yoshida6_step(q, p, jnp.float64(h))
        return (q, p), None

    (qf, pf), _ = jax.lax.scan(step_fn, (q0, p0), None, length=n_steps)

    dq = jnp.linalg.norm((qf - q0).ravel())
    dp = jnp.linalg.norm((pf - p0).ravel())
    total = jnp.linalg.norm(jnp.concatenate([(qf - q0).ravel(), (pf - p0).ravel()]))

    E0 = total_energy(q0, p0)
    Ef = total_energy(qf, pf)
    dE = jnp.abs((Ef - E0) / E0)

    return total, dq, dp, dE


def main():
    print("=" * 70)
    print("VALIDATION: Known Suvakov-Dmitrasinovic Periodic Orbits")
    print("Testing Yoshida-6 integrator closure on 15 known orbits")
    print("=" * 70)

    results = []

    print(f"\n{'Name':<20} {'T':>10} {'n_steps':>8} {'||F||':>12} "
          f"{'||dq||':>12} {'||dp||':>12} {'|dE/E|':>12} {'Pass?':>6}")
    print("-" * 95)

    for orbit in SUVAKOV_ORBITS:
        q0, p0 = make_ic(orbit)
        T = orbit["T"]

        # Try increasing n_steps until convergence or give up
        best_total = float('inf')
        best_result = None

        for n_steps in [10_000, 20_000, 50_000, 100_000]:
            total, dq, dp, dE = integrate_and_check(q0, p0, T, n_steps)
            total, dq, dp, dE = float(total), float(dq), float(dp), float(dE)

            if total < best_total:
                best_total = total
                best_result = {
                    "name": orbit["name"],
                    "T": T,
                    "n_steps": n_steps,
                    "h": T / n_steps,
                    "total_residual": total,
                    "dq_norm": dq,
                    "dp_norm": dp,
                    "dE_rel": dE,
                    "pass_1e8": total < 1e-8,
                    "pass_1e6": total < 1e-6,
                }

            if total < 1e-8:
                break

        r = best_result
        passed = "YES" if r["pass_1e8"] else ("near" if r["pass_1e6"] else "NO")
        print(f"{r['name']:<20} {r['T']:10.4f} {r['n_steps']:8d} {r['total_residual']:12.4e} "
              f"{r['dq_norm']:12.4e} {r['dp_norm']:12.4e} {r['dE_rel']:12.4e} {passed:>6}")

        results.append(r)

    # Summary
    n_pass_8 = sum(1 for r in results if r["pass_1e8"])
    n_pass_6 = sum(1 for r in results if r["pass_1e6"])
    print(f"\n{'='*70}")
    print(f"SUMMARY: {n_pass_8}/15 pass at 1e-8, {n_pass_6}/15 pass at 1e-6")
    print(f"{'='*70}")

    if n_pass_8 > 0:
        print("\nYoshida-6 CAN close known periodic orbits at double precision.")
        print("The problem is candidate selection, not integration accuracy.")
    else:
        print(f"\nBest residual: {min(r['total_residual'] for r in results):.2e}")
        if n_pass_6 > 0:
            print("Orbits close to 1e-6 but not 1e-8. Higher precision may help.")
        else:
            print("Integration accuracy may be limiting. Consider higher-order method.")

    # Also compute shape sphere points for these orbits (for Step 2: entropy mapping)
    print(f"\n{'='*70}")
    print("Shape sphere points for known orbits (for entropy mapping):")
    print(f"{'='*70}")
    for orbit in SUVAKOV_ORBITS:
        q0, _ = make_ic(orbit)
        n = config_to_shape(q0)
        print(f"  {orbit['name']:<20} n = ({float(n[0]):8.5f}, {float(n[1]):8.5f}, {float(n[2]):8.5f})")

    # Save results
    with open(OUT_DIR / "known_orbit_validation.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {OUT_DIR / 'known_orbit_validation.json'}")


if __name__ == "__main__":
    main()
