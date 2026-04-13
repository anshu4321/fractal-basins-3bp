"""Experiment 02: MGOY uncertainty exponent measurement.

McDonald-Grebogi-Ott-Yorke (1985) algorithm:
  1. Sample N_pair random ICs uniformly on the phase space.
  2. For each IC x, generate x' = x + ε·u_hat (random tangent direction).
  3. Integrate both. Call the pair "uncertain" if final labels disagree.
  4. Measure f(ε) = fraction of uncertain pairs.
  5. Repeat for multiple ε values spanning ~2.5 decades.
  6. Fit log f(ε) = α log ε + const.
  7. Uncertainty dimension: D_u = d - α.

Runs in both 2D (at-rest shape sphere) and 6D (full phase space).

Outputs:
    results/02_uncertainty_exponent/2d.json
    results/02_uncertainty_exponent/6d.json
    figures/02_uncertainty_fit.png / .pdf
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from mega3bp.batch import integrate_batch_diag
from mega3bp.shape_sphere import (
    sample_momentum_4ball,
    sample_shape_sphere,
    shape_to_config,
    jacobi_momenta_to_body,
)
from mega3bp.style import PALETTE, apply_dirac_style

N_PAIRS = 100_000
SEED = 0x4D474F59  # "MGOY"
H = 0.01
T_MAX = 500.0
N_STEPS = int(round(T_MAX / H))
R_ESCAPE = 5.0
BINARY_FACTOR = 2.0
R_CLOSE = 1e-3
CHUNK = 50_000
R_MIN_THRESHOLD = 0.01

EPSILONS_2D = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
EPSILONS_6D = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]


def perturb_on_sphere(pts, epsilon, key):
    """Tangent-plane perturbation on S^2, then re-project."""
    noise = jax.random.normal(key, pts.shape, dtype=jnp.float64)
    dot = jnp.sum(noise * pts, axis=-1, keepdims=True)
    tangent = noise - dot * pts
    tangent = tangent / jnp.linalg.norm(tangent, axis=-1, keepdims=True)
    moved = pts + epsilon * tangent
    return moved / jnp.linalg.norm(moved, axis=-1, keepdims=True)


def perturb_6d(shape_pts, pi_jacobi, epsilon, key):
    """Perturb in 6D: tangent-plane on S^2 + scaled direction in R^4."""
    k1, k2 = jax.random.split(key)
    shape_new = perturb_on_sphere(shape_pts, epsilon, k1)
    noise_m = jax.random.normal(k2, pi_jacobi.shape, dtype=jnp.float64)
    noise_m = noise_m / jnp.linalg.norm(noise_m, axis=-1, keepdims=True)
    pi_new = pi_jacobi + epsilon * noise_m
    return shape_new, pi_new


def integrate_and_label(q, p, chunk_size=CHUNK):
    """Integrate, return labels. Exclude close encounters."""
    N = q.shape[0]
    n_chunks = (N + chunk_size - 1) // chunk_size
    all_labels = np.empty(N, dtype=np.int8)
    all_rmin = np.empty(N, dtype=np.float64)

    for i in range(n_chunks):
        lo = i * chunk_size
        hi = min(lo + chunk_size, N)
        actual = hi - lo
        q_c, p_c = q[lo:hi], p[lo:hi]
        if actual < chunk_size:
            pad = chunk_size - actual
            q_c = jnp.concatenate([q_c, jnp.repeat(q_c[:1], pad, axis=0)])
            p_c = jnp.concatenate([p_c, jnp.repeat(p_c[:1], pad, axis=0)])
        out = integrate_batch_diag(q_c, p_c, H, N_STEPS,
                                   r_escape=R_ESCAPE, binary_factor=BINARY_FACTOR,
                                   r_close=R_CLOSE, strict_escape=True)
        jax.block_until_ready(out["label"])
        all_labels[lo:hi] = np.asarray(out["label"])[:actual]
        all_rmin[lo:hi] = np.asarray(out["r_min_ever"])[:actual]
    return all_labels, all_rmin


def run_mgoy(name, shape_pts, pi_jacobi, epsilons, key, is_6d=False):
    """Run the full MGOY algorithm for one dimensionality."""
    N = shape_pts.shape[0]

    if is_6d:
        q = jax.vmap(lambda n: shape_to_config(n, inertia=1.0))(shape_pts)
        p = jax.vmap(lambda pi: jacobi_momenta_to_body(pi[:2], pi[2:]))(pi_jacobi)
    else:
        q = shape_to_config(shape_pts, inertia=1.0)
        p = jnp.zeros_like(q)

    print(f"  integrating {N} base ICs...")
    t0 = time.perf_counter()
    labels_base, rmin_base = integrate_and_label(q, p)
    print(f"  base done in {time.perf_counter() - t0:.1f}s")

    reliable = rmin_base >= R_MIN_THRESHOLD
    n_reliable = int(reliable.sum())
    print(f"  reliable (r_min >= {R_MIN_THRESHOLD}): {n_reliable}/{N} ({100*n_reliable/N:.1f}%)")

    results = {"name": name, "n_pairs": N, "n_reliable": n_reliable,
               "epsilons": [], "f_uncertain": [], "n_uncertain": [],
               "n_both_reliable": []}

    for eps in epsilons:
        key, subkey = jax.random.split(key)
        t0 = time.perf_counter()

        if is_6d:
            shape_pert, pi_pert = perturb_6d(shape_pts, pi_jacobi, eps, subkey)
            q_pert = jax.vmap(lambda n: shape_to_config(n, inertia=1.0))(shape_pert)
            p_pert = jax.vmap(lambda pi: jacobi_momenta_to_body(pi[:2], pi[2:]))(pi_pert)
        else:
            shape_pert = perturb_on_sphere(shape_pts, eps, subkey)
            q_pert = shape_to_config(shape_pert, inertia=1.0)
            p_pert = jnp.zeros_like(q_pert)

        labels_pert, rmin_pert = integrate_and_label(q_pert, p_pert)
        dt = time.perf_counter() - t0

        both_reliable = reliable & (rmin_pert >= R_MIN_THRESHOLD)
        n_both = int(both_reliable.sum())
        disagree = (labels_base != labels_pert) & both_reliable
        n_disagree = int(disagree.sum())
        f_unc = n_disagree / max(n_both, 1)

        results["epsilons"].append(float(eps))
        results["f_uncertain"].append(float(f_unc))
        results["n_uncertain"].append(n_disagree)
        results["n_both_reliable"].append(n_both)

        print(f"  ε={eps:.1e}  f(ε)={f_unc:.4f}  "
              f"({n_disagree}/{n_both} disagree)  [{dt:.1f}s]")

    log_eps = np.log(np.array(results["epsilons"]))
    log_f = np.log(np.array(results["f_uncertain"]))

    valid = np.isfinite(log_f) & (np.array(results["f_uncertain"]) > 0)
    if valid.sum() >= 3:
        slope, intercept = np.polyfit(log_eps[valid], log_f[valid], 1)
        residuals = log_f[valid] - (slope * log_eps[valid] + intercept)
        n_jack = int(valid.sum())
        jack_slopes = []
        for j in range(n_jack):
            mask_j = np.ones(n_jack, dtype=bool)
            mask_j[j] = False
            le = log_eps[valid][mask_j]
            lf = log_f[valid][mask_j]
            s_j, _ = np.polyfit(le, lf, 1)
            jack_slopes.append(s_j)
        jack_slopes = np.array(jack_slopes)
        jack_mean = jack_slopes.mean()
        jack_var = (n_jack - 1) / n_jack * np.sum((jack_slopes - jack_mean)**2)
        jack_2sigma = 2 * np.sqrt(jack_var)

        results["alpha"] = float(slope)
        results["alpha_2sigma"] = float(jack_2sigma)
        results["intercept"] = float(intercept)
        results["r_squared"] = float(1 - np.sum(residuals**2) / np.sum((log_f[valid] - log_f[valid].mean())**2))

        d = 6 if is_6d else 2
        results["D_uncertainty"] = float(d - slope)

        print(f"\n  α = {slope:.4f} ± {jack_2sigma:.4f} (2σ jackknife)")
        print(f"  D_u = {d} - {slope:.4f} = {d - slope:.4f}")
        print(f"  R² = {results['r_squared']:.4f}")
    else:
        results["alpha"] = None
        results["alpha_2sigma"] = None
        print("  insufficient valid points for fit")

    return results


def main() -> int:
    apply_dirac_style()

    out_results = PROJECT_ROOT / "results" / "02_uncertainty_exponent"
    out_figs = PROJECT_ROOT / "figures"
    out_results.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    print(f"jax backend: {jax.default_backend()}")
    print(f"N_pairs: {N_PAIRS:,}")
    print()

    key = jax.random.PRNGKey(SEED)

    # ---- 2D at-rest ----
    print("=== 2D at-rest shape sphere ===")
    key, k1 = jax.random.split(key)
    shape_2d = sample_shape_sphere(k1, N_PAIRS)
    key, k2 = jax.random.split(key)
    res_2d = run_mgoy("2d", shape_2d, None, EPSILONS_2D, k2, is_6d=False)

    with open(out_results / "2d.json", "w") as f:
        json.dump(res_2d, f, indent=2)
    print()

    # ---- 6D ----
    print("=== 6D full phase space ===")
    key, k3 = jax.random.split(key)
    shape_6d = sample_shape_sphere(k3, N_PAIRS)
    key, k4 = jax.random.split(key)
    pi_6d = sample_momentum_4ball(k4, N_PAIRS, p_max=3.0)
    key, k5 = jax.random.split(key)
    res_6d = run_mgoy("6d", shape_6d, pi_6d, EPSILONS_6D, k5, is_6d=True)

    with open(out_results / "6d.json", "w") as f:
        json.dump(res_6d, f, indent=2)
    print()

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    fig.patch.set_facecolor(PALETTE["bg_deep"])

    for ax, res, dim_label, d in [(axes[0], res_2d, "2D at-rest", 2),
                                   (axes[1], res_6d, "6D phase space", 6)]:
        eps_arr = np.array(res["epsilons"])
        f_arr = np.array(res["f_uncertain"])
        valid = f_arr > 0

        ax.plot(eps_arr[valid], f_arr[valid], "o", color=PALETTE["cyan"], ms=8, zorder=5)

        if res["alpha"] is not None:
            eps_fit = np.logspace(np.log10(eps_arr[valid].min()),
                                  np.log10(eps_arr[valid].max()), 50)
            f_fit = np.exp(res["intercept"]) * eps_fit ** res["alpha"]
            ax.plot(eps_fit, f_fit, "--", color=PALETTE["lime"], lw=2,
                    label=rf"$\alpha = {res['alpha']:.3f} \pm {res['alpha_2sigma']:.3f}$")
            ax.set_title(f"{dim_label}: $D_u = {d} - {res['alpha']:.3f} = {d - res['alpha']:.3f}$",
                         color=PALETTE["text"], fontsize=13)
        else:
            ax.set_title(f"{dim_label}: fit failed", color=PALETTE["coral"])

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"perturbation $\varepsilon$")
        ax.set_ylabel(r"uncertain fraction $f(\varepsilon)$")
        ax.legend(fontsize=11)

    fig.suptitle("MGOY uncertainty exponent measurement",
                 color=PALETTE["text"], fontsize=15, y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_figs / f"02_uncertainty_fit.{ext}",
                    facecolor=PALETTE["bg_deep"], bbox_inches="tight")
    plt.close(fig)
    print("saved figures/02_uncertainty_fit.png/pdf")

    # ---- Summary ----
    print("\n=== Summary ===")
    if res_2d["alpha"] is not None:
        print(f"  2D: α = {res_2d['alpha']:.4f} ± {res_2d['alpha_2sigma']:.4f}")
        print(f"       D_u = {2 - res_2d['alpha']:.4f}")
        print(f"       (box-counting gave D = 1.592, predicts α = 0.408)")
    if res_6d["alpha"] is not None:
        print(f"  6D: α = {res_6d['alpha']:.4f} ± {res_6d['alpha_2sigma']:.4f}")
        print(f"       D_u = {6 - res_6d['alpha']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
