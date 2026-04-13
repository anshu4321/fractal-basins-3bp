"""Experiment 02b: Extended MGOY with finer ε resolution.

Re-runs the 2D uncertainty exponent measurement with:
  - ε extended down to 10^{-6}
  - 12 epsilon values for better coverage
  - Automatic saturation detection: fits only the power-law regime
  - Sliding-window fit to identify the onset of saturation

Also re-runs 6D with extended range for consistency.

Outputs:
    results/02_uncertainty_exponent/2d_extended.json
    results/02_uncertainty_exponent/6d_extended.json
    figures/02_uncertainty_fit_extended.png / .pdf
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

from mega3bp.batch import integrate_batch_diag
from mega3bp.shape_sphere import (
    sample_momentum_4ball,
    sample_shape_sphere,
    shape_to_config,
    jacobi_momenta_to_body,
)
from mega3bp.style import PALETTE, apply_dirac_style

N_PAIRS = 100_000
SEED = 0x4D474F5A
H = 0.01
T_MAX = 500.0
N_STEPS = int(round(T_MAX / H))
R_ESCAPE = 5.0
BINARY_FACTOR = 2.0
R_CLOSE = 1e-3
CHUNK = 50_000
R_MIN_THRESHOLD = 0.01

EPSILONS_EXTENDED = [
    1e-6, 3e-6, 1e-5, 3e-5,
    1e-4, 3e-4, 1e-3, 3e-3,
    1e-2, 3e-2, 1e-1, 3e-1,
]


def perturb_on_sphere(pts, epsilon, key):
    noise = jax.random.normal(key, pts.shape, dtype=jnp.float64)
    dot = jnp.sum(noise * pts, axis=-1, keepdims=True)
    tangent = noise - dot * pts
    tangent = tangent / jnp.linalg.norm(tangent, axis=-1, keepdims=True)
    moved = pts + epsilon * tangent
    return moved / jnp.linalg.norm(moved, axis=-1, keepdims=True)


def perturb_6d(shape_pts, pi_jacobi, epsilon, key):
    k1, k2 = jax.random.split(key)
    shape_new = perturb_on_sphere(shape_pts, epsilon, k1)
    noise_m = jax.random.normal(k2, pi_jacobi.shape, dtype=jnp.float64)
    noise_m = noise_m / jnp.linalg.norm(noise_m, axis=-1, keepdims=True)
    pi_new = pi_jacobi + epsilon * noise_m
    return shape_new, pi_new


def integrate_and_label(q, p, chunk_size=CHUNK):
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


def fit_alpha(log_eps, log_f, label=""):
    """Fit slope with jackknife errors. Returns (alpha, 2sigma, intercept, r2)."""
    valid = np.isfinite(log_f)
    if valid.sum() < 3:
        return None, None, None, None
    le, lf = log_eps[valid], log_f[valid]
    slope, intercept = np.polyfit(le, lf, 1)
    resid = lf - (slope * le + intercept)
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((lf - lf.mean())**2)
    r2 = 1 - ss_res / max(ss_tot, 1e-30)

    n = len(le)
    jack_slopes = []
    for j in range(n):
        mask = np.ones(n, dtype=bool)
        mask[j] = False
        s, _ = np.polyfit(le[mask], lf[mask], 1)
        jack_slopes.append(s)
    jack_slopes = np.array(jack_slopes)
    jack_var = (n - 1) / n * np.sum((jack_slopes - jack_slopes.mean())**2)
    sigma2 = 2 * np.sqrt(jack_var)

    return slope, sigma2, intercept, r2


def run_mgoy_extended(name, shape_pts, pi_jacobi, epsilons, key, is_6d=False):
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
    print(f"  reliable: {n_reliable}/{N} ({100*n_reliable/N:.1f}%)")

    eps_list, f_list, n_unc_list, n_both_list = [], [], [], []

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

        eps_list.append(float(eps))
        f_list.append(float(f_unc))
        n_unc_list.append(n_disagree)
        n_both_list.append(n_both)

        print(f"  ε={eps:.1e}  f(ε)={f_unc:.5f}  ({n_disagree}/{n_both})  [{dt:.1f}s]")

    log_eps = np.log(np.array(eps_list))
    f_arr = np.array(f_list)
    log_f = np.log(np.where(f_arr > 0, f_arr, 1e-30))

    # Full fit
    alpha_full, sig_full, inter_full, r2_full = fit_alpha(log_eps, log_f, "full")

    # Fit only small-ε (unsaturated): use points where f < 0.8 * f_max
    f_max = f_arr.max()
    unsaturated = f_arr < 0.8 * f_max
    if unsaturated.sum() >= 3:
        alpha_unsat, sig_unsat, inter_unsat, r2_unsat = fit_alpha(
            log_eps[unsaturated], log_f[unsaturated], "unsaturated")
    else:
        alpha_unsat, sig_unsat, inter_unsat, r2_unsat = None, None, None, None

    # Sliding window fit (5-point windows)
    window = 5
    sliding = []
    for start in range(len(eps_list) - window + 1):
        end = start + window
        sl, sg, si, sr = fit_alpha(log_eps[start:end], log_f[start:end])
        if sl is not None:
            mid_eps = np.exp(log_eps[start:end].mean())
            sliding.append({"mid_eps": float(mid_eps), "alpha": float(sl),
                            "r2": float(sr)})

    d = 6 if is_6d else 2
    results = {
        "name": name,
        "n_pairs": N,
        "n_reliable": n_reliable,
        "epsilons": eps_list,
        "f_uncertain": f_list,
        "n_uncertain": n_unc_list,
        "n_both_reliable": n_both_list,
        "fit_full": {
            "alpha": float(alpha_full) if alpha_full else None,
            "alpha_2sigma": float(sig_full) if sig_full else None,
            "intercept": float(inter_full) if inter_full else None,
            "r_squared": float(r2_full) if r2_full else None,
            "D_u": float(d - alpha_full) if alpha_full else None,
        },
        "fit_unsaturated": {
            "alpha": float(alpha_unsat) if alpha_unsat else None,
            "alpha_2sigma": float(sig_unsat) if sig_unsat else None,
            "intercept": float(inter_unsat) if inter_unsat else None,
            "r_squared": float(r2_unsat) if r2_unsat else None,
            "D_u": float(d - alpha_unsat) if alpha_unsat else None,
            "n_points": int(unsaturated.sum()),
        },
        "sliding_window": sliding,
    }

    print(f"\n  Full fit:        α = {alpha_full:.4f} ± {sig_full:.4f}  R²={r2_full:.4f}  D_u={d-alpha_full:.4f}" if alpha_full else "  Full fit: failed")
    if alpha_unsat:
        print(f"  Unsaturated fit: α = {alpha_unsat:.4f} ± {sig_unsat:.4f}  R²={r2_unsat:.4f}  D_u={d-alpha_unsat:.4f} ({int(unsaturated.sum())} pts)")
    if sliding:
        print(f"  Sliding window fits:")
        for sw in sliding:
            print(f"    mid_ε={sw['mid_eps']:.2e}  α={sw['alpha']:.4f}  R²={sw['r2']:.4f}")

    return results, log_eps, log_f, f_arr


def main() -> int:
    apply_dirac_style()
    out_results = PROJECT_ROOT / "results" / "02_uncertainty_exponent"
    out_figs = PROJECT_ROOT / "figures"
    out_results.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    print(f"jax backend: {jax.default_backend()}")
    print(f"N_pairs: {N_PAIRS:,}")
    print(f"ε range: {EPSILONS_EXTENDED[0]:.0e} to {EPSILONS_EXTENDED[-1]:.0e} ({len(EPSILONS_EXTENDED)} points)")
    print()

    key = jax.random.PRNGKey(SEED)

    # 2D
    print("=== 2D at-rest (extended ε) ===")
    key, k1 = jax.random.split(key)
    shape_2d = sample_shape_sphere(k1, N_PAIRS)
    key, k2 = jax.random.split(key)
    res_2d, le_2d, lf_2d, f_2d = run_mgoy_extended(
        "2d_extended", shape_2d, None, EPSILONS_EXTENDED, k2, is_6d=False)
    with open(out_results / "2d_extended.json", "w") as f:
        json.dump(res_2d, f, indent=2)
    print()

    # 6D
    print("=== 6D (extended ε) ===")
    key, k3 = jax.random.split(key)
    shape_6d = sample_shape_sphere(k3, N_PAIRS)
    key, k4 = jax.random.split(key)
    pi_6d = sample_momentum_4ball(k4, N_PAIRS, p_max=3.0)
    key, k5 = jax.random.split(key)
    res_6d, le_6d, lf_6d, f_6d = run_mgoy_extended(
        "6d_extended", shape_6d, pi_6d, EPSILONS_EXTENDED, k5, is_6d=True)
    with open(out_results / "6d_extended.json", "w") as f:
        json.dump(res_6d, f, indent=2)
    print()

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    fig.patch.set_facecolor(PALETTE["bg_deep"])

    for ax, res, le, lf, f_arr, dim_label, d in [
        (axes[0], res_2d, le_2d, lf_2d, f_2d, "2D at-rest", 2),
        (axes[1], res_6d, le_6d, lf_6d, f_6d, "6D phase space", 6),
    ]:
        eps_arr = np.array(res["epsilons"])
        valid = f_arr > 0

        ax.plot(eps_arr[valid], f_arr[valid], "o", color=PALETTE["cyan"], ms=7, zorder=5)

        # Full fit
        ff = res["fit_full"]
        if ff["alpha"] is not None:
            eps_fit = np.logspace(np.log10(eps_arr[valid].min()),
                                  np.log10(eps_arr[valid].max()), 50)
            f_fit = np.exp(ff["intercept"]) * eps_fit ** ff["alpha"]
            ax.plot(eps_fit, f_fit, "--", color=PALETTE["text_mute"], lw=1.5,
                    label=rf"full: $\alpha={ff['alpha']:.3f}$")

        # Unsaturated fit
        fu = res["fit_unsaturated"]
        if fu["alpha"] is not None:
            f_max = f_arr.max()
            unsat = f_arr < 0.8 * f_max
            eps_u = eps_arr[unsat & valid]
            if len(eps_u) >= 2:
                eps_fit2 = np.logspace(np.log10(eps_u.min()), np.log10(eps_u.max()), 50)
                f_fit2 = np.exp(fu["intercept"]) * eps_fit2 ** fu["alpha"]
                ax.plot(eps_fit2, f_fit2, "-", color=PALETTE["lime"], lw=2.5,
                        label=rf"unsaturated: $\alpha={fu['alpha']:.3f} \pm {fu['alpha_2sigma']:.3f}$")

        # Box-counting prediction for 2D
        if d == 2:
            ax.axhline(y=None)  # placeholder
            eps_pred = np.logspace(-6, -0.5, 50)
            f_pred = 0.05 * eps_pred ** 0.408
            ax.plot(eps_pred, f_pred, ":", color=PALETTE["coral"], lw=1.5,
                    label=r"predicted $\alpha=0.408$ (box-counting)")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"perturbation $\varepsilon$")
        ax.set_ylabel(r"uncertain fraction $f(\varepsilon)$")
        ax.set_title(dim_label, color=PALETTE["text"], fontsize=13)
        ax.legend(fontsize=10, loc="lower right")

    fig.suptitle("MGOY uncertainty exponent (extended ε range)",
                 color=PALETTE["text"], fontsize=15, y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_figs / f"02_uncertainty_fit_extended.{ext}",
                    facecolor=PALETTE["bg_deep"], bbox_inches="tight")
    plt.close(fig)

    print("=== Final summary ===")
    for name, res, d in [("2D", res_2d, 2), ("6D", res_6d, 6)]:
        ff = res["fit_full"]
        fu = res["fit_unsaturated"]
        print(f"  {name}:")
        if ff["alpha"]:
            print(f"    full:        α={ff['alpha']:.4f} ± {ff['alpha_2sigma']:.4f}  D_u={ff['D_u']:.4f}")
        if fu["alpha"]:
            print(f"    unsaturated: α={fu['alpha']:.4f} ± {fu['alpha_2sigma']:.4f}  D_u={fu['D_u']:.4f} ({fu['n_points']} pts)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
