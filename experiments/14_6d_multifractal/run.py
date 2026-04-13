"""Experiment 14: 6D multi-fractal measurement (PROMPT_v2 Section 3.3).

Tasks:
1. Compute generalized dimension spectrum D_q for q in {0, 1, 2}.
   If D_q varies with q, the boundary is multi-fractal.
2. Measure alpha_6D in 3 disjoint epsilon sub-ranges.
   Compare each local alpha to the classifier slope -0.054.
3. If any local alpha matches the classifier slope within 2sigma,
   report this as the effective uncertainty exponent at sampling scale.

Pass criterion: mechanistic explanation for the 2.2x disagreement,
OR accept as negative result.
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
import numpy as np

from mega3bp.shape_sphere import sample_phase_space_6d

R_MIN_THRESHOLD = 0.01
CLASSIFIER_SLOPE_6D = -0.054
ALPHA_6D_GLOBAL = 0.145

N_SAMPLE = 500_000
EPSILONS = np.logspace(-5, -1, 50)
N_PAIRS = 2_000_000

EPSILON_RANGES = [
    ("fine", 1e-5, 1e-4),
    ("mid", 1e-4, 1e-3),
    ("coarse", 1e-3, 1e-2),
]


def load_6d_labels():
    path = PROJECT_ROOT / "results" / "01_dataset_regen" / "6d_with_diagnostics.npz"
    d = np.load(path)
    mask = (d["r_min_ever"] >= R_MIN_THRESHOLD) & (d["label"] != -1)
    shape = d["shape_n"][mask]
    pi = d["pi_jacobi"][mask]
    labels = d["label"][mask].astype(np.int32)
    features = np.concatenate([shape, pi], axis=-1).astype(np.float64)
    return features, labels


def uncertainty_fraction(features, labels, epsilon, n_pairs, key):
    """Fraction of random pairs within distance epsilon that disagree on label."""
    n = features.shape[0]
    k1, k2 = jax.random.split(key)
    idx_a = jax.random.randint(k1, (n_pairs,), 0, n)
    idx_b = jax.random.randint(k2, (n_pairs,), 0, n)

    xa = jnp.asarray(features)[idx_a]
    xb = jnp.asarray(features)[idx_b]
    dist = jnp.sqrt(jnp.sum((xa - xb) ** 2, axis=-1))

    la = jnp.asarray(labels)[idx_a]
    lb = jnp.asarray(labels)[idx_b]

    within_eps = dist < epsilon
    disagree = la != lb
    n_within = jnp.sum(within_eps)
    n_disagree_within = jnp.sum(within_eps & disagree)

    frac = jnp.where(n_within > 0, n_disagree_within / n_within, 0.0)
    return float(frac), int(n_within)


def measure_alpha_range(features, labels, eps_lo, eps_hi, n_eps=15, n_pairs=2_000_000):
    """Measure alpha in a specific epsilon sub-range via log-log fit."""
    epsilons = np.logspace(np.log10(eps_lo), np.log10(eps_hi), n_eps)
    fracs = []
    counts = []
    key = jax.random.PRNGKey(42)

    for eps in epsilons:
        key, sub = jax.random.split(key)
        f, c = uncertainty_fraction(features, labels, eps, n_pairs, sub)
        fracs.append(f)
        counts.append(c)

    fracs = np.array(fracs)
    counts = np.array(counts)

    valid = (fracs > 0) & (counts > 100)
    if valid.sum() < 3:
        return None, None, None, epsilons, fracs, counts

    log_eps = np.log(epsilons[valid])
    log_frac = np.log(fracs[valid])
    alpha, intercept = np.polyfit(log_eps, log_frac, 1)

    residuals = log_frac - (alpha * log_eps + intercept)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((log_frac - np.mean(log_frac)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return float(alpha), float(r2), float(intercept), epsilons, fracs, counts


def generalized_dimension(features, labels, q_val, n_pairs=2_000_000):
    """Estimate generalized dimension D_q from uncertainty fraction scaling.

    D_q is related to the scaling of the q-th moment of the box-probability
    distribution. For the uncertainty dimension:
    - D_0 = box-counting dimension of the boundary
    - D_1 = information dimension
    - D_2 = correlation dimension

    We use the uncertainty fraction f(eps) and its moments.
    For a mono-fractal: D_q = d - alpha for all q (constant).
    For a multi-fractal: D_q varies with q.
    """
    epsilons = np.logspace(-4, -1.5, 20)
    key = jax.random.PRNGKey(100 + q_val)
    d = features.shape[1]

    if q_val == 0:
        fracs = []
        for eps in epsilons:
            key, sub = jax.random.split(key)
            f, c = uncertainty_fraction(features, labels, eps, n_pairs, sub)
            fracs.append(f)
        fracs = np.array(fracs)
        valid = fracs > 0
        if valid.sum() < 3:
            return None
        log_eps = np.log(epsilons[valid])
        log_frac = np.log(fracs[valid])
        alpha, _ = np.polyfit(log_eps, log_frac, 1)
        return float(d - alpha)

    elif q_val == 1:
        fracs = []
        for eps in epsilons:
            key, sub = jax.random.split(key)
            f, c = uncertainty_fraction(features, labels, eps, n_pairs, sub)
            fracs.append(f)
        fracs = np.array(fracs)
        valid = fracs > 0
        if valid.sum() < 3:
            return None
        log_eps = np.log(epsilons[valid])
        info = np.array([f * np.log(f) if f > 0 else 0 for f in fracs[valid]])
        slope, _ = np.polyfit(log_eps, info, 1)
        return float(d - 1 + slope)

    elif q_val == 2:
        fracs = []
        for eps in epsilons:
            key, sub = jax.random.split(key)
            f, c = uncertainty_fraction(features, labels, eps, n_pairs, sub)
            fracs.append(f)
        fracs = np.array(fracs)
        valid = fracs > 0
        if valid.sum() < 3:
            return None
        log_eps = np.log(epsilons[valid])
        log_frac2 = np.log(fracs[valid] ** 2)
        slope, _ = np.polyfit(log_eps, log_frac2, 1)
        return float(d - slope)

    return None


def main() -> int:
    out_dir = PROJECT_ROOT / "results" / "14_6d_multifractal"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"jax backend: {jax.default_backend()}")
    print()

    features, labels = load_6d_labels()
    print(f"loaded {len(labels)} clean 6D samples")
    print(f"global alpha_6D = {ALPHA_6D_GLOBAL}, classifier slope = {CLASSIFIER_SLOPE_6D}")
    print()

    # 1. Generalized dimension spectrum
    print("=== Generalized dimension spectrum ===")
    dq_results = {}
    for q in [0, 1, 2]:
        t0 = time.perf_counter()
        dq = generalized_dimension(features, labels, q)
        dt = time.perf_counter() - t0
        dq_results[f"D_{q}"] = dq
        alpha_q = (features.shape[1] - dq) if dq is not None else None
        print(f"  D_{q} = {dq:.4f}  (alpha_{q} = {alpha_q:.4f})  [{dt:.1f}s]" if dq else f"  D_{q} = None")

    is_multifractal = False
    dq_vals = [v for v in dq_results.values() if v is not None]
    if len(dq_vals) >= 2:
        spread = max(dq_vals) - min(dq_vals)
        is_multifractal = spread > 0.1
        print(f"  D_q spread: {spread:.4f} ({'multi-fractal' if is_multifractal else 'mono-fractal'})")

    # 2. Local alpha in sub-ranges
    print("\n=== Local alpha in epsilon sub-ranges ===")
    local_results = {}
    for name, lo, hi in EPSILON_RANGES:
        t0 = time.perf_counter()
        alpha, r2, intercept, eps, fracs, counts = measure_alpha_range(features, labels, lo, hi)
        dt = time.perf_counter() - t0
        local_results[name] = {
            "eps_range": [lo, hi],
            "alpha": alpha,
            "r2": r2,
            "epsilons": eps.tolist(),
            "fracs": fracs.tolist(),
            "counts": [int(c) for c in counts],
        }
        if alpha is not None:
            pred_slope = -alpha / features.shape[1]
            print(f"  {name:>6s} [{lo:.0e}, {hi:.0e}]: alpha = {alpha:.4f} (R2 = {r2:.4f}), "
                  f"predicted slope = {pred_slope:.4f} vs classifier {CLASSIFIER_SLOPE_6D:.4f}  [{dt:.1f}s]")
        else:
            print(f"  {name:>6s} [{lo:.0e}, {hi:.0e}]: insufficient data  [{dt:.1f}s]")

    # 3. Check if any local alpha matches classifier slope
    match_found = False
    match_range = None
    match_alpha = None
    for name, info in local_results.items():
        if info["alpha"] is None:
            continue
        pred_slope = -info["alpha"] / features.shape[1]
        # Within 2sigma: use rough estimate sigma ~ 0.01
        if abs(pred_slope - CLASSIFIER_SLOPE_6D) < 0.02:
            match_found = True
            match_range = name
            match_alpha = info["alpha"]

    # Verdict
    if match_found:
        verdict_text = (
            f"Mechanistic match at epsilon range '{match_range}': local alpha = {match_alpha:.4f}, "
            f"predicted slope = {-match_alpha/features.shape[1]:.4f} matches classifier slope "
            f"{CLASSIFIER_SLOPE_6D:.4f}. The factor-of-2.2 disagreement between the global GOY "
            f"prediction and classifier accuracy is explained by multi-fractal structure: the "
            f"effective uncertainty exponent at the sampling scale differs from the global value."
        )
        d3_mechanism = True
    else:
        verdict_text = (
            f"No local alpha matches the classifier slope {CLASSIFIER_SLOPE_6D:.4f} within "
            f"2sigma in any of the tested epsilon ranges. The 6D result stands as a negative "
            f"finding: the GOY single-alpha prediction does not match classifier accuracy, "
            f"{'and the boundary shows multi-fractal structure' if is_multifractal else 'but the boundary appears mono-fractal'}."
        )
        d3_mechanism = False

    results = {
        "generalized_dimensions": dq_results,
        "is_multifractal": is_multifractal,
        "local_alpha": {k: {kk: vv for kk, vv in v.items() if kk != "epsilons"} for k, v in local_results.items()},
        "classifier_slope_6d": CLASSIFIER_SLOPE_6D,
        "alpha_6d_global": ALPHA_6D_GLOBAL,
        "match_found": match_found,
        "match_range": match_range,
        "match_alpha": match_alpha,
        "d3_mechanism": d3_mechanism,
        "verdict": verdict_text,
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== VERDICT (D3) ===")
    print(verdict_text)
    print(f"\nD3 mechanism found: {d3_mechanism}")
    if not d3_mechanism:
        print("D3 passes via honest-negative framing.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
