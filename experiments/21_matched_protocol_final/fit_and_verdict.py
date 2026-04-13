"""Phase E Steps 3.3 + 3.4: Slope fitting, uncertainty, and verdict.

Reads results.json. Fits log(test_error) vs log(N) per architecture.
Computes jackknife + bootstrap uncertainty. Evaluates pass criteria.
Writes fit.json and VERDICT.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR = PROJECT_ROOT / "experiments" / "21_matched_protocol_final"

N_VALUES = [10_000, 30_000, 100_000, 300_000]
SEEDS = [0, 1, 2, 3, 4]
N_BOOT = 10_000
ALPHA_2D = 0.259
ALPHA_2D_SIGMA = 0.019
GOY_SLOPE = -ALPHA_2D / 2  # -0.1295
GOY_SLOPE_SIGMA = ALPHA_2D_SIGMA / 2  # 0.0095
MAX_CI_WIDTH = 0.04


def build_error_matrix(runs, arch_name):
    matrix = np.full((len(N_VALUES), len(SEEDS)), np.nan)
    for r in runs:
        if r["arch"] != arch_name:
            continue
        try:
            ni = N_VALUES.index(r["N"])
            si = SEEDS.index(r["seed"])
        except ValueError:
            continue
        matrix[ni, si] = r["test_error"]
    return matrix


def fit_slope(N_vals, error_matrix):
    mean_errors = np.nanmean(error_matrix, axis=1)
    log_N = np.log(np.array(N_vals, dtype=np.float64))
    log_e = np.log(mean_errors.astype(np.float64))
    slope, intercept = np.polyfit(log_N, log_e, 1)
    fitted = slope * log_N + intercept
    ss_res = np.sum((log_e - fitted) ** 2)
    ss_tot = np.sum((log_e - np.mean(log_e)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return float(slope), float(intercept), float(r2)


def jackknife_sigma(N_vals, error_matrix):
    n_seeds = error_matrix.shape[1]
    log_N = np.log(np.array(N_vals, dtype=np.float64))
    jack_slopes = []
    for drop in range(n_seeds):
        kept = np.delete(error_matrix, drop, axis=1)
        me = np.nanmean(kept, axis=1)
        le = np.log(me.astype(np.float64))
        s, _ = np.polyfit(log_N, le, 1)
        jack_slopes.append(s)
    jack_var = (n_seeds - 1) / n_seeds * np.sum(
        (np.array(jack_slopes) - np.mean(jack_slopes)) ** 2)
    return float(np.sqrt(jack_var)), jack_slopes


def bootstrap_sigma(N_vals, error_matrix, n_boot=N_BOOT):
    n_seeds = error_matrix.shape[1]
    log_N = np.log(np.array(N_vals, dtype=np.float64))
    rng = np.random.default_rng(12345)
    boot_slopes = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_seeds, size=n_seeds)
        resampled = error_matrix[:, idx]
        me = np.nanmean(resampled, axis=1)
        le = np.log(me.astype(np.float64))
        s, _ = np.polyfit(log_N, le, 1)
        boot_slopes.append(s)
    return float(np.std(boot_slopes)), boot_slopes


def main():
    results_path = EXP_DIR / "results.json"
    if not results_path.exists():
        print("ERROR: results.json not found. Run run_sweep.py first.")
        return 1

    with open(results_path) as f:
        data = json.load(f)
    runs = data["runs"]

    fit_results = {}

    for arch_name in ["MLP", "S3BodyNet"]:
        matrix = build_error_matrix(runs, arch_name)
        if np.any(np.isnan(matrix)):
            missing = np.argwhere(np.isnan(matrix))
            print(f"WARNING: {arch_name} has {len(missing)} missing cells: {missing.tolist()}")

        slope, intercept, r2 = fit_slope(N_VALUES, matrix)
        jack_sig, jack_slopes = jackknife_sigma(N_VALUES, matrix)
        boot_sig, boot_slopes = bootstrap_sigma(N_VALUES, matrix)
        sigma = max(jack_sig, boot_sig)
        ci_width = 2 * sigma

        fit_results[arch_name] = {
            "slope": slope,
            "intercept": intercept,
            "r2": r2,
            "jackknife_sigma": jack_sig,
            "bootstrap_sigma": boot_sig,
            "sigma_conservative": sigma,
            "ci_2sigma": ci_width,
            "slope_lower": slope - ci_width / 2,
            "slope_upper": slope + ci_width / 2,
            "error_matrix": matrix.tolist(),
            "mean_errors": np.nanmean(matrix, axis=1).tolist(),
        }

        print(f"{arch_name}: slope={slope:.4f} +/- {sigma:.4f} (2sigma={ci_width:.4f}), R2={r2:.6f}")

    # --- Pass criteria evaluation ---
    mlp = fit_results["MLP"]
    s3b = fit_results["S3BodyNet"]

    # Combined uncertainty: sqrt(measured_sigma^2 + GOY_sigma^2), then 2x for 2sigma
    mlp_combined_sigma = np.sqrt(mlp["sigma_conservative"]**2 + GOY_SLOPE_SIGMA**2)
    s3b_combined_sigma = np.sqrt(s3b["sigma_conservative"]**2 + GOY_SLOPE_SIGMA**2)

    # Criterion 1: MLP within 2sigma of GOY
    mlp_gap = abs(mlp["slope"] - GOY_SLOPE)
    mlp_threshold = 2 * mlp_combined_sigma
    c1_pass = mlp_gap < mlp_threshold

    # Criterion 2: S3BodyNet within 2sigma of GOY
    s3b_gap = abs(s3b["slope"] - GOY_SLOPE)
    s3b_threshold = 2 * s3b_combined_sigma
    c2_pass = s3b_gap < s3b_threshold

    # Criterion 3: inter-architecture agreement
    arch_gap = abs(mlp["slope"] - s3b["slope"])
    arch_combined_sigma = np.sqrt(mlp["sigma_conservative"]**2 + s3b["sigma_conservative"]**2)
    arch_threshold = 2 * arch_combined_sigma
    c3_pass = arch_gap < arch_threshold

    # Criterion 4: CI width hard floor
    c4_mlp = mlp["ci_2sigma"] <= MAX_CI_WIDTH
    c4_s3b = s3b["ci_2sigma"] <= MAX_CI_WIDTH
    c4_pass = c4_mlp and c4_s3b

    verdict = c1_pass and c2_pass and c3_pass and c4_pass
    verdict_str = "PASS" if verdict else "FAIL"

    criteria = {
        "c1_mlp_goy_agreement": {
            "pass": c1_pass,
            "mlp_slope": mlp["slope"],
            "goy_slope": GOY_SLOPE,
            "gap": mlp_gap,
            "threshold_2sigma": mlp_threshold,
            "combined_sigma": mlp_combined_sigma,
        },
        "c2_s3b_goy_agreement": {
            "pass": c2_pass,
            "s3b_slope": s3b["slope"],
            "goy_slope": GOY_SLOPE,
            "gap": s3b_gap,
            "threshold_2sigma": s3b_threshold,
            "combined_sigma": s3b_combined_sigma,
        },
        "c3_inter_architecture": {
            "pass": c3_pass,
            "gap": arch_gap,
            "threshold_2sigma": arch_threshold,
            "combined_sigma": arch_combined_sigma,
        },
        "c4_ci_width_floor": {
            "pass": c4_pass,
            "mlp_ci_width": mlp["ci_2sigma"],
            "s3b_ci_width": s3b["ci_2sigma"],
            "max_allowed": MAX_CI_WIDTH,
        },
        "verdict": verdict_str,
    }

    fit_output = {
        "MLP": fit_results["MLP"],
        "S3BodyNet": fit_results["S3BodyNet"],
        "goy_prediction": {
            "slope": GOY_SLOPE,
            "sigma": GOY_SLOPE_SIGMA,
            "ci_2sigma_lower": GOY_SLOPE - 2 * GOY_SLOPE_SIGMA,
            "ci_2sigma_upper": GOY_SLOPE + 2 * GOY_SLOPE_SIGMA,
        },
        "criteria": criteria,
    }

    def to_native(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            return to_native(obj)

    with open(EXP_DIR / "fit.json", "w") as f:
        json.dump(fit_output, f, indent=2, cls=NumpyEncoder)
    print(f"\nWrote fit.json")

    # --- Write VERDICT.md ---
    lines = [
        "# Phase E Verdict",
        "",
        "## Measured slopes",
        "",
        f"- MLP: {mlp['slope']:.4f} +/- {mlp['sigma_conservative']:.4f} "
        f"(2sigma = {mlp['ci_2sigma']:.4f}), R2 = {mlp['r2']:.6f}",
        f"- S3BodyNet: {s3b['slope']:.4f} +/- {s3b['sigma_conservative']:.4f} "
        f"(2sigma = {s3b['ci_2sigma']:.4f}), R2 = {s3b['r2']:.6f}",
        "",
        "## GOY prediction",
        "",
        f"- Predicted slope: -alpha/d = -{ALPHA_2D}/{2} = {GOY_SLOPE:.4f}",
        f"- Propagated 2sigma CI: +/-{2*GOY_SLOPE_SIGMA:.4f}, "
        f"giving [{GOY_SLOPE - 2*GOY_SLOPE_SIGMA:.4f}, {GOY_SLOPE + 2*GOY_SLOPE_SIGMA:.4f}]",
        "",
        "## Pass criteria evaluation",
        "",
        f"1. MLP-GOY agreement: |{mlp['slope']:.4f} - ({GOY_SLOPE:.4f})| = {mlp_gap:.4f} "
        f"vs 2sigma threshold {mlp_threshold:.4f} -> {'PASS' if c1_pass else 'FAIL'}",
        f"2. S3BodyNet-GOY agreement: |{s3b['slope']:.4f} - ({GOY_SLOPE:.4f})| = {s3b_gap:.4f} "
        f"vs 2sigma threshold {s3b_threshold:.4f} -> {'PASS' if c2_pass else 'FAIL'}",
        f"3. Inter-architecture: |{mlp['slope']:.4f} - ({s3b['slope']:.4f})| = {arch_gap:.4f} "
        f"vs 2sigma threshold {arch_threshold:.4f} -> {'PASS' if c3_pass else 'FAIL'}",
        f"4. CI width floor: MLP {mlp['ci_2sigma']:.4f}, S3BodyNet {s3b['ci_2sigma']:.4f}, "
        f"max allowed {MAX_CI_WIDTH} -> {'PASS' if c4_pass else 'FAIL'}",
        "",
        f"## Verdict: {verdict_str}",
    ]

    verdict_path = EXP_DIR / "VERDICT.md"
    with open(verdict_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {verdict_path}")
    print(f"\nVERDICT: {verdict_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
