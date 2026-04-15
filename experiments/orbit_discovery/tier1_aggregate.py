"""Aggregate Tier 1 per-seed results into 95% CI summary + paired stats.

Inputs:  tier1_ablation/results/seed{0..9}_{method}.json
Outputs: tier1_ablation/aggregated.json   (machine-readable)
         tier1_ablation/RESULT.md         (human-readable)

Metrics per method:
    mean, std, 95% CI (t-distribution with n-1 df)
    for n_verified and hit_rate_verify

Paired comparison (same seeds compared):
    Paired t-test (two-sided and one-sided)
    Cohen's d (paired)
    Wilcoxon signed-rank (nonparametric)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "tier1_ablation"
RESULTS_DIR = OUT_DIR / "results"
AGG_PATH = OUT_DIR / "aggregated.json"
RESULT_MD = OUT_DIR / "RESULT.md"

METHODS = ["label_disagreement", "classifier_entropy"]


def ci_95(values, label="n_verified"):
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n < 2:
        return {"n": n, "mean": float(x.mean()) if n else None,
                "std": None, "ci_lo": None, "ci_hi": None,
                "ci_half_width": None, "t_critical": None}
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    half = t_crit * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "std": sd,
            "ci_lo": mean - half, "ci_hi": mean + half,
            "ci_half_width": half, "t_critical": t_crit}


def paired_stats(a_values, b_values):
    """Paired tests for a > b (we expect label_disagreement > classifier_entropy)."""
    a = np.asarray(a_values, dtype=float)
    b = np.asarray(b_values, dtype=float)
    assert a.shape == b.shape
    diff = a - b
    n = len(diff)
    if n < 2:
        return {"n": n, "mean_diff": float(diff.mean()) if n else None}

    # Paired t-test (two-sided p from scipy; we report one-sided by halving when mean_diff>0)
    t_stat, p_two = stats.ttest_rel(a, b)
    if diff.mean() > 0:
        p_one = p_two / 2.0
    else:
        p_one = 1.0 - p_two / 2.0
    # Cohen's d for paired = mean(diff) / std(diff, ddof=1)
    diff_sd = float(diff.std(ddof=1))
    cohens_d = float(diff.mean() / diff_sd) if diff_sd > 0 else None

    # Wilcoxon signed-rank (nonparametric). Skip if all zeros.
    try:
        w_stat, p_wilcox = stats.wilcoxon(a, b, zero_method="wilcox",
                                          alternative="two-sided")
        w_stat = float(w_stat)
        p_wilcox = float(p_wilcox)
    except Exception as e:
        w_stat, p_wilcox = None, None

    return {
        "n": n,
        "mean_diff": float(diff.mean()),
        "std_diff": diff_sd,
        "t_statistic": float(t_stat),
        "p_two_sided": float(p_two),
        "p_one_sided_greater": float(p_one),
        "cohens_d_paired": cohens_d,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p_two_sided": p_wilcox,
        "per_seed_diff": diff.tolist(),
    }


def load_results(results_dir):
    per = {m: {} for m in METHODS}  # method -> seed -> dict
    found_seeds = set()
    for method in METHODS:
        for path in sorted(results_dir.glob(f"seed*_{method}.json")):
            try:
                data = json.loads(path.read_text())
                seed = int(data.get("seed"))
                per[method][seed] = data
                found_seeds.add(seed)
            except Exception as e:
                print(f"WARN: {path}: {e}", file=sys.stderr)
    return per, sorted(found_seeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--out-json", default=str(AGG_PATH))
    ap.add_argument("--out-md", default=str(RESULT_MD))
    args = ap.parse_args()

    per, seeds = load_results(Path(args.results_dir))

    per_method = {}
    total_wall = 0.0
    for m in METHODS:
        runs = [per[m][s] for s in seeds if s in per[m]]
        nvs = [r["n_verified"] for r in runs]
        hrs = [r["hit_rate_verify"] for r in runs]
        wts = [r.get("total_time_s", r.get("wall_time_s", 0.0)) for r in runs]
        total_wall += sum(wts)
        per_method[m] = {
            "seeds_used": [int(r["seed"]) for r in runs],
            "n_verified": {
                "values": nvs,
                **ci_95(nvs),
            },
            "hit_rate_verify": {
                "values": hrs,
                **ci_95(hrs),
            },
            "wall_time_s_total": sum(wts),
            "wall_time_s_mean": float(np.mean(wts)) if wts else None,
        }

    # Paired tests on seeds present in both methods
    common = sorted(set(per["label_disagreement"]) & set(per["classifier_entropy"]))
    if len(common) >= 2:
        a = [per["label_disagreement"][s]["n_verified"] for s in common]
        b = [per["classifier_entropy"][s]["n_verified"] for s in common]
        paired_nv = paired_stats(a, b)
        paired_nv["seeds"] = common
        a2 = [per["label_disagreement"][s]["hit_rate_verify"] for s in common]
        b2 = [per["classifier_entropy"][s]["hit_rate_verify"] for s in common]
        paired_hr = paired_stats(a2, b2)
        paired_hr["seeds"] = common
    else:
        paired_nv = {"n": len(common), "note": "need >=2 paired samples"}
        paired_hr = {"n": len(common), "note": "need >=2 paired samples"}

    agg = {
        "methods": METHODS,
        "seeds_present": seeds,
        "per_method": per_method,
        "paired_n_verified_ld_vs_ce": paired_nv,
        "paired_hit_rate_verify_ld_vs_ce": paired_hr,
        "total_compute_s": total_wall,
    }

    Path(args.out_json).write_text(json.dumps(agg, indent=2, default=str))
    print(f"Wrote {args.out_json}")

    # Markdown
    def fmt_ci(d):
        if d.get("mean") is None:
            return "n/a"
        if d.get("ci_half_width") is None:
            return f"{d['mean']:.3f} (n={d['n']})"
        return (f"{d['mean']:.3f}  (95% CI [{d['ci_lo']:.3f}, {d['ci_hi']:.3f}], "
                f"std {d['std']:.3f}, n={d['n']})")

    lines = []
    lines.append("# Tier 1 ablation: 10-seed reliability study")
    lines.append("")
    lines.append("Compares `label_disagreement` vs `classifier_entropy` candidate "
                 "priors for 3-body periodic-orbit discovery, holding all other "
                 "pipeline parameters at the paper baseline (k=6, MLP hidden=128, "
                 "n_layers=4, N_CANDIDATES=5000, N_REFINE_TOP=50).")
    lines.append("")
    lines.append(f"Seeds present: {seeds}")
    lines.append(f"Total compute: {total_wall:.0f} s "
                 f"({total_wall/3600:.2f} h)")
    lines.append("")
    lines.append("## Per-method summary")
    lines.append("")
    lines.append("| Method | n_verified (mean ± 95% CI) | hit_rate_verify (mean ± 95% CI) | Wall / run (mean, s) |")
    lines.append("|---|---|---|---|")
    for m in METHODS:
        nv = per_method[m]["n_verified"]
        hr = per_method[m]["hit_rate_verify"]
        w = per_method[m]["wall_time_s_mean"]
        lines.append(f"| {m} | {fmt_ci(nv)} | {fmt_ci(hr)} | "
                     f"{w:.0f} |")
    lines.append("")

    lines.append("## Paired comparison: label_disagreement − classifier_entropy")
    lines.append("")
    lines.append(f"Paired seeds: {paired_nv.get('seeds', [])}")
    lines.append("")
    lines.append("### n_verified")
    if paired_nv.get("mean_diff") is None:
        lines.append("  Not enough paired samples.")
    else:
        lines.append(f"- mean(ld − ce) = {paired_nv['mean_diff']:+.3f}  "
                     f"(std {paired_nv['std_diff']:.3f})")
        lines.append(f"- paired t = {paired_nv['t_statistic']:.3f}")
        lines.append(f"- two-sided p = {paired_nv['p_two_sided']:.5f}")
        lines.append(f"- one-sided p (ld > ce) = {paired_nv['p_one_sided_greater']:.5f}")
        if paired_nv.get("cohens_d_paired") is not None:
            lines.append(f"- Cohen's d (paired) = {paired_nv['cohens_d_paired']:+.3f}")
        if paired_nv.get("wilcoxon_p_two_sided") is not None:
            lines.append(f"- Wilcoxon signed-rank: stat={paired_nv['wilcoxon_stat']:.2f}, "
                         f"two-sided p={paired_nv['wilcoxon_p_two_sided']:.5f}")
        lines.append(f"- per-seed diffs: {paired_nv['per_seed_diff']}")
    lines.append("")
    lines.append("### hit_rate_verify")
    if paired_hr.get("mean_diff") is None:
        lines.append("  Not enough paired samples.")
    else:
        lines.append(f"- mean(ld − ce) = {paired_hr['mean_diff']:+.5f}  "
                     f"(std {paired_hr['std_diff']:.5f})")
        lines.append(f"- paired t = {paired_hr['t_statistic']:.3f}")
        lines.append(f"- two-sided p = {paired_hr['p_two_sided']:.5f}")
        lines.append(f"- one-sided p (ld > ce) = {paired_hr['p_one_sided_greater']:.5f}")
        if paired_hr.get("cohens_d_paired") is not None:
            lines.append(f"- Cohen's d (paired) = {paired_hr['cohens_d_paired']:+.3f}")
        if paired_hr.get("wilcoxon_p_two_sided") is not None:
            lines.append(f"- Wilcoxon signed-rank: stat={paired_hr['wilcoxon_stat']:.2f}, "
                         f"two-sided p={paired_hr['wilcoxon_p_two_sided']:.5f}")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if paired_nv.get("p_one_sided_greater") is not None:
        p1 = paired_nv["p_one_sided_greater"]
        d_cohen = paired_nv.get("cohens_d_paired") or 0.0
        md = paired_nv["mean_diff"]
        ce_mean = per_method["classifier_entropy"]["n_verified"]["mean"]
        pct = (md / ce_mean * 100) if ce_mean else 0
        if p1 < 0.05 and md > 0:
            verdict = (f"label_disagreement finds ~{pct:+.1f}% more verified orbits "
                       f"than classifier_entropy (mean Δ = {md:+.2f}, "
                       f"one-sided paired-t p = {p1:.4f}, "
                       f"Cohen's d = {d_cohen:+.2f}). "
                       f"The paper's claim is supported at n=10.")
        elif md > 0:
            verdict = (f"label_disagreement shows a positive mean advantage "
                       f"(Δ = {md:+.2f}, {pct:+.1f}%) but it is not significant "
                       f"at α=0.05 (one-sided p = {p1:.4f}). Larger seed "
                       f"count may be needed.")
        else:
            verdict = (f"No advantage for label_disagreement (Δ = {md:+.2f}, "
                       f"one-sided p = {p1:.4f}). The paper's claim would need "
                       f"revision or additional support.")
        lines.append(verdict)
    lines.append("")

    Path(args.out_md).write_text("\n".join(lines))
    print(f"Wrote {args.out_md}")

    # Also print summary to stdout
    print()
    print("=" * 60)
    for m in METHODS:
        nv = per_method[m]["n_verified"]
        print(f"{m:24s}  n_verified: {fmt_ci(nv)}")
    if paired_nv.get("mean_diff") is not None:
        print(f"\nPaired (ld - ce): Δ={paired_nv['mean_diff']:+.3f}, "
              f"one-sided p={paired_nv['p_one_sided_greater']:.5f}, "
              f"Cohen's d={paired_nv.get('cohens_d_paired', 0):+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
