"""Experiment 01b: Time-cutoff sensitivity analysis.

Subsamples 10^4 ICs from the at-rest dataset. Reruns each at
t_max in {500, 2000, 10000}. Plots bound fraction vs t_max.
Fits power law if bound fraction is still dropping.

Outputs:
    results/01_dataset_regen/time_cutoff_sensitivity.json
    figures/01_time_cutoff.png
    figures/01_time_cutoff.pdf
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
from mega3bp.shape_sphere import sample_shape_sphere, shape_to_config
from mega3bp.style import PALETTE, apply_dirac_style

N_SUB = 10_000
SEED = 0xFEEDBEEF
H = 0.01
T_MAX_VALUES = [500.0, 2000.0, 10000.0]
R_ESCAPE = 5.0
BINARY_FACTOR = 2.0
R_CLOSE = 1e-3
CHUNK = 10_000


def run_at_tmax(q, p, t_max):
    n_steps = int(round(t_max / H))
    out = integrate_batch_diag(
        q, p, H, n_steps,
        r_escape=R_ESCAPE, binary_factor=BINARY_FACTOR, r_close=R_CLOSE,
        strict_escape=True,
    )
    jax.block_until_ready(out["label"])
    labels = np.asarray(out["label"])
    return labels


def main() -> int:
    apply_dirac_style()

    out_results = PROJECT_ROOT / "results" / "01_dataset_regen"
    out_figs = PROJECT_ROOT / "figures"
    out_results.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    print(f"jax backend: {jax.default_backend()}")
    print(f"subsampling {N_SUB} ICs from seed {SEED:#x}")

    key = jax.random.PRNGKey(SEED)
    shape_pts = sample_shape_sphere(key, N_SUB)
    q = shape_to_config(shape_pts, inertia=1.0)
    p = jnp.zeros_like(q)

    results = {"t_max": [], "bound_frac": [], "escape_frac": [], "ambiguous_frac": [],
               "per_class": []}

    for t_max in T_MAX_VALUES:
        t0 = time.perf_counter()
        labels = run_at_tmax(q, p, t_max)
        dt = time.perf_counter() - t0

        bound = float((labels == 0).mean())
        escape = float(((labels >= 1) & (labels <= 3)).mean())
        ambig = float((labels == -1).mean())

        unique, counts = np.unique(labels, return_counts=True)
        per_class = {int(l): int(c) for l, c in zip(unique, counts)}

        results["t_max"].append(t_max)
        results["bound_frac"].append(bound)
        results["escape_frac"].append(escape)
        results["ambiguous_frac"].append(ambig)
        results["per_class"].append(per_class)

        print(f"  t_max={t_max:>8.0f}  bound={bound:.4f}  escape={escape:.4f}  "
              f"ambig={ambig:.4f}  [{dt:.1f}s]")

    t_arr = np.array(results["t_max"])
    b_arr = np.array(results["bound_frac"])

    if b_arr[-1] < b_arr[0] - 0.005:
        try:
            def power_law(t, a, gamma):
                return a * t ** (-gamma)
            popt, _ = curve_fit(power_law, t_arr, b_arr, p0=[1.0, 0.1], maxfev=5000)
            results["power_law_fit"] = {"a": float(popt[0]), "gamma": float(popt[1])}
            print(f"  power law fit: bound ~ {popt[0]:.3f} * t^(-{popt[1]:.4f})")
        except Exception as e:
            results["power_law_fit"] = {"error": str(e)}
            print(f"  power law fit failed: {e}")
    else:
        results["power_law_fit"] = None
        print("  bound fraction stable, no power law fit needed")

    reclassified = 0
    if len(T_MAX_VALUES) >= 2:
        labels_short = run_at_tmax(q, p, T_MAX_VALUES[0])
        labels_long = run_at_tmax(q, p, T_MAX_VALUES[-1])
        was_bound = labels_short == 0
        now_escape = (labels_long >= 1) & (labels_long <= 3)
        reclassified = float((was_bound & now_escape).mean())
        results["frac_reclassified_500_to_10000"] = reclassified
        print(f"  reclassified bound@500 -> escape@10000: {reclassified:.4%}")

    json_path = out_results / "time_cutoff_sensitivity.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {json_path}")

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor(PALETTE["bg_deep"])
    ax.plot(t_arr, b_arr, "o-", color=PALETTE["cyan"], lw=2, ms=8, label="bound fraction")
    ax.plot(t_arr, np.array(results["escape_frac"]), "s-", color=PALETTE["lime"],
            lw=2, ms=8, label="escape fraction")
    if results["power_law_fit"] and "gamma" in results["power_law_fit"]:
        t_fit = np.linspace(t_arr[0], t_arr[-1], 100)
        a, g = results["power_law_fit"]["a"], results["power_law_fit"]["gamma"]
        ax.plot(t_fit, a * t_fit ** (-g), "--", color=PALETTE["coral"], lw=1.5,
                label=f"fit: $t^{{-{g:.3f}}}$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$t_{\max}$")
    ax.set_ylabel("fraction")
    ax.set_title("Time-cutoff sensitivity (at-rest, 10k ICs)", color=PALETTE["text"])
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_figs / f"01_time_cutoff.{ext}", facecolor=PALETTE["bg_deep"])
    plt.close(fig)
    print(f"saved figures/01_time_cutoff.png/pdf")

    return 0


if __name__ == "__main__":
    sys.exit(main())
