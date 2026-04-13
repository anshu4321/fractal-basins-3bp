"""Generate all paper figures from result artifacts."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
FIG_DIR = Path(__file__).resolve().parent

WONG = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "cyan": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
}

def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def fig_F3_scaling_law():
    """F3: Phase C vs Phase E scaling law comparison."""
    N_vals_c = [10_000, 30_000, 100_000, 300_000]
    log_N = np.log10(N_vals_c)

    with open(ROOT / "results" / "13_scaling_2d_s3bodynet" / "results.json") as f:
        dc = json.load(f)

    with open(ROOT / "experiments" / "21_matched_protocol_final" / "results.json") as f:
        de = json.load(f)

    goy_slope = -0.1295
    goy_sigma = 0.0095

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.2), sharey=False)

    # --- Phase C (left) ---
    for arch, color, marker in [("MLP", WONG["blue"], "o"), ("S3BodyNet", WONG["orange"], "s")]:
        em = np.array(dc[arch]["error_matrix"])
        mean_err = np.mean(em, axis=1)
        std_err = np.std(em, axis=1)
        log_err = np.log10(mean_err)
        slope = dc[arch]["slope"]
        ax1.errorbar(log_N, log_err, yerr=std_err / (mean_err * np.log(10)),
                     fmt=marker, color=color, markersize=5, capsize=3, label=arch, zorder=3)
        x_fit = np.linspace(log_N[0] - 0.1, log_N[-1] + 0.1, 50)
        intercept = np.mean(log_err) - slope / np.log(10) * np.mean(np.log(N_vals_c))
        y_fit = slope / np.log(10) * np.log(np.power(10, x_fit)) + intercept
        # Use polyfit in log10 space
        coef = np.polyfit(log_N, log_err, 1)
        y_fit = np.polyval(coef, x_fit)
        ax1.plot(x_fit, y_fit, color=color, linewidth=1, alpha=0.7)

    # GOY band
    log_N_band = np.linspace(log_N[0] - 0.1, log_N[-1] + 0.1, 50)
    anchor = np.mean(np.log10(np.mean(np.array(dc["MLP"]["error_matrix"]), axis=1)))
    anchor_x = np.mean(log_N)
    for s, label in [(goy_slope, "GOY prediction")]:
        coef_goy = [s / np.log(10), anchor - s / np.log(10) * anchor_x]
        y_goy = np.polyval(coef_goy, log_N_band)
        y_upper = np.polyval([coef_goy[0] + goy_sigma / np.log(10), coef_goy[1]], log_N_band)
        y_lower = np.polyval([coef_goy[0] - goy_sigma / np.log(10), coef_goy[1]], log_N_band)
        ax1.fill_between(log_N_band, y_lower, y_upper, alpha=0.15, color=WONG["green"])
        ax1.plot(log_N_band, y_goy, "--", color=WONG["green"], linewidth=1, alpha=0.8, label=label)

    ax1.set_xlabel(r"$\log_{10} N$")
    ax1.set_ylabel(r"$\log_{10}$(test error)")
    ax1.set_title("(a) Architecture-specific training", fontsize=9)
    ax1.legend(fontsize=7, loc="upper right")

    # --- Phase E (right) ---
    runs = de["runs"]
    for arch, color, marker in [("MLP", WONG["blue"], "o"), ("S3BodyNet", WONG["orange"], "s")]:
        mean_errs = []
        std_errs = []
        for N in N_vals_c:
            errs = [1 - r["accuracy"] for r in runs if r["arch"] == arch and r["N"] == N]
            mean_errs.append(np.mean(errs))
            std_errs.append(np.std(errs))
        mean_errs = np.array(mean_errs)
        std_errs = np.array(std_errs)
        log_err = np.log10(mean_errs)
        ax2.errorbar(log_N, log_err, yerr=std_errs / (mean_errs * np.log(10)),
                     fmt=marker, color=color, markersize=5, capsize=3, label=arch, zorder=3)
        coef = np.polyfit(log_N, log_err, 1)
        x_fit = np.linspace(log_N[0] - 0.1, log_N[-1] + 0.1, 50)
        y_fit = np.polyval(coef, x_fit)
        ax2.plot(x_fit, y_fit, color=color, linewidth=1, alpha=0.7)

    # GOY band (same anchor as left panel for visual reference)
    anchor2 = np.mean(np.log10(mean_errs))
    anchor_x2 = np.mean(log_N)
    coef_goy2 = [goy_slope / np.log(10), anchor2 - goy_slope / np.log(10) * anchor_x2]
    y_goy2 = np.polyval(coef_goy2, log_N_band)
    y_u2 = np.polyval([coef_goy2[0] + goy_sigma / np.log(10), coef_goy2[1]], log_N_band)
    y_l2 = np.polyval([coef_goy2[0] - goy_sigma / np.log(10), coef_goy2[1]], log_N_band)
    ax2.fill_between(log_N_band, y_l2, y_u2, alpha=0.15, color=WONG["green"])
    ax2.plot(log_N_band, y_goy2, "--", color=WONG["green"], linewidth=1, alpha=0.8, label="GOY prediction")

    ax2.set_xlabel(r"$\log_{10} N$")
    ax2.set_title("(b) Matched-compute protocol", fontsize=9)
    ax2.legend(fontsize=7, loc="upper right")

    fig.tight_layout(w_pad=2)
    fig.savefig(FIG_DIR / "F3_scaling_law.pdf")
    fig.savefig(FIG_DIR / "F3_scaling_law.png")
    plt.close(fig)
    print("Saved F3_scaling_law")


def fig_F4_diagnostic():
    """F4: Phase E diagnostic training curves, 2x2 grid."""
    with open(ROOT / "experiments" / "21_matched_protocol_final" / "diagnostic_logs.json") as f:
        logs = json.load(f)

    TOTAL = 58593
    cells = [
        ("MLP_N10000", "MLP, $N{=}10$k", "OVERFIT"),
        ("MLP_N300000", "MLP, $N{=}300$k", "UNDERTRAINED"),
        ("S3BodyNet_N10000", "S3BodyNet, $N{=}10$k", "UNDERTRAINED"),
        ("S3BodyNet_N300000", "S3BodyNet, $N{=}300$k", "UNDERTRAINED"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7, 5))
    for idx, (key, title, cls) in enumerate(cells):
        ax = axes[idx // 2][idx % 2]
        d = logs[key]
        steps = np.array(d["step"])
        loss = np.array(d["train_loss"])
        f1 = np.array(d["val_f1"])

        ax_loss = ax
        ax_f1 = ax.twinx()

        l1, = ax_loss.semilogy(steps, loss, color=WONG["blue"], linewidth=0.6, alpha=0.8)
        l2, = ax_f1.plot(steps, f1, color=WONG["orange"], linewidth=0.6, alpha=0.8)

        ax_loss.set_title(f"{title} ({cls})", fontsize=8)
        ax_loss.set_ylabel("Training loss", fontsize=7, color=WONG["blue"])
        ax_f1.set_ylabel("Val macro-F1", fontsize=7, color=WONG["orange"])
        ax_loss.tick_params(axis="y", labelcolor=WONG["blue"], labelsize=6)
        ax_f1.tick_params(axis="y", labelcolor=WONG["orange"], labelsize=6)
        ax_loss.tick_params(axis="x", labelsize=6)

        if idx >= 2:
            ax_loss.set_xlabel("Gradient updates", fontsize=7)

    fig.tight_layout(h_pad=1.5, w_pad=2)
    fig.savefig(FIG_DIR / "F4_diagnostic_curves.pdf")
    fig.savefig(FIG_DIR / "F4_diagnostic_curves.png")
    plt.close(fig)
    print("Saved F4_diagnostic_curves")


def fig_F2_active_learning():
    """F2: Active learning 2D vs 6D. Data from existing figure."""
    # Data points read from the existing figure
    budgets_2d = [200_000, 500_000]
    uniform_2d = [0.331, 0.368]
    al_2d = [0.369, 0.398]
    uniform_2d_err = [0.005, 0.004]
    al_2d_err = [0.006, 0.003]

    budgets_6d = [200_000, 400_000, 700_000]
    uniform_6d = [0.621, 0.683, 0.693]
    al_6d = [0.619, 0.669, 0.685]
    uniform_6d_err = [0.003, 0.003, 0.002]
    al_6d_err = [0.004, 0.004, 0.003]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    ax1.errorbar(np.array(budgets_2d)/1e3, al_2d, yerr=al_2d_err,
                 fmt="s-", color=WONG["green"], markersize=5, capsize=3, label="Active learning")
    ax1.errorbar(np.array(budgets_2d)/1e3, uniform_2d, yerr=uniform_2d_err,
                 fmt="o-", color=WONG["blue"], markersize=5, capsize=3, label="Uniform")
    ax1.set_xlabel("Total budget ($\\times 10^3$)")
    ax1.set_ylabel("Test macro-F1")
    ax1.set_title("(a) 2D at rest", fontsize=9)
    ax1.legend(fontsize=7)

    ax2.errorbar(np.array(budgets_6d)/1e3, al_6d, yerr=al_6d_err,
                 fmt="s-", color=WONG["green"], markersize=5, capsize=3, label="Active learning")
    ax2.errorbar(np.array(budgets_6d)/1e3, uniform_6d, yerr=uniform_6d_err,
                 fmt="o-", color=WONG["blue"], markersize=5, capsize=3, label="Uniform")
    ax2.set_xlabel("Total budget ($\\times 10^3$)")
    ax2.set_title("(b) 6D phase space", fontsize=9)
    ax2.legend(fontsize=7)

    fig.tight_layout(w_pad=2)
    fig.savefig(FIG_DIR / "F2_al_2d_6d.pdf")
    fig.savefig(FIG_DIR / "F2_al_2d_6d.png")
    plt.close(fig)
    print("Saved F2_al_2d_6d")


def fig_F1_basin_map():
    """F1: Basin map on shape sphere (Mollweide projection)."""
    npz_path = ROOT / "results" / "01_dataset_regen" / "at_rest_with_diagnostics.npz"
    if not npz_path.exists():
        print("SKIP F1: dataset not available locally")
        return

    d = np.load(npz_path)
    shape_n = d["shape_n"]
    labels = d["label"]
    r_min = d["r_min_ever"]

    mask = (r_min >= 0.01) & (labels != -1)
    n1, n2, n3 = shape_n[mask, 0], shape_n[mask, 1], shape_n[mask, 2]
    lab = labels[mask]

    # Shape sphere coords to Mollweide: lat = arcsin(n3), lon = arctan2(n2, n1)
    lat = np.arcsin(np.clip(n3, -1, 1))
    lon = np.arctan2(n2, n1)

    colors = {0: WONG["blue"], 1: WONG["orange"], 2: WONG["green"], 3: WONG["red"]}
    color_names = {0: "Bound", 1: "Body 1 escapes", 2: "Body 2 escapes", 3: "Body 3 escapes"}

    fig = plt.figure(figsize=(7, 3.5))
    ax = fig.add_subplot(111, projection="mollweide")

    subsample = np.random.default_rng(42).choice(len(lat), size=min(200_000, len(lat)), replace=False)
    for cls in [0, 1, 2, 3]:
        idx = subsample[lab[subsample] == cls]
        ax.scatter(lon[idx], lat[idx], c=colors[cls], s=0.02, alpha=0.3,
                   rasterized=True, label=color_names[cls])

    ax.set_title("")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=6, loc="lower right", markerscale=20)
    ax.set_xlabel("")
    ax.set_ylabel("")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "F1_basin_map.pdf", dpi=300)
    fig.savefig(FIG_DIR / "F1_basin_map.png", dpi=300)
    plt.close(fig)
    print("Saved F1_basin_map")


if __name__ == "__main__":
    setup_style()
    fig_F1_basin_map()
    fig_F2_active_learning()
    fig_F3_scaling_law()
    fig_F4_diagnostic()
    print("All figures generated.")
