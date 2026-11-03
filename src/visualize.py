"""Create figures from saved trials"""

import argparse
from ast import literal_eval
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from src.metrics import (
    summarize_results, wilson_interval, summarize_failure_modes, summarize_runtime,
)


PLOT_CONFIG = {"confidence": 0.95, "diagnostic_theta": 1.3, "dpi": 300}
REGIME_TITLES = {"linear": "Linear", "sublinear": "Sublinear", "fractional": "Fractional power"}
RUNS = {
    "Paper setting": "full",
    "Five weak effects": "weak_signals",
    "Larger dimensions": "larger_p",
}


def read_run(directory):
    """Read a completed run and its saved settings, not today's edited settings."""
    config = literal_eval((directory / "config_used.txt").read_text())
    metadata = literal_eval((directory / "metadata.txt").read_text())
    raw = pd.read_csv(directory / "raw_results.csv")
    if not metadata["complete"] or len(raw) != metadata["expected_trials"]:
        raise ValueError(f"The simulation in {directory} is incomplete.")
    if not raw["success"].isin([0, 1]).all():
        raise ValueError(f"Invalid success indicators in {directory}.")
    errors = raw[["false_positives", "false_negatives", "sign_errors"]].sum(axis=1)
    if not np.array_equal(raw["success"], (errors == 0).astype(int)):
        raise ValueError(f"Recovery indicators and error counts disagree in {directory}.")
    counts = raw.groupby(["design", "regime", "p", "theta"]).size()
    if not counts.eq(config["repetitions"]).all():
        raise ValueError(f"Unequal repetition counts in {directory}.")
    return raw, config


def save_figure(fig, output_dir, stem):
    """Save a high-resolution preview and a vector PDF with embedded fonts."""
    for extension in ["png", "pdf"]:
        fig.savefig(Path(output_dir) / f"{stem}.{extension}",
                    dpi=PLOT_CONFIG["dpi"], bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_recovery(summary, config, output_dir):
    """recovery curves and marginal Monte Carlo intervals."""
    styles = {
        "Paper setting": {"color": "#0072B2", "linestyle": "-", "marker": "o"},
        "Five weak effects": {"color": "#D55E00", "linestyle": "--", "marker": "s"},
    }
    for number, design in enumerate(config["designs"], start=1):
        fig, axes = plt.subplots(len(config["regimes"]), len(config["p_values"]),
                                 figsize=(10.8, 9.2), sharex=True, sharey=True, squeeze=False)
        fig.subplots_adjust(left=.10, right=.98, bottom=.18, top=.82, hspace=.38, wspace=.14)
        for row, regime in enumerate(config["regimes"]):
            for col, p in enumerate(config["p_values"]):
                ax = axes[row, col]
                subset = summary[(summary.design == design["name"])
                                 & (summary.regime == regime) & (summary.p == p)]
                for label, style in styles.items():
                    curve = subset[subset.setting == label].sort_values("theta")
                    ax.fill_between(curve.theta, curve.ci_lower, curve.ci_upper,
                                    color=style["color"], alpha=.16, linewidth=0)
                    ax.plot(curve.theta, curve.success_probability, label=label,
                            linewidth=1.7, markersize=3, **style)
                for threshold in design["reference_lines"]:
                    ax.axvline(threshold, color=".45", linestyle=":", linewidth=1, zorder=0)
                k = int(subset.k.iloc[0])
                ax.set(title=f"{REGIME_TITLES[regime]} | p = {p}, k = {k}",
                       xlim=(0, 2.4), ylim=(-.025, 1.025),
                       xticks=[0, .5, 1, 1.5, 2], yticks=[0, .25, .5, .75, 1])
                ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
                ax.grid(axis="y", color=".9", linewidth=.65)
        fig.suptitle("Exact Lasso recovery with strong and weak effects", fontsize=16, y=.985)
        covariance = "Independent Gaussian predictors" if design["mu"] == 0 else f"Gaussian Toeplitz predictors: adjacent correlation = {design['mu']:g}"
        fig.text(.54, .945, f"{covariance}  |  Noise SD = {config['noise_sigma']:g}",
                 ha="center", fontsize=10)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.54, .92),
                   ncol=2, frameon=False, fontsize=10)
        fig.supxlabel(r"Sample-size multiplier $\theta$ (larger values give more observations)", y=.12)
        fig.supylabel("Probability of exact signed-support recovery", x=.015)
        reference = ("Dotted line: asymptotic identity threshold, θ = 1."
                     if design["mu"] == 0 else "Dotted lines: asymptotic Toeplitz bounds, θ ≈ 0.46 and 1.84.")
        caption = (
            "Paper setting: all active magnitudes = 1. Weak setting: five active magnitudes = 0.25; the rest = 1.\n"
            f"Points: {config['repetitions']} trials per condition. Shading: {100 * PLOT_CONFIG['confidence']:g}% pointwise Wilson Monte Carlo intervals.\n"
            "Recovery requires every support location AND sign to be correct (estimated |β| ≤ 10⁻⁷ counts as zero).\n"
            "n = floor[2θk log(p − k)] + k + 1; random support and ± signs; natural logs. "
            "Sparsity uses α = 0.4, γ = 0.75.\n"
            + reference + " These are large-sample references, not finite-sample guarantees."
        )
        fig.text(.10, .008, caption, va="bottom", fontsize=7.7, linespacing=1.5)
        save_figure(fig, output_dir, f"{number:02d}_recovery_{design['name']}")


def plot_diagnostics(raw, config, output_dir, tables_dir):
    theta = PLOT_CONFIG["diagnostic_theta"]
    settings = ["Paper setting", "Five weak effects"]
    colors = {"Exact recovery": "#B3B3B3", "Extra only": "#0072B2",
              "Missed only": "#D55E00", "Extra + missed": "#CC79A7", "Wrong sign(s)": "#000000"}
    summary = summarize_failure_modes(raw, theta)
    summary.to_csv(Path(tables_dir) / "diagnostic_failure_modes.csv", index=False)
    order = [(p, setting) for p in sorted(config["p_values"]) for setting in settings]
    labels = [f"{'Paper' if setting == settings[0] else 'Weak'}\np={p}" for p, setting in order]
    fig, axes = plt.subplots(len(config["regimes"]), len(config["designs"]),
                             figsize=(11, 9), sharey=True, squeeze=False)
    for row, regime in enumerate(config["regimes"]):
        for col, design in enumerate(config["designs"]):
            ax = axes[row, col]
            facet = summary[(summary["design"] == design["name"]) & (summary["regime"] == regime)]
            facet = facet.set_index(["p", "setting"]).reindex(order)
            bottom = np.zeros(len(order))
            for category, color in colors.items():
                values = facet[category].to_numpy()
                ax.bar(np.arange(len(order)), values, bottom=bottom, color=color, label=category, width=.78)
                for position, (value, base) in enumerate(zip(values, bottom)):
                    if value >= .12:
                        text_color = "white" if color in ["#0072B2", "#D55E00", "#000000"] else "#222222"
                        ax.text(position, base + value / 2, f"{100 * value:.0f}%",
                                ha="center", va="center", fontsize=8, color=text_color)
                bottom += values
            ax.set(title=f"{REGIME_TITLES[regime]} — {design['name'].capitalize()}", ylim=(0, 1))
            ax.set_xticks(np.arange(len(order)), labels, fontsize=8)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_axisbelow(True)
            ax.grid(axis="y", alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(.5, .905), fontsize=9)
    fig.suptitle("Why exact Lasso recovery fails", y=.995, fontsize=16)
    fig.text(.5, .945, rf"$\theta={theta:g}$; paper: active effects $\pm1$; weak: five at $\pm0.25$, rest $\pm1$; "
             rf"noise SD $={config['noise_sigma']:g}$", ha="center", fontsize=10)
    fig.supylabel("Fraction of simulations")
    fig.text(.02, .015,
             f"{summary['trials'].iloc[0]:,} simulation trials per bar. Exact recovery requires every support location and sign to be correct.\n"
             "Extra = a selected null variable; missed = an omitted true variable. Wrong-sign cases take priority; all categories sum to 100%.\n"
             "Gaussian predictors: independent (left) or Toeplitz with adjacent correlation 0.1 (right); random support and signs.\n"
             "n = floor[2θk log(p − k)] + k + 1; sparsity uses α = 0.4, γ = 0.75. Frequencies are descriptive; segments ≥12% are labeled.",
             ha="left", va="bottom", fontsize=7.7, linespacing=1.4)
    fig.tight_layout(rect=(.025, .13, 1, .875))
    save_figure(fig, output_dir, "03_failure_modes")


def plot_runtime(raw, config, output_dir, tables_dir):
    """Show the distribution of fit times over the larger-p experiment."""
    runtime = raw.assign(fit_ms=1000 * raw["fit_seconds"])
    summary = summarize_runtime(raw)
    summary.to_csv(Path(tables_dir) / "runtime_distribution.csv", index=False)
    dimensions = sorted(config["p_values"])
    fig, axes = plt.subplots(len(config["regimes"]), len(config["designs"]),
                             figsize=(11, 6.5), sharey=True, squeeze=False)
    for row, regime in enumerate(config["regimes"]):
        for col, design in enumerate(config["designs"]):
            ax = axes[row, col]
            facet = runtime[(runtime["design"] == design["name"]) & (runtime["regime"] == regime)]
            values = [facet.loc[facet["p"] == p, "fit_ms"].to_numpy() for p in dimensions]
            boxes = ax.boxplot(values, tick_labels=[str(p) for p in dimensions], showfliers=False,
                               patch_artist=True, widths=.5, medianprops={"color": "black", "linewidth": 1.3})
            for box, color in zip(boxes["boxes"], ["#0072B2", "#CC79A7"]):
                box.set_facecolor(color)
                box.set_alpha(.75)
            ratio = summary.loc[(summary["design"] == design["name"])
                                & (summary["regime"] == regime)
                                & (summary["p"] == dimensions[-1]), "median_ratio"].iloc[0]
            ax.text(.04, .94, f"Median {dimensions[-1]}/{dimensions[0]} = {ratio:.2f}×",
                    transform=ax.transAxes, va="top", fontsize=9,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85})
            ax.set_yscale("log")
            ax.set_title(f"{REGIME_TITLES[regime]} — {design['name'].capitalize()}")
            ax.set_axisbelow(True)
            ax.grid(axis="y", alpha=.2)
    fig.suptitle("Distribution of Lasso fit times as dimension increases", fontsize=16)
    fig.supylabel("Lasso fit time (ms; logarithmic scale)")
    fig.supxlabel("Dimension p", y=.155)
    fig.text(.02, .015, f"Each box pools {summary['fits'].iloc[0]:,} fits over θ = 0.1, 0.3, …, 2.3 "
             f"({config['repetitions']} trials per value); both n and k grow with p.\n"
             "Gaussian predictors: independent (left) or Toeplitz, adjacent correlation 0.1 (right); noise SD = 0.5; active magnitudes = 1.\n"
             "n = floor[2θk log(p − k)] + k + 1; α = 0.4, γ = 0.75. Timing includes fitting only, on the same machine and software.\n"
             "Boxes: middle 50%; line: median; whiskers: data within 1.5 × IQR fences. Outlier markers hidden; all fits retained in summaries.\n"
             "Boxes describe variability among fits, not confidence intervals for the medians.",
             ha="left", va="bottom", fontsize=7.7, linespacing=1.4)
    fig.tight_layout(rect=(.025, .21, 1, .95))
    save_figure(fig, output_dir, "04_runtime")


def create_figures(results_dir="results"):
    """Build all figures."""
    results_dir = Path(results_dir)
    output_dir = results_dir / "figures"
    tables_dir = results_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    runs = {label: read_run(results_dir / "raw" / name) for label, name in RUNS.items()}
    labels = ["Paper setting", "Five weak effects"]
    raw = pd.concat([runs[label][0].assign(setting=label) for label in labels], ignore_index=True)
    baseline_config = runs["Paper setting"][1]
    summary = pd.concat([summarize_results(runs[label][0]).assign(setting=label)
                         for label in labels], ignore_index=True)
    summary["ci_lower"], summary["ci_upper"] = wilson_interval(
        summary.successes, summary.trials, PLOT_CONFIG["confidence"])
    summary.to_csv(tables_dir / "recovery_with_intervals.csv", index=False)

    with plt.rc_context({
        "font.size": 9, "axes.titlesize": 9,
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }):
        plot_recovery(summary, baseline_config, output_dir)
        plot_diagnostics(raw, baseline_config, output_dir, tables_dir)
        large_raw, large_config = runs["Larger dimensions"]
        plot_runtime(large_raw, large_config, output_dir, tables_dir)
    print(f"Saved four final figures in {output_dir}; plotting tables in {tables_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results", help="Results root containing raw runs.")
    args = parser.parse_args()
    create_figures(args.results)
