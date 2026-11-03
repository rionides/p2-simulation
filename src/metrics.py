"""Recovery measures and numerical summaries for the simulation study."""

from statistics import NormalDist

import numpy as np


FAILURE_MODES = ["Exact recovery", "Extra only", "Missed only", "Extra + missed", "Wrong sign(s)"]


def evaluate_recovery(beta_true, beta_hat, zero_tolerance):
    """Success requires exactly the correct nonzero locations AND signs."""
    true_sign = np.sign(beta_true)
    estimated_sign = np.where(np.abs(beta_hat) > zero_tolerance, np.sign(beta_hat), 0)
    active = true_sign != 0
    selected = estimated_sign != 0
    return {
        "success": int(np.array_equal(true_sign, estimated_sign)),
        "false_positives": int(np.sum(selected & ~active)),
        "false_negatives": int(np.sum(~selected & active)),
        "sign_errors": int(np.sum(selected & active & (true_sign != estimated_sign))),
    }


def summarize_results(raw):
    """Estimates success probability and its binomial Monte Carlo standard error."""
    summary = raw.groupby(["design", "mu", "regime", "p", "k", "theta", "n", "penalty"],
                          as_index=False).agg(
        trials=("success", "size"),
        successes=("success", "sum"),
        success_probability=("success", "mean"),
    )
    probability = summary["success_probability"]
    summary["mcse"] = np.sqrt(probability * (1 - probability) / summary["trials"])
    return summary


def wilson_interval(successes, trials, confidence=0.95):
    """Pointwise binomial confidence intervals"""
    successes, trials = np.asarray(successes), np.asarray(trials)
    if np.any(trials <= 0) or np.any(successes < 0) or np.any(successes > trials):
        raise ValueError("Requires 0 <= successes <= trials.")
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    probability = successes / trials
    denominator = 1 + z**2 / trials
    center = (probability + z**2 / (2 * trials)) / denominator
    half_width = z * np.sqrt(probability * (1 - probability) / trials
                             + z**2 / (4 * trials**2)) / denominator
    return np.clip(center - half_width, 0, 1), np.clip(center + half_width, 0, 1)


def classify_failures(raw):
    extra = raw["false_positives"] > 0
    missed = raw["false_negatives"] > 0
    return np.select(
        [raw["sign_errors"] > 0, extra & missed, extra, missed],
        ["Wrong sign(s)", "Extra + missed", "Extra only", "Missed only"],
        default="Exact recovery",
    )


def summarize_failure_modes(raw, theta):
    """Gives category fractions per theta"""
    settings = ["Paper setting", "Five weak effects"]
    selected = raw.loc[np.isclose(raw["theta"], theta) & raw["setting"].isin(settings)].copy()
    if selected.empty:
        raise ValueError(f"No saved trials at diagnostic theta={theta}.")
    selected["failure_mode"] = classify_failures(selected)
    counts = selected.groupby(["design", "regime", "p", "setting", "failure_mode"]).size()
    summary = counts.unstack("failure_mode", fill_value=0).reindex(columns=FAILURE_MODES, fill_value=0)
    summary["trials"] = summary.sum(axis=1)
    summary[FAILURE_MODES] = summary[FAILURE_MODES].div(summary["trials"], axis=0)
    summary = summary.reset_index()
    summary["theta"] = theta
    return summary


def summarize_runtime(raw):
    """Summarizes fit times and ratios to the smallest p within each design/regime."""
    runtime = raw.assign(fit_ms=1000 * raw["fit_seconds"])
    summary = runtime.groupby(["design", "regime", "p"])["fit_ms"].agg(
        fits="size", median_ms="median", q1_ms=lambda x: x.quantile(.25), q3_ms=lambda x: x.quantile(.75),
    ).reset_index()
    summary = summary.sort_values(["design", "regime", "p"]).reset_index(drop=True)
    reference = summary.groupby(["design", "regime"])["median_ms"].transform("first")
    summary["median_ratio"] = summary["median_ms"] / reference
    return summary
