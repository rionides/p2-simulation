"""Run independent trials or summarize their saved results."""

import argparse
from ast import literal_eval
import csv
from importlib.metadata import version
from itertools import product
import logging
from pathlib import Path
import platform
from pprint import pformat
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from src.config import CONFIGS
from src.dgps import generate_data, sample_size, sparsity_size
from src.metrics import evaluate_recovery, summarize_results
from src.methods import fit_lasso, lasso_penalty


def validate_config(config):
    for key in ["seed", "repetitions", "max_iterations", "log_every"]:
        value = config[key]
        if not isinstance(value, int) or value < (0 if key == "seed" else 1):
            raise ValueError(f"{key} must be an integer.")
    for key in ["p_values", "theta_values", "regimes", "designs"]:
        if not config[key]:
            raise ValueError(f"{key} cannot be empty.")
    if not 0 < config["alpha"] < 1 or not 0 < config["gamma"] < 1:
        raise ValueError("alpha, gamma must be in [0, 1]")
    if config["noise_sigma"] < 0 or config["zero_tolerance"] < 0:
        raise ValueError("noise and zero tolerance must be >= 0")
    if config["solver_tolerance"] <= 0:
        raise ValueError("solver tolerance must be >0")
    if any(not -1 < d["mu"] < 1 for d in config["designs"]):
        raise ValueError("designs must have -1 < mu < 1.")
    weak_count = config.get("weak_count", 0)
    if not isinstance(weak_count, int) or weak_count < 0:
        raise ValueError("weak_count must be a nonnegative integer.")
    if not 0 < config.get("weak_magnitude", 0.25) <= 1:
        raise ValueError("weak_magnitude must be between zero and one.")
    for p, regime in product(config["p_values"], config["regimes"]):
        k = sparsity_size(p, regime, config["alpha"], config["gamma"])
        if weak_count > k:
            raise ValueError("weak_count cannot exceed the support size k.")
        for theta in config["theta_values"]:
            sample_size(p, k, theta)


def run_simulations(config, output_dir):
    """
    Generate data, fit Lasso, and save scores for every setting and trial.
    """
    validate_config(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / "simulation.log", mode="w")],
        force=True,
    )
    
    total = config["repetitions"]
    for key in ["designs", "regimes", "p_values", "theta_values"]:
        total *= len(config[key])
    (output_dir / "config_used.txt").write_text(pformat(config, sort_dicts=False) + "\n")
    
    metadata = {
        "complete": False,
        "expected_trials": total,
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ["numpy", "scikit-learn", "pandas", "matplotlib"]},
    }
    metadata_path = output_dir / "metadata.txt"
    metadata_path.write_text(pformat(metadata, sort_dicts=False) + "\n")
    
    fields = [
        "design", "mu", "regime", "p", "k", "theta", "n", "penalty", "trial", "seed",
        "weak_count", "weak_magnitude", "success", "false_positives", "false_negatives",
        "sign_errors", "n_iter", "dual_gap", "fit_seconds",
    ]
    completed = 0
    settings = product(config["designs"], config["regimes"], config["p_values"], config["theta_values"])
    logging.info("Starting %s trials; raw results: %s", total, output_dir / "raw_results.csv")
    
    
    # A solver failure stops the run instead of counting an inaccurate fit as a failure.
    with warnings.catch_warnings(), (output_dir / "raw_results.csv").open("w", newline="") as file:
        warnings.simplefilter("error", ConvergenceWarning)
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for design, regime, p, theta in settings:
            k = sparsity_size(p, regime, config["alpha"], config["gamma"])
            n = sample_size(p, k, theta)
            penalty = lasso_penalty(n, p, k)
            for trial in range(1, config["repetitions"] + 1):
                seed = config["seed"] + completed
                rng = np.random.default_rng(seed)
                X, y, beta = generate_data(
                    n, p, k, config["noise_sigma"], design["mu"], rng,
                    weak_count=config.get("weak_count", 0),
                    weak_magnitude=config.get("weak_magnitude", 0.25),
                )
                start = perf_counter()
                model = fit_lasso(X, y, penalty, config["solver_tolerance"], config["max_iterations"])
                fit_seconds = perf_counter() - start
                scores = evaluate_recovery(beta, model.coef_, config["zero_tolerance"])
                writer.writerow({
                    "design": design["name"], "mu": design["mu"], "regime": regime,
                    "p": p, "k": k, "theta": theta, "n": n, "penalty": penalty,
                    "trial": trial, "seed": seed, **scores,
                    "weak_count": config.get("weak_count", 0),
                    "weak_magnitude": config.get("weak_magnitude", 0.25),
                    "n_iter": model.n_iter_, "dual_gap": model.dual_gap_,
                    "fit_seconds": fit_seconds,
                })
                completed += 1
                if completed % config["log_every"] == 0 or completed == total:
                    file.flush()
                    logging.info("[%s/%s] %s, %s, p=%s, theta=%.2f", completed, total,
                                 design["name"], regime, p, theta)
    metadata["complete"] = True
    metadata["completed_trials"] = completed
    metadata_path.write_text(pformat(metadata, sort_dicts=False) + "\n")
    logging.info("All done, %s trial results saved.", completed)


def summarize_run(raw_dir, tables_dir):
    """Use saved results; refuse to summarize an interrupted simulation."""
    raw_dir = Path(raw_dir)
    tables_dir = Path(tables_dir)
    # These files contain the Python dictionaries saved with the simulation.
    config = literal_eval((raw_dir / "config_used.txt").read_text())
    metadata = literal_eval((raw_dir / "metadata.txt").read_text())
    raw = pd.read_csv(raw_dir / "raw_results.csv")
    expected = config["repetitions"]
    for key in ["designs", "regimes", "p_values", "theta_values"]:
        expected *= len(config[key])
    if not metadata["complete"] or len(raw) != expected or metadata["expected_trials"] != expected:
        raise ValueError("The simulation is incomplete. Rerun make simulate first.")
    summary = summarize_results(raw)
    tables_dir.mkdir(parents=True, exist_ok=True)
    output = tables_dir / f"{raw_dir.name}_summary.csv"
    summary.to_csv(output, index=False)
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["simulate", "summarize"])
    parser.add_argument("--config", choices=["all", *CONFIGS], default="all")
    parser.add_argument("--results", default="results", help="Root directory for all outputs.")
    args = parser.parse_args()
    names = ["full", "weak_signals", "larger_p"] if args.config == "all" else [args.config]
    for name in names:
        raw_dir = Path(args.results) / "raw" / name
        if args.stage == "simulate":
            run_simulations(CONFIGS[name], raw_dir)
        else:
            summarize_run(raw_dir, Path(args.results) / "tables")


if __name__ == "__main__":
    main()
