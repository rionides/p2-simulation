from ast import literal_eval
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.metrics import summarize_results


class SummaryTests(unittest.TestCase):
    def test_summary_matches_hand_calculation(self):
        rows = []
        for theta, outcomes in [(.5, [0, 0, 0, 0]), (1., [1, 0, 1, 0]), (2., [1, 1, 1, 1])]:
            for success in outcomes:
                rows.append(dict(design="identity", mu=0, regime="linear", p=128,
                                 k=51, theta=theta, n=495, penalty=.2, success=success))
        summary = summarize_results(pd.DataFrame(rows)).sort_values("theta")
        np.testing.assert_array_equal(summary["trials"], [4, 4, 4])
        np.testing.assert_array_equal(summary["successes"], [0, 2, 4])
        np.testing.assert_allclose(summary["success_probability"], [0, .5, 1])
        np.testing.assert_allclose(summary["mcse"], [0, .25, 0])


class SavedOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        results = Path(__file__).resolve().parents[1] / "results"
        output = results / "raw" / "full"
        cls.raw = pd.read_csv(output / "raw_results.csv")
        cls.summary = pd.read_csv(results / "tables" / "full_summary.csv")
        cls.config = literal_eval((output / "config_used.txt").read_text())
        cls.metadata = literal_eval((output / "metadata.txt").read_text())

    def assert_valid_outputs(self, raw, summary, config, metadata):
        self.assertTrue(metadata["complete"])
        expected = config["repetitions"]
        for key in ["designs", "regimes", "p_values", "theta_values"]:
            expected *= len(config[key])
        self.assertEqual(len(raw), expected)
        self.assertEqual(metadata["expected_trials"], expected)
        counts = raw.groupby(["design", "regime", "p", "theta"]).size()
        self.assertTrue(counts.eq(config["repetitions"]).all())
        self.assertTrue(raw["seed"].is_unique)
        self.assertTrue(raw["success"].isin([0, 1]).all())
        for name, default in [("weak_count", 0), ("weak_magnitude", .25)]:
            if name in raw:
                self.assertTrue(raw[name].eq(config.get(name, default)).all())
        for name, upper in [("false_positives", raw.p - raw.k),
                            ("false_negatives", raw.k), ("sign_errors", raw.k)]:
            self.assertTrue(((raw[name] >= 0) & (raw[name] <= upper)).all())
        errors = raw[["false_positives", "false_negatives", "sign_errors"]].sum(axis=1)
        np.testing.assert_array_equal(raw["success"], (errors == 0).astype(int))
        self.assertTrue(np.isfinite(summary[["success_probability", "mcse"]]).all().all())
        self.assertTrue(summary["success_probability"].between(0, 1).all())
        self.assertTrue((summary["mcse"] >= 0).all())
        self.assertTrue((summary["mcse"] <= .5 / np.sqrt(summary["trials"]) + 1e-12).all())
        pd.testing.assert_frame_equal(summary, summarize_results(raw), atol=1e-12, rtol=1e-12)

    def test_saved_counts_and_metric_ranges(self):
        self.assert_valid_outputs(self.raw, self.summary, self.config, self.metadata)

    def test_saved_weak_signal_outputs(self):
        root = Path(__file__).resolve().parents[1]
        name = "weak_signals"
        output = root / "results" / "raw" / name
        required = ["raw_results.csv", "config_used.txt", "metadata.txt"]
        if not all((output / file).is_file() for file in required):
            self.skipTest(f"Complete {name}.")
        metadata = literal_eval((output / "metadata.txt").read_text())
        if not metadata["complete"]:
            self.skipTest(f"The simulation in {name} is incomplete !!")
        config = literal_eval((output / "config_used.txt").read_text())
        raw = pd.read_csv(output / "raw_results.csv")
        summary = pd.read_csv(root / "results" / "tables" / f"{name}_summary.csv")
        self.assert_valid_outputs(raw, summary, config, metadata)
        columns = ["seed", "design", "mu", "regime", "p", "k", "theta", "n", "penalty"]
        pd.testing.assert_frame_equal(self.raw[columns], raw[columns], check_exact=True)

    def test_broad_agreement_with_paper_threshold_pattern(self):
        # qualitative finite sample checks to see if everything is basically working as advertised in the paper
        paper = dict(repetitions=200, p_values=[128, 256, 512], alpha=.4, gamma=.75,
                     noise_sigma=.5, regimes=["linear", "sublinear", "fractional"],
                     theta_values=[.1, .3, .5, .7, .9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3])
        designs = {d["name"]: d["mu"] for d in self.config["designs"]}
        if any(self.config[k] != v for k, v in paper.items()) or designs != {"identity": 0, "toeplitz": .1}:
            self.skipTest("need full-study settings to compare with paper")
        for (design, regime, p), curve in self.summary.groupby(["design", "regime", "p"]):
            with self.subTest(design=design, regime=regime, p=p):
                low = curve[curve.theta <= .3].success_probability
                high = curve[curve.theta >= 2.1].success_probability
                self.assertFalse(low.empty or high.empty)
                self.assertTrue((low <= .1).all())
                self.assertTrue((high >= .9).all())
                crossing = curve[curve.success_probability >= .5].theta.min()
                lower, upper = (.6, 1.4) if design == "identity" else (.46, 1.84)
                self.assertTrue(lower <= crossing <= upper)


if __name__ == "__main__":
    unittest.main()
