"""Check uncertainty calculations and performance summaries."""

import unittest

import numpy as np
import pandas as pd

from src.metrics import (
    FAILURE_MODES, classify_failures, summarize_failure_modes,
    wilson_interval,
)


class MetricTests(unittest.TestCase):
    def test_wilson_intervals_match_reference_values(self):
        # Reference: scipy.stats.binomtest(k, 200).proportion_ci(method="wilson").
        lower, upper = wilson_interval(np.array([0, 100, 200]), 200)
        np.testing.assert_allclose(lower, [0, .43136085960389187, .9811546736227335], atol=1e-12)
        np.testing.assert_allclose(upper, [.018845326377266575, .5686391403961082, 1], atol=1e-12)
        self.assertTrue(np.all(upper > lower))

    def test_wilson_intervals_reject_invalid_counts(self):
        for successes, trials in [(0, 0), (-1, 200), (201, 200)]:
            with self.subTest(successes=successes, trials=trials), self.assertRaises(ValueError):
                wilson_interval(successes, trials)

    def test_diagnostic_categories_are_exclusive(self):
        raw = pd.DataFrame({
            "false_positives": [0, 2, 0, 1, 0, 2],
            "false_negatives": [0, 0, 2, 1, 0, 2],
            "sign_errors": [0, 0, 0, 0, 1, 1],
        })
        np.testing.assert_array_equal(classify_failures(raw), [
            "Exact recovery", "Extra only", "Missed only", "Extra + missed",
            "Wrong sign(s)", "Wrong sign(s)",
        ])

    def test_failure_summary_matches_known_fractions(self):
        raw = pd.DataFrame({
            "false_positives": [0, 2, 0, 1, 0, 2],
            "false_negatives": [0, 0, 2, 1, 0, 2],
            "sign_errors": [0, 0, 0, 0, 1, 1],
            "design": "identity", "regime": "linear", "p": 128,
            "setting": "Paper setting", "theta": 1.3,
        })
        # Other sample-size conditions must not enter this diagnostic.
        other = raw.assign(theta=2.3)
        summary = summarize_failure_modes(pd.concat([raw, other]), theta=1.3)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary["trials"].iloc[0], 6)
        np.testing.assert_allclose(summary[FAILURE_MODES].iloc[0], [1/6, 1/6, 1/6, 1/6, 2/6])
        self.assertAlmostEqual(summary[FAILURE_MODES].iloc[0].sum(), 1)

if __name__ == "__main__":
    unittest.main()
