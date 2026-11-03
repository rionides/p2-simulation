"""check uncertainty calculations + failute"""

import unittest

import numpy as np
import pandas as pd

from visualize import classify_failures, wilson_interval


class VisualizationTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
