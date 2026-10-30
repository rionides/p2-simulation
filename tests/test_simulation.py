import unittest

import numpy as np

from dgp import generate_data, sample_size, sparsity_size
from evaluation import evaluate_recovery
from methods import fit_lasso, lasso_penalty


class SimulationTests(unittest.TestCase):
    def test_paper_parameters(self):
        expected = {"linear": 51, "sublinear": 13, "fractional": 15}
        for regime, k in expected.items():
            self.assertEqual(sparsity_size(128, regime, .4, .75), k)
        self.assertEqual(sample_size(128, 51, 1), 495)
        self.assertAlmostEqual(lasso_penalty(495, 128, 51), .1857503782842714)

    def test_dgp_seed_and_noiseless_response(self):
        first = generate_data(30, 12, 4, 0, .1, np.random.default_rng(607))
        second = generate_data(30, 12, 4, 0, .1, np.random.default_rng(607))
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        X, y, beta = first
        self.assertEqual(np.count_nonzero(beta), 4)
        self.assertTrue(np.all(np.isin(beta, [-1, 0, 1])))
        np.testing.assert_allclose(y, X @ beta)

    def test_gaussian_covariances(self):
        for mu in [0, .6]:
            X, y, beta = generate_data(
                100_000, 4, 2, .5, mu, np.random.default_rng(607)
            )
            expected = mu ** np.abs(np.arange(4)[:, None] - np.arange(4))
            np.testing.assert_allclose(np.cov(X, rowvar=False), expected, atol=.025)
            np.testing.assert_allclose(X.mean(axis=0), 0, atol=.015)
            noise = y - X @ beta
            self.assertAlmostEqual(noise.mean(), 0, delta=.01)
            self.assertAlmostEqual(noise.std(), .5, delta=.01)
            np.testing.assert_allclose(X.T @ noise / len(noise), 0, atol=.01)

    def test_weak_effects_preserve_paired_design_signs_and_noise(self):
        X, y, beta = generate_data(50, 20, 8, .5, .1, np.random.default_rng(607))
        weak_X, weak_y, weak_beta = generate_data(
            50, 20, 8, .5, .1, np.random.default_rng(607), weak_count=5,
            weak_magnitude=.25,
        )
        self.assertEqual(np.count_nonzero(np.abs(weak_beta) == .25), 5)
        self.assertEqual(np.count_nonzero(np.abs(weak_beta) == 1), 3)
        self.assertEqual(np.count_nonzero(weak_beta), 8)
        np.testing.assert_array_equal(weak_X, X)
        np.testing.assert_array_equal(np.sign(weak_beta), np.sign(beta))
        np.testing.assert_allclose(weak_y - weak_X @ weak_beta, y - X @ beta, atol=1e-14)

    def test_invalid_weak_effect_parameters(self):
        for parameters in [{"weak_count": 9}, {"weak_count": -1}, {"weak_count": 1.5},
                           {"weak_magnitude": -.25}, {"weak_magnitude": 0},
                           {"weak_magnitude": 1.1}]:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                generate_data(50, 20, 8, .5, .1, np.random.default_rng(607), **parameters)

    def test_lasso_matches_orthogonal_soft_threshold(self):
        # X.T @ X / n = I
        X = 2 * np.eye(4)
        beta = np.array([2., -1., .1, 0.])
        fit = fit_lasso(X, X @ beta, .3, 1e-10, 1000)
        np.testing.assert_allclose(fit.coef_, [1.7, -.7, 0, 0], atol=1e-12)

    def test_lasso_recovers_simple_noiseless_signal(self):
        X, y, beta = generate_data(200, 12, 3, 0, 0, np.random.default_rng(607))
        fit = fit_lasso(X, y, .05, 1e-10, 1000)
        self.assertEqual(evaluate_recovery(beta, fit.coef_, 1e-7)["success"], 1)
        np.testing.assert_allclose(fit.coef_, beta, atol=.1)

    def test_recovery_requires_correct_signs(self):
        true = np.array([1., -1., 0., 0.])
        wrong_sign = evaluate_recovery(true, np.array([.8, .7, 0, 0]), 1e-8)
        self.assertEqual(wrong_sign["success"], 0)
        self.assertEqual(wrong_sign["false_positives"], 0)
        self.assertEqual(wrong_sign["false_negatives"], 0)
        self.assertEqual(wrong_sign["sign_errors"], 1)

    def test_recovery_tolerance_and_selection_errors(self):
        true = np.array([1., -1., 0., 0.])
        tiny = np.array([.8, -.7, 1e-10, 0])
        recovered = evaluate_recovery(true, tiny, 1e-8)
        self.assertEqual(recovered["success"], 1)
        self.assertEqual(evaluate_recovery(true, tiny, 0)["false_positives"], 1)
        errors = evaluate_recovery(true, np.array([0., -1., .2, 0]), 1e-8)
        self.assertEqual(errors["success"], 0)
        self.assertEqual(errors["false_positives"], 1)
        self.assertEqual(errors["false_negatives"], 1)
        self.assertEqual(errors["sign_errors"], 0)


if __name__ == "__main__":
    unittest.main()
