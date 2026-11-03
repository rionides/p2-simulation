"""Data-generating functions"""

import numpy as np


def sparsity_size(p, regime, alpha, gamma):
    """Calculate ks, rounding down (paper does not specify rounding)."""
    if regime == "linear":
        value = alpha * p
    elif regime == "sublinear":
        value = alpha * p / np.log(alpha * p)
    elif regime == "fractional":
        value = alpha * p**gamma
    else:
        raise ValueError(f"Unknown sparsity regime: {regime}")
    k = int(np.floor(value))
    if not 2 <= k < p - 1:
        raise ValueError("parameters must be in range 2 <= k < p - 1.")
    return k


def sample_size(p, k, theta):
    """Use Section 4's sample size."""
    if theta <= 0:
        raise ValueError("theta must be positive")
    return int(np.floor(2 * theta * k * np.log(p - k))) + k + 1


def generate_data(n, p, k, sigma, mu, rng, weak_count=0, weak_magnitude=0.25):
    """Returns X, y, beta for y = X beta + N(0, sigma**2). """
    if not 0 <= sigma or not -1 < mu < 1:
        raise ValueError("requires sigma >= 0 and -1 < mu < 1.")
    if not isinstance(weak_count, (int, np.integer)) or not 0 <= weak_count <= k:
        raise ValueError("weak_count must be an integer between 0 and k.")
    if not 0 < weak_magnitude <= 1:
        raise ValueError("weak_magnitude must be in (0, 1].")
    X = rng.standard_normal((n, p))
    if mu != 0:
        # exact Gaussian sampler for Sigma[i,j] = mu**abs(i-j) (paper, eq. 39).
        innovation_scale = np.sqrt(1 - mu**2)
        for j in range(1, p):
            X[:, j] = mu * X[:, j - 1] + innovation_scale * X[:, j]

    beta = np.zeros(p)
    support = rng.choice(p, size=k, replace=False)#
    beta[support] = rng.choice([-1.0, 1.0], size=k)
    beta[support[:weak_count]] *= weak_magnitude
    y = X @ beta + rng.normal(0, sigma, size=n)
    return X, y, beta
