""" Lasso penalty."""

import numpy as np
from sklearn.linear_model import Lasso


def lasso_penalty(n, p, k):
    """Section 4's rho_n = sqrt(log(p-k) * log(k) / n)."""
    return float(np.sqrt(np.log(p - k) * np.log(k) / n))


def fit_lasso(X, y, penalty, tolerance, max_iter):
    """Minimize ||y-Xb||^2/(2n) + penalty*||b||_1, as in equation (3).

    scikit-learn's alpha equals the paper's rho_n directly so this is just a sklearn wrapper
    """
    model = Lasso(
        alpha=penalty,
        fit_intercept=False,
        tol=tolerance,
        max_iter=max_iter,
        selection="cyclic",
    )
    return model.fit(X, y)
