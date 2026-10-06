"""The GD and SGD portion of EPFL CS-433 Project 1.

Merge these functions into the team's implementations.py. The other four
required methods are implemented by teammates.
"""

import numpy as np


def _mse_loss(y, tx, w):
    """Return the full-data loss ||y - tx @ w||^2 / (2 * N)."""
    residual = y - tx @ w
    return np.mean(residual**2) / 2


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Fit linear regression with full-batch gradient descent.

    Args:
        y: Targets, shape (N,).
        tx: Design matrix, shape (N, D), including an intercept if wanted.
        initial_w: Initial weights, shape (D,). This array is not modified.
        max_iters: Number of gradient updates; may be zero.
        gamma: Learning rate.

    Returns:
        (w, loss): Final weights of shape (D,) and the scalar full-data MSE
        loss with the course's factor of 1/2, evaluated at those weights.
    """
    w = np.array(initial_w, dtype=float, copy=True)
    n_samples = len(y)

    for _ in range(max_iters):
        residual = y - tx @ w
        gradient = -(tx.T @ residual) / n_samples
        w = w - gamma * gradient

    return w, _mse_loss(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Fit linear regression with single-sample stochastic gradient descent.

    Args:
        y: Targets, shape (N,).
        tx: Design matrix, shape (N, D), including an intercept if wanted.
        initial_w: Initial weights, shape (D,). This array is not modified.
        max_iters: Number of single-sample updates, not number of epochs.
        gamma: Learning rate.

    Returns:
        (w, loss): Final weights of shape (D,) and the scalar full-data MSE
        loss with a factor of 1/2, evaluated at those weights.

    Each update samples one row uniformly with replacement. Set the NumPy
    random seed in the calling experiment to make results reproducible.
    """
    w = np.array(initial_w, dtype=float, copy=True)
    n_samples = len(y)

    for _ in range(max_iters):
        index = np.random.randint(n_samples)
        x_i = tx[index]
        residual_i = y[index] - x_i @ w
        gradient = -x_i * residual_i
        w = w - gamma * gradient

    return w, _mse_loss(y, tx, w)
