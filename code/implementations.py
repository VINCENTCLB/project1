"""GD and SGD for EPFL CS-433 Project 1."""

import numpy as np


def _mse_loss(y, tx, w):
    """Return the full-data MSE loss (with 1/2 factor)."""
    residual = y - tx @ w
    return np.mean(residual**2) / 2


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression via full-batch gradient descent.

    Args:
        y: Targets, shape (N,).
        tx: Design matrix, shape (N, D).
        initial_w: Initial weights, shape (D,); not modified.
        max_iters: Number of updates (may be 0).
        gamma: Learning rate.

    Returns:
        (w, loss): Final weights and full-data MSE loss.
    """
    w = np.array(initial_w, dtype=float, copy=True)
    n_samples = len(y)

    for _ in range(max_iters):
        residual = y - tx @ w
        gradient = -(tx.T @ residual) / n_samples
        w = w - gamma * gradient

    return w, _mse_loss(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression via single-sample SGD.

    Each update samples one row uniformly with replacement; seed
    np.random in the caller for reproducibility.

    Args:
        y: Targets, shape (N,).
        tx: Design matrix, shape (N, D).
        initial_w: Initial weights, shape (D,); not modified.
        max_iters: Number of single-sample updates (not epochs).
        gamma: Learning rate.

    Returns:
        (w, loss): Final weights and full-data MSE loss.
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
