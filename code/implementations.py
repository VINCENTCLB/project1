import numpy as np

def least_squares(y, tx):
    """Least squares regression via the normal equations.

    Args:
        y: Targets, shape (N,).
        tx: Design matrix, shape (N, D).

    Returns:
        (w, loss): Optimal weights, shape (D,), and full-data MSE loss.
    """
    lhs_matrix  = tx.T @ tx
    rhs = tx.T @ y
    w = np.linalg.solve(lhs_matrix , rhs)

    return w, _mse_loss(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Ridge regression via the normal equations.

    Minimises L(w) + lambda_ * ||w||^2, where L is the MSE with 1/2 factor.
    Setting the gradient to zero gives
        (X^T X + 2 N lambda_ I) w = X^T y

    Args:
        y: Targets, shape (N,).
        tx: Design matrix, shape (N, D).
        lambda_: Regularization parameter.

    Returns:
        (w, loss): Optimal weights, shape (D,), and full-data MSE loss
    """
    n_samples, n_features = tx.shape
    lambda_prime = 2 * n_samples * lambda_
    lhs_matrix  = tx.T @ tx + lambda_prime * np.eye(n_features)
    rhs = tx.T @ y
    w = np.linalg.solve(lhs_matrix , rhs)

    return w, _mse_loss(y, tx, w)