import numpy as np



def sigmoid(x):
    """Compute the sigmoid function for the input x.
    Args:
        x (np.array): Input array
    Returns:
        np.array: Sigmoid of the input array
    """
    return 1/(1+ np.exp(-x))

def cross_entropy(y, pred, eps=1e-15):
    """Compute the cross-entropy loss.
    Args:
        y (np.array): True labels
        pred (np.array): Predicted probabilities
        eps (float): Epsilon value to clip predictions
    Returns:
        float: Cross-entropy loss
    """
    n = y.shape[0]
    pred = np.clip(pred, eps, 1 - eps)
    return - ((y.T @ np.log(pred)) + ((np.ones(y.shape)-y).T @ np.log(np.ones(pred.shape) - pred)))/ n

def cross_entropy_grad(y, tx, pred):
    """Compute the gradient of the cross-entropy loss.
    Args:
        y (np.array): True labels
        tx (np.array): Features
        pred (np.array): Predicted probabilities
    Returns:
        np.array: Gradient of the cross-entropy loss
    """
    n = y.shape[0]
    return tx.T @ (pred - y)/ n


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

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent.

    Args:
        y (np.array): Labels
        tx (np.array): Features
        initial_w (np.array): Initial weights
        max_iters (int): Maximum number of iterations
        gamma (float): Learning rate

    Returns:
        w (np.array): Final weights
        loss (float): Final loss
    """
    w = initial_w
    for i in range(max_iters):
        pred = sigmoid(tx @ w)
        w = w - gamma * cross_entropy_grad(y, tx, pred)
    loss = cross_entropy(y, sigmoid(tx @ w))
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent.

    Args:
        y (np.array): Labels
        tx (np.array): Features
        lambda_ (float): Regularization parameter
        initial_w (np.array): Initial weights
        max_iters (int): Maximum number of iterations
        gamma (float): Learning rate

    Returns:
        w (np.array): Final weights
        loss (float): Final loss
    """
    w = initial_w
    for i in range(max_iters):
        pred = sigmoid(tx @ w)
        w = w - gamma * (cross_entropy_grad(y, tx, pred) + 2 * lambda_ * w)
        
    loss = cross_entropy(y, sigmoid(tx @ w))
    return w, loss

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