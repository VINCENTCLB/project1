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
    N = y.shape[0]
    pred = np.clip(pred, eps, 1 - eps)
    return - ((y.T @ np.log(pred)) + ((np.ones(y.shape)-y).T @ np.log(np.ones(pred.shape) - pred)))/ N

def cross_entropy_grad(y, tx, pred):
    """Compute the gradient of the cross-entropy loss.
    Args:
        y (np.array): True labels
        tx (np.array): Features
        pred (np.array): Predicted probabilities
    Returns:
        np.array: Gradient of the cross-entropy loss
    """
    N = y.shape[0]
    return tx.T @ (pred - y)/ N


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