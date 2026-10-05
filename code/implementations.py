import NumPy as np



def sigmoid(x):
    return 1/(1+ np.exp(-x))

def cross_entropy(y, pred):
    return - np.mean((y.T @ np.log(pred)) + ((np.ones(y.shape)-y).T @ np.log(np.ones(pred.shape) - pred)))

def cross_entropy_grad(y, tx, pred):
    return tx.T @ (pred - y)


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
    loss = 0
    for i in range(max_iters):
        pred = sigmoid(tx @ w)
        loss = cross_entropy(y, pred)
        w = w - gamma * cross_entropy_grad(y, tx, pred)
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
    loss = 0
    for i in range(max_iters):
        pred = sigmoid(tx @ w)
        loss = cross_entropy(y, pred) + lambda_ * np.sum(w**2)
        w = w - gamma * (cross_entropy_grad(y, tx, pred) + 2 * lambda_ * w)
    return w, loss