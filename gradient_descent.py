import numpy as np

def mse(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    return np.mean((y_true - y_pred) ** 2)

def r_squared(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return 1 - mse(y_true, y_pred) / np.var(y_true)

def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    # Make sure univariate data still works
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_samples = X.shape[0]

    X = np.column_stack((np.ones(n_samples), X))

    # Initialize parameters to zero
    n_features = X.shape[1]
    weights = np.zeros(n_features)
    loss_history = []

    for i in range(n_iterations):
        y_pred = np.dot(X, weights)
        error = y_pred - y
        gradient = (2 / n_samples) * np.dot(X.T, error)
        weights = weights - learning_rate * gradient
        loss = mse(y, y_pred)
        loss_history.append(loss)

    return weights, loss_history

def predict(X, weights):
    X = np.asarray(X, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    X = np.column_stack((np.ones(X.shape[0]), X))
    return np.dot(X, weights)

def evaluate_model(X, y, weights):
    y_pred = predict(X, weights)
    return {
        'R_squared': r_squared(y, y_pred),
        'MSE': mse(y, y_pred)
    }