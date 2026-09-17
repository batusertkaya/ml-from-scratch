import numpy as np

def predict(X, w, b):
    return X @ w + b

def compute_loss(X, y, w, b):
    n = X.shape[0]
    y_predict = predict(X, w, b)
    e = y_predict - y
    return (1/n) * np.sum(e**2)

def fit_gd(X, y, lr, n_iter):
    w = np.zeros(X.shape[1])
    b = 0
    lossList=[]
    n = X.shape[0]
    for i in range(n_iter):
        y_predict = predict(X, w, b)
        e = y_predict - y
        loss = compute_loss(X, y, w, b)
        lossList.append(loss)
        gradient_w = (2/n) * X.T @ e
        gradient_b = (2/n) * np.sum(e)
        w -= lr * gradient_w
        b -= lr * gradient_b

    return w, b, lossList

def fit_normal_equation(X, y):
    X_bias = np.pad(X, ((0,0),(0,1)), mode='constant', constant_values=1)

    w_bias = np.linalg.solve(X_bias.T @ X_bias, X_bias.T @ y)

    b = w_bias[-1]
    w = w_bias[:-1]

    return w, b

