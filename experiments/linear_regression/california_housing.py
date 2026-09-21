import numpy as np
from sklearn.datasets import fetch_california_housing
from models.linear_regression import fit_gd, predict, compute_loss, fit_normal_equation

def main():

    # Fetch the california housing data.
    # Shuffle the samples while preserving feature-target pairs.
    rng = np.random.default_rng(0)
    data = fetch_california_housing()
    X, y = data.data, data.target
    idx = rng.permutation(X.shape[0])    
    X_shuffled = X[idx]
    y_shuffled = y[idx]

    # Use 80% of the data to train our model and rest for testing.
    n_train = int(X.shape[0] * 0.8)
    X_train = X_shuffled[:n_train]
    X_test = X_shuffled[n_train:]
    y_train = y_shuffled[:n_train]
    y_test = y_shuffled[n_train:]

    # Standardize features to similar scales so gradient descent converges more efficiently.
    avg = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    X_train_norm = (X_train - avg) / std
    X_test_norm = (X_test - avg) / std

    # Learn the model parameters with gradient descent on training dataset.
    # Take learning rate as 0.01 and iterate 1000 times.
    w_found, b_found, lossList = fit_gd(X_train_norm, y_train, 0.01, 1000)

    # Evaluate MSE on the training data.
    train_loss = compute_loss(X_train_norm, y_train, w_found, b_found)

    # Evaluate MSE on the test data to assess generalization.
    test_loss = compute_loss(X_test_norm, y_test, w_found, b_found)

    print(f"Train Loss: {train_loss:.6f}")
    print(f"Test Loss: {test_loss:.6f}")

    # Compute test MSE for a baseline that predicts the mean training target for every sample.
    baseline_test_mse = np.mean((y_test - y_train.mean()) ** 2)
    print(f"Baseline Loss: {baseline_test_mse:.6f}")

if __name__ == "__main__":
    main()