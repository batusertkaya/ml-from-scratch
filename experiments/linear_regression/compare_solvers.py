import numpy as np
import matplotlib.pyplot as plt

from models.linear_regression import fit_gd, predict, compute_loss, fit_normal_equation

def main():
    sample_size = 200

    rng = np.random.default_rng(0)
    X = rng.standard_normal((sample_size,2)) #Producing random X
    w_real = np.array([2.0, 6.0]) #Setting up a weight matrix
    b_real = 10.0 #Setting up bias

    y = X @ w_real + b_real + rng.normal(0, 2.0, size=sample_size) #Finding y using our pre-dedicated values and with noise

    w_gd, b_gd, loss_history = fit_gd(X, y, 0.01, 1000) #Predicting parameters by using gradient descent with learning rate=0.01 over 1000 iterations
    w_normal, b_normal = fit_normal_equation(X, y) #Predicting parameters by using normal equation
    y_gd = predict(X, w_gd, b_gd) #Predicting the output by using parameters found with GD
    y_normal = predict(X, w_normal, b_normal) #Predicting the output by using parameters found with normal equation

    gd_loss = compute_loss(X, y, w_gd, b_gd) #Computing the MSE of GD
    normal_loss = compute_loss(X, y, w_normal, b_normal) #Computing the MSE of normal equation

    print(f"Mean Square Error of Gradient Descent: {gd_loss}")
    print(f"Mean Square Error of Normal Equation: {normal_loss}\n")

    prediction_mse = np.mean((y_gd - y_normal) ** 2) #Computing the MSE between GD and normal equation
    print(f"Mean Square Error Between Two Predictions: {prediction_mse}\n")

    print("Parameters Found by Gradient Descent:")
    print(f"w = {w_gd}")
    print(f"b = {b_gd}\n")

    print("Parameters Found by Normal Equation:")
    print(f"w = {w_normal}")
    print(f"b = {b_normal}")

    #Plotting Loss vs Iteration graph to show the loss difference between GD and normal equation over iterations
    plt.plot(loss_history, label='Gradient Descent')
    plt.axhline(y=normal_loss, color='r', linestyle='--', label='Normal Equation')

    plt.xlabel("Iteration")
    plt.ylabel("Loss (MSE)")
    plt.title("GD Convergence vs Normal Equation")
    plt.legend()

    plt.savefig("GD_vs_normal.png", dpi=150, bbox_inches="tight")
    plt.show()



if __name__ == "__main__":
    main()
 