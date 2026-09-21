# ml-from-scratch
A learning project where I implement fundamental machine learning
algorithms using Python and NumPy.

My goal is to understand the mathematics behind the models,
verify my implementations, and explore their behavior through experiments.

##Implementations

-Linear Regression
-KNN

##Experiments and Results

###Gradient Descent vs Normal Equation

I compared both methods on the same synthetic dataset with 200 samples, two features, and noise.

| Method | Training MSE |
|--------|-------------:|
| Gradient descent | 3.932201 |
| Normal equation | 3.932201 |

The mean squared difference between their predictions was
approximately 1.60e-14. Gradient descent therefore reached nearly
the same solution as the normal equation in this experiment.

| Parameter | True Value | Gradient Descent   | Normal Equation   |
|-----------|-----------:|-------------------:|------------------:|
| w₁        | 2.0        | 1.89951185         | 1.89951195        |
| w₂        | 6.0        | 6.23891666         | 6.23891662        |
| b         | 10.0       | 9.994443176324776   | 9.994443261031517  |

Both methods produced nearly identical parameter estimates.
The estimates differ from the true values because noise was
added to the targets.


