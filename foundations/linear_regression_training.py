import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # For each iteration:
        #   1. Compute predictions with get_model_prediction(X, weights)
        #   2. For each weight index j, compute gradient with get_derivative()
        #   3. Update: weights[j] -= learning_rate * gradient
        # Return np.round(final_weights, 5)
        
        #make a copy of initial weights to avoid modifying
        weights = np.array(initial_weights, dtype=np.float64).copy()
        N = len(X) #if X = [[1, 2, 3], [1, 1, 1]] then 2 rows so N = 2
        for _ in range(num_iterations):
            predictions = self.get_model_prediction(X,weights)
            gradients = np.zeros(len(weights), dtype=np.float64)
            #for each forward pass calculate gradient
            for j in range(len(weights)):
                """
                Each loop calculates the gradient for one weight.
                The inner loop does not process one row at a time. 
                Instead, it processes one weight at a time using all training rows.
                """
                gradients[j] = self.get_derivative(predictions,Y,N,X,j)
            weights -= self.learning_rate * gradients
        return np.round(weights, 5)


