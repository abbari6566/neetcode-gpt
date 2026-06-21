import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)

        # result = []
        # for i in range(len(z)):
        #     Z = 1/(1 + math.exp(-z[i]))
        #     result.append(np.round(Z,5))
        
        # instead of for loop use numpy np that can simultaneously calculate element wise
        ans = 1 / (1 + np.exp(-z))
        return np.round(ans, 5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        return np.maximum(0, z).round(2)
