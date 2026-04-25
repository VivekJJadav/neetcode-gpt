import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        z_stable = z - np.max(z)   
        exp_z = np.exp(z_stable)
        return np.round(exp_z / np.sum(exp_z), 4)
     
