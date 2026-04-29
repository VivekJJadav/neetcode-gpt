import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        x = np.array(x, dtype=float)
        W1 = np.array(W1, dtype=float)
        b1 = np.array(b1, dtype=float)
        W2 = np.array(W2, dtype=float)
        b2 = np.array(b2, dtype=float)
        y_true = np.array(y_true, dtype=float)

        z1 = np.dot(W1, x) + b1

        a1 = np.maximum(0, z1)

        y_pred = np.dot(W2, a1) + b2

        loss = np.mean((y_pred - y_true) ** 2)

        n = len(y_true)

        dz2 = 2 * (y_pred - y_true) / n

        dW2 = np.outer(dz2, a1)

        db2 = dz2

        da1 = np.dot(W2.T, dz2)

        relu_mask = z1 > 0

        dz1 = da1 * relu_mask

        dW1 = np.outer(dz1, x)

        db1 = dz1

        def clean(arr):
            arr = np.round(arr, 4)
            arr[np.isclose(arr, 0)] = 0.0
            return arr.tolist()

        return {
            "loss": round(float(loss), 4),
            "dW1": clean(dW1),
            "db1": clean(db1),
            "dW2": clean(dW2),
            "db2": clean(db2)
        }
