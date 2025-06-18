"""
A module with utility functions and classes
"""
import numpy as np
from numpy import ndarray


################################################################
# functions
################################################################


def mse(y_pred: ndarray, y_true: ndarray) -> float:
    """ Returns the MSE given predicted and expected values"""
    if y_pred.size != y_true.size:
        raise ValueError(
            "Shape mismatch: y_pred and y_true",
            "must have the same number of elements."
        )
    n = y_pred.size
    error = (y_pred - y_true)
    sqr_error = error ** 2
    out = float(sqr_error.sum() / n)
    return out


################################################################
# Classes
################################################################


class LinearEquation():

    def __init__(self, B: ndarray):
        self.B = B  # [b0, b1, ..., bn]

    def calculate(self, X: ndarray) -> ndarray:
        
        b0, *B = self.B
        
        if not B:
            # make sure B has a value
            B = np.array([0])
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        n = X.shape[1] # dimsions of x
        
        if n > (len(B)):
            # pad B with zeros
            zeros_needed = n - len(B)
            np.append(B, [[0]*zeros_needed])
            
        if n < (len(B)):
            # shorten B to match X
            B = B[:n]
            
        y = b0 + X @ B
        return y

