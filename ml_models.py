import numpy as np
from numpy import ndarray


class SimpleLinearRegression():
    """A class to solve simple (1D variable) linear regression"""

    def __init__(self):
        self.b0 = 0
        self.b1 = 0

    def solve(self, x: ndarray, y: ndarray) -> None:
        x_mean = x.mean()
        y_mean = y.mean()
        self.b1 = np.sum(((x - x_mean) * (y - y_mean))) / np.sum((x - x_mean)**2)
        self.b0 = y_mean - (self.b1 * x_mean)

    def predict(self, x: ndarray):
        y_pred = self.b0 + self.b1 * x
        return y_pred

    def get_r_squared(self):
        pass
        
    def __str__(self):
        return f"(b0: {self.b0}, b1: {self.b1})"
    
    def __repr__(self):
        return self.__str__()