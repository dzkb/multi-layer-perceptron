import numpy as np
from BaseFunction import BaseFunction


class Sigmoid(BaseFunction):
    def __init__(self, b):
        self.b = b

    def __call__(self, x):
        x = np.clip(x, -100, 100)
        return 1.0 / (1.0 + np.exp(self.b * x))

    def derivative(self, x):
        return self(x) * (1.0 - self(x))
