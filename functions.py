import math
from BaseFunction import BaseFunction


class Linear(BaseFunction):
    def __call__(self, x):
        return x

    def derivative(self, x):
        return 1


class Sigmoid(BaseFunction):
    def __init__(self, b):
        self.b = b

    def __call__(self, x):
        return 1 / (1 + math.exp(self.b * x))

    def derivative(self, x):
        return self(x) * (1 - self(x))


# def sigmoid(b):
#     return lambda x:
#
#
# def tanh(b):
#     return lambda x: (2 / (1 + math.exp(-b * x))) - 1