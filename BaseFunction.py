from typing import *


class BaseFunction:
    def __call__(self, x) -> float:
        return x

    def derivative(self, x) -> Callable:
        return x
