import numpy as np

class Tanh:
    @staticmethod
    def calc(x):
        return np.tanh(x)

    @staticmethod
    def deriv(x):
        return 1 - np.tanh(x) ** 2


class ReLU:
    @staticmethod
    def calc(x):
        return np.maximum(x, 0)

    @staticmethod
    def deriv(x):
        return (x > 0) * 1.0