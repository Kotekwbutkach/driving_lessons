import numpy as np


class NeuralNetwork:
    def __init__(self, size: int, critical_distance: float):
        self.weights = np.random.normal(0, 1, size)
        self.bias = np.random.randn()
        self.critical_distance = critical_distance

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_deriv(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def predict(self, input_vector):
        if input_vector[0] < self.critical_distance:
            return 0
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2
        return prediction

