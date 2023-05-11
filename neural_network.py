import numpy as np


class NeuralNetwork:
    def __init__(self, size: int, critical_distance: float, min_acceleration: float):
        self.weights = np.random.normal(0, 1, (size, size//2))
        self.weights2 = np.random.normal(0, 1, size//2)
        self.bias = np.random.randn()
        self.bias2 = np.random.randn()
        self.critical_distance = critical_distance
        self.min_acceleration = min_acceleration

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_deriv(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def predict(self, input_vector):
        if input_vector[0] < self.critical_distance:
            return self.min_acceleration
        r = 0#np.random.normal(0, 10)
        layer_1 = np.dot(input_vector, self.weights) + self.bias + r
        layer_2 = self._sigmoid(layer_1)
        r = 0 #np.random.normal(0, 10)
        layer_3 = np.dot(layer_2, self.weights2) + self.bias2 + r
        layer_4 = self._sigmoid(layer_3)
        prediction = layer_4
        return prediction

