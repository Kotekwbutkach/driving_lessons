from typing import Union

import numpy as np


class NeuralNetwork:
    def __init__(self,
                 size: int,
                 mutation_rate: float,
                 imported_weights: Union[np.array, None] = None):
        self.weights = imported_weights if imported_weights is not None else np.random.normal(0, 1, size)
        self.active_weights = np.copy(self.weights)
        self.bias = np.random.randn()
        self.active_bias = self.bias
        self.mutation_rate = mutation_rate
        self.highest_score = 0
        self.try_shift()

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_deriv(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def predict(self, input_vector, mutation_shift):
        if mutation_shift:
            layer_1 = np.dot(input_vector, self.active_weights) + self.active_bias
        else:
            layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2
        return prediction

    def try_shift(self):
        self.active_bias = self.bias + np.random.normal(0, self.mutation_rate)
        self.active_weights = self.weights + np.random.normal(0, self.mutation_rate, self.weights.shape)

    def confirm_shift(self):
        self.bias = self.active_bias
        self.weights = self.active_weights.copy()

    def assess_shift(self, score):
        result = False

        # print(f"recent_score: {self.recent_score} score {score}")
        # print(f"score: {score} recent_score: {self.recent_score}")
        # print(f"active_weights {self.active_weights} weights: {self.weights}")
        # print(f"active_bias {self.active_bias} bias: {self.bias}")
        if self.highest_score <= score:
            self.confirm_shift()
            result = True
            self.highest_score = score
        self.try_shift()
        # print(f"active_weights {self.active_weights} weights: {self.weights}")
        # print(f"active_bias {self.active_bias} bias: {self.bias}")
        return result
