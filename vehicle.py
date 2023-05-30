from typing import Union, NamedTuple, Tuple

import numpy as np
from neural_network import NeuralNetwork


class VehicleParams(NamedTuple):
    max_acceleration: float
    min_acceleration: float
    max_velocity: float
    min_velocity: float
    awareness: int
    reaction_steps: int
    mutation_rate: float


class Vehicle:
    transform: np.array  # transform[0] == position, transform[1] == velocity, transform[2] == acceleration
    controller_network: NeuralNetwork
    max_acceleration: float
    min_acceleration: float
    max_velocity: float
    min_velocity: float
    awareness: int
    reaction_steps: int
    rounds: int = 0
    has_crashed: bool = False

    def __init__(self, vehicle_params: VehicleParams, imported_weights: Union[np.array, None] = None):
        self.max_acceleration = vehicle_params.max_acceleration
        self.min_acceleration = vehicle_params.min_acceleration
        self.max_velocity = vehicle_params.max_velocity
        self.min_velocity = vehicle_params.min_velocity
        self.awareness = vehicle_params.awareness
        self.reaction_steps = vehicle_params.reaction_steps
        mutation_rate = vehicle_params.mutation_rate

        self.transform = np.zeros(3).astype(float)
        self.controller_network = NeuralNetwork(
            self.awareness * 4,
            mutation_rate,
            imported_weights)

    def update(self, delta_time, input_vector, road_length, mutation_shift):
        predict = self.controller_network.predict(input_vector, mutation_shift)
        acceleration = self.min_acceleration + predict * (self.max_acceleration - self.min_acceleration)
        if self.transform[1] + acceleration * delta_time > self.max_velocity:
            acceleration = (self.max_velocity - self.transform[1]) / delta_time
        elif self.transform[1] + acceleration * delta_time < self.min_velocity:
            acceleration = (self.min_velocity - self.transform[1]) / delta_time
        self.transform[2] = acceleration
        self.transform[1] += self.transform[2] * delta_time
        self.transform[1] = min(self.transform[1], self.max_velocity)
        self.transform[1] = max(self.transform[1], self.min_velocity)
        self.transform[0] += self.transform[1] * delta_time + self.transform[2] * (delta_time ** 2) / 2

        # account for road circularity
        if self.transform[0] > road_length:
            self.transform[0] -= road_length
            self.rounds += 1
        if self.transform[0] < 0:
            self.transform[0] += road_length
            self.rounds -= 1

    def reset(self, transform):
        self.transform = transform
        self.rounds = 0
        self.has_crashed = False

    def export_nn_parameters(self):
        return self.controller_network.active_weights.copy(), self.controller_network.active_bias

    def import_nn_parameters(self, imported_parameters: Tuple[np.array, float]):
        self.controller_network.weights = imported_parameters[0].copy()
        self.controller_network.active_weights = imported_parameters[0].copy()
        self.controller_network.bias = imported_parameters[1]
        self.controller_network.active_bias = imported_parameters[1]
