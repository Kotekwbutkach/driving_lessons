import numpy as np
from neural_network import NeuralNetwork


class Vehicle:
    transform: np.array  # transform[0] == position, transform[1] == velocity, transform[2] == acceleration
    controller_network: NeuralNetwork
    max_acceleration: float
    min_acceleration: float
    max_speed: float
    min_speed: float
    max_velocity: float
    min_velocity: float
    awareness: int
    learning_rate: float
    reaction_steps: float
    rounds: int = 0
    has_crashed: bool = False

    def __init__(self,
                 max_acceleration: float,
                 min_acceleration: float,
                 max_speed: float,
                 min_speed: float,
                 awareness: int,
                 critical_distance: float,
                 long_distance: float,
                 reaction_steps: int,
                 mutation_rate: float):
        self.transform = np.zeros(3).astype(float)
        self.max_acceleration = max_acceleration
        self.min_acceleration = min_acceleration
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.awareness = awareness
        self.controller_network = NeuralNetwork(awareness * 4, critical_distance, long_distance,  mutation_rate)
        self.reaction_steps = reaction_steps

    def update(self, delta_time, input_vector, road_length, evolution_train):
        predict = self.controller_network.predict(input_vector, evolution_train)
        acceleration = self.min_acceleration + predict * (self.max_acceleration - self.min_acceleration)
        self.transform[2] = acceleration
        self.transform[1] += self.transform[2] * delta_time
        self.transform[1] = min(self.transform[1], self.max_speed)
        self.transform[1] = max(self.transform[1], self.min_speed)
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
