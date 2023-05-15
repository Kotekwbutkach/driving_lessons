import random
import numpy as np
from typing import List

from road import Road
from vehicle import Vehicle


class DriverSchool:
    road: Road
    vehicles: List[Vehicle]
    learning_rate: float
    initial_distance: float

    def __init__(self, road, vehicles, learning_rate, initial_distance):
        self.road = road
        self.vehicles = vehicles
        self.learning_rate = learning_rate
        self.initial_distance = initial_distance

    def get_scores(self, crash_penalty=10):
        scores = dict()
        for i, v in enumerate(self.vehicles):
            scores[v] = np.mean(self.road.road_data[:, i][:, 1])
            if v.has_crashed:
                scores[v] -= crash_penalty
        return scores

    def teach(self):
        self.evolve()
        self.learn()

    def evolve(self):
        score = self.road.time_step
        for v in self.vehicles:
            v.controller_network.assess_shift(score)

    def learn(self):
        scores = self.get_scores()
        for i, v in enumerate(self.vehicles):
            score_difference_sum = 0
            for w in self.vehicles:
                if scores[v] < scores[w]:
                    score_difference_sum += scores[w] - scores[v]
            if score_difference_sum == 0:
                continue
            weights_change = np.zeros(v.controller_network.weights.shape)
            bias_change = 0
            for w in self.vehicles:
                if scores[v] < scores[w]:
                    coefficient = (scores[w] - scores[v]) / score_difference_sum * self.learning_rate
                    weights_change += coefficient * np.subtract(w.controller_network.weights, v.controller_network.weights)
                    bias_change += coefficient * (w.controller_network.bias - v.controller_network.bias)
            v.controller_network.weights = np.add(v.controller_network.weights, weights_change)
            v.controller_network.bias += bias_change
