import numpy as np
from typing import List

from road import Road
from vehicle import Vehicle


class DriverSchool:
    road: Road
    vehicles: List[Vehicle]
    learning_rate: float

    def __init__(self, road, vehicles, learning_rate):
        self.road = road
        self.vehicles = vehicles
        self.learning_rate = learning_rate

    def get_scores(self, crash_penalty=20, backward_penalty=100):
        scores = dict()
        for i, v in enumerate(self.vehicles):
            data_to_score = self.road.road_data[:, i][:, 1]
            data_to_score = data_to_score[data_to_score != 0]
            # print(f" {i} data_to_score: {data_to_score}")
            scores[v] = np.mean(data_to_score)
            if v.has_crashed:
                scores[v] -= crash_penalty
            if v.transform[1] < 0:
                scores[v] -= backward_penalty
        return scores

    def teach(self):
        return self.evolve(), self.learn()

    def evolve(self):
        score = self.road.time_step
        return [v.controller_network.assess_shift(score) for v in self.vehicles]

    def learn(self):
        scores = self.get_scores()
        for i, v in enumerate(self.vehicles):
            score_difference_sum = 0
            for w in self.vehicles:
                if scores[v] < scores[w]:
                    score_difference_sum += scores[w] - scores[v]
            if score_difference_sum == 0:
                continue
            weights_change = np.zeros(v.controller_network.active_weights.shape)
            bias_change = 0
            # print(f"v: {i} score: {scores[v]}")
            # print(score_difference_sum)
            j = -1
            for w in self.vehicles:
                j += 1
                # print(f"score w: {scores[w]}")
                if scores[v] < scores[w] and scores[w] > 1:
                    coefficient = (scores[w] - scores[v]) / score_difference_sum * self.learning_rate
                    # print(f"dla v: {i} i w: {j} coefficient to {coefficient}")
                    weights_change += coefficient * np.subtract(w.controller_network.active_weights,
                                                                v.controller_network.active_weights)
                    bias_change += coefficient * (w.controller_network.active_bias - v.controller_network.active_bias)
            # print(f"aktualne: {v.controller_network.active_weights} zmienione: {weights_change}")
            v.controller_network.active_weights = np.add(v.controller_network.active_weights, weights_change)
            v.controller_network.active_bias += bias_change
