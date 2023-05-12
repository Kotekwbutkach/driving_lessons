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

    def teach(self):
        good_examples = []
        bad_examples = []
        for i, v in enumerate(self.vehicles):
            input_vector = self.road.get_input_data(i, v.awareness, v.reaction_steps)
            if v.has_crashed:
                bad_examples.append(v)
            elif input_vector[0] > 1.2 * self.initial_distance:  # 0.8 i 1.5
                bad_examples.append(v)
            elif self.road.road_data[self.road.time_step, i][1] < 0:
                bad_examples.append(v)
            else:
                good_examples.append(v)
        print(f"good: {len(good_examples)}, bad: {len(bad_examples)}")

        good_bias = 0
        good_bias2 = 0
        good_weight = np.zeros((4, 2))
        good_weight2 = np.zeros(2)

        for g in good_examples:
            good_bias += g.controller_network.bias
            good_bias2 += g.controller_network.bias2
            good_weight = np.add(good_weight, g.controller_network.weights)
            good_weight2 = np.add(good_weight2, g.controller_network.weights2)
        good_bias = good_bias / len(good_examples)
        good_weight = np.divide(good_weight, len(good_examples))
        for w in self.vehicles:
            if w in bad_examples:
                w.controller_network.bias = w.controller_network.bias * (1-self.learning_rate)
                w.controller_network.bias += good_bias * self.learning_rate
                w.controller_network.bias2 = w.controller_network.bias2 * (1-self.learning_rate)
                w.controller_network.bias2 += good_bias2 * self.learning_rate

                w.controller_network.weights = np.add(w.controller_network.weights * (1-self.learning_rate),
                                                      good_weight * self.learning_rate)
                w.controller_network.weights2 = np.add(w.controller_network.weights2 * (1-self.learning_rate),
                                                       good_weight2 * self.learning_rate)
