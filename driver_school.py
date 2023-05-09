import random
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
            input_vector = self.road.get_input_data(i, v.awareness)
            if v.has_crashed:
                bad_examples.append(v)
            elif input_vector[0] > 1.2*self.initial_distance: #0.8 i 1.5
                bad_examples.append(v)
            else:
                good_examples.append(v)
        print(f"good: {len(good_examples)}, bad: {len(bad_examples)}")
        for w in self.vehicles:
            if w in bad_examples:
                role_model = good_examples[random.randint(0, len(good_examples) - 1)]
                w.controller_network.bias *= (1-self.learning_rate)
                w.controller_network.bias += role_model.controller_network.bias * self.learning_rate
                w.controller_network.weights *= (1-self.learning_rate)
                w.controller_network.weights += role_model.controller_network.weights * self.learning_rate