import random
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

    def teach(self):
        good_examples = []
        for v in self.vehicles:
            if not v.has_crashed:
                good_examples.append(v)
        for w in self.vehicles:
            if w.has_crashed:
                role_model = good_examples[random.randint(0, len(good_examples) - 1)]
                w.controller_network.bias *= (1-self.learning_rate)
                w.controller_network.bias += role_model.controller_network.bias * self.learning_rate
                w.controller_network.weights *= (1-self.learning_rate)
                w.controller_network.weights += role_model.controller_network.weights * self.learning_rate