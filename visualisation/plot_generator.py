from typing import Union

import matplotlib.pyplot as plt

from road import Road
from visualisation import from_id


class PlotGenerator:
    road: Road
    name: str

    def __init__(self,
                 road: Road,
                 name: str = "plot",
                 start: int = 0,
                 end: Union[int, None] = None):
        self.road = road
        self.name = name
        self.start = start
        self.end = end

    def plot_vehicle_data(self):
        end = self.road.time_step if self.end is None else min(self.end, self.road.time_step)

        value_types = [(0, "position"), (1, "velocity"), (2, "acceleration")]

        for value_id, value_name in value_types:
            for vehicle_id in range(self.road.number_of_vehicles):
                plt.plot(list(range(self.start, end)),
                         self.road.road_data[self.start:end, vehicle_id][:, value_id],
                         ",",
                         color=from_id(vehicle_id, sat=0.75))
            plt.savefig(f"graphs/{self.name}_{value_name}.png", format="png")
            plt.clf()
