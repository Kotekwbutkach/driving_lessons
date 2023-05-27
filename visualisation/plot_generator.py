from typing import Union

import matplotlib.pyplot as plt

from road import Road
from visualisation import from_id


class PlotGenerator:
    @staticmethod
    def plot_vehicle_data(road: Road, name: str, start: int = 0, end: Union[int, None] = None):
        end = road.time_step if end is None else min(end, road.time_step)
        value_types = [(0, "position"), (1, "velocity"), (2, "acceleration")]

        for value_id, value_name in value_types:
            for vehicle_id in range(road.number_of_vehicles):
                plt.plot(list(range(start, end)), road.road_data[start:end, vehicle_id][:, value_id],
                         ",", color=from_id(vehicle_id, sat=0.75))
            plt.savefig(f"graphs/{name}_{value_name}.png", format="png")
            plt.clf()
