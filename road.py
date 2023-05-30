from typing import NamedTuple

import numpy as np


class RoadParams(NamedTuple):
    length: float
    number_of_vehicles: int
    time_horizon: int
    update_time: float


class Road:
    road_data: np.array
    length: float
    number_of_vehicles: int
    time_horizon: int
    time_step: int = -1
    update_time: float
    crashed_at: int = -1

    def __init__(self, road_params: RoadParams):
        self.length = road_params.length
        self.number_of_vehicles = road_params.number_of_vehicles
        self.time_horizon = road_params.time_horizon
        self.update_time = road_params.update_time
        self.reset()

    def reset(self):
        self.time_step = -1
        self.crashed_at = -1
        self.road_data = np.zeros([self.time_horizon, self.number_of_vehicles, 3]).astype(float)

    def get_input_data(self, vehicle_id, awareness, reaction_steps):
        all_vehicle_indices = list(range(vehicle_id + 1, self.number_of_vehicles)) + list(range(vehicle_id))
        vehicle_indices = all_vehicle_indices[:awareness] + all_vehicle_indices[-awareness:]
        data = self.road_data[self.time_step-reaction_steps, vehicle_indices][:, [0, 1]] -\
            np.full((awareness, 2), (self.road_data[self.time_step-reaction_steps, vehicle_id][[0, 1]]))
        data[:awareness, 0] += np.array([self.length if data[x, 0] < 0 else 0 for x in range(awareness)])
        data[awareness:, 0] -= np.array([self.length if data[x+awareness, 0] > 0 else 0 for x in range(awareness)])
        vector = data.flatten()
        return vector

    def add_vehicle_data(self, data):
        self.time_step += 1
        self.road_data[self.time_step, :, :] = data

    def __repr__(self):
        repr_string = ""
        for t in range(self.time_step + 1):
            repr_string += f"Time {t * self.update_time} s:\n"
            for v in range(self.number_of_vehicles):
                repr_string += f"  Vehicle {v}: " \
                               f"{self.road_data[t, v, 0]} m | " \
                               f"{self.road_data[t, v, 1]} m/s | " \
                               f"{self.road_data[t, v, 2]} m/s^2\n"
            repr_string += "\n"
        return repr_string
