import numpy as np


class Road:
    road_data: np.array
    number_of_vehicles: int
    time_horizon: int
    length: float
    time_step: int = -1
    update_time: float
    crashed_at: int = -1

    def __init__(self, length, number_of_vehicles, time_horizon, update_time):
        self.length = length
        self.number_of_vehicles = number_of_vehicles
        self.time_horizon = time_horizon
        self.update_time = update_time
        self.reset()

    def reset(self):
        self.time_step = -1
        self.road_data = np.zeros([self.time_horizon, self.number_of_vehicles, 3]).astype(float)

    def get_input_data(self, vehicle_id, awareness):
        vehicle_indices0 = list(range(vehicle_id + 1, self.number_of_vehicles)) + list(range(vehicle_id))
        vehicle_indices = vehicle_indices0[:awareness]
        k = 2
        data = self.road_data[self.time_step-k, vehicle_indices][:, [0, 1]] -\
            np.full((awareness, 2), (self.road_data[self.time_step-k, vehicle_id][[0, 1]]))
        vehicle_indices2 = vehicle_indices0[-awareness:]
        data2 = self.road_data[self.time_step-k, vehicle_indices2][:, [0, 1]] -\
            np.full((awareness, 2), (self.road_data[self.time_step-k, vehicle_id][[0, 1]]))
        data[:, 0] += np.array([self.length if data[x, 0] < 0 else 0 for x in range(awareness)])
        data2[:,0] += np.array([self.length if data2[x, 0] < 0 else 0 for x in range(awareness)])
        data = np.concatenate((data, data2), axis=1)
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
