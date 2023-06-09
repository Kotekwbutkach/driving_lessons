from typing import List

import numpy as np

from road import Road
from traffic_supervisor import TrafficSupervisor
from vehicle import Vehicle


class TrafficController:
    road: Road
    vehicles: List[Vehicle]
    number_of_vehicles: int
    update_time: float
    time_steps_horizon: int
    initial_positions: List[float]
    supervisor: TrafficSupervisor

    def __init__(self,
                 road: Road,
                 vehicles: List[Vehicle],
                 initial_positions: List[float]):
        self.road = road
        self.vehicles = vehicles
        self.number_of_vehicles = road.number_of_vehicles
        self.update_time = road.update_time
        self.time_steps_horizon = road.time_horizon
        self.supervisor = TrafficSupervisor(road, vehicles)
        self.initial_positions = initial_positions

        self.reset()

    def reset(self):
        for i, v in enumerate(self.vehicles):
            v.reset(np.array([self.initial_positions[i], 0, 0]).astype(float))
        self.road.reset()
        self.road.add_vehicle_data(self.get_vehicles_data())

    def get_vehicles_data(self):
        return np.array([v.transform for v in self.vehicles])

    def update(self, evolution_train):
        for i, vehicle in enumerate(self.vehicles):
            input_vector = self.road.get_input_data(i, vehicle.awareness, vehicle.reaction_steps)
            self.vehicles[i].update(self.update_time, input_vector, self.road.length, evolution_train)
        self.road.add_vehicle_data(self.get_vehicles_data())
        return not self.supervisor.check_for_crashes()

    def run(self, mutation_shift):
        for t in range(self.time_steps_horizon - 1):
            if not self.update(mutation_shift):
                self.road.crashed_at = t
                break
        return not self.supervisor.check_for_crashes(), [v.export_nn_parameters() for v in self.vehicles]

    def print_status(self):
        if self.road.crashed_at == -1:
            print(f"Success (t = {self.time_steps_horizon * self.update_time} s)")
            return
        print(f"CRASH on t = {self.road.crashed_at * self.update_time} s")
        for i, v in enumerate(self.vehicles):
            if v.has_crashed:
                print(f"  Vehicle {i}")
