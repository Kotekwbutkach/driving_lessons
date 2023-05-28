from typing import List

from road import Road
from vehicle import Vehicle


class TrafficSupervisor:
    road: Road
    vehicles: List[Vehicle]

    def __init__(self, road: Road, vehicles: List[Vehicle]):
        self.road = road
        self.vehicles = vehicles

    def check_for_crashes(self):
        for i in range(self.road.number_of_vehicles):
            next_i = i + 1
            rounds_correction = 0
            if next_i == self.road.number_of_vehicles:
                next_i = 0
                rounds_correction += 1  # the next vehicle should have at least as many rounds unless it is the last one
            if self.vehicles[i].rounds > self.vehicles[next_i].rounds + rounds_correction:
                return True
            if self.vehicles[i].rounds < self.vehicles[next_i].rounds + rounds_correction:
                continue
            time_step = self.road.time_step
            distance = self.road.road_data[time_step, next_i, 0] - self.road.road_data[time_step, i, 0]

            if distance < 0:
                self.vehicles[i].has_crashed = self.vehicles[(i + 1) % self.road.number_of_vehicles].has_crashed = True
                return True
        return False
