from typing import List

import numpy as np

from driver_school import DriverSchool
from road import Road
from traffic_controller import TrafficController
from vehicle import Vehicle
from visualisation import RoadAnimation, PlotGenerator

road_length = 100
time_horizon = 10000
delta_time = 0.1

number_of_vehicles = 10
awareness = 1
initial_distance = 10
max_speed = 50
min_speed = -10
max_acceleration = 1
min_acceleration = -0.2
long_distance = 1.5*initial_distance
critical_distance = 0.2*initial_distance
reaction_steps = 2
mutation_rate = 0.1

learning_rate = 0.5

road = Road(road_length, number_of_vehicles, time_horizon, delta_time)
vehicles = [Vehicle(max_acceleration,
                    min_acceleration,
                    max_speed,
                    min_speed,
                    awareness,
                    critical_distance,
                    long_distance,
                    reaction_steps,
                    mutation_rate)
            for x in range(number_of_vehicles)]

traffic_controller = TrafficController(road, vehicles, [initial_distance * x for x in range(number_of_vehicles)])
driver_school = DriverSchool(road, vehicles, 0.1, initial_distance)

road_animation = RoadAnimation(road)

best_weights: List[np.array]
for i in range(1000):
    traffic_controller.run()
    driver_school.teach()
    traffic_controller.print_status()
    if traffic_controller.is_success():
        traffic_controller.reset()
        best_weights = traffic_controller.run(False)
        print(f"Training successful after {i} iterations")
        break
    traffic_controller.reset()

for w in best_weights:
    print(w)

vehicles = [Vehicle(max_acceleration,
                    min_acceleration,
                    max_speed,
                    min_speed,
                    awareness,
                    critical_distance,
                    long_distance,
                    reaction_steps,
                    mutation_rate,
                    best_weights[x])
            for x in range(number_of_vehicles)]

traffic_controller = TrafficController(road, vehicles, [initial_distance * x for x in range(number_of_vehicles)])

traffic_controller.run(False)
traffic_controller.print_status()
PlotGenerator.plot_vehicle_data(road, "experiment")
road_animation.show()

