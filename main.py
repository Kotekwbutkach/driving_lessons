import numpy as np

from driver_school import DriverSchool
from road import Road
from traffic_controller import TrafficController
from vehicle import Vehicle
from visualisation import RoadAnimation

road_length = 100
time_horizon = 1000
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

t_max = 0
for i in range(11):
    if i % 8:
        traffic_controller.run()
        driver_school.teach()
    else:
        traffic_controller.run(False)
        traffic_controller.print_status()
        road_animation.show()
    traffic_controller.reset()

traffic_controller.run(False)
traffic_controller.print_status()
road_animation.show()

