from road import Road
from traffic_controller import TrafficController
from vehicle import Vehicle

road_length = 100
time_horizon = 100
delta_time = 0.1

number_of_vehicles = 10
awareness = 1
initial_distance = 10
max_acceleration = 1
min_acceleration = -0.2

learning_rate = 0.1

road = Road(road_length, number_of_vehicles, time_horizon, delta_time)
vehicles = [Vehicle(max_acceleration,
                    min_acceleration,
                    awareness,
                    learning_rate)
            for x in range(number_of_vehicles)]

traffic_controller = TrafficController(road, vehicles, [initial_distance * x for x in range(number_of_vehicles)])

for i in range(10):
    traffic_controller.run()
    traffic_controller.reset()
    # learn() :c
