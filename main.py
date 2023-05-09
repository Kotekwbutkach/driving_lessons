from driver_school import DriverSchool
from road import Road
from traffic_controller import TrafficController
from vehicle import Vehicle
from visualisation import RoadAnimation

road_length = 100
time_horizon = 1000
delta_time = 0.1

number_of_vehicles = 20
awareness = 1
initial_distance = 5
max_acceleration = 1
min_acceleration = -0.2
critical_distance = 2

learning_rate = 0.1

road = Road(road_length, number_of_vehicles, time_horizon, delta_time)
vehicles = [Vehicle(max_acceleration,
                    min_acceleration,
                    awareness,
                    critical_distance)
            for x in range(number_of_vehicles)]

traffic_controller = TrafficController(road, vehicles, [initial_distance * x for x in range(number_of_vehicles)])
driver_school = DriverSchool(road, vehicles, 0.1, initial_distance)

road_animation = RoadAnimation(road)

for i in range(100):
    traffic_controller.run()
    driver_school.teach()
    if not i % 50:
        road_animation.show()
    traffic_controller.reset()
