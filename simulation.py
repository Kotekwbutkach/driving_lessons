from typing import Union, List

import numpy as np

from driver_school import DriverSchool
from road import Road, RoadParams
from traffic_controller import TrafficController
from traffic_supervisor import TrafficSupervisor
from vehicle import VehicleParams, Vehicle
from visualisation import RoadAnimation, PlotGenerator


class Simulation:
    road: Road
    vehicles: List[Vehicle]
    traffic_controller: TrafficController
    traffic_supervisor: TrafficSupervisor
    driver_school: DriverSchool
    road_animation: RoadAnimation
    plot_generator: PlotGenerator
    number_of_vehicles: int

    def __init__(self,
                 road_params: RoadParams,
                 vehicle_params: Union[VehicleParams, List[VehicleParams]],
                 initial_distance: Union[float, List[float]],
                 learning_rate: float,
                 plot_generator: Union[PlotGenerator, None] = None):

        self.number_of_vehicles = road_params.number_of_vehicles

        self.road = Road(road_params)
        if type(vehicle_params) is VehicleParams:
            self.vehicles = [Vehicle(vehicle_params) for _ in range(road_params.number_of_vehicles)]
        elif type(vehicle_params is List[VehicleParams]):
            self.vehicles = [Vehicle(vehicle_params[i]) for i in range(road_params.number_of_vehicles)]
        else:
            raise Exception("Invalid vehicle parameters")

        if type(initial_distance) is float:
            positions = [initial_distance * x for x in range(self.number_of_vehicles)]
            self.traffic_controller = TrafficController(self.road, self.vehicles, positions)
        elif type(initial_distance) is List[float]:
            self.traffic_controller = TrafficController(self.road, self.vehicles, initial_distance)
        else:
            raise Exception("Invalid initial distance")

        self.driver_school = DriverSchool(self.road, self.vehicles, learning_rate)
        self.road_animation = RoadAnimation(self.road)
        self.plot_generator = plot_generator if plot_generator is not None else PlotGenerator(self.road)

    def import_weights(self, imported_weights: List[Union[np.array, None]]):
        for vehicle, weights in zip(self.vehicles, imported_weights):
            vehicle.import_weights(weights)

    def run(self,
            should_learn: bool = True,
            should_shift: bool = True,
            should_print_status: bool = False,
            should_plot: bool = False,
            should_show: bool = False):
        self.traffic_controller.reset()
        result, weights = self.traffic_controller.run(mutation_shift=should_shift)
        if should_learn:
            self.driver_school.teach()
        if should_print_status:
            self.traffic_controller.print_status()
        if should_plot:
            self.plot_generator.plot_vehicle_data()
        if should_show:
            self.road_animation.show()
        return result, weights

    def run_batch(self, n: int,
                  should_learn: Union[bool, List[bool]],
                  should_shift: Union[bool, List[bool]],
                  should_print_status: Union[bool, List[bool]] = False,
                  should_plot: Union[bool, List[bool]] = False,
                  should_show: Union[bool, List[bool]] = False):
        def validate_setting(setting: Union[bool, List[bool]]):
            if type(setting) is bool:
                return [setting] * n
            if type(setting) is List[bool]:
                return setting
            raise Exception("Invalid setting_value in Simulation.run_batch")

        should_learn = validate_setting(should_learn)
        should_shift = validate_setting(should_shift)
        should_print_status = validate_setting(should_print_status)
        should_plot = validate_setting(should_plot)
        should_show = validate_setting(should_show)

        return [self.run(should_learn[i],
                         should_shift[i],
                         should_print_status[i],
                         should_plot[i],
                         should_show[i]) for i in range(n)]

    def run_until_success(self,
                          limit: int,
                          should_learn: bool = True,
                          should_shift: bool = True,
                          should_print_status: bool = False,
                          should_plot: bool = False,
                          should_show: bool = False):
        run_results = list()
        for _ in range(limit):
            result, weights = self.run(should_learn, should_shift, should_print_status, should_plot, should_show)
            run_results.append((result, weights))
            if result:
                return True, len(run_results), run_results
        return False, limit, run_results
