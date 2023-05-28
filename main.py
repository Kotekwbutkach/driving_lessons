from typing import List

import numpy as np

from road import RoadParams
from simulation import Simulation
from traffic_controller import TrafficController
from vehicle import Vehicle, VehicleParams
from visualisation import PlotGenerator

road_params = RoadParams(
    length=100.,
    number_of_vehicles=10,
    time_horizon=1000,
    update_time=0.1)

vehicle_params = VehicleParams(
    max_acceleration=1,
    min_acceleration=-0.2,
    max_velocity=10,
    min_velocity=-2,
    awareness=1,
    reaction_steps=2,
    mutation_rate=0.1)

initial_distance = 10.
learning_rate = 0.5

stats = dict()
for i in range(100):
    simulation = Simulation(road_params, vehicle_params, initial_distance, learning_rate)
    success, tries, results = simulation.run_until_success(1000,
                                                           should_learn=True,
                                                           should_shift=True,
                                                           should_print_status=False,
                                                           should_plot=False,
                                                           should_show=False)
    # print(f"{'Success' if success else 'Failure'} after {tries} tries")
    if tries in stats.keys():
        stats[tries] += 1
    else:
        stats[tries] = 1
print(stats)
