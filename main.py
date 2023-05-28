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

simulation = Simulation(road_params, vehicle_params, initial_distance, learning_rate)

simulation.run_batch(10, should_learn=True, should_shift=True, should_print_status=True, should_plot=False, should_show=False)

simulation.run(should_learn=False, should_shift=False, should_print_status=True, should_plot=True, should_show=True)
