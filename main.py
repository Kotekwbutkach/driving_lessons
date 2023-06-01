import random

import numpy as np
import sys

from road import RoadParams
from simulation import Simulation
from vehicle import VehicleParams
from visualisation import PlotGenerator

FAIL_ID = -1
SEED = 7

np.random.seed(SEED)
random.seed(SEED)

number_of_runs = 1000
number_of_tries = 100

road_params = RoadParams(
    length=100.,
    number_of_vehicles=10,
    time_horizon=1000,
    update_time=0.1)

vehicle_params = VehicleParams(
    max_acceleration=0.5,
    min_acceleration=-0.2,
    max_velocity=40,
    min_velocity=-2,
    awareness=1,
    reaction_steps=20,
    mutation_rate=0.1,
    prediction_error=0.1)

initial_distance = road_params.length/road_params.number_of_vehicles
learning_rate = 0.2

simulation = Simulation(road_params, vehicle_params, initial_distance, learning_rate)
# success, tries, weights = simulation.run_until_success(number_of_tries,
#                                                should_learn=True,
#                                                should_shift=True,
#                                                should_print_status=True,
#                                                should_plot=False,
#                                                should_show=False)
success, weights = simulation.run_batch(number_of_tries,
                                               should_learn=True,
                                               should_shift=True,
                                               should_print_status=False,
                                               should_plot=False,
                                               should_show=False)[-1]

if True:
    print(f"Learning successful for {road_params.number_of_vehicles} vehicles.")
    road_params = RoadParams(
        length=100.,
        number_of_vehicles=15,
        time_horizon=1000,
        update_time=0.1)
    print(f"Testing for {road_params.number_of_vehicles}:")
    initial_distance = road_params.length/road_params.number_of_vehicles / 1.8
    # weights = results[-1][1][:]
    # weights = [weights[np.random.randint(0, 9)] for i in range(road_params.number_of_vehicles)]
    # korzystamy z kierowców

    # losujemy wagi kierowców i tworzymy nowych:
    weights2 = []
    for i in range(road_params.number_of_vehicles):
        wagi = []
        while sum(wagi) == 0: #na wypadek gdy * random.randint(0,1) da same 0
            wagi = []
            for j in range(len(weights)):
                wagi.append(random.randint(0, 100) * random.randint(0, 1))

        suma = sum(wagi)
        wagi = [x / suma for x in wagi]
        wagi2, bias2 = np.zeros(shape=(4 * vehicle_params.awareness, 1)), 0
        for j in range(len(weights)):
            for k in range(4 * vehicle_params.awareness):
                wagi2[k] += wagi[j] * weights[j][0][k]
            bias2 += wagi[j]*weights[j][1]
        lista = [wagi2, bias2]
        weights2.append(lista)
    weights = weights2

    simulation = Simulation(road_params, vehicle_params, initial_distance, learning_rate)
    simulation.plot_generators.append(PlotGenerator(simulation.road, f"plot_macro_seed_{SEED}", 200, 800))
    simulation.import_weights(weights)
    success, weights = simulation.run(should_learn=False,
                                      should_shift=False,
                                      should_print_status=True,
                                      should_plot=True,
                                      should_show=False)
    # print(success)
    # print(weights)


stats = dict()


def number_of_failures():
    return stats[FAIL_ID] if FAIL_ID in stats.keys() else 0


number_of_runs = 100
number_of_tries = 1000

road_params = RoadParams(
    length=100.,
    number_of_vehicles=10,
    time_horizon=1000,
    update_time=0.1)

vehicle_params = VehicleParams(
    max_acceleration=0.5,
    min_acceleration=-0.2,
    max_velocity=40,
    min_velocity=-2,
    awareness=1,
    reaction_steps=20,
    mutation_rate=0.1,
    prediction_error=0.1)

initial_distance = road_params.length/road_params.number_of_vehicles
learning_rate = 0.5

for i in range(number_of_runs):
    simulation = Simulation(road_params, vehicle_params, initial_distance, learning_rate)
    success, tries, results = simulation.run_until_success(number_of_tries,
                                                           should_learn=True,
                                                           should_shift=True,
                                                           should_print_status=False,
                                                           should_plot=False,
                                                           should_show=False)
    # print(f"{'Success' if success else 'Failure'} after {tries} tries")

    if not success:
        if FAIL_ID in stats.keys():  # failed learning sessions
            stats[FAIL_ID] += 1
        else:
            stats[FAIL_ID] = 1
    elif tries in stats.keys():
        stats[tries] += 1
    else:
        stats[tries] = 1

    sys.stdout.write(f"\rSimulation {i+1}/{number_of_runs} ({i + 1 - number_of_failures()} successful)")
    sys.stdout.flush()
print()
number_of_tries = sorted(stats.keys())
for n in number_of_tries:
    print(f"{stats[n]} run{'' if stats[n] == 1 else 's'} successful after {n} tries")
print(f"{number_of_failures()} run{'' if number_of_failures() == 1 else 's'} failed")

