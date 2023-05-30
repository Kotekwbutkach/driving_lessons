import sys

from road import RoadParams
from simulation import Simulation
from vehicle import VehicleParams

FAIL_ID = -1

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


def number_of_failures():
    return stats[FAIL_ID] if FAIL_ID in stats.keys() else 0


number_of_runs = 1000
number_of_tries = 1000
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

    sys.stdout.write(f"\rSimulation {i}/{number_of_runs} ({i - number_of_failures()} successful)")
    sys.stdout.flush()

number_of_tries = sorted(stats.keys())
for n in number_of_tries:
    print(f"{stats[n]} run{'' if stats[n] == 1 else 's'} successful after {n} tries")
print(f"{number_of_failures()} run{'' if number_of_failures() == 1 else 's'} failed")
