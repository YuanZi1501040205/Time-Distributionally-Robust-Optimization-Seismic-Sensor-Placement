import numpy as np
import pandas as pd
import chama
import csv
from tqdm import tqdm
from sympy import symbols, solve


def leak_simulation(grid, atm, leak_positions, leak_heights, leak_rates, event_name='S'):
    """input: grid, atmosphere model, leak lists"""
    source = chama.simulation.Source(leak_positions[0][0], leak_positions[0][1], leak_heights[0],
                                     leak_rates[0])
    gauss_plume = chama.simulation.GaussianPlume(grid, source, atm)
    gauss_plume.run()
    signals = gauss_plume.conc
    signals = signals.drop(columns='S')
    # print(signals.head(5))
    scenario_worst_impacts = []
    scenario_names = []
    scenario_probs = []
    Undetected_impact_scale = 100
    total_simulation_scenario_num = leak_positions.__len__()
    for i, (leak_point, leak_h, leak_r) in tqdm(enumerate(zip(leak_positions, leak_heights, leak_rates)), total=len(leak_positions), leave=False):
        # print('simulation: ', i)
        source = chama.simulation.Source(leak_point[0], leak_point[1], leak_h, leak_r)
        gauss_plume = chama.simulation.GaussianPlume(grid, source, atm)
        gauss_plume.run()
        signal = gauss_plume.conc
        scenario_name = event_name + str(i)
        signals[scenario_name] = signal['S']
        # print(signal.head(5))
        scenario_worst_impacts.append(24 * Undetected_impact_scale)
        scenario_names.append(scenario_name)
        scenario_probs.append(1 / total_simulation_scenario_num)

    scenario = pd.DataFrame(
        {'Scenario': scenario_names, 'Undetected Impact': scenario_worst_impacts, 'Probability': scenario_probs})
    return signals, scenario


def sample_wind_speed(num_samples, wind_mean, wind_std):
    """sample wind speed traces from normal distribution(assumed true distribution of wind speed)"""
    wind_speed_samples = []
    for sample_idx in range(num_samples):
        wind_speed_trace = []
        for i, (wind_speed_time_point) in enumerate(zip(wind_mean)):
            wind_speed_trace.append(np.random.normal(wind_speed_time_point, wind_std)[0])
        wind_speed_samples.append(wind_speed_trace)
    return wind_speed_samples


def generate_sensor_candidates(sensor_threshold):
    sensors = dict()
    sensor_names = []
    count = 0
    sensor_costs = []
    for i in range(9):
        for j in range(9):
            for m in range(10):
                for n in range(1):
                    sensor_name = 'A_' + str(i) + '_' + str(j) + '_' + str(m) + '_' + str(n)
                    sensor_names.append(sensor_name)
                    pos = chama.sensors.Stationary(location=((i + 1) * 10 + 1, (j + 1) * 10, m))
                    if n == 0:
                        sensor_threshold = sensor_threshold  # regular sensor
                        sensor_cost = 10000
                    # elif n == 1:
                    #     sensor_threshold = 0.01  # high sensitivity sensor
                    #     sensor_cost = 100000

                    sensor_costs.append(sensor_cost)
                    det = chama.sensors.Point(threshold=sensor_threshold, sample_times=list(range(24)))
                    stationary_pt_sensor = chama.sensors.Sensor(position=pos, detector=det)
                    sensors[sensor_name] = stationary_pt_sensor
                    # print('count: ', count)
                    count = count + 1

    sensor_cost_pairs = pd.DataFrame({'Sensor': sensor_names, 'Cost': sensor_costs})
    print('sensors candidates have been configured!')
    return sensors, sensor_cost_pairs



def extract_min_detect_time(signal, sensors):
    det_times = chama.impact.extract_detection_times(signal, sensors)

    #  extract statistic of detection time
    det_time_stats = chama.impact.detection_time_stats(det_times)
    #  extract the min detect time
    min_det_time = det_time_stats[['Scenario', 'Sensor', 'Min']]
    min_det_time = min_det_time.rename(columns={'Min': 'Impact'})
    if min_det_time.__len__() == 0:
        raise ValueError('No signal was detected by any sensor candidates')
    return min_det_time


def optimize_sensor(min_det_time, sensor_cost_pairs, scenario, sensor_budget):
    #  optimization impact formulation
    impactform = chama.optimize.ImpactFormulation()
    results = impactform.solve(impact=min_det_time, sensor_budget=sensor_budget,
                               sensor=sensor_cost_pairs, scenario=scenario,
                               use_scenario_probability=True,
                               use_sensor_cost=True)
    print('sensor placement strategy has been generated!')
    return results


def eval_sensor_placement(min_det_time, sensor_cost_pairs, scenario_prob, placed_sensor_set):
    #  optimization impact formulation
    impactform = chama.optimize.ImpactFormulation()
    model = impactform.create_pyomo_model(impact=min_det_time, sensor=sensor_cost_pairs, scenario=scenario_prob)
    for sensor_placed in placed_sensor_set:
        impactform.add_grouping_constraint([sensor_placed], min_select=1)
    # impactform.add_grouping_constraint(['A_3_2_3_0'], min_select=1)
    impactform.solve_pyomo_model(sensor_budget=len(placed_sensor_set))
    results = impactform.create_solution_summary()
    return results


def Wasserstein_upper_bound(distribution, gamma=0.9):
    # distribution: empirical distribution
    # gamma: Confidence level
    # (high gamma, more sample)->high shift
    len_distribution = float(distribution.__len__())
    S = len_distribution # Number of historical data for empirical distribution
    H = len_distribution # Number of bins for empirical distribution
    kappa = (H / (2 * S)) * np.log(2 * H / (1 - gamma))

    # solve |x-d| = kappa wasserstein function (quadratic function choose max solution)
    dro_det_time = symbols('dro_det_time')
    a = len_distribution
    b = -2 * np.sum(distribution)
    c = np.sum(distribution ** 2) - (kappa * len_distribution) ** 2
    f = a * dro_det_time ** 2 + b * dro_det_time + c
    result = solve(f)
    result = complex(result[-1]).real
    return result


def eliminate_missing_leak_events(opt_results, leak_positions, leak_heights, leak_rates):
    covered_scenario_number = int(np.where(opt_results['Assessment']['Sensor'].values == None)[0][0])
    print('Detected scenarios: ', covered_scenario_number)
    scenario_names_cache = opt_results['Assessment']['Scenario'].values[:covered_scenario_number]
    new_leak_positions = []
    new_leak_heights = []
    new_leak_rates = []
    for detect_scenario in scenario_names_cache:
        s_idx = int(detect_scenario.split('S')[-1])
        leak_point = leak_positions[s_idx]
        leak_h = leak_heights[s_idx]
        leak_r = leak_rates[s_idx]
        new_leak_positions.append(leak_point)
        new_leak_heights.append(leak_h)
        new_leak_rates.append(leak_r)
    return new_leak_positions, new_leak_heights, new_leak_rates


# %% parameter and configuration preparing
# environment: grid
random_seed = 800
gamma=0.9
TOTAL_EVENTS_NUM = 50
x_grid = np.linspace(0, 99, 100)
y_grid = np.linspace(0, 99, 100)
z_grid = np.linspace(0, 9, 10)
grid = chama.simulation.Grid(x_grid, y_grid, z_grid)
# environment: wind
wind_speed_mean = [6.72, 8.87, 9.73, 9.27, 7.43, 6.73, 6.05, 6.36, 7.89, 8.78, 9.09, 8.29, 8.44, 8.93, 8.38, 10.71,
                   7.95, 7.64, 6.17, 6.26, 5.65, 8.63, 7.83, 7.18]
wind_speed_std = 1.3837437106919512
wind_direction = [177.98, 185.43, 185.43, 184.68, 183.19, 182.45, 175.75, 178.72, 180.96, 198.09, 212.98, 224.15,
                  268.09, 277.77, 272.55, 272.55, 275.53, 281.49, 282.98, 298.62, 284.47, 332.13, 341.06, 337.34]
time_points = 24
atm = pd.DataFrame({'Wind Direction': wind_speed_mean,
                    'Wind Speed': wind_direction,
                    'Stability Class': ['A'] * 24}, index=list(np.array(range(24))))
# test wind speed samples
num_test_sample = 5
wind_speed_test = sample_wind_speed(num_test_sample, wind_speed_mean, wind_speed_std)

# source: potential source location
leak_positions = [[25, 75], [75, 75], [65, 60], [25, 50], [45, 50], [75, 50], [25, 25], [40, 35], [60, 45], [75, 25]]
leak_positions = leak_positions * int(TOTAL_EVENTS_NUM / 10)

# source: potential leak rate
with open('./data/raw/leak_pdf.csv') as leakfile:
    csvreader = csv.reader(leakfile)
    rows = []
    for row in csvreader:
        rows.append(row)
leak_rates = []
for rate in rows[1:]:
    leak_rates.append([float(rate[0]), float(rate[1])])

leak_rate = np.array(leak_rates)[:, 0]
leak_prob = np.array(leak_rates)[:, 1]

grid = chama.simulation.Grid(x_grid, y_grid, z_grid)
# Choose elements with different probabilities
np.random.seed(random_seed)
leak_rates = np.random.choice(list(leak_rate), TOTAL_EVENTS_NUM, p=list(leak_prob) / leak_prob.sum())
print(leak_rates)
# Choose elements with different probabilities
np.random.seed(random_seed)
leak_heights = np.random.choice([0, 1, 2], TOTAL_EVENTS_NUM, p=[0.33, 0.33, 0.34])
print(leak_heights)
# sensor
sensor_threshold = 0.001
sensor_budget = 100000
# %% optimize based on the mean wind time
"""input grid, mean_atm, leak_positions, sampleLeakHeights, sampleLeakRates"""

signals_m, scenario_m = leak_simulation(grid, atm, leak_positions, leak_heights, leak_rates)
sensors, sensor_cost_pairs_m = generate_sensor_candidates(sensor_threshold)
min_det_time_m = extract_min_detect_time(signals_m, sensors)

opt_result_m = optimize_sensor(min_det_time_m, sensor_cost_pairs_m, scenario_m, sensor_budget)

# repeat simulation to eliminate missing events
leak_positions, leak_heights, leak_rates = eliminate_missing_leak_events(opt_result_m, leak_positions, leak_heights, leak_rates)
signals_m, scenario_m = leak_simulation(grid, atm, leak_positions, leak_heights, leak_rates)
min_det_time_m = extract_min_detect_time(signals_m, sensors)
opt_result_m = optimize_sensor(min_det_time_m, sensor_cost_pairs_m, scenario_m, sensor_budget)
#%% check optimization result on basic leak events set (associate to the mean wind sample trace)
print('check optimization result on basic leak events set (associate to the mean wind sample trace)')
print('sensor placement for mean wind: ', opt_result_m['Sensors'])
print('Objective of mean sensor placement: ', opt_result_m['Objective'])

covered_scenario_number_m = int(np.where(opt_result_m['Assessment']['Sensor'].values != None)[0][-1] + 1)
total_scenario_number = opt_result_m['Assessment']['Sensor'].values.__len__()
print('covered_scenario_number_m: ', covered_scenario_number_m)
print('total_scenario_number: ', total_scenario_number)
# %%  test on the noisy test data (perturbated wind speed)


for i in tqdm(range(num_test_sample), leave=False):
    # use perturbated wind speed
    wind_speed = wind_speed_test[i]
    atm = pd.DataFrame({'Wind Direction': wind_speed,
                        'Wind Speed': wind_direction,
                        'Stability Class': ['A'] * 24}, index=list(np.array(range(24))))

    signals, scenario = leak_simulation(grid, atm, leak_positions, leak_heights, leak_rates, event_name='Sample'+str(i)+'S') #

    min_det_time = extract_min_detect_time(signals, sensors)


    if i == 0:
        # prepare test events dataframe
        scenario_samples = scenario
        min_det_time_samples = min_det_time
        scenario_idxs = scenario['Scenario']
        # prepare statistic distribution of detection for DRO correction
        # scenario stat
        scenario_stat = min_det_time_m['Scenario']
        # sensors stat
        sensors_stat = np.expand_dims(min_det_time['Sensor'].array, axis=1)
        # impact stat
        impact_stat = np.expand_dims(min_det_time['Impact'].array, axis=1)
    else:
        # prepare test events dataframe
        scenario_samples = scenario_samples.append(scenario)
        min_det_time_samples = min_det_time_samples.append(min_det_time)

        # prepare statistic distribution of detection for DRO correction
        # sensor stat
        sensor_array = np.expand_dims(min_det_time['Sensor'].array, axis=1)
        sensors_stat = np.concatenate((sensors_stat, sensor_array), axis=1)
        # impact stat
        impact_array = np.expand_dims(min_det_time['Impact'].array, axis=1)
        impact_stat = np.concatenate((impact_stat, impact_array), axis=1)

scenario_samples['Probability'] = scenario_samples['Probability']/scenario.__len__()
mean_method_on_noise_eval_result = eval_sensor_placement(min_det_time_samples, sensor_cost_pairs_m, scenario_samples, opt_result_m['Sensors'])
# %% check mean wind optimization result on test events set (perturbated wind speed)
print('check mean wind optimization result on test events set (perturbated wind speed)')
print('sensor placement for mean wind: ', mean_method_on_noise_eval_result['Sensors'])
print('Objective of mean sensor placement: ', mean_method_on_noise_eval_result['Objective'])

covered_scenario_number_m = int(np.where(mean_method_on_noise_eval_result['Assessment']['Sensor'].values != None)[0][-1] + 1)
total_scenario_number = mean_method_on_noise_eval_result['Assessment']['Sensor'].values.__len__()
print('covered_scenario_number_m: ', covered_scenario_number_m)
print('total_scenario_number: ', total_scenario_number)
# %% optimize placement based on the DRO modified detect time and test on the test data
# statistic samples conbine stat series to dataframe
stat_df_min_det_time = pd.DataFrame({'Scenario': scenario_stat,
                                     'Sensor': list(sensors_stat),
                                     'Impact': list(impact_stat)})
dro_sensors = []
dro_impact = []
for i in range(stat_df_min_det_time.__len__()):
    # extract the most reliable sensor which detect the most samples
    act_sensor_list_per_event = stat_df_min_det_time.iloc[i]['Sensor']
    robust_sensor = pd.value_counts(act_sensor_list_per_event).keys()[0]
    robust_sensor_idxs = list(np.where(act_sensor_list_per_event == robust_sensor)[0])
    # extract this sensor's impact distribution
    impact_distribution = np.array([stat_df_min_det_time.iloc[i]['Impact'][j] for j in robust_sensor_idxs])
    # DRO impact correction
    robust_impact_value = Wasserstein_upper_bound(impact_distribution, gamma=gamma)

    dro_sensors.append(robust_sensor)
    dro_impact.append(robust_impact_value)

dro_min_det_time = pd.DataFrame({'Scenario': scenario_stat,
                                     'Sensor': list(dro_sensors),
                                     'Impact': list(dro_impact)})

# %%

sensors_stat = np.expand_dims(min_det_time_m['Sensor'].array, axis=1)
sensor_array = np.expand_dims(min_det_time['Sensor'].array, axis=1)
sensors_stat = np.concatenate((sensors_stat, sensor_array), axis = 1)


for event in min_det_time['Scenario']:
    active_sensor_list = set()
# %%
min_det_time_samples
min_det_time_m
det_times = chama.impact.extract_detection_times(signal, sensors)
opt_result_m = optimize_sensor(min_det_time_m, sensor_cost_pairs_m, scenario_m, sensor_budget)
active_sensor_list = list(set(min_det_time_samples['Sensor']))
# exp 6: optimize based on the test data and test on the test data
