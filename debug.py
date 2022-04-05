import numpy as np
import pandas as pd
import chama
import csv
from tqdm import tqdm
from sympy import symbols, solve
from scipy.stats import wasserstein_distance
from scipy.optimize import fsolve


def leak_simulation(grid, atmosphere, leak_positions, leak_heights, leak_rates, event_name='S'):
    """input: grid, atmosphere model, leak lists"""
    source = chama.simulation.Source(leak_positions[0][0], leak_positions[0][1], leak_heights[0],
                                     leak_rates[0])
    gauss_plume = chama.simulation.GaussianPlume(grid, source, atmosphere)
    gauss_plume.run()
    sigs = gauss_plume.conc
    sigs = sigs.drop(columns='S')
    # print(signals.head(5))
    scenario_worst_impacts = []
    scenario_names = []
    scenario_probs = []
    undetected_impact_scale = 100
    total_simulation_scenario_num = leak_positions.__len__()
    for idx, (leak_point, leak_h, leak_r) in tqdm(enumerate(zip(leak_positions, leak_heights, leak_rates)),
                                                  total=len(leak_positions), leave=False):
        # print('simulation: ', i)
        source = chama.simulation.Source(leak_point[0], leak_point[1], leak_h, leak_r)
        gauss_plume = chama.simulation.GaussianPlume(grid, source, atmosphere)
        gauss_plume.run()
        signal = gauss_plume.conc
        scenario_name = event_name + str(idx)
        sigs[scenario_name] = signal['S']
        # print(signal.head(5))
        scenario_worst_impacts.append(24 * undetected_impact_scale)
        scenario_names.append(scenario_name)
        scenario_probs.append(1 / total_simulation_scenario_num)

    events = pd.DataFrame(
        {'Scenario': scenario_names, 'Undetected Impact': scenario_worst_impacts, 'Probability': scenario_probs})
    return sigs, events


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
    sensor_candidates = dict()
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
                    sensor_candidates[sensor_name] = stationary_pt_sensor
                    # print('count: ', count)
                    count = count + 1

    sensor_cost_pairs = pd.DataFrame({'Sensor': sensor_names, 'Cost': sensor_costs})
    print('sensors candidates have been configured!')
    return sensor_candidates, sensor_cost_pairs


def extract_min_detect_time(signal, sensor_candidates):
    det_times = chama.impact.extract_detection_times(signal, sensor_candidates)

    #  extract statistic of detection time
    det_time_stats = chama.impact.detection_time_stats(det_times)
    #  extract the min detect time
    min_det_time = det_time_stats[['Scenario', 'Sensor', 'Min']]
    min_det_time = min_det_time.rename(columns={'Min': 'Impact'})
    if min_det_time.__len__() == 0:
        raise ValueError('No signal was detected by any sensor candidates')
    return min_det_time


def optimize_sensor(min_det_time, sens_cost_pairs, event_probs, sens_budget):
    #  optimization impact formulation
    impactform = chama.optimize.ImpactFormulation()
    results = impactform.solve(impact=min_det_time, sensor_budget=sens_budget,
                               sensor=sens_cost_pairs, scenario=event_probs,
                               use_scenario_probability=True,
                               use_sensor_cost=True)
    print('sensor placement strategy has been generated!')
    return results


def eval_sensor_placement(min_det_time, sens_cost_pairs, event_probs, placed_sensor_set):
    #  optimization impact formulation
    impactform = chama.optimize.ImpactFormulation()
    model = impactform.create_pyomo_model(impact=min_det_time, sensor=sens_cost_pairs, scenario=event_probs)
    for sensor_placed in placed_sensor_set:
        impactform.add_grouping_constraint([sensor_placed], min_select=1)
    # impactform.add_grouping_constraint(['A_3_2_3_0'], min_select=1)
    impactform.solve_pyomo_model(sensor_budget=len(placed_sensor_set))
    results = impactform.create_solution_summary()
    return results


def wasserstein_upper_bound(distribution, confidential_level=0.9):
    # distribution: empirical distribution
    # gamma: Confidence level
    # (high gamma, more sample)->high shift
    len_distribution = float(distribution.__len__())
    S = len_distribution  # Number of historical data for empirical distribution
    H = len_distribution  # Number of bins for empirical distribution
    kappa = (H / (2 * S)) * np.log(2 * H / (1 - confidential_level))

    def func(x):
        return [wasserstein_distance([x[0]], distribution) - kappa]

    spike_distribution = fsolve(func, [max(distribution)])
    if spike_distribution >= np.mean(distribution):
        upper_bound_result = spike_distribution
    else:
        raise ValueError('wasserstein_distance is not upper bond.')

    return upper_bound_result[0]


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
# %%
leak_pdf_file = './data/raw/leak_pdf.csv'

# TOTAL_EVENTS_NUM = 50
# num_test_sample = 5
# random_seed = 800
# gamma = 0.9

TOTAL_EVENTS_NUM = 500
num_train_sample = 8
num_test_sample = 16
random_seed = 1997
gamma = 0.8

x_grid = np.linspace(0, 99, 100)
y_grid = np.linspace(0, 99, 100)
z_grid = np.linspace(0, 9, 10)
grid_init = chama.simulation.Grid(x_grid, y_grid, z_grid)
# environment: wind
wind_speed_mean = [6.72, 8.87, 9.73, 9.27, 7.43, 6.73, 6.05, 6.36, 7.89, 8.78, 9.09, 8.29, 8.44, 8.93, 8.38, 10.71,
                   7.95, 7.64, 6.17, 6.26, 5.65, 8.63, 7.83, 7.18]
wind_speed_test_std = 1.3837437106919512
wind_speed_train_std = 0.6918718553459756
wind_direction = [177.98, 185.43, 185.43, 184.68, 183.19, 182.45, 175.75, 178.72, 180.96, 198.09, 212.98, 224.15,
                  268.09, 277.77, 272.55, 272.55, 275.53, 281.49, 282.98, 298.62, 284.47, 332.13, 341.06, 337.34]
time_points = 24
atm_mean = pd.DataFrame({'Wind Direction': wind_direction,
                         'Wind Speed': wind_speed_mean,
                         'Stability Class': ['A'] * time_points}, index=list(np.array(range(time_points))))
# test wind speed samples
np.random.seed(random_seed)
wind_speed_test = sample_wind_speed(num_test_sample, wind_speed_mean, wind_speed_test_std)
wind_speed_train = sample_wind_speed(num_train_sample, wind_speed_mean, wind_speed_train_std)

# source: potential source location
leak_positions_init = [[25, 75], [75, 75], [65, 60], [25, 50], [45, 50], [75, 50], [25, 25], [40, 35], [60, 45],
                       [75, 25]]
leak_positions_init = leak_positions_init * int(TOTAL_EVENTS_NUM / 10)

# source: potential leak rate
with open(leak_pdf_file) as leakfile:
    csvreader = csv.reader(leakfile)
    rows = []
    for row in csvreader:
        rows.append(row)
leak_pdf = []
for rate in rows[1:]:
    leak_pdf.append([float(rate[0]), float(rate[1])])

leak_rate_list = np.array(leak_pdf)[:, 0]
leak_prob_list = np.array(leak_pdf)[:, 1]

# Choose elements with different probabilities
np.random.seed(random_seed)
leak_rates_init = np.random.choice(list(leak_rate_list), TOTAL_EVENTS_NUM,
                                   p=list(leak_prob_list) / leak_prob_list.sum())
# print(leak_rates_init)
# Choose elements with different probabilities
np.random.seed(random_seed)
leak_heights_init = np.random.choice([0, 1, 2], TOTAL_EVENTS_NUM, p=[0.33, 0.33, 0.34])
# print(leak_heights_init)
# sensor
sensor_thsh = 0.1
sensor_budget = 100000
# %% optimize sensor placement based on the mean wind speed (mean optimization)|(basic dataset)
"""input grid, mean_atm, leak_positions, sampleLeakHeights, sampleLeakRates"""

signals_m, scenario_m = leak_simulation(grid_init, atm_mean, leak_positions_init, leak_heights_init,
                                        leak_rates_init)
sensors, sensor_cost_pairs_m = generate_sensor_candidates(sensor_thsh)
min_det_time_m = extract_min_detect_time(signals_m, sensors)

opt_result_m = optimize_sensor(min_det_time_m, sensor_cost_pairs_m, scenario_m, sensor_budget)

# repeat simulation to eliminate missing events
leak_positions_set, leak_heights_set, leak_rates_set = eliminate_missing_leak_events(opt_result_m,
                                                                                     leak_positions_init,
                                                                                     leak_heights_init,
                                                                                     leak_rates_init)
signals_m, scenario_probs_m = leak_simulation(grid_init, atm_mean, leak_positions_set, leak_heights_set,
                                              leak_rates_set)
min_det_time_m = extract_min_detect_time(signals_m, sensors)
opt_result_m = optimize_sensor(min_det_time_m, sensor_cost_pairs_m, scenario_probs_m, sensor_budget)
# %% check optimization result on basic leak events set (associate to the mean wind sample trace)
covered_scenario_number_m = int(np.where(opt_result_m['Assessment']['Sensor'].values != None)[0][-1] + 1)
total_scenario_number_m = opt_result_m['Assessment']['Sensor'].values.__len__()

accuracy_basic = covered_scenario_number_m / total_scenario_number_m

# print('check optimization result on basic leak events set (associate to the mean wind sample trace)')
# print('sensor placement for mean wind: ', opt_result_m['Sensors'])
# print('Objective of mean sensor placement: ', opt_result_m['Objective'])
# print('covered_scenario_number_m: ', covered_scenario_number_m)
# print('total_scenario_number: ', total_scenario_number)
# %%  test on the noisy test data: perturbation wind speed (basic dataset * number of testing samples)
for i in tqdm(range(num_test_sample), leave=False):
    # use perturbation wind speed
    wind_speed = wind_speed_test[i]
    atm = pd.DataFrame({'Wind Direction': wind_direction,
                        'Wind Speed': wind_speed,
                        'Stability Class': ['A'] * time_points}, index=list(np.array(range(time_points))))

    # current sample's impact dataframe-basic dataset with one wind sample (a line) for test
    signals, scenario = leak_simulation(grid_init, atm, leak_positions_set, leak_heights_set, leak_rates_set,
                                        event_name='Sample' + str(i) + '_S')
    min_det_time_currt_sample = extract_min_detect_time(signals, sensors)
    if i == 0:
        # prepare test events dataframe
        scenario_samples_test = scenario
        min_det_time_samples_test = min_det_time_currt_sample
        # prepare statistic distribution of detection for DRO correction
    else:
        # prepare test events dataframe
        scenario_samples_test = scenario_samples_test.append(scenario)
        min_det_time_samples_test = min_det_time_samples_test.append(min_det_time_currt_sample)
# eval
scenario_samples_test['Probability'] = scenario_samples_test['Probability'] / scenario_samples_test.__len__()
mean_test_result = eval_sensor_placement(min_det_time_samples_test, sensor_cost_pairs_m,
                                         scenario_samples_test,
                                         opt_result_m['Sensors'])
# %% check mean wind optimization result on testing events set (perturbation wind speed)
covered_scenario_number_m_test = int(np.where(
    mean_test_result['Assessment']['Sensor'].values is not None)[0][-1] + 1)
total_scenario_number_m_test = mean_test_result['Assessment']['Sensor'].values.__len__()

accuracy_m_test = covered_scenario_number_m_test / total_scenario_number_m_test

# print('check mean wind optimization result on test events set (perturbated wind speed)')
# print('sensor placement for mean wind: ', mean_method_on_noise_eval_result['Sensors'])
# print('Objective of mean sensor placement: ', mean_method_on_noise_eval_result['Objective'])
# print('covered_scenario_number_m: ', covered_scenario_number_m)
# print('total_scenario_number: ', total_scenario_number_m)
# print('accuracy of mean method: ', accuracy_m)
# %% prepare samples stat for DRO training and multiple samples optimization method
impact_stat_dict = {}  # dictionary to store distribution of each event-sensor's impact
for i in tqdm(range(num_train_sample), leave=False):
    # use perturbation wind speed
    wind_speed = wind_speed_train[i]
    atm = pd.DataFrame({'Wind Direction': wind_direction,
                        'Wind Speed': wind_speed,
                        'Stability Class': ['A'] * time_points}, index=list(np.array(range(time_points))))

    # current sample's impact dataframe-basic dataset with one wind sample (a line) for train
    signals, scenario = leak_simulation(grid_init, atm, leak_positions_set, leak_heights_set, leak_rates_set,
                                        event_name='Sample' + str(i) + '_S')
    min_det_time_currt_sample = extract_min_detect_time(signals, sensors)

    # update distribution of each event-sensor pair
    for j in tqdm(range(min_det_time_currt_sample.__len__())):
        row = min_det_time_currt_sample.iloc[j]
        name_sample_sensor_pair = row['Scenario'].split('_')[-1] + '|' + row['Sensor']
        if name_sample_sensor_pair in impact_stat_dict:
            impact_stat_dict[name_sample_sensor_pair].append(row['Impact'])

        else:
            impact_stat_dict[name_sample_sensor_pair] = [row['Impact']]

    # training dataset build
    if i == 0:
        # prepare test events dataframe
        scenario_samples_train = scenario
        min_det_time_samples_train = min_det_time_currt_sample
        # prepare statistic distribution of detection for DRO correction
    else:
        # prepare test events dataframe
        scenario_samples_train = scenario_samples_train.append(scenario)
        min_det_time_samples_train = min_det_time_samples_train.append(min_det_time_currt_sample)

    # statistic dataframe of impact distributions (training dataset)
    scenario_stat_list = []
    sensor_stat_list = []
    impact_stat_list = []
    for item in impact_stat_dict:
        scenario_name = item.split('|')[0]
        sensor_name = item.split('|')[1]
        impact_det = impact_stat_dict[item]
        scenario_stat_list.append(scenario_name)
        sensor_stat_list.append(sensor_name)
        impact_stat_list.append(impact_det)

    stat_df_min_det_time = pd.DataFrame({'Scenario': scenario_stat_list,
                                         'Sensor': sensor_stat_list,
                                         'Impact': impact_stat_list})
# %% DRO sensor placement on training dataset
dro_sensors = []
dro_impact = []
for i in range(stat_df_min_det_time.__len__()):
    # extract the active sensors
    # act_sensor_list_per_event = stat_df_min_det_time.iloc[i]['Sensor']
    # robust_sensor = pd.value_counts(act_sensor_list_per_event).keys()[0]
    # robust_sensor_idxs = list(np.where(act_sensor_list_per_event == robust_sensor)[0])
    # extract this sensor's impact distribution
    # impact_distribution = np.array([stat_df_min_det_time.iloc[i]['Impact'][j] for j in robust_sensor_idxs])
    impact_distribution = stat_df_min_det_time.iloc[i]['Impact']

    # DRO impact correction
    robust_impact_value = wasserstein_upper_bound(impact_distribution, gamma)

    # dro_sensors.append(robust_sensor)
    dro_impact.append(robust_impact_value)
# dro corrected detect impact dataframe
min_det_time_dro = pd.DataFrame({'Scenario': scenario_stat_list,
                                 'Sensor': sensor_stat_list,
                                 'Impact': list(dro_impact)})
opt_result_dro = optimize_sensor(min_det_time_dro, sensor_cost_pairs_m, scenario_probs_m, sensor_budget)
# %% evaluate dro placement strategy on test dataset

dro_test_result = eval_sensor_placement(min_det_time_samples_test, sensor_cost_pairs_m, scenario_samples_test,
                                        opt_result_dro['Sensors'])
# %% check dro optimization result on test events set (perturbated wind speed)
covered_scenario_number_dro = int(
    np.where(dro_test_result['Assessment']['Sensor'].values != None)[0][-1] + 1)
total_scenario_number_dro = dro_test_result['Assessment']['Sensor'].values.__len__()
accuracy_dro_test = covered_scenario_number_dro / total_scenario_number_dro

# print('check dro optimization result on test events set (perturbated wind speed)')
# print('sensor placement for mean wind: ', dro_method_on_noise_eval_result['Sensors'])
# print('Objective of mean sensor placement: ', dro_method_on_noise_eval_result['Objective'])
# print('covered_scenario_number_dro: ', covered_scenario_number_dro)
# print('total_scenario_number_dro: ', total_scenario_number_dro)
# print('accuracy of dro method: ', accuracy_dro)
# %% naive optimization placement on training dataset
scenario_samples_train['Probability'] = scenario_samples_train['Probability'] / scenario_samples_train.__len__()
naive_opt_train_result = optimize_sensor(min_det_time_samples_train, sensor_cost_pairs_m,
                                         scenario_samples_train, sensor_budget)
covered_scenario_number_naive_opt_train = int(np.where(
    naive_opt_train_result['Assessment']['Sensor'].values is not None)[0][-1] + 1)
total_scenario_number_naive_opt_train = naive_opt_train_result['Assessment']['Sensor'].values.__len__()
accuracy_naive_opt_train = covered_scenario_number_naive_opt_train / total_scenario_number_naive_opt_train

# %% naive optimization placement (train on training dataset) test on testing dataset

naive_opt_test_result = eval_sensor_placement(min_det_time_samples_test, sensor_cost_pairs_m, scenario_samples_test,
                                              naive_opt_train_result['Sensors'])
covered_scenario_number_naive_opt_test = int(np.where(
    naive_opt_test_result['Assessment']['Sensor'].values is not None)[0][-1] + 1)
total_scenario_number_naive_opt_test = naive_opt_test_result['Assessment']['Sensor'].values.__len__()
accuracy_naive_opt_test = covered_scenario_number_naive_opt_test / total_scenario_number_naive_opt_test

# %%
result_list = [opt_result_m, mean_test_result, dro_test_result, naive_opt_test_result, naive_opt_train_result,
               accuracy_basic, accuracy_m_test, accuracy_dro_test,
               accuracy_naive_opt_test, accuracy_naive_opt_train]