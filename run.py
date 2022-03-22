import numpy as np
import pandas as pd
import chama
import csv
from tqdm import tqdm
from sympy import symbols, solve


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

    # solve |x-d| = kappa wasserstein function (quadratic function choose max solution)
    dro_det_time = symbols('dro_det_time')
    a = len_distribution
    b = -2 * np.sum(distribution)
    c = np.sum(distribution ** 2) - (kappa * len_distribution) ** 2
    f = a * dro_det_time ** 2 + b * dro_det_time + c
    upper_bound_result = solve(f)
    upper_bound_result = complex(upper_bound_result[-1]).real
    return upper_bound_result


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


def main(TOTAL_EVENTS_NUM, num_test_sample, random_seed, gamma):
    # %% parameter and configuration preparing
    # environment: grid
    leak_pdf_file = './data/raw/leak_pdf.csv'

    # TOTAL_EVENTS_NUM = 50
    # num_test_sample = 5
    # random_seed = 800
    # gamma = 0.9

    TOTAL_EVENTS_NUM = TOTAL_EVENTS_NUM
    num_test_sample = num_test_sample
    random_seed = random_seed
    gamma = gamma

    x_grid = np.linspace(0, 99, 100)
    y_grid = np.linspace(0, 99, 100)
    z_grid = np.linspace(0, 9, 10)
    grid_init = chama.simulation.Grid(x_grid, y_grid, z_grid)
    # environment: wind
    wind_speed_mean = [6.72, 8.87, 9.73, 9.27, 7.43, 6.73, 6.05, 6.36, 7.89, 8.78, 9.09, 8.29, 8.44, 8.93, 8.38, 10.71,
                       7.95, 7.64, 6.17, 6.26, 5.65, 8.63, 7.83, 7.18]
    wind_speed_std = 1.3837437106919512
    wind_direction = [177.98, 185.43, 185.43, 184.68, 183.19, 182.45, 175.75, 178.72, 180.96, 198.09, 212.98, 224.15,
                      268.09, 277.77, 272.55, 272.55, 275.53, 281.49, 282.98, 298.62, 284.47, 332.13, 341.06, 337.34]
    time_points = 24
    atm_mean = pd.DataFrame({'Wind Direction': wind_speed_mean,
                             'Wind Speed': wind_direction,
                             'Stability Class': ['A'] * 24}, index=list(np.array(range(24))))
    # test wind speed samples
    wind_speed_test = sample_wind_speed(num_test_sample, wind_speed_mean, wind_speed_std)

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
    # %% optimize sensor placement based on the mean wind speed
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

    accuracy_basic = covered_scenario_number_m/total_scenario_number_m

    # print('check optimization result on basic leak events set (associate to the mean wind sample trace)')
    # print('sensor placement for mean wind: ', opt_result_m['Sensors'])
    # print('Objective of mean sensor placement: ', opt_result_m['Objective'])
    # print('covered_scenario_number_m: ', covered_scenario_number_m)
    # print('total_scenario_number: ', total_scenario_number)
    # %%  test on the noisy test data (perturbed wind speed)
    for i in tqdm(range(num_test_sample), leave=False):
        # use perturbated wind speed
        wind_speed = wind_speed_test[i]
        atm = pd.DataFrame({'Wind Direction': wind_speed,
                            'Wind Speed': wind_direction,
                            'Stability Class': ['A'] * 24}, index=list(np.array(range(24))))

        signals, scenario = leak_simulation(grid_init, atm, leak_positions_set, leak_heights_set, leak_rates_set,
                                            event_name='Sample' + str(i) + 'S')  #

        min_det_time_currt_sample = extract_min_detect_time(signals, sensors)

        if i == 0:
            # prepare test events dataframe
            scenario_samples = scenario
            min_det_time_samples = min_det_time_currt_sample
            # prepare statistic distribution of detection for DRO correction
            # scenario stat
            scenario_stat = min_det_time_m['Scenario']
            # sensors stat
            sensors_stat = np.expand_dims(min_det_time_currt_sample['Sensor'].array, axis=1)
            # impact stat
            impact_stat = np.expand_dims(min_det_time_currt_sample['Impact'].array, axis=1)
        else:
            # prepare test events dataframe
            scenario_samples = scenario_samples.append(scenario)
            min_det_time_samples = min_det_time_samples.append(min_det_time_currt_sample)

            # prepare statistic distribution of detection for DRO correction
            # sensor stat
            sensor_array = np.expand_dims(min_det_time_currt_sample['Sensor'].array, axis=1)
            sensors_stat = np.concatenate((sensors_stat, sensor_array), axis=1)
            # impact stat
            impact_array = np.expand_dims(min_det_time_currt_sample['Impact'].array, axis=1)
            impact_stat = np.concatenate((impact_stat, impact_array), axis=1)

    scenario_samples['Probability'] = scenario_samples['Probability'] / scenario.__len__()
    mean_method_on_noise_eval_result = eval_sensor_placement(min_det_time_samples, sensor_cost_pairs_m,
                                                             scenario_samples,
                                                             opt_result_m['Sensors'])

    # statistic samples combine stat series to pandas dataframe
    stat_df_min_det_time = pd.DataFrame({'Scenario': scenario_stat,
                                         'Sensor': list(sensors_stat),
                                         'Impact': list(impact_stat)})
    # %% check mean wind optimization result on test events set (perturbated wind speed)
    covered_scenario_number_m = int(
        np.where(mean_method_on_noise_eval_result['Assessment']['Sensor'].values != None)[0][-1] + 1)
    total_scenario_number_m = mean_method_on_noise_eval_result['Assessment']['Sensor'].values.__len__()
    accuracy_m = covered_scenario_number_m / total_scenario_number_m

    # print('check mean wind optimization result on test events set (perturbated wind speed)')
    # print('sensor placement for mean wind: ', mean_method_on_noise_eval_result['Sensors'])
    # print('Objective of mean sensor placement: ', mean_method_on_noise_eval_result['Objective'])
    # print('covered_scenario_number_m: ', covered_scenario_number_m)
    # print('total_scenario_number: ', total_scenario_number_m)
    # print('accuracy of mean method: ', accuracy_m)
    # %% optimize placement based on the DRO modified detect impact
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
        robust_impact_value = wasserstein_upper_bound(impact_distribution, gamma=gamma)

        dro_sensors.append(robust_sensor)
        dro_impact.append(robust_impact_value)
    # dro corrected detect impact dataframe
    min_det_time_dro = pd.DataFrame({'Scenario': scenario_stat,
                                     'Sensor': list(dro_sensors),
                                     'Impact': list(dro_impact)})
    opt_result_dro = optimize_sensor(min_det_time_dro, sensor_cost_pairs_m, scenario_m, sensor_budget)
    # %% evaluate dro placement strategy
    dro_method_on_noise_eval_result = eval_sensor_placement(min_det_time_samples, sensor_cost_pairs_m, scenario_samples,
                                                            opt_result_dro['Sensors'])
    # %% check dro optimization result on test events set (perturbated wind speed)
    covered_scenario_number_dro = int(
        np.where(dro_method_on_noise_eval_result['Assessment']['Sensor'].values != None)[0][-1] + 1)
    total_scenario_number_dro = dro_method_on_noise_eval_result['Assessment']['Sensor'].values.__len__()
    accuracy_dro = covered_scenario_number_dro / total_scenario_number_dro

    # print('check dro optimization result on test events set (perturbated wind speed)')
    # print('sensor placement for mean wind: ', dro_method_on_noise_eval_result['Sensors'])
    # print('Objective of mean sensor placement: ', dro_method_on_noise_eval_result['Objective'])
    # print('covered_scenario_number_dro: ', covered_scenario_number_dro)
    # print('total_scenario_number_dro: ', total_scenario_number_dro)
    # print('accuracy of dro method: ', accuracy_dro)

    result_list = [opt_result_m, mean_method_on_noise_eval_result, dro_method_on_noise_eval_result, accuracy_basic, accuracy_m, accuracy_dro]
    return result_list

# %%


if __name__ == "__main__":

    TOTAL_EVENTS_NUM_list = [100, 500, 1000]
    num_test_sample_list = [5, 10, 20]
    random_seed_list = [1947, 1997, 2008, 2022]
    gamma_list = [0.7, 0.8, 0.9]

    best_accuracy_gap_test_train = np.inf
    best_accuracy_gap_dro_mean = 0
    best_obj_gap_test_train = 0
    best_obj_gap_dro_mean = np.inf
    best_num_events = 0
    for TOTAL_EVENTS_NUM in TOTAL_EVENTS_NUM_list:
        for num_test_sample in num_test_sample_list:
            for random_seed in random_seed_list:
                for gamma in gamma_list:
                    result_list = main(TOTAL_EVENTS_NUM, num_test_sample, random_seed, gamma)

                    accracy_gap_test_train = result_list[-2] - result_list[-3]
                    accracy_gap_dro_mean = result_list[-1] - result_list[-2]
                    obj_gap_test_train = result_list[1]['Objective'] - result_list[0]['Objective']
                    obj_gap_dro_mean = result_list[2]['Objective'] - result_list[1]['Objective']
                    num_events = result_list[1]['Assessment']['Sensor'].values.__len__()

                    if accracy_gap_test_train<=best_accuracy_gap_test_train and accracy_gap_dro_mean>=best_accuracy_gap_dro_mean and obj_gap_test_train>=best_obj_gap_test_train and obj_gap_dro_mean<=best_obj_gap_dro_mean and num_events>=best_num_events:

                        best_accuracy_gap_test_train = accracy_gap_test_train
                        best_accuracy_gap_dro_mean = accracy_gap_dro_mean
                        best_obj_gap_test_train = obj_gap_test_train
                        best_obj_gap_dro_mean = obj_gap_dro_mean
                        best_num_events = num_events

                        f2 = open('./best_parameters_log.txt','r+')
                        f2.read()
                        f2.write('\nbest_accuracy_gap_test_train')
                        f2.write('\n' + best_accuracy_gap_test_train)
                        f2.write('\nbest_accuracy_gap_dro_mean')
                        f2.write('\n' + best_accuracy_gap_dro_mean)
                        f2.write('\nbest_obj_gap_test_train')
                        f2.write('\n' + best_obj_gap_test_train)
                        f2.write('\nbest_obj_gap_dro_mean')
                        f2.write('\n' + best_obj_gap_dro_mean)
                        f2.write('\nbest_num_events')
                        f2.write('\n' + best_num_events)
                        f2.write('\nbest parameters')
                        f2.write('\n' + TOTAL_EVENTS_NUM + ' ' + num_test_sample + ' ' + random_seed+' ' + gamma)
                        f2.close()



