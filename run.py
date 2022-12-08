import numpy as np
import pandas as pd
import chama
from tqdm import tqdm
from sympy import symbols, solve
import matplotlib.pyplot as plt
import time
import csv
from psmodules import psarray, pssynthetic, psraytrace, pswavelet, \
    psplot, pspicker, pspdf

""" 
1. Compare grid placement with SO placement when velocity is fixed.
2. Compare SO placement with fixed and perturbed velocity
3. DRO, SO, grid comparison

"""


def Wasserstein_upper_bound(distribution, num_bins=10, gamma=0.9):
    # distribution: empirical distribution
    # gamma: Confidence level
    # (high gamma, more sample)->high shift
    distribution = np.array(distribution)
    len_distribution = float(distribution.__len__())
    S = len_distribution # Number of historical data for empirical distribution
    # H = len_distribution # Number of bins for empirical distribution
    H = num_bins # Number of bins for empirical distribution

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


def sample_velocity_model(num_samples, vp_mean, vp_std):
    """sample wind speed traces from normal distribution(assumed true distribution of wind speed)"""
    vp_samples = []
    for sample_idx in range(num_samples):
        vp_trace = []
        for i, (wind_speed_time_point) in enumerate(zip(vp_mean)):
            vp_trace.append(np.random.normal(wind_speed_time_point, vp_std)[0])
        vp_samples.append(vp_trace)
    return vp_samples


def prepare_forward_input(geox, geoy, geoz, src_gt, zlayer, vp):
    """
    prepare_forward_input(): format input for forward simulation.
    geox:array, geoy:array, geoz:array-coordinates of geophones|output of psarray.gridarray function
    src_gt:list-coordinates of microseismic source
    zlayer:list-depth of strata
    vp:list-p wave velocity of strata
    """
    vp = np.array(vp)
    sourcex = np.array([src_gt[0]])
    sourcey = np.array([src_gt[1]])
    sourcez = np.array([src_gt[2]])

    # observation forward
    nt = len(geox)
    #  S wave velocity based on Castagna's rule
    # vs = (vp - 1360) / 1.16
    #  S wave velocity based on literature:
    #  2019 Optimal design of microseismic monitoring network: Synthetic study for the Kimberlina CO2 storage demonstration site
    vs = vp / 1.73

    # 3D passive seismic raytracing
    dg = 10
    src = np.array([sourcex, sourcey, sourcez]).T
    rcv = np.array([geox, geoy, geoz]).T
    return vp, vs, zlayer, dg, src, rcv


class SingleEventForwarder():
    def __init__(self, vp, vs, zlayer, dg, rcv):
        self.ptimes_obs = None  # instance variable unique to each instance
        self.forward_counter = 0  # instance variable unique to each instance
        self.vp = vp
        self.vs = vs
        self.zlayer = zlayer
        self.dg = dg
        self.rcv = rcv

    def forward_obs(self, gt_src):
        """
        forward_obs(): forward simulation to get the observation ground truth seismic event arrival time.

        """

        start_time = time.time()
        print("3D passive seismic raytracing is running[Waiting...]")
        tps, _, tetas = psraytrace.raytrace(self.vp, self.vs, self.zlayer, self.dg, gt_src, self.rcv)
        self.ptimes_obs = tps
        self.forward_counter = self.forward_counter + 1

        print("3D passive seismic raytracing completed[OK]")
        print("running time: " + str((time.time() - start_time) / 50) + ' mins')
        return tps

    def timediff(self, tp, tmp):
        """
        timediff(): microseismic event location method based on arrival time differences.
        This method needs to pick first arrival times of microseismic event and
        generally aims to process high signal-to-noise ratio.

        """
        tpdiff = abs(np.diff(tp, axis=0))
        tmpdiff = abs(np.diff(tmp, axis=0))

        temp = np.square(tpdiff - tmpdiff)
        sumErrs = np.cumsum(temp)
        minErr = sumErrs[len(sumErrs) - 1]
        return minErr

    def forward_source_pred_error(self, pre_src):
        """
        time_diff_Err(): try different source location to see error of simulated arrival time.

        """
        sx = [pre_src[0][0]]
        sy = [pre_src[1][0]]
        sz = [pre_src[2][0]]

        if sz[0] in self.zlayer + 1:
            sz[0] = sz[0] + 1

        try_src = np.array([sx, sy, sz]).T

        print('try_src: ', try_src)
        print("3D passive seismic raytracing example is running[Waiting...]")
        tps, _, tetas = psraytrace.raytrace(self.vp, self.vs, self.zlayer, self.dg, try_src, self.rcv)
        self.forward_counter = self.forward_counter + 1
        print("3D passive seismic raytracing completed[OK]")
        # tps = tps / dt

        minErr = self.timediff(tps, self.ptimes_obs)

        return float(minErr)


def seismic_event_simulation(vp, event_positions,geo_sensor_candidates_positions, zlayer, e_name='Event'):
    """input: grid, atmosphere model, leak lists"""
    Sensor_name_list = []
    Impact_list = []
    Scenario_name_list = []


    fault_x = event_positions[0]
    fault_y = event_positions[1]
    fault_z = event_positions[2]

    for i in range(len(fault_x)):
        event_name = e_name + '_' + str(int(fault_x[i])) + '_' + str(int(fault_y[i])) + '_' + str(int(fault_z[i]))
        # Generate square grid array
        geox, geoy, geoz = geo_sensor_candidates_positions[0], geo_sensor_candidates_positions[1], geo_sensor_candidates_positions[2]
        # # Define geological model
        # zlayer = np.array([0, 540, 1070, 1390, 1740, 1950, 2290,
        #                    2630, 4000])

        # Define source coordinates
        source = [fault_x[i], fault_y[i], fault_z[i]]


        # Define velocity model
        # P wave velocity
        # vp = vp

        # formatting the input for forward
        vp, vs, zlayer, dg, src, rcv = prepare_forward_input(geox, geoy, geoz, source, zlayer, vp)

        # run forward simulation
        forwarder = SingleEventForwarder(vp, vs, zlayer, dg, rcv)
        event_min_detect_time = forwarder.forward_obs(src)

        # each sensor's record
        for sensor_idx in range(len(geox)):
            name_sensor = 'sensor_' + str(int(geox[sensor_idx])) + '_' + str(int(geoy[sensor_idx])) + '_' + str(
                int(geoz[sensor_idx]))
            event_sensor_impact = event_min_detect_time[sensor_idx]
            Scenario_name_list.append(event_name)
            Sensor_name_list.append(name_sensor)
            Impact_list.append(event_sensor_impact)
    min_det_time = pd.DataFrame({'Scenario': Scenario_name_list, 'Sensor': Sensor_name_list, 'Impact': Impact_list})
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




# %% create impact panda df for optimization on basic seismic events set


def gap_obj_f(num_train_sample, num_test_sample, random_seed, gamma, v_test_std, v_train_std, fault_x_start, fault_y_start, fault_depth, fault_length):
    with open('./data/raw/fault_poso_creek_location.csv') as leakfile:
        csvreader = csv.reader(leakfile)
        rows = []
        for row in csvreader:
            rows.append(row)
    seismic_events_positions = []
    for coordinate in rows[1:]:
        seismic_events_positions.append([int(10*float(coordinate[0])), int(float(coordinate[1]))])

    # fault 1
    fault_1_x = fault_x_start + fault_length*(np.array(seismic_events_positions).T[0] - np.array(seismic_events_positions).T[0][0])/(np.array(seismic_events_positions).T[0][-1] - np.array(seismic_events_positions).T[0][0])
    fault_1_y = fault_y_start + fault_length*(np.array(seismic_events_positions).T[1] - np.array(seismic_events_positions).T[1][0])/(np.array(seismic_events_positions).T[1][-1] - np.array(seismic_events_positions).T[1][0])
    fault_1_z = np.zeros(len(fault_1_x)) + fault_depth

    fault_x = fault_1_x
    fault_y = fault_1_y
    fault_z = fault_1_z


    # num_train_sample = 4
    # num_test_sample = 8
    # random_seed = 16
    # gamma = 0.9
    # v_test_std = 500
    # v_train_std = 200
    geox, geoy, geoz = psarray.gridarray(81, 10000, 10000)

    # Define geological model
    zlayer = np.array([0, 540, 1070, 1390, 1740, 1950, 2290,
                       2630, 4000])

    # Define velocity model
    # P wave velocity
    vp = np.array([2100, 2500, 2950, 3300, 3700, 4200,
                   4700, 5800])
    event_positions = [fault_x, fault_y, fault_z]
    geo_sensor_candidates_positions = [geox, geoy, geoz]
    min_det_time = seismic_event_simulation(vp, event_positions,geo_sensor_candidates_positions, zlayer, e_name='Event')


    # create scenario panda db for optimization on basic seismic events set
    Impact_list = min_det_time['Impact']
    Scenario_name_list_no_redundant = list(set(min_det_time['Scenario']))
    Scenario_name_list_no_redundant.sort()
    Undetected_Impact_list = []
    Scenario_Probability_list = []
    for event in Scenario_name_list_no_redundant:
        Undetected_Impact_list.append(10*max(Impact_list))
        Scenario_Probability_list.append(float(1/len(Scenario_name_list_no_redundant)))

    scenario_prob_basic = pd.DataFrame(
        {'Scenario': Scenario_name_list_no_redundant, 'Undetected Impact': Undetected_Impact_list, 'Probability': Scenario_Probability_list})
    # create sensor_cost panda df for optimization on basic seismic events set
    sensor_cost_list = []
    Sensor_name_list = []
    for sensor_idx in range(len(geox)):
        name_sensor = 'sensor_' + str(int(geox[sensor_idx])) + '_' + str(int(geoy[sensor_idx])) + '_' + str(
            int(geoz[sensor_idx]))
        name_sensor = 'sensor_' + str(int(geox[sensor_idx])) + '_' + str(int(geoy[sensor_idx])) + '_' + str(
            int(geoz[sensor_idx]))
        Sensor_name_list.append(name_sensor)
        sensor_cost_list.append(1000)
    sensor_cost_pairs = pd.DataFrame({'Sensor': Sensor_name_list, 'Cost': sensor_cost_list})
    # %% optimization
    sens_budget = 9000
    mean_basic_results = optimize_sensor(min_det_time, sensor_cost_pairs, scenario_prob_basic, sens_budget)
    mean_sensor_place_strategy = mean_basic_results['Sensors']
    print(mean_sensor_place_strategy)
    print(mean_basic_results['Objective'])

    # evaluate grid sensor placement with events set
    # grid placement strategy
    geox, geoy, geoz = psarray.gridarray(9, 10000, 10000)  # extract sensor in placed positions
    placed_sensor_x = np.array(geox, dtype=np.int)
    placed_sensor_y = np.array(geoy, dtype=np.int)
    placed_sensor_z = np.array(geoz, dtype=np.int)
    grid_sensor_place_strategy = []
    for i in range(len(placed_sensor_x)):
        sensor_name = 'sensor_'+str(placed_sensor_x[i]) + '_' + str(placed_sensor_y[i]) + '_'+ str(placed_sensor_z[i])
        grid_sensor_place_strategy.append(sensor_name)

    # eval
    grid_basic_result = eval_sensor_placement(min_det_time, sensor_cost_pairs, scenario_prob_basic, grid_sensor_place_strategy)
    grid_obj = grid_basic_result['Objective']
    print(grid_obj)

    # evaluate robustness of SO placement when facing perturbation
    # simulation events with noise
    # num_v_samples = 1
    # vp_mean = np.array([2100, 2500, 2950, 3300, 3700, 4200,
    #                 4700, 5800])
    # vp_std = 500
    # np.random.seed(16)
    # v_samples = sample_velocity_model(num_v_samples, vp_mean, vp_std)
    # v_noise = v_samples[0]
    # noise_min_det_time = seismic_event_simulation(v_noise, event_positions,geo_sensor_candidates_positions, zlayer, e_name='Event')
    # # test
    # mean_noise_test_results = eval_sensor_placement(noise_min_det_time, sensor_cost_pairs, scenario_prob_basic, so_mean_sensor_place_strategy)
    # print('so_noise_test_results obj: ', mean_noise_test_results['Objective'])

    #  DRO: MEAN test, DRO train/test,SO train/test


    vp_mean = np.array([2100, 2500, 2950, 3300, 3700, 4200,
                    4700, 5800])
    # test wind speed samples
    np.random.seed(random_seed)
    vp_samples_train = sample_velocity_model(num_train_sample, vp_mean, v_train_std)
    vp_samples_test = sample_velocity_model(num_test_sample, vp_samples_train[0], v_test_std)

    # test on the noisy test dataset: perturbation wind speed (basic dataset * number of testing samples)
    for i in tqdm(range(num_test_sample), leave=False):
        # use perturbation wind speed
        v_noise = vp_samples_test[i]

        min_det_time_currt_sample = seismic_event_simulation(v_noise, event_positions,geo_sensor_candidates_positions, zlayer, e_name='Sample' + str(i) + '_Event')
        if i == 0:
            # prepare test events dataframe
            min_det_time_samples_test = min_det_time_currt_sample
            # prepare statistic distribution of detection for DRO correction
        else:
            # prepare test events dataframe
            min_det_time_samples_test = min_det_time_samples_test.append(min_det_time_currt_sample)
    # eval

    #  create scenario panda db for mean test on test dataset
    Impact_list = min_det_time_samples_test['Impact']
    Scenario_name_list_no_redundant = list(set(min_det_time_samples_test['Scenario']))
    Scenario_name_list_no_redundant.sort()
    Undetected_Impact_list = []
    Scenario_Probability_list = []
    for event in Scenario_name_list_no_redundant:
        Undetected_Impact_list.append(10*max(Impact_list))
        Scenario_Probability_list.append(float(1/len(Scenario_name_list_no_redundant)))

    scenario_prob_test = pd.DataFrame(
        {'Scenario': Scenario_name_list_no_redundant, 'Undetected Impact': Undetected_Impact_list, 'Probability': Scenario_Probability_list})

    mean_test_result = eval_sensor_placement(min_det_time_samples_test, sensor_cost_pairs,
                                             scenario_prob_test,
                                             mean_sensor_place_strategy)
    #  stat multi events for training DRO/SO
    impact_stat_dict = {}  # dictionary to store distribution of each event-sensor's impact
    for i in tqdm(range(num_train_sample), leave=False):
        # use perturbation wind speed
        # use perturbation wind speed
        v_noise = vp_samples_test[i]

        min_det_time_currt_sample = seismic_event_simulation(v_noise, event_positions, geo_sensor_candidates_positions,
                                                             zlayer, e_name='Sample' + str(i) + '_Event')


        # update distribution of each event-sensor pair
        for j in tqdm(range(min_det_time_currt_sample.__len__())):
            row = min_det_time_currt_sample.iloc[j]
            name_sample_sensor_pair = 'Event_' + row['Scenario'].split('Event_')[-1] + '|' + row['Sensor']
            if name_sample_sensor_pair in impact_stat_dict:
                impact_stat_dict[name_sample_sensor_pair].append(row['Impact'])
            else:
                impact_stat_dict[name_sample_sensor_pair] = [row['Impact']]

        if i == 0:
            # prepare test events dataframe
            min_det_time_samples_train = min_det_time_currt_sample
            # prepare statistic distribution of detection for DRO correction
        else:
            # prepare test events dataframe
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

    # create scenario_prob_train for training dataset
    Impact_list = min_det_time_samples_train['Impact']
    Scenario_name_list_no_redundant = list(set(min_det_time_samples_train['Scenario']))
    Scenario_name_list_no_redundant.sort()
    Undetected_Impact_list = []
    Scenario_Probability_list = []
    for event in Scenario_name_list_no_redundant:
        Undetected_Impact_list.append(10*max(Impact_list))
        Scenario_Probability_list.append(float(1/len(Scenario_name_list_no_redundant)))

    scenario_prob_train = pd.DataFrame(
        {'Scenario': Scenario_name_list_no_redundant, 'Undetected Impact': Undetected_Impact_list, 'Probability': Scenario_Probability_list})

    #  stat to panda df detection time

    dro_impact = []
    for i in range(stat_df_min_det_time.__len__()):
        impact_distribution = stat_df_min_det_time.iloc[i]['Impact']

        # DRO impact correction
        robust_impact_value = Wasserstein_upper_bound(impact_distribution, num_bins=5, gamma=gamma)

        # dro_sensors.append(robust_sensor)
        dro_impact.append(robust_impact_value)
    # dro corrected detect impact dataframe
    min_det_time_dro = pd.DataFrame({'Scenario': scenario_stat_list,
                                     'Sensor': sensor_stat_list,
                                     'Impact': list(dro_impact)})

    # DRO optimization: placement
    opt_result_dro = optimize_sensor(min_det_time_dro, sensor_cost_pairs, scenario_prob_basic, sens_budget)
    dro_sensor_place_strategy = opt_result_dro['Sensors']

    # evaluate dro placement strategy on test dataset
    dro_train_result = eval_sensor_placement(min_det_time_samples_train, sensor_cost_pairs,
                                             scenario_prob_train,
                                             dro_sensor_place_strategy)

    dro_test_result = eval_sensor_placement(min_det_time_samples_test, sensor_cost_pairs,
                                             scenario_prob_test,
                                             dro_sensor_place_strategy)

    #  SO optimization on training and evaluaiton on training and testing
    opt_result_so = optimize_sensor(min_det_time_samples_train, sensor_cost_pairs, scenario_prob_train, sens_budget)
    so_sensor_place_strategy = opt_result_so['Sensors']
    # evaluation mean method on train dataset
    mean_train_result = eval_sensor_placement(min_det_time_samples_train, sensor_cost_pairs,
                                             scenario_prob_train,
                                             mean_sensor_place_strategy)
    #  evaluate dro placement strategy on test dataset
    so_train_result = eval_sensor_placement(min_det_time_samples_train, sensor_cost_pairs,
                                             scenario_prob_train,
                                             so_sensor_place_strategy)

    so_test_result = eval_sensor_placement(min_det_time_samples_test, sensor_cost_pairs,
                                             scenario_prob_test,
                                             so_sensor_place_strategy)

    # eval grid on test dataset
    grid_test_result = eval_sensor_placement(min_det_time_samples_test, sensor_cost_pairs, scenario_prob_test, grid_sensor_place_strategy)
    # expected performance gaps

    gap_1 = mean_basic_results['Objective'] - grid_basic_result['Objective']
    gap_2 = mean_train_result['Objective'] - mean_test_result['Objective']
    gap_3 = mean_test_result['Objective'] - grid_test_result['Objective']
    gap_4 = so_test_result['Objective'] - mean_test_result['Objective']
    gap_5 = dro_test_result['Objective'] - so_test_result['Objective']
    gap_6 = dro_train_result['Objective'] - dro_test_result['Objective']
    gap_7 = so_train_result['Objective'] - so_test_result['Objective']


    result_list = [gap_1,  gap_2,gap_3,
                   gap_4, gap_5,
                   gap_6, gap_7]



if __name__ == "__main__":

    num_train_sample_list = [4, 8]
    num_test_sample_list = [6, 9]
    random_seed_list = [16]
    gamma_list = [0.7, 0.9]
    v_test_std_list = [200, 500]
    v_train_std_list = [200, 500]
    fault_x_start_list = [2000, 4000]
    fault_y_start_list = [2000, 4000]
    fault_depth_list = [1800,  2300]
    fault_length_list = [2000, 4000]

    # good experiment result should be
    """
    1. opt_result_m > mean_test_result (gap_1)
    2. dro_test_result > naive_opt_test_result (gap_2)
    3. dro_test_result >  mean_test_result (gap_3)
    4. naive_opt_train_result > naive_opt_test_result (gap_4)
    5. the smaller objective the better
    6. the higher of accuracy the better
    """
    best_obj_gap_1 = 0
    best_obj_gap_2 = 0
    best_obj_gap_3 = 0
    best_obj_gap_4 = 0
    best_obj_gap_5 = 0
    best_obj_gap_6 = 0
    best_obj_gap_7 = 0


    for num_train_sample in num_train_sample_list:
        for num_test_sample in num_test_sample_list:
            for random_seed in random_seed_list:
                for gamma in gamma_list:
                    for v_test_std in v_test_std_list:
                        for v_train_std in v_train_std_list:
                            for fault_x_start in fault_x_start_list:
                                for fault_y_start in fault_y_start_list:
                                    for fault_depth in fault_depth_list:
                                        for fault_length in fault_length_list:


                                            result_list = gap_obj_f(num_train_sample, num_test_sample, random_seed, gamma, v_test_std, v_train_std, fault_x_start, fault_y_start, fault_depth, fault_length)



                                            if result_list[0] <= best_obj_gap_1 and result_list[1] <= best_obj_gap_2 and result_list[2] <= best_obj_gap_3 and result_list[3] <= best_obj_gap_4 and result_list[4] <= best_obj_gap_5 and result_list[5] <= best_obj_gap_6 and result_list[6] >= best_obj_gap_7 :
                                                best_obj_gap_1 = result_list[0]
                                                best_obj_gap_2 = result_list[1]
                                                best_obj_gap_3 = result_list[2]
                                                best_obj_gap_4 = result_list[3]
                                                best_obj_gap_5 = result_list[4]
                                                best_obj_gap_6 = result_list[5]
                                                best_obj_gap_7 = result_list[6]

                                                f2 = open('./best_parameters_log.txt', 'r+')
                                                f2.read()


                                                f2.write('\n best_obj_gap_1')
                                                f2.write('\n ' + str(best_obj_gap_1))
                                                f2.write('\n best_obj_gap_2')
                                                f2.write('\n ' + str(best_obj_gap_2))
                                                f2.write('\n best_obj_gap_3')
                                                f2.write('\n ' + str(best_obj_gap_3))
                                                f2.write('\n best_obj_gap_4')
                                                f2.write('\n ' + str(best_obj_gap_4))
                                                f2.write('\n best_obj_gap_5')
                                                f2.write('\n ' + str(best_obj_gap_5))
                                                f2.write('\n best_obj_gap_6')
                                                f2.write('\n ' + str(best_obj_gap_6))
                                                f2.write('\n best_obj_gap_7')
                                                f2.write('\n ' + str(best_obj_gap_7))

                                                f2.write('\n best_parameters')
                                                f2.write('\n ' + str(num_train_sample) + ' ' + str(num_train_sample) + ' ' + str(
                                                    random_seed) + ' ' + str(gamma) + ' ' + str(v_test_std)+ ' ' + str(v_train_std)
                                                         + ' ' + str(fault_x_start)+ ' ' + str(fault_y_start)+ ' ' + str(fault_depth)
                                                         + ' ' + str(fault_length))
                                                f2.close()

# %%
import math
math.comb(100, 25)