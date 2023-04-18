import numpy as np
import pandas as pd
import chama
from tqdm import tqdm
from sympy import symbols, solve
import matplotlib.pyplot as plt
import time

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

# %% read leak position
import csv
with open('./data/raw/fault_poso_creek_location.csv') as leakfile:
    csvreader = csv.reader(leakfile)
    rows = []
    for row in csvreader:
        rows.append(row)
seismic_events_positions = []
for coordinate in rows[1:]:
    seismic_events_positions.append([int(10*float(coordinate[0])), int(float(coordinate[1]))])

# fault 1
fault_1_x = 3000 + 4000*(np.array(seismic_events_positions).T[0] - np.array(seismic_events_positions).T[0][0])/(np.array(seismic_events_positions).T[0][-1] - np.array(seismic_events_positions).T[0][0])
fault_1_y = 2000 + 4000*(np.array(seismic_events_positions).T[1] - np.array(seismic_events_positions).T[1][0])/(np.array(seismic_events_positions).T[1][-1] - np.array(seismic_events_positions).T[1][0])
fault_1_z = np.zeros(len(fault_1_x)) + 2100

fault_2_x = 2500 + 4000*(np.array(seismic_events_positions).T[1] - np.array(seismic_events_positions).T[1][0])/(np.array(seismic_events_positions).T[1][-1] - np.array(seismic_events_positions).T[1][0])
fault_2_y = 3500 + 4000*(np.array(seismic_events_positions).T[0] - np.array(seismic_events_positions).T[0][0])/(np.array(seismic_events_positions).T[0][-1] - np.array(seismic_events_positions).T[0][0])
fault_2_z = np.zeros(len(fault_1_x)) + 2100

fault_x = np.concatenate((fault_1_x, fault_2_x), axis=0)
fault_y = np.concatenate((fault_1_y, fault_2_y), axis=0)
fault_z = np.concatenate((fault_1_z, fault_2_z), axis=0)
# %%
# single fault source
fault_x = np.array([fault_1_x[0]])
fault_y = np.array([fault_1_y[0]])
fault_z = np.array([fault_1_z[0]])
# %% create impact panda df for optimization on basic seismic events set
from psmodules import psarray, pssynthetic, psraytrace, pswavelet, \
    psplot, pspicker, pspdf

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

# %% DRO: MEAN test, DRO train/test,SO train/test

num_train_sample = 4
num_test_sample = 8
random_seed = 0
gamma = 0.8
v_test_std = 500
v_train_std = 500
vp_mean = np.array([2100, 2500, 2950, 3300, 3700, 4200,
                4700, 5800])
# test wind speed samples
np.random.seed(random_seed)
vp_samples_train = sample_velocity_model(num_train_sample, vp_mean, v_train_std)
vp_samples_test = sample_velocity_model(num_test_sample, vp_samples_train[0], v_test_std)


#  create scenario panda db for optimization on basic seismic events set
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
# stat multi events for training DRO/SO
impact_stat_dict = {}  # dictionary to store distribution of each event-sensor's impact
for i in tqdm(range(num_train_sample), leave=False):
    # use perturbation velocity
    v_noise = vp_samples_train[i]

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


#
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
# %% drop sensor
import random
random.seed(0)
dropped_impact_stat_list = []
for impact_list in impact_stat_list:
    # Creating a number list
    dropped_impact_list = random.choices(impact_list, k=random.randint(1, int(len(impact_list)/4)))
    # dropped_impact_list = random.choices(impact_list, k=int(len(impact_list)/2))
    dropped_impact_stat_list.append(dropped_impact_list)
dropped_stat_df_min_det_time = pd.DataFrame({'Scenario': scenario_stat_list,
                                     'Sensor': sensor_stat_list,
                                     'Impact': dropped_impact_stat_list})
# %%
# stat to panda df detection time
gamma = 0.9
num_bins=5
dro_impact = []
for i in range(dropped_stat_df_min_det_time.__len__()):
    impact_distribution = dropped_stat_df_min_det_time.iloc[i]['Impact']

    # DRO impact correction
    robust_impact_value = Wasserstein_upper_bound(impact_distribution, num_bins=num_bins, gamma=gamma)
    # robust_impact_value = max(impact_distribution)

    # dro_sensors.append(robust_sensor)
    dro_impact.append(robust_impact_value)
# dro corrected detect impact dataframe
min_det_time_dro = pd.DataFrame({'Scenario': scenario_stat_list,
                                 'Sensor': sensor_stat_list,
                                 'Impact': list(dro_impact)})


#  SO optimization
so_impact = []
for i in range(dropped_stat_df_min_det_time.__len__()):
    impact_distribution = dropped_stat_df_min_det_time.iloc[i]['Impact']

    # DRO impact correction
    so_impact_value = np.mean(impact_distribution)

    # dro_sensors.append(robust_sensor)
    so_impact.append(so_impact_value)
# dro corrected detect impact dataframe
min_det_time_so = pd.DataFrame({'Scenario': scenario_stat_list,
                                 'Sensor': sensor_stat_list,
                                 'Impact': list(so_impact)})

#  evaluate dro placement strategy on test dataset

# dro_train_result = eval_sensor_placement(min_det_time_samples_train, sensor_cost_pairs,
#                                          scenario_prob_train,
#                                          dro_sensor_place_strategy)
#
# dro_test_result = eval_sensor_placement(min_det_time_samples_test, sensor_cost_pairs,
#                                          scenario_prob_test,
#                                          dro_sensor_place_strategy)

#  create sensor_cost panda df
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
#  DRO optimization: placement

# obj_dro_list = []
# for i in range(9):
#     sens_budget = 1000 * (i+1)
#     opt_result_dro = optimize_sensor(min_det_time_dro, sensor_cost_pairs, scenario_prob_basic, sens_budget)
#     dro_sensor_place_strategy = opt_result_dro['Sensors']
#     obj_dro_list.append(opt_result_dro['Objective'])
# print('obj_dro_list: ', obj_dro_list)
#
sens_budget = 2000
opt_result_dro = optimize_sensor(min_det_time_dro, sensor_cost_pairs, scenario_prob_basic, sens_budget)
dro_sensor_place_strategy = opt_result_dro['Sensors']
opt_result_so = optimize_sensor(min_det_time_so, sensor_cost_pairs, scenario_prob_basic, sens_budget)
so_sensor_place_strategy = opt_result_so['Sensors']

print('dro_sensor_place_strategy: ', dro_sensor_place_strategy)
print('so_sensor_place_strategy: ', so_sensor_place_strategy)
# %%
import matplotlib.pyplot as plt
import numpy as np
# plt.style.use(['ieee'])
# Fixing random state for reproducibility
x = range(len(dro_impact))

plt.plot(x, so_impact, label='SO')
plt.plot(x, dro_impact, label='DRO')

plt.xlabel("sensors")
plt.ylabel("Detection time")
# plt.ylim([0, 2])
plt.legend()
plt.tight_layout()
# plt.savefig('performance_lines.png')
plt.show()