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

# read leak position
import csv
# read microseismic locations
with open('./data/raw/fault_three_location.csv') as leakfile:
    csvreader = csv.reader(leakfile)
    rows = []
    for row in csvreader:
        rows.append(row)
seismic_events_positions = []
for coordinate in rows[1:]:
    seismic_events_positions.append([int(float(coordinate[0])), int(float(coordinate[1]))])

fault_x = np.array(seismic_events_positions).T[0]
fault_y = np.array(seismic_events_positions).T[1]
fault_z = np.zeros(len(fault_x)) + 2100

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
min_det_time = seismic_event_simulation(vp, event_positions, geo_sensor_candidates_positions, zlayer, e_name='Event')

#  create scenario panda db for optimization on basic seismic events set
Impact_list = min_det_time['Impact']
Scenario_name_list_no_redundant = list(set(min_det_time['Scenario']))
Scenario_name_list_no_redundant.sort()
Undetected_Impact_list = []
Scenario_Probability_list = []
for event in Scenario_name_list_no_redundant:
    Undetected_Impact_list.append(10 * max(Impact_list))
    Scenario_Probability_list.append(float(1 / len(Scenario_name_list_no_redundant)))

scenario_prob_basic = pd.DataFrame(
    {'Scenario': Scenario_name_list_no_redundant, 'Undetected Impact': Undetected_Impact_list,
     'Probability': Scenario_Probability_list})
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
# %%
grid_obj_list = []
so_obj_list = []
mean_sensor_place_strategy_list = []
for i in [2, 3, 5, 9]:

    # optimization
    sens_budget = 1000 * (i**2)
    mean_basic_results = optimize_sensor(min_det_time, sensor_cost_pairs, scenario_prob_basic, sens_budget)
    mean_sensor_place_strategy = mean_basic_results['Sensors']
    print(mean_sensor_place_strategy)
    print(mean_basic_results['Objective'])
    so_obj_list.append(mean_basic_results['Objective'])
    mean_sensor_place_strategy_list.append(mean_sensor_place_strategy)

    #  evaluate grid sensor placement with events set
    # grid placement strategy
    geox, geoy, geoz = psarray.gridarray(int(i**2), 10000, 10000)  # extract sensor in placed positions
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
    grid_obj_list.append(grid_obj)


# %%
print('grid_obj_list: ', grid_obj_list)
print('so_obj_list: ', so_obj_list)
print('mean_sensor_place_strategy_list: ', mean_sensor_place_strategy_list)
# %% plot performance lines
import matplotlib.pyplot as plt
import numpy as np

plt.style.available
# %%
plt.style.use(['ieee'])
# Fixing random state for reproducibility
x = [4, 9, 25, 81]

plt.plot(x, so_obj_list, label='SO')
plt.plot(x, grid_obj_list, label='Regular')

plt.xlabel("Number of sensors")
plt.ylabel("Detection time")
plt.ylim([0, 2])
plt.legend()
plt.tight_layout()
plt.savefig('seg_performance_lines.png')
plt.show()
# %%
x_grid_place = []
y_grid_place = []
for i in [2, 3, 5, 9]:
    geox, geoy, geoz = psarray.gridarray(int(i**2), 10000, 10000)  # extract sensor in placed positions
# %% so placement
x_so_place = []
y_so_place = []
for sensor in mean_sensor_place_strategy_list[1]:
    x_sensor = int(sensor.split('_')[1])
    y_sensor = int(sensor.split('_')[2])
    x_so_place.append(x_sensor)
    y_so_place.append(y_sensor)
# %%
geox, geoy, geoz = psarray.gridarray(25, 10000, 10000)
so_solution = np.array([False, False, False,  True, False,  True, False, False, False,
       False, False,  True, False, False, False, False,  True, False,
        True,  True, False, False,  True,  True,  True])

so_placed_sensor_index = so_solution.nonzero()[0]
geox_so_loc = geox[so_placed_sensor_index]
geoy_so_loc = geoy[so_placed_sensor_index]

# %% plot sensor placement

# import numpy as np
# import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
plt.style.use(['ieee'])
figure(figsize=(5, 5), dpi=300)
plt.grid(True)

x_leak = fault_x
y_leak = fault_y

plt.xlim([-500, 10500])
plt.ylim([-500, 10500])
plt.scatter(x_leak, y_leak, s=40, marker='X', color='#3C4E72', label='Sources')

type_marker = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
# plot grid strategy

geox, geoy, geoz = psarray.gridarray(9, 10000, 10000)  # extract sensor in placed positions
plt.scatter(geox, geoy, s=120, marker='d', color='#D98F4E', label='Regular network')

plt.scatter(x_so_place, y_so_place, s=80, marker='o', color='#C0B69B', label='Detection time Optimization')
#
# plt.scatter(geox_so_loc, geoy_so_loc, s=60, marker='*', c='g', label='Source localization Optimization')

plt.xlabel("X", fontweight='bold',  fontsize=15)
plt.ylabel("Y", fontweight='bold',  fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=10, loc='lower left')
plt.tight_layout()
plt.savefig('seg_placement_vis.png')
plt.show()
plt.close('all')


