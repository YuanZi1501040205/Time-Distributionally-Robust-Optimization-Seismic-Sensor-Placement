import numpy as np
import pandas as pd
import chama
from tqdm import tqdm
from sympy import symbols, solve
import matplotlib.pyplot as plt
import time
""" setting:  1 seismic source position, multiple velocity traces, output multiple detect time for each sensor positions
 1: plot sampled velocity distribution and detected time distribution of some sensors.
 2: plot mean velocity and associated time compared to the detected time distribution.
 3: plot DRO shifted max spike time distribution and compare to the sampled time distribution.
 
 experience when perform experiments:
 1. high sensitivity sensor->low detect time->even the wind has perturbation, the detection time has low variance
 2. low sensitivity sensor->low signal in grid points->many sensor candidators position can not detect any signal
 3. high confidential level->more conservative->bigger time delay shift for dro correction
 4. more samples->
 
"""


def Wasserstein_upper_bound(distribution, num_bins=10, gamma=0.9):
    # distribution: empirical distribution
    # gamma: Confidence level
    # (high gamma, more sample)->high shift
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

        if sz[0] in zlayer + 1:
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

# %% create impact panda df
from psmodules import psarray, pssynthetic, psraytrace, pswavelet, \
    psplot, pspicker, pspdf
Sensor_name_list = []
Impact_list = []
Scenario_name_list = []


# simulation
event_list = ['Event_1']

for event in event_list:
    # Generate square grid array
    geox, geoy, geoz = psarray.gridarray(25, 10000, 10000)

    # Define source coordinates
    src_gt = [5000, 5000,  2100]
    # Define geological model
    zlayer = np.array([0, 540, 1070, 1390, 1740, 1950, 2290,
                    2630, 4000])

    # Define velocity model
    # P wave velocity
    vp = np.array([2100, 2500, 2950, 3300, 3700, 4200,
                4700, 5800])

    # formatting the input for forward
    vp, vs, zlayer, dg, src, rcv = prepare_forward_input(geox, geoy, geoz, src_gt, zlayer, vp)

    # run forward simulation
    forwarder = SingleEventForwarder(vp, vs, zlayer, dg, rcv)
    event_min_detect_time = forwarder.forward_obs(src)

    # each sensor's record
    for sensor_idx in range(len(geox)):
        name_sensor = 'sensor_' + str(int(geox[sensor_idx])) + '_' + str(int(geoy[sensor_idx])) + '_' + str(int(geoz[sensor_idx]))
        event_sensor_impact = event_min_detect_time[sensor_idx]
        Scenario_name_list.append(event)
        Sensor_name_list.append(name_sensor)
        Impact_list.append(event_sensor_impact)
min_det_time = pd.DataFrame({'Scenario': Scenario_name_list, 'Sensor': Sensor_name_list, 'Impact': Impact_list})
# %% create scenario panda db

Undetected_Impact_list = []
Scenario_Probability_list = []
for event in event_list:
    Undetected_Impact_list.append(10000)
    Scenario_Probability_list.append(float(1))


scenario_prob = pd.DataFrame({'Scenario': event_list, 'Undetected Impact': Undetected_Impact_list, 'Probability': Scenario_Probability_list})
# %% create sensor_cost panda df
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
impactform = chama.optimize.ImpactFormulation()
results = impactform.solve(impact=min_det_time, sensor_budget=1000,
                             sensor=sensor_cost_pairs, scenario=scenario_prob,
                             use_scenario_probability=True,
                             use_sensor_cost=True)

# %% multiple samples of min detection time distribution

num_v_samples = 85
vp_mean = np.array([2100, 2500, 2950, 3300, 3700, 4200,
                4700, 5800])
vp_std = 1000
np.random.seed(0)
v_samples = sample_velocity_model(num_v_samples, vp_mean, vp_std)

Sensor_name_list = []
Impact_list = []
Scenario_name_list = []

for i in tqdm(range(num_v_samples)):
    # formatting the input for forward
    event_name = 'Sample_' + str(i)
    vp, vs, zlayer, dg, src, rcv = prepare_forward_input(geox, geoy, geoz, src_gt, zlayer, v_samples[i])

    # run forward simulation
    forwarder = SingleEventForwarder(vp, vs, zlayer, dg, rcv)
    event_min_detect_time = forwarder.forward_obs(src)
    # each sensor's record
    for sensor_idx in range(len(geox)):
        name_sensor = 'sensor_' + str(int(geox[sensor_idx])) + '_' + str(int(geoy[sensor_idx])) + '_' + str(int(geoz[sensor_idx]))
        event_sensor_impact = event_min_detect_time[sensor_idx]
        Scenario_name_list.append(event_name)
        Sensor_name_list.append(name_sensor)
        Impact_list.append(event_sensor_impact)



min_det_time_samples = pd.DataFrame({'Scenario': Scenario_name_list, 'Sensor': Sensor_name_list, 'Impact': Impact_list})
# %% stat min detect time per sensor
sensor_list = list(set(min_det_time_samples['Sensor']))
active_sensor_det_time_distribution_list = []
dro_det_time_list_3 = []
dro_det_time_list_5 = []
dro_det_time_list_9 = []
for active_sensor in sensor_list:
    det_time_distribution = min_det_time_samples[min_det_time_samples['Sensor'] == active_sensor]['Impact'].values
    active_sensor_det_time_distribution_list.append(det_time_distribution)
    active_sensor_dro_det_time_3 = Wasserstein_upper_bound(det_time_distribution, num_bins=5, gamma=0.3)
    active_sensor_dro_det_time_5 = Wasserstein_upper_bound(det_time_distribution, num_bins=5, gamma=0.5)
    active_sensor_dro_det_time_9 = Wasserstein_upper_bound(det_time_distribution, num_bins=5, gamma=0.9)
    dro_det_time_list_3.append(active_sensor_dro_det_time_3)
    dro_det_time_list_5.append(active_sensor_dro_det_time_5)
    dro_det_time_list_9.append(active_sensor_dro_det_time_9)

# %% plot empirical distribution and the wasserstein upper bound distribution
visualiz_sensor_idx = 10
visualize_sensor = sensor_list[visualiz_sensor_idx]
visualize_distribution = active_sensor_det_time_distribution_list[visualiz_sensor_idx]
len_visualize_distribution = visualize_distribution.__len__()
dro_det_time_3 = dro_det_time_list_3[visualiz_sensor_idx]
dro_det_time_5 = dro_det_time_list_5[visualiz_sensor_idx]
dro_det_time_9 = dro_det_time_list_9[visualiz_sensor_idx]
mean_wind_det_time = min_det_time[min_det_time['Sensor'] == visualize_sensor]['Impact'].values[0]


fig, ax = plt.subplots(figsize=(15, 12), dpi=80)
plt.hist(visualize_distribution,bins=5, color='b', edgecolor='k', alpha=0.6, label='Empirical distribution')
plt.hist([dro_det_time_3] * len_visualize_distribution, color='r', alpha=0.8, label='DRO with 30% confidential level')
plt.hist([dro_det_time_5] * len_visualize_distribution, color='c', alpha=0.8, label='DRO with 50% confidential level')
plt.hist([dro_det_time_9] * len_visualize_distribution, color='y', alpha=0.8, label='DRO with 90% confidential level')
plt.hist([mean_wind_det_time] * len_visualize_distribution, color='g', alpha=0.8, label='Baseline')

plt.xlabel('Detection time/s', fontsize=36)
plt.ylabel('Number of events', fontsize=36)
plt.xticks(fontsize=36)
plt.yticks(fontsize=36)
plt.legend(fontsize=36)
plt.savefig('detection_time_distribution_dro.png')
plt.show()
plt.clf()


