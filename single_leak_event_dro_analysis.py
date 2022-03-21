import numpy as np
import pandas as pd
import chama
from tqdm import tqdm
from sympy import symbols, solve
import matplotlib.pyplot as plt
""" setting:  1 leak position, 1 leak rate, mutiple wind speed traces, 1 wind direction, output mutiple detect time for each sensor positions
 1: plot sampled wind distribution and detected time distribution of some sensors.
 2: plot mean wind and associated time compared to the detected time distribution.
 3: plot DRO shifted max spike time distribution and compare to the sampled time distribution.
 4: plot DRO shifted max spike time distribution and compare to the sampled time distribution.
 
 experience when perform experiments:
 1. high sensitivity sensor->low detect time->even the wind has perturbation, the detection time has low variance
 2. low sensitivity sensor->low signal in grid points->many sensor candidators position can not detect any signal
 3. high confidential level->more conservative->bigger time delay shift for dro correction
 
"""

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
    print('sensors generated!')
    return sensors, sensor_cost_pairs


def extract_min_detect_time(signal, sensors):
    det_times = chama.impact.extract_detection_times(signal, sensors)

    #  extract statistic of detection time
    det_time_stats = chama.impact.detection_time_stats(det_times)
    #  extract the min detect time
    min_det_time = det_time_stats[['Scenario', 'Sensor', 'Min']]
    min_det_time = min_det_time.rename(columns={'Min': 'Impact'})
    return min_det_time


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
    return result

# %% define the grid
x_grid = np.linspace(0, 99, 100)
y_grid = np.linspace(0, 99, 100)
z_grid = np.linspace(0, 9, 10)
grid = chama.simulation.Grid(x_grid, y_grid, z_grid)

# %% atmospheric conditions
wind_speed_mean = [6.72, 8.87, 9.73, 9.27, 7.43, 6.73, 6.05, 6.36, 7.89, 8.78, 9.09, 8.29, 8.44, 8.93, 8.38, 10.71,
                   7.95, 7.64, 6.17, 6.26, 5.65, 8.63, 7.83, 7.18]
wind_speed_std = 1.3837437106919512
wind_direction = [177.98, 185.43, 185.43, 184.68, 183.19, 182.45, 175.75, 178.72, 180.96, 198.09, 212.98, 224.15,
                  268.09, 277.77, 272.55, 272.55, 275.53, 281.49, 282.98, 298.62, 284.47, 332.13, 341.06, 337.34]
time_points = 24
atm = pd.DataFrame({'Wind Direction': wind_speed_mean,
                    'Wind Speed': wind_direction,
                    'Stability Class': ['A'] * 24}, index=list(np.array(range(24))))

# %% Initialize the Gaussian plume model
#  1 source
source = chama.simulation.Source(65, 60, 5, 5)
detect_threshold = 0.1
gauss_plume = chama.simulation.GaussianPlume(grid, source, atm)
gauss_plume.run()
signal_total = gauss_plume.conc
print(signal_total.head(5))
# where can be detected by 0.01 sensor
active_points_set = set()
active_points = np.where(signal_total['S'].values >= detect_threshold)[0]
for point in list(active_points):
    x = int(signal_total.iloc[point].X)
    y = int(signal_total.iloc[point].Y)
    z = int(signal_total.iloc[point].Z)
    active_points_set.add(str(x) + '_' + str(y) + '_' + str(z))
print('point: ', active_points_set)
# %% generate sensor
detect_threshold = 0.1
sensors, sensor_cost_pairs = generate_sensor_candidates(sensor_threshold=detect_threshold)
# %% 1 sample
min_det_time = extract_min_detect_time(signal_total, sensors)
print('min_det_time', min_det_time)
# %% plot leak signal propagation
chama.graphics.signal_xsection(signal_total, 'S', threshold=0.001, x_range=(50,100), y_range=(50,70), z_range=(4,9))

# %% multiple samples of min detection time distribution
source = chama.simulation.Source(65, 60, 5, 5)

gauss_plume = chama.simulation.GaussianPlume(grid, source, atm)
gauss_plume.run()
signals = gauss_plume.conc
signals = signals.drop(columns='S')
print(signals.head(5))

num_wind_samples = 25
np.random.seed(1997)
wind_speed_samples = sample_wind_speed(num_wind_samples, wind_speed_mean, wind_speed_std)

for i in tqdm(range(num_wind_samples)):
    atm['Wind Speed'] = wind_speed_samples[i]
    # propagation
    gauss_plume = chama.simulation.GaussianPlume(grid, source, atm)
    gauss_plume.run()
    signal = gauss_plume.conc
    scenario_name = 'Sample' + str(i)
    signals[scenario_name] = signal['S']
    i = i + 1

min_det_time_samples = extract_min_detect_time(signals, sensors)
# %% stat min detect time per sensor
sensor_list = list(set(min_det_time_samples['Sensor']))
active_sensor_det_time_distribution_list = []
dro_det_time_list = []
for active_sensor in sensor_list:
    det_time_distribution = min_det_time_samples[min_det_time_samples['Sensor'] == active_sensor]['Impact'].values
    active_sensor_det_time_distribution_list.append(det_time_distribution)
    active_sensor_dro_det_time = complex(Wasserstein_upper_bound(det_time_distribution, gamma=0.1)[-1]).real
    dro_det_time_list.append(active_sensor_dro_det_time)

# %% plot empirical distribution and the wasserstein upper bound distribution
visualiz_sensor_idx = 2
visualize_sensor = sensor_list[visualiz_sensor_idx]
visualize_distribution = active_sensor_det_time_distribution_list[visualiz_sensor_idx]
len_visualize_distribution = visualize_distribution.__len__()
dro_det_time = dro_det_time_list[visualiz_sensor_idx]
mean_wind_det_time = min_det_time[min_det_time['Sensor'] == visualize_sensor]['Impact'].values[0]


plt.hist(visualize_distribution, color='b', edgecolor='k', alpha=0.6, label='empirical distribution')
plt.hist([dro_det_time] * len_visualize_distribution, color='r', alpha=0.6, label='dro')
plt.hist([mean_wind_det_time] * len_visualize_distribution, color='g', alpha=0.6, label='baseline')
plt.legend()
plt.savefig('gamma10.png')
plt.show()
plt.clf()


