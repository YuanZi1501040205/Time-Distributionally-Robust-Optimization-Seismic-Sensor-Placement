import numpy as np
import pandas as pd
import chama
import csv
# %% define the grid
TOTAL_SENARIO_NUM = 50
x_grid = np.linspace(0, 99, 100)
y_grid = np.linspace(0, 99, 100)
z_grid = np.linspace(0, 9, 10)
grid = chama.simulation.Grid(x_grid, y_grid, z_grid)
# %% source
source = chama.simulation.Source(25, 75, 1, 2)
# %% atmospheric conditions
atm = pd.DataFrame({'Wind Direction': [177.98,185.43,185.43,184.68,183.19,182.45,175.75,178.72,180.96,198.09,212.98,224.15,268.09,277.77,272.55,272.55,275.53,281.49,282.98,298.62,284.47,332.13,341.06,337.34],
                    'Wind Speed': [6.72,8.87,9.73,9.27,7.43,6.73,6.05,6.36,7.89,8.78,9.09,8.29,8.44,8.93,8.38,10.71,7.95,7.64,6.17,6.26,5.65,8.63,7.83,7.18],
                   'Stability Class': ['A']*24}, index=list(np.array(range(24))))
# %% Initialize the Gaussian plume model
gauss_plume = chama.simulation.GaussianPlume(grid, source, atm)
gauss_plume.run()
signal_total = gauss_plume.conc
signal_total = signal_total.drop(columns='S')
print(signal_total.head(5))
# %% read leak position
with open('./data/raw/leak_positions.csv') as leakfile:
    csvreader = csv.reader(leakfile)
    rows = []
    for row in csvreader:
        rows.append(row)
leak_positions = []
for coordinate in rows[1:]:
    leak_positions.append([int(coordinate[0]), int(coordinate[1])])
# %% leak positions times 10 for 100 scenarios
leak_positions = leak_positions*int(TOTAL_SENARIO_NUM/10)
# %% pdf
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

# %% cdf
# import matplotlib.pyplot as plt
# # Compute the CDF
# d_x = leak_rate[1]
# CY = np.cumsum(leak_prob * d_x)
# plt.plot(leak_rate, CY)
# plt.show()
# plt.close()
# %% sample 100 leak rates from PDF of
import numpy as np

# Choose elements with different probabilities
np.random.seed(0)
sampleLeakRates = np.random.choice(list(leak_rate), TOTAL_SENARIO_NUM, p=list(leak_prob)/leak_prob.sum())
print(sampleLeakRates)
# %% scenarios probability
import numpy as np

# Choose elements with different probabilities
np.random.seed(0)
sampleLeakHeights = np.random.choice([0,1,2], TOTAL_SENARIO_NUM, p=[0.33,0.33,0.34])
print(sampleLeakHeights)
# %% Simulation 1000 scenarios

scenario_worst_impacts = []
scenario_names = []
scenario_probs = []
for i, (leak_point, leak_h, leak_r)  in enumerate(zip(leak_positions, sampleLeakHeights, sampleLeakRates)):
    print('simulation: ', i)
    # source
    source = chama.simulation.Source(leak_point[0], leak_point[1], leak_h, leak_r)
    gauss_plume = chama.simulation.GaussianPlume(grid, source, atm)
    gauss_plume.run()
    signal = gauss_plume.conc
    scenario_name = 'S' + str(i)
    signal_total[scenario_name] = signal['S']
    i = i + 1
    print(signal_total.head(5))
    scenario_worst_impacts.append(24*TOTAL_SENARIO_NUM)
    scenario_names.append(scenario_name)
    scenario_probs.append(1/leak_positions.__len__())

scenario =pd.DataFrame({'Scenario': scenario_names,'Undetected Impact': scenario_worst_impacts, 'Probability': scenario_probs})
# %% save simulation result to csv
# signal_total.to_csv('scenarios.csv', index = False)
# %% x sections
# chama.graphics.signal_xsection(signal_total, 'S1', threshold=0.01)
# %% place 9*9*10*2 (raw*column*height*sensor types) stationary sensor
sensors = dict()
sensor_names = []
count = 0
sensor_costs = []
for i in range(9):
    for j in range(9):
        for m in range(10):
            for n in range(1):
                sensor_name = 'A_'+str(i)+'_'+str(j)+'_'+str(m)+'_'+str(n)
                sensor_names.append(sensor_name)
                pos = chama.sensors.Stationary(location=((i+1)*10,(j+1)*10,m))
                if n == 0:
                    sensor_threshold = 0.1 #regular sensor
                    sensor_cost = 10000
                elif n == 1:
                    sensor_threshold = 0.01 #high sensitivity sensor
                    sensor_cost = 100000

                sensor_costs.append(sensor_cost)
                det = chama.sensors.Point(threshold=sensor_threshold, sample_times=list(range(24)))
                stationary_pt_sensor = chama.sensors.Sensor(position=pos, detector=det)
                sensors[sensor_name] = stationary_pt_sensor
                print('count: ', count)
                count = count + 1

sensor =pd.DataFrame({'Sensor': sensor_names,'Cost': sensor_costs})



# %% extract detection time
det_times = chama.impact.extract_detection_times(signal_total, sensors)
# print('det_times: ', det_times)
# %% extract statistic of detection time
det_time_stats = chama.impact.detection_time_stats(det_times)
# %% extract the min detect time
min_det_time = det_time_stats[['Scenario','Sensor','Min']]
min_det_time = min_det_time.rename(columns={'Min':'Impact'})
# %% optimization impact formulation
impactform = chama.optimize.ImpactFormulation()
results = impactform.solve(impact=min_det_time, sensor_budget=100000,
                           sensor=sensor, scenario=scenario,
                             use_scenario_probability=True,
                           use_sensor_cost=True)
# %% eliminate non detected scenarios
covered_scenario_number = int(np.where(results['Assessment']['Sensor'].values == None)[0][0])
print('Detected scenarios: ', covered_scenario_number)
scenario_names_cache = results['Assessment']['Scenario'].values[:covered_scenario_number]
new_leak_positions = []
new_sampleLeakHeights = []
new_sampleLeakRates = []
for detect_scenario in scenario_names_cache:
    s_idx = int(detect_scenario.split('S')[-1])
    leak_point = leak_positions[s_idx]
    leak_h = sampleLeakHeights[s_idx]
    leak_r = sampleLeakRates[s_idx]
    new_leak_positions.append(leak_point)
    new_sampleLeakHeights.append(leak_h)
    new_sampleLeakRates.append(leak_r)
#
# %%  COVERED SIMULATION source
source = chama.simulation.Source(25, 75, 1, 2)
# %% atmospheric conditions
atm = pd.DataFrame({'Wind Direction': [177.98,185.43,185.43,184.68,183.19,182.45,175.75,178.72,180.96,198.09,212.98,224.15,268.09,277.77,272.55,272.55,275.53,281.49,282.98,298.62,284.47,332.13,341.06,337.34],
                    'Wind Speed': [6.72,8.87,9.73,9.27,7.43,6.73,6.05,6.36,7.89,8.78,9.09,8.29,8.44,8.93,8.38,10.71,7.95,7.64,6.17,6.26,5.65,8.63,7.83,7.18],
                   'Stability Class': ['A']*24}, index=list(np.array(range(24))))
# %% Initialize the Gaussian plume model
gauss_plume = chama.simulation.GaussianPlume(grid, source, atm)
gauss_plume.run()
signal_vanilla = gauss_plume.conc
signal_vanilla = signal_vanilla.drop(columns='S')
print(signal_vanilla.head(5))
# %% simulation covered scenarios
scenario_worst_impacts = []
scenario_names = []
scenario_probs = []
for i, (leak_point, leak_h, leak_r)  in enumerate(zip(new_leak_positions, new_sampleLeakHeights, new_sampleLeakRates)):
    print('simulation: ', i)
    # source
    source = chama.simulation.Source(leak_point[0], leak_point[1], leak_h, leak_r)
    gauss_plume = chama.simulation.GaussianPlume(grid, source, atm)
    gauss_plume.run()
    signal = gauss_plume.conc
    scenario_name = 'S' + str(i)
    signal_vanilla[scenario_name] = signal['S']
    print(signal_vanilla.head(5))
    scenario_worst_impacts.append(24*new_leak_positions.__len__())
    scenario_names.append(scenario_name)
    scenario_probs.append(1/new_leak_positions.__len__())

scenario =pd.DataFrame({'Scenario': scenario_names,'Undetected Impact': scenario_worst_impacts, 'Probability': scenario_probs})

# %% extract detection time
det_times = chama.impact.extract_detection_times(signal_vanilla, sensors)
# print('det_times: ', det_times)
# %% extract statistic of detection time
det_time_stats = chama.impact.detection_time_stats(det_times)
# %% extract the min detect time
min_det_time = det_time_stats[['Scenario','Sensor','Min']]
min_det_time = min_det_time.rename(columns={'Min':'Impact'})
# %% optimization impact formulation
impactform = chama.optimize.ImpactFormulation()
results = impactform.solve(impact=min_det_time, sensor_budget=100000,
                           sensor=sensor, scenario=scenario,
                             use_scenario_probability=True,
                           use_sensor_cost=True)
# %% optimization result
print(results['Sensors'])
print(results['Objective'])
print(results['Assessment'])
# %% arrange sensors for plot
result_sensors = dict()
count = 0
for name_sensor in results['Sensors']:
    sensor_x = int(name_sensor.split('_')[1] * 10)
    sensor_y = int(name_sensor.split('_')[2] * 10)
    sensor_h = int(name_sensor.split('_')[3])
    sensor_type = int(name_sensor.split('_')[4])
    pos = chama.sensors.Stationary(location=(sensor_x,sensor_y,sensor_h))
    if sensor_type == 0:
        sensor_threshold = 0.1 #regular sensor
        sensor_cost = 10000
    elif sensor_type == 1:
        sensor_threshold = 0.01 #high sensitivity sensor
        sensor_cost = 100000
    det = chama.sensors.Point(threshold=sensor_threshold, sample_times=list(range(24)))
    stationary_pt_sensor = chama.sensors.Sensor(position=pos, detector=det)
    result_sensors[name_sensor] = stationary_pt_sensor
    print('count: ', count)
    count = count + 1
# %% plot sensor
# chama.graphics.sensor_locations(result_sensors)
# %% impact static plot
# results['Assessment'].plot(kind='bar')
# plt.show()
# %% atmospheric conditions + perturbation
atm = pd.DataFrame({'Wind Direction': [177.98,185.43,185.43,184.68,183.19,182.45,175.75,178.72,180.96,198.09,212.98,224.15,268.09,277.77,272.55,272.55,275.53,281.49,282.98,298.62,284.47,332.13,341.06,337.34],
                    'Wind Speed': [8.67,8.67,9.75,8.18,9.81,10.27,5.14,7.96,5.86,10.27,8.59,8.04,5.56,10.54,8.49,11.28,8.39,7.24,7.32,6.86,8.41,11.40,7.76,8.40],
                   'Stability Class': ['A']*24}, index=list(np.array(range(24))))
# %% Initialize the Gaussian plume model with perturbed wind
gauss_plume = chama.simulation.GaussianPlume(grid, source, atm)
gauss_plume.run()
signal_noise = gauss_plume.conc
signal_noise = signal_noise.drop(columns='S')
print(signal_noise.head(5))
# %% simulation covered scenarios with perturbed wind
scenario_worst_impacts = []
scenario_names = []
scenario_probs = []
for i, (leak_point, leak_h, leak_r)  in enumerate(zip(new_leak_positions, new_sampleLeakHeights, new_sampleLeakRates)):
    print('simulation: ', i)
    # source
    source = chama.simulation.Source(leak_point[0], leak_point[1], leak_h, leak_r)
    gauss_plume = chama.simulation.GaussianPlume(grid, source, atm)
    gauss_plume.run()
    signal = gauss_plume.conc
    scenario_name = 'S' + str(i)
    signal_noise[scenario_name] = signal['S']
    i = i + 1
    print(signal_noise.head(5))
    scenario_worst_impacts.append(24*new_leak_positions.__len__())
    scenario_names.append(scenario_name)
    scenario_probs.append(1/new_leak_positions.__len__())

scenario =pd.DataFrame({'Scenario': scenario_names,'Undetected Impact': scenario_worst_impacts, 'Probability': scenario_probs})
# %% extract detection time with perturbated wind
noise_det_times = chama.impact.extract_detection_times(signal_noise, sensors)
# %% extract statistic of detection time with perturbated wind
noise_det_time_stats = chama.impact.detection_time_stats(noise_det_times)
# %% extract the min detect time
noise_min_det_time = noise_det_time_stats[['Scenario','Sensor','Min']]
noise_min_det_time = noise_min_det_time.rename(columns={'Min':'Impact'})
# %% optimization impact formulation with perturbated wind
impactform = chama.optimize.ImpactFormulation()
noise_optimal_results = impactform.solve(impact=noise_min_det_time, sensor_budget=100000,
                           sensor=sensor, scenario=scenario,
                             use_scenario_probability=True,
                           use_sensor_cost=True)

# test vanilla placement strategy on noise wind data
# %% place stationary sensor based on the clean wind optimization result

sensors = dict()
sensor_names = []
count = 0
sensor_costs = []
for sensor_name in results['Sensors']:
    sensor_names.append(sensor_name)
    i = int(sensor_name.split('_')[1])
    j = int(sensor_name.split('_')[2])
    m = int(sensor_name.split('_')[3])
    n = int(sensor_name.split('_')[4])
    pos = chama.sensors.Stationary(location=((i+1)*10,(j+1)*10,m))
    if n == 0:
        sensor_threshold = 0.1 #regular sensor
        sensor_cost = 10000
    elif n == 1:
        sensor_threshold = 0.01 #high sensitivity sensor
        sensor_cost = 100000

    sensor_costs.append(sensor_cost)
    det = chama.sensors.Point(threshold=sensor_threshold, sample_times=list(range(24)))
    stationary_pt_sensor = chama.sensors.Sensor(position=pos, detector=det)
    sensors[sensor_name] = stationary_pt_sensor
    print('count: ', count)
    count = count + 1

sensor =pd.DataFrame({'Sensor': sensor_names,'Cost': sensor_costs})


naive_noise_det_times = chama.impact.extract_detection_times(signal_noise, sensors)
# %% extract statistic of detection time with perturbated wind
naive_noise_det_time_stats = chama.impact.detection_time_stats(naive_noise_det_times)
# %% extract the min detect time
min_naive_noise_det_time = naive_noise_det_time_stats[['Scenario','Sensor','Min']]
min_naive_noise_det_time = min_naive_noise_det_time.rename(columns={'Min':'Impact'})


# %% noise optimization result
print(results['Sensors'])
print(results['Objective'])

print('old strategy on noise objective function: ', min_naive_noise_det_time['Impact'].mean())

print(noise_optimal_results['Sensors'])
print(noise_optimal_results['Objective'])

# %% DRO
S = 3
K = 3
gamma = 0.9
kappa = (K/(2*S))*np.log(2*K/(1-gamma))
print('kappa: ', kappa)
# %%  COVERED SIMULATION source
source = chama.simulation.Source(25, 75, 1, 2)
# %% atmospheric conditions
atm = pd.DataFrame({'Wind Direction': [177.98,185.43,185.43,184.68,183.19,182.45,175.75,178.72,180.96,198.09,212.98,224.15,268.09,277.77,272.55,272.55,275.53,281.49,282.98,298.62,284.47,332.13,341.06,337.34],
                    'Wind Speed': [6.72,8.87,9.73,9.27,7.43,6.73,6.05,6.36,7.89,8.78,9.09,8.29,8.44,8.93,8.38,10.71,7.95,7.64,6.17,6.26,5.65,8.63,7.83,7.18],
                   'Stability Class': ['A']*24}, index=list(np.array(range(24))))
# %% Initialize the Gaussian plume model
gauss_plume = chama.simulation.GaussianPlume(grid, source, atm)
gauss_plume.run()
signal_dro = gauss_plume.conc
signal_dro = signal_dro.drop(columns='S')
print(signal_dro.head(5))
# %% simulation covered scenarios
scenario_worst_impacts = []
scenario_names = []
scenario_probs = []
for i, (leak_point, leak_h, leak_r)  in enumerate(zip(new_leak_positions, new_sampleLeakHeights, new_sampleLeakRates)):
    print('simulation: ', i)
    # source
    source = chama.simulation.Source(leak_point[0], leak_point[1], leak_h, leak_r)
    gauss_plume = chama.simulation.GaussianPlume(grid, source, atm)
    gauss_plume.run()
    signal = gauss_plume.conc
    scenario_name = 'S' + str(i)
    signal_dro[scenario_name] = signal['S']
    i = i + 1
    print(signal_dro.head(5))
    scenario_worst_impacts.append(24*new_leak_positions.__len__())
    scenario_names.append(scenario_name)
    scenario_probs.append(1/new_leak_positions.__len__())

scenario =pd.DataFrame({'Scenario': scenario_names,'Undetected Impact': scenario_worst_impacts, 'Probability': scenario_probs})
# %% extract detection time
signal_dro = signal_vanilla
det_times_dro = chama.impact.extract_detection_times(signal_dro, sensors)
# print('det_times: ', det_times)
# %% extract statistic of detection time
det_time_stats_dro = chama.impact.detection_time_stats(det_times_dro)
# %% extract the min detect time
min_det_time_dro = det_time_stats_dro[['Scenario','Sensor','Min']]
min_det_time_dro = min_det_time_dro.rename(columns={'Min':'Impact'})
min_det_time_dro['Impact'] = min_det_time_dro['Impact'] + phi


sensors = dict()
sensor_names = []
count = 0
sensor_costs = []
for i in range(9):
    for j in range(9):
        for m in range(10):
            for n in range(1):
                sensor_name = 'A_'+str(i)+'_'+str(j)+'_'+str(m)+'_'+str(n)
                sensor_names.append(sensor_name)
                pos = chama.sensors.Stationary(location=((i+1)*10,(j+1)*10,m))
                if n == 0:
                    sensor_threshold = 0.1 #regular sensor
                    sensor_cost = 10000
                elif n == 1:
                    sensor_threshold = 0.01 #high sensitivity sensor
                    sensor_cost = 100000

                sensor_costs.append(sensor_cost)
                det = chama.sensors.Point(threshold=sensor_threshold, sample_times=list(range(24)))
                stationary_pt_sensor = chama.sensors.Sensor(position=pos, detector=det)
                sensors[sensor_name] = stationary_pt_sensor
                print('count: ', count)
                count = count + 1

sensor =pd.DataFrame({'Sensor': sensor_names,'Cost': sensor_costs})

# %% optimization impact formulation
impactform = chama.optimize.ImpactFormulation()
dro_results = impactform.solve(impact=min_det_time_dro, sensor_budget=100000,
                           sensor=sensor, scenario=scenario,
                             use_scenario_probability=True,
                           use_sensor_cost=True)
# %% optimization result
print(dro_results['Sensors'])
print(dro_results['Objective'])
# print(dro_results['Assessment'])

# DRO on noise test
# %% place stationary sensor based on the clean wind optimization result

sensors = dict()
sensor_names = []
count = 0
sensor_costs = []
for sensor_name in dro_results['Sensors']:
    sensor_names.append(sensor_name)
    i = int(sensor_name.split('_')[1])
    j = int(sensor_name.split('_')[2])
    m = int(sensor_name.split('_')[3])
    n = int(sensor_name.split('_')[4])
    pos = chama.sensors.Stationary(location=((i+1)*10,(j+1)*10,m))
    if n == 0:
        sensor_threshold = 0.1 #regular sensor
        sensor_cost = 10000
    elif n == 1:
        sensor_threshold = 0.01 #high sensitivity sensor
        sensor_cost = 100000

    sensor_costs.append(sensor_cost)
    det = chama.sensors.Point(threshold=sensor_threshold, sample_times=list(range(24)))
    stationary_pt_sensor = chama.sensors.Sensor(position=pos, detector=det)
    sensors[sensor_name] = stationary_pt_sensor
    print('count: ', count)
    count = count + 1

sensor =pd.DataFrame({'Sensor': sensor_names,'Cost': sensor_costs})


dro_noise_det_times = chama.impact.extract_detection_times(signal_noise, sensors)
# %% extract statistic of detection time with perturbated wind
dro_noise_det_time_stats = chama.impact.detection_time_stats(dro_noise_det_times)
# %% extract the min detect time
min_dro_noise_det_time = dro_noise_det_time_stats[['Scenario','Sensor','Min']]
min_dro_noise_det_time = min_dro_noise_det_time.rename(columns={'Min':'Impact'})

# DRO on clean test
# %% place stationary sensor based on the clean wind optimization result

dro_clean_det_times = chama.impact.extract_detection_times(signal_vanilla, sensors)
# %% extract statistic of detection time with perturbated wind
dro_clean_det_time_stats = chama.impact.detection_time_stats(dro_clean_det_times)
# %% extract the min detect time
min_dro_clean_det_time = dro_clean_det_time_stats[['Scenario','Sensor','Min']]
min_dro_clean_det_time = min_dro_clean_det_time.rename(columns={'Min':'Impact'})


# %% noise optimization result
print(results['Sensors'])
print(results['Objective'])


def impact2objective(min_det_time, num_scenarios = new_leak_positions.__len__()):
    obj_detect = 0
    for idx_scenario in range(num_scenarios):
        obj_detect = obj_detect + min_det_time[min_det_time['Scenario']=='S'+str(idx_scenario)]['Impact'].min()
    obj_detect = obj_detect/num_scenarios
    return obj_detect

obj_naive_optimal_normal = impact2objective(min_det_time, num_scenarios = new_leak_positions.__len__())
obj_naive_optimal_noise = impact2objective(min_naive_noise_det_time, num_scenarios = new_leak_positions.__len__())
obj_optimal_noise = impact2objective(noise_min_det_time, num_scenarios = new_leak_positions.__len__())
obj_dro_normal = impact2objective(min_det_time_dro, num_scenarios = new_leak_positions.__len__())
obj_dro_noise = impact2objective(min_dro_noise_det_time, num_scenarios = new_leak_positions.__len__())


print('naive placement on noise objective function: ', obj_naive_optimal_noise)
print('dro placement on noise objective function: ', obj_dro_noise)
print('optimal placement on noise objective function: ', obj_optimal_noise)
print('dro placement on clean objective function: ', obj_dro_normal)
print('naive placement on clean objective function: ', obj_naive_optimal_normal)


print(noise_optimal_results['Sensors'])
print(noise_optimal_results['Objective'])



