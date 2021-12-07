import numpy as np
import pandas as pd
import chama
import csv
# %% define the grid
x_grid = np.linspace(0, 100, 100)
y_grid = np.linspace(0, 100, 100)
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
leak_positions = leak_positions*10
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
import matplotlib.pyplot as plt
# Compute the CDF
d_x = leak_rate[1]
CY = np.cumsum(leak_prob * d_x)
plt.plot(leak_rate, CY)
plt.show()
# %% sample 100 leak rates from PDF of
import numpy as np

# Choose elements with different probabilities
sampleLeakRates = np.random.choice(list(leak_rate), 100, p=list(leak_prob)/leak_prob.sum())
print(sampleLeakRates)
# %%
import numpy as np

# Choose elements with different probabilities
sampleLeakHeights = np.random.choice([0,1,2], 100, p=[0.33,0.33,0.34])
print(sampleLeakHeights)
# %% Simulation 100 scenarios


for i, (leak_point, leak_h, leak_r)  in enumerate(zip(leak_positions, sampleLeakHeights, sampleLeakRates)):
    print('simulation: ', i)
    # source
    source = chama.simulation.Source(leak_point[0], leak_point[1], leak_h, leak_r)
    gauss_plume = chama.simulation.GaussianPlume(grid, source, atm)
    gauss_plume.run()
    signal = gauss_plume.conc
    signal_total['S' + str(i)] = signal['S']
    i = i + 1
    print(signal_total.head(5))
# %% save simulation result to csv
signal_total.to_csv('scenarios.csv', index = False)
# %% x sections
chama.graphics.signal_xsection(signal_total, 'S18', threshold=0.001)
# %% place 9*9*10*2 (raw*column*height*sensor types) stationary sensor
sensors = dict()
count =0
for i in range(9):
    for j in range(9):
        for m in range(10):
            for n in range(2):
                pos = chama.sensors.Stationary(location=((i+1)*10,(j+1)*10,m))
                if n == 0:
                    sensor_threshold = 0.1 #regular sensor
                elif n == 1:
                    sensor_threshold = 0.01 #high sensitivity sensor
                det = chama.sensors.Point(threshold=sensor_threshold, sample_times=list(range(24)))
                stationary_pt_sensor = chama.sensors.Sensor(position=pos, detector=det)
                sensors['A_'+str(i)+'_'+str(j)+'_'+str(m)+'_'+str(n)] = stationary_pt_sensor
                print('count: ', count)
                count = count + 1

# %% extract detection time
det_times = chama.impact.extract_detection_times(signal, sensors)
# print('det_times: ', det_times)
# %% extract the min detect time
det_time_stats = chama.impact.detection_time_stats(det_times)
print(det_time_stats)
min_det_time = det_time_stats[['Scenario','Sensor','Min']]
min_det_time = min_det_time.rename(columns={'Min':'Impact'})
print(min_det_time)
# %% optimization
# >>> print(min_det_time)
#   Scenario Sensor  Impact
# 0       S1      A     2.0
# 1       S2      A     3.0
# 2       S3      B     4.0
# 3       S4      C     1.0
# 4       S5      D     2.0
# >>> print(sensor)
#   Sensor   Cost
# 0      A  100.0
# 1      B  200.0
# 2      C  400.0
# 3      D  500.0
# >>> print(scenario)
#   Scenario  Undetected Impact  Probability
# 0       S1               50.0         0.15
# 1       S2              250.0         0.50
# 2       S3              100.0         0.05
# 3       S4               75.0         0.20
# 4       S5              225.0         0.10
#
# >>> impactform = chama.optimize.ImpactFormulation()
# >>> results = impactform.solve(impact=min_det_time, sensor_budget=1000,
# ...                              sensor=sensor, scenario=scenario,
# ...                              use_scenario_probability=True,
# ...                              use_sensor_cost=True)
#
# >>> print(results['Sensors'])
# ['A', 'C', 'D']
# >>> print(results['Objective'])
# 7.2
# >>> print(results['Assessment'])
#   Scenario Sensor  Impact
# 0       S1      A     2.0
# 1       S2      A     3.0
# 2       S4      C     1.0
# 3       S5      D     2.0
# 4       S3   None   100.0
