import numpy as np
import pandas as pd
import chama
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
signal = gauss_plume.conc
print(signal.head(5))
# %% visualize convex hull
import chama
chama.graphics.signal_convexhull(signal, scenarios=['S','S'], threshold=1e-100)
# %% x sections
chama.graphics.signal_xsection(signal, 'S', threshold=0.001)
# %% place one stationary sensor
pos1 = chama.sensors.Stationary(location=(70,70,1))
det1 = chama.sensors.Point(threshold=1e-5, sample_times=[0,10])
stationary_pt_sensor = chama.sensors.Sensor(position=pos1, detector=det1)
# %% extract detection time
sensors = dict()
sensors['A'] = stationary_pt_sensor
det_times = chama.impact.extract_detection_times(signal, sensors)
print('det_times: ', det_times)
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
