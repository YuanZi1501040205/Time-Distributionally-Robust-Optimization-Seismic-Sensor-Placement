import numpy as np
import pandas as pd
import chama
from tqdm import tqdm
from sympy import symbols, solve
import matplotlib.pyplot as plt
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
import time
from psmodules import psarray, pssynthetic, psraytrace, pswavelet, \
    psplot, pspicker, pspdf
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


# %% read leak position
with open('./data/raw/fault_poso_creek_location.csv') as leakfile:
    csvreader = csv.reader(leakfile)
    rows = []
    for row in csvreader:
        rows.append(row)
seismic_events_positions = []
for coordinate in rows[1:]:
    seismic_events_positions.append([int(10*float(coordinate[0])), int(float(coordinate[1]))])

# %%normalization
fault_x = 3000 + 4000*(np.array(seismic_events_positions).T[0] - np.array(seismic_events_positions).T[0][0])/(np.array(seismic_events_positions).T[0][-1] - np.array(seismic_events_positions).T[0][0])
fault_y = 2000 + 4000*(np.array(seismic_events_positions).T[1] - np.array(seismic_events_positions).T[1][0])/(np.array(seismic_events_positions).T[1][-1] - np.array(seismic_events_positions).T[1][0])
fault_x = fault_x/1000
fault_y = fault_y/1000
# %% plot overview
x_grid = np.linspace(0, 10, 11)
y_grid = np.linspace(0, 10, 11)
z_grid = np.linspace(0.002, 0.002, 1)
grid = chama.simulation.Grid(x_grid, y_grid, z_grid)
leak_positions_init = seismic_events_positions
x_leak_grid = fault_x
y_leak_grid = fault_y
z_leak_grid = [2.1]

x_list = []
y_list = []
z_list = []

for x in x_grid:
    for y in y_grid:
        for z in z_grid:
            x_list.append(x)
            y_list.append(y)
            z_list.append(z)

x_list_leak = []
y_list_leak = []
z_list_leak = []
for i in range(len(fault_x)):
    x = fault_x[i]
    y = fault_y[i]
    for z in z_leak_grid:
        x_list_leak.append(x)
        y_list_leak.append(y)
        z_list_leak.append(z)

# 绘制散点图
fig = plt.figure(figsize=(5, 5), dpi=120)
# plt.gca().invert_yaxis()
ax = Axes3D(fig)
ax.scatter(x_list, y_list, z_list, alpha = 0.8, c='y', label='Sensor Candidates')
ax.scatter(x_list_leak, y_list_leak, z_list_leak, marker='^',c='b', label='Potential Leak Points')
plt.xlim(-1, 11)
plt.ylim(-1, 11)



# 添加坐标轴(顺序是Z, Y, X)

ax.set_zlim(0, 4)
ax.invert_zaxis()
ax.legend(loc='upper right', fontsize=20)

ax.set_zlabel('\nZ\km', fontdict={'size': 20}, linespacing=1)
ax.set_ylabel('\nY\km', fontdict={'size': 20}, linespacing=1)
ax.set_xlabel('\nX\km', fontdict={'size': 20}, linespacing=1)
ax.dist = 13
plt.savefig('./over view.png')

plt.show()
# %%
# plot event detection time contours
#导入模块
import numpy as np
import matplotlib.pyplot as plt

# Generate square grid array
geox, geoy, geoz = psarray.gridarray(121, 10000, 10000)

# Define source coordinates
src_gt = [3000, 2000, 2100]
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

# %% plot detection time contour
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
CS = plt.tricontour(geox, geoy, event_min_detect_time, 15, linewidths=2.5, colors='k')
CS2 = plt.tricontourf(geox, geoy, event_min_detect_time, 15, cmap='hot')

ax.clabel(CS, fontsize=25)
plt.xlabel('X/m', fontsize=25, weight='bold')
plt.ylabel('Y/m', fontsize=25, weight='bold')
plt.colorbar(orientation='horizontal').set_label(label='Detection time/s',size=25,weight='bold')

fig.axes[0].tick_params(axis="both", labelsize=25)
fig.axes[1].tick_params(axis="x", labelsize=25)
plt.savefig('./event_example.png')
plt.show()
plt.clf()
# %% plot scatter of detection time of surface sensor
import matplotlib.pyplot as plt

rcParams.update({'figure.autolayout': True})
fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

plt.scatter(geox,geoy,c=event_min_detect_time, cmap='hot')


plt.xlabel('X/m', fontsize=25, weight='bold')
plt.ylabel('Y/m', fontsize=25, weight='bold')
plt.colorbar(orientation='horizontal').set_label(label='Detection time/s',size=25,weight='bold')

fig.axes[0].tick_params(axis="both", labelsize=25)
fig.axes[1].tick_params(axis="x", labelsize=25)
plt.savefig('./event_example_sensors.png')
plt.show()
plt.clf()
