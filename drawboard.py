from wannier import Wannier
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import SymLogNorm
import numpy as np
import datetime

lattice_vec = np.array([
    [2.71175, 2.71175, 2.71175],
    [-2.71175, 2.71175, 2.71175],
    [-2.71175, -2.71175, 2.71175]
]
)
system = Wannier({'hr': 'hr.dat', 'rr': 'rr.dat', 'rndegen': 'rndegen.dat'}, lattice_vec)
system.read_all()

kpt_list = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0],
        [0, 0, 0],
        [0, 0, 1],
        [0.5,0, 0.5],
        [0, 0, 0],
        [0.5,0.5,0.5],
        [0.5,0.0,5]
    ]
)

kpt_flatten, eig = system.plot_band(kpt_list, 1000)
plt.plot(kpt_flatten, eig)
plt.show()
print('done')
'''
N = 100
dx = 1/N
dz = 1/N
x = np.linspace(0.00, 1.00, N)
z = np.linspace(0.00, 1.00, N)
kpt_list = np.zeros((N**2, 3))
cnt = 0
for i in range(N):
    for k in range(N):
        kpt_list[cnt, 0] = x[i]
        kpt_list[cnt, 1] = 0
        kpt_list[cnt, 2] = z[k]
        cnt += 1
system.kpt_list = kpt_list
system.fermi_energy = 12.627900
system.calculate('eigenvalue')
x_a = []
z_a = []
for i in range(N**2):
    if (abs(system.kpt_data['eigenvalue'][:, i] - system.fermi_energy) < 0.02).any():
        x_a += [kpt_list[i, 0]]
        z_a += [kpt_list[i, 2]]
plt.scatter(x_a, z_a)
plt.show()
'''
'''
berry_curv = system.cal_berry_curv(0, 1)
x, z = np.meshgrid(x, z)
berry_curv = berry_curv.reshape((N, N))
cmap = plt.get_cmap('coolwarm')
# contours are *point* based plots, so convert our bound into point
# centers
fig = plt.figure()
plot = plt.contourf(x, z, berry_curv, cmap=cmap, norm=SymLogNorm(0.01))
fig.colorbar(plot)
plt.show()
print('done')
'''