from wannier import Wannier
from matplotlib import pyplot as plt
import numpy as np
import datetime

lattice_vec = np.array([
    [2.71175, 2.71175, 2.71175],
    [-2.71175, 2.71175, 2.71175],
    [-2.71175, -2.71175, 2.71175]
]
) * 0.5293

system = Wannier({'hr': 'data/hr_Fe.dat', 'rr': 'data/rr_Fe.dat', 'rndegen': 'data/rndegen_Fe.dat'}, lattice_vec)
system.read_all()
kpt_list = np.array(
    [
        [0.1, 0.2, 0.3],
    ]
)
b1 = np.array([0.5, -0.5, -0.5])
b2 = np.array([0.5, 0.5, 0.5])
kpt = 91 / 200 * b1 + 65 / 200 * b2
system.kpt_list = kpt.reshape((1, 3))
system.fermi_energy = 12.627900
system.calculate('omega', 0, 1)
system.calculate('A_h_ind_ind', 0, 1)
system.calculate('A_h_ind_ind', 1, 0)
print(system.kpt_data['omega'][15, 15, 0, 1, 0])
print((system.kpt_data['A_h_ind_ind'][:, :, 1, 0, 0] - system.kpt_data['A_h_ind_ind'][:, :, 0, 1, 0])[15, 15])
'''
# band plot

kpt_list = np.array(
    [
        [0, 0, 0],
        [0, 0.5, 0],
        [0.5, 0.5, 0],
        [0, 0, 0],
        [0, 0, 0.5],
        [0, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0, 0, 0.5]
    ]
)
kpt_flatten, eig = system.plot_band(kpt_list, 100)
plt.plot(kpt_flatten, eig)
plt.show()
'''