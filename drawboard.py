from wannier import Wannier
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm
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
        [0.1, 0.2, 0.3]
    ]
)
system.kpt_list = kpt_list
system.fermi_energy = 12.627900
system.calculate('shift_integrand', 2, 2)
print(system.kpt_data['shift_integrand'][9, 6, 2, 2, 0])
kpt_list = np.array(
    [
        [-0.1, -0.2, -0.3]
    ]
)
system.kpt_list = kpt_list
system.fermi_energy = 12.627900
system.calculate('shift_integrand', 2, 2)
print(system.kpt_data['shift_integrand'][9, 6, 2, 2, 0])
# band plot
'''
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
        [0.5,0,0.5]
    ]
)

kpt_flatten, eig = system.plot_band(kpt_list, 100)
plt.plot(kpt_flatten, eig)
plt.show()
print('done')
'''
