from wannier import Wannier
from utility import cal_shift_cond_3D
from matplotlib import pyplot as plt
import numpy as np
import datetime
'''
lattice_vec = np.array([
    [2.71175, 2.71175, 2.71175],
    [-2.71175, 2.71175, 2.71175],
    [-2.71175, -2.71175, 2.71175]
]
) * 0.5293
'''
lattice_vec = np.array([
    [3.999800000000001, 0.000000000000000, 0.000000000000000],
    [0.000000000000000, 3.999800000000001, 0.000000000000000],
    [0.000000000000000, 0.000000000000000, 4.018000000000000],
]
)
system = Wannier(lattice_vec,
    {'hr': 'data/hr_BTO.dat', 'rr': 'data/rr_BTO.dat', 'rndegen': 'data/rndegen_BTO.dat'}
    )
system.read_all()
system.set_fermi_energy(2.4398)
print(cal_shift_cond_3D(system, 4, 2, 2, 10, 2, 1000))
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