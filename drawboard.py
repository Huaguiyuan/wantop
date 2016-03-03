from wannier import Wannier
from utility import cal_shift_cond, plot_band
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
    [2.4684162140, 0.0000000000, 0.0000000000],
    [-1.2342081070, 2.1377111484, 0.0000000000],
    [0.0000000000, 0.0000000000, 9.9990577698],
]
)
system = Wannier(lattice_vec,
                 {'hr': 'hr.dat', 'rr': 'rr.dat', 'rndegen': 'rndegen.dat'}
                 )
system.read_all()

# band plot

kpt_list = np.array(
    [
        [0.5, 0.0, 0.0],
        #[0.333333333333333333333, 0.33333333333333333, 0.0],
        [-0.5, 0.0, 0.0]
    ]
)
kpt_flatten, eig = plot_band(system, kpt_list, 1000)
plt.plot(kpt_flatten, eig)
plt.show()

