from wannier import Wannier
from matplotlib import pyplot as plt
import numpy as np
import datetime

lattice_vec = np.array([
    [3.999800000000001, 0.000000000000000, 0.000000000000000],
    [0.000000000000000, 3.999800000000001, 0.000000000000000],
    [0.000000000000000, 0.000000000000000, 4.018000000000000],
]
)
system = Wannier({'hr': 'hr.dat', 'rr': 'rr.dat', 'weight': 'weight.dat'}, lattice_vec)
system.read_all()

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
print('done')
