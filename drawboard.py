from wannier import Wannier
from utility import cal_shift_cond, plot_band
from matplotlib import pyplot as plt
import numpy as np
import datetime

lattice_vec = np.array([
    [1.5000000, -0.8660254, 0.0000000],
    [1.5000000, 0.8660254, 0.0000000],
    [0.0000000, 0.0000000, 10.000000],
]
)

system = Wannier(lattice_vec,
                 {'hr': 'hr.dat', 'rr': 'rr.dat', 'rndegen': 'rndegen.dat'}
                 )
system.read_hr()
system.read_rndegen()
'''
system.set_fermi_energy(0)

kpt_list = np.array(
    [
        [-0.1, 0.3, 0],
    ]
)

kpt_list = np.array(
    [
        [0, 0.5, 0],
    ]
)

system.set_kpt_list(kpt_list)
system.calculate('shift_integrand', 0, 0)
print(system.kpt_list)
print(system.kpt_data['A_h_ind'][0][:, :, 0])
print(system.kpt_data['A_h_ind_ind'][0][0][:, :,0])
print(system.kpt_data['shift_integrand'][0][0][:, :, 0])
'''
# band plot
'''
kpt_list = np.array(
    [
        [1, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)
kpt_flatten, eig = plot_band(system, kpt_list, 1000)
plt.plot(kpt_flatten, eig)
plt.show()
'''
kpt_list = np.array(
    [
        [-0.1, 0.3, 0],
    ]
)
