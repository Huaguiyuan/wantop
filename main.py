from wannier import Wannier
from utility import cal_shift_cond
import numpy as np
lattice_vec = np.array([
    [3.999800000000001, 0.000000000000000, 0.000000000000000],
    [0.000000000000000, 3.999800000000001, 0.000000000000000],
    [0.000000000000000, 0.000000000000000, 4.018000000000000],
]
)
system = Wannier(lattice_vec, {'hr': 'hr.dat', 'rr': 'rr.dat', 'rndegen': 'rndegen.dat'})
system.read_all()
N = 200
x = np.linspace(0.0, 1.0, N, endpoint=False)
y = np.linspace(0.0, 1.0, N, endpoint=False)
z = np.linspace(0.0, 1.0, N, endpoint=False)
kpt_list = np.zeros((N**3, 3))
cnt = 0
for i in range(N):
    for j in range(N):
        for k in range(N):
            kpt_list[cnt, 0] = x[i]
            kpt_list[cnt, 1] = y[j]
            kpt_list[cnt, 2] = z[k]
            cnt += 1
kpt_list = kpt_list[0:1000000, :]
system.set_kpt_list(kpt_list)
system.fermi_energy = 2.4115
omega_list = np.linspace(2, 6, 40)
file = open('sigma.dat', 'w')
for omega in omega_list:
    sigma = cal_shift_cond(system, omega, 2, 2)
    file.write(str(omega))
    file.write('    ')
    file.write(str(sigma))
    file.write('\n')
    file.flush()
file.close()