from wannier import Wannier
import numpy as np
import datetime
lattice_vec = np.array([
    [3.999800000000001, 0.000000000000000, 0.000000000000000],
    [0.000000000000000, 3.999800000000001, 0.000000000000000],
    [0.000000000000000, 0.000000000000000, 4.018000000000000],
]
)
system = Wannier({'hr': 'hr.dat', 'rr': 'rr.dat', 'rndegen': 'rndegen.dat'}, lattice_vec)
system.read_all()
N = 20
x = np.linspace(0.01, 1.01, N)
y = np.linspace(0.01, 1.01, N)
z = np.linspace(0.01, 1.01, N)
kpt_list = np.zeros((N**3, 3))
cnt = 0
for i in range(N):
    for j in range(N):
        for k in range(N):
            kpt_list[cnt, 0] = x[i]
            kpt_list[cnt, 1] = y[j]
            kpt_list[cnt, 2] = z[k]
            cnt += 1
system.kpt_list = kpt_list
system.fermi_energy = 3
omega_list = np.linspace(1, 9, 1000)
file = open('sigma.dat', 'w')
for omega in omega_list:
    sigma = system.cal_shift_cond(omega, 2, 2)
    file.write(str(omega))
    file.write('    ')
    file.write(str(sigma))
    file.write('\n')
    file.flush()
file.close()
file = open('integrand.dat', 'w')
np.save(file, system.kpt_data['shift_integrand'][:, :, 2, 2, :])
file.close()
print('done')