from wannier import Wannier
from matplotlib import pyplot as plt
import numpy as np
import datetime

lattice_vec = np.array(
        [[4.0771999, 0.0000000, 0.0000000],
         [0.0214194, 4.0771437, 0.0000000],
         [0.0214194, 0.0213072, 4.0770880]]

)
system = Wannier({'hr': 'wannier90_hr.dat', 'rr': 'wannier90_rr.dat'}, lattice_vec)
system.read_hr()
system.read_rr()
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
integrand_list = system.cal_shift_integrand(kpt_list, fermi_energy=4, alpha=0, beta=2)
file = open('integrand', 'wb')
np.save(file, integrand_list)
file.close()