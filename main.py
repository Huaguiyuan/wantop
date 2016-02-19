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
'''
kpt_list = np.array(
    [
        [0.5, 0.5, 0],
        [0, 0, 0],
        [0.5, 0.5, 0.5]
    ]
)
system.plot_band(kpt_list, 100)
'''

'''
#result = system.cal_A_w(kpt, 1, 0)
#result = system.cal_A_h(kpt, v, 1, 1)
# read u
u_file = open('U.dat')
u_list = np.zeros((system.num_wann, system.num_wann, 0), dtype='complex')
while True:
    u_buffer = u_file.readline()
    if u_buffer:
        u_temp = np.zeros((system.num_wann, system.num_wann, 1), dtype='complex')
        for i in range(system.num_wann ** 2):
            u_buffer = u_file.readline()
            u_buffer = [float(item) for item in u_buffer.split()]
            u_temp[i // system.num_wann, i % system.num_wann, 0] = u_buffer[0] + 1j * u_buffer[1]
        u_list = np.concatenate((u_list, u_temp), axis=2)
    else:
        break
u = u_list[:, :, 0]

kpt = np.array([0.1, 0.2, 0.3])
(w, v) = system.cal_eig(kpt)
result = system.cal_A_h(kpt, v, 2, 1)
'''
'''
N = 100
omega= np.linspace(3, 9, N)
shift = np.zeros(N)
for i in range(len(omega)):
    print(i)
    time_1 = datetime.datetime.now()
    shift[i] = np.real(system.cal_shift_cond(omega[i], 0, 0, 2, 4, 10))
    time_2 = datetime.datetime.now()
    print((time_2 - time_1).total_seconds())
plt.plot(omega, shift)
plt.show()
'''
print(np.real(system.cal_shift_cond(4, 0, 0, 2, 4, 10)))
print('done')
