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
'''
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
'''
#result = system.__cal_A_w(kpt, 1, 0)
#result = system.__cal_A_h(kpt, v, 1, 1)
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
'''
'''
kpt = np.array([0.1, 0.2, 0.3])
(w, v) = system.__cal_eig(kpt)
result_1 = system.__cal_A_h(kpt, v, 2, 0, 1)
kpt = np.array([0.1, 0.2, 0.3])
(w, v) = system.__cal_eig(kpt)
result_2 = system.__cal_A_h(kpt, v, 2, 0, 1)
result = result_1- result_2
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
'''
system.fermi_energy = 4
time_1 = datetime.datetime.now()
N = 10
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
print(np.real(system.cal_shift_cond(4, alpha=0, beta=2, epsilon=1e-4)))
time_2 = datetime.datetime.now()
print((time_2 - time_1).total_seconds())
'''
N = 100
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
file = open('integrand', 'rb')
system.import_data(file, 'shift_integrand', 0, 2)
file.close()
system.cal_shift_cond(4, 0, 2)
print('done')
