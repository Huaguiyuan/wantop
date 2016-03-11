from wannier import Wannier
from utility import cal_shift_cond, plot_band
from matplotlib import pyplot as plt
import numpy as np
from numpy import linalg as LA
import datetime
'''
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


kpt_list = np.array(
    [
        [0.1, 0.2, 0],
    ]
)
system.set_kpt_list(kpt_list)
system.calculate('H_w')
system.calculate('eigenvalue')
print(system.kpt_data['eigenvalue'][:, 0])
N = 600
print(system.kpt_data['H_w'][2:4, 2:4, 0])
H_00 = system.kpt_data['H_w'][2:4, 2:4, 0]
print(system.kpt_data['H_w'][2:4, 4:6, 0])
print(system.kpt_data['H_w'][4:6, 2:4, 0])
H_01 = system.kpt_data['H_w'][2:4, 4:6, 0]
H = np.zeros((2 * N, 2 * N), dtype='complex')
delta = 1e-2
for i in range(N):
    H[2 * i: 2 * i + 2, 2 * i: 2 * i + 2] = H_00
    if i != N - 1:
        H[2 * i: 2 * i + 2, 2 * i + 2: 2 * i + 4] = H_01
        H[2 * i + 2: 2 * i + 4, 2 * i: 2 * i + 2] = H_01.conj().T
#print(np.sort(LA.eig(H)[0]))
#print(np.sort(LA.eig(1.40510429*np.eye(2*N) - H)[0]))
print(LA.inv((1.40510429+ 1j* delta)*np.eye(2*N) - H)[0:2,0:2])

for omega in np.linspace(0, 4, 1000):
    if np.max(np.abs(np.imag(np.diagonal(LA.inv(omega + 1j*delta -H)[0:2,0:2])))) > 1e-3:
        print(omega)

#    if np.imag(np.sum(np.diagonal(LA.inv(omega + 1j*delta -H)[0:2,0:2]))) > 1e-6:
#        print(omega)
print('done')
#
k_ndiv = 200
fermi_energy = 0
alpha = 0
beta = 0
system = Wannier(lattice_vec, {'hr': 'hr.dat', 'rr': 'rr.dat', 'rndegen': 'rndegen.dat'})
system.read_all()
system.set_fermi_energy(fermi_energy)
x = np.linspace(0.0, 1.0, k_ndiv, endpoint=False)
y = np.linspace(0.0, 1.0, k_ndiv, endpoint=False)
z = np.linspace(0.0, 1.0, k_ndiv, endpoint=False)
kpt_list = np.zeros((k_ndiv ** 3, 3))
cnt = 0
for i in range(k_ndiv):
    for j in range(k_ndiv):
        for k in range(k_ndiv):
            kpt_list[cnt, 0] = x[i]
            kpt_list[cnt, 1] = y[j]
            kpt_list[cnt, 2] = z[k]
            cnt += 1
system.set_kpt_list(kpt_list)
system.calculate('shift_integrand')
np.save('kpt_list', system.kpt_list)
np.save('shift_integrand', system.kpt_data['shift_integrand'][alpha][beta])
lattice_vec = np.array([
    [1.0000000, 0.0000000, 0.0000000],
    [0.0000000, 1.0000000, 0.0000000],
    [0.0000000, 0.0000000, 1.0000000],
]
)
system = Wannier(lattice_vec,
                 {'hr': 'hr.dat', 'rr': 'rr.dat', 'rndegen': 'rndegen.dat'}
                 )
system.read_all()
system.set_fermi_energy(0)
kpt_list = np.array(
    [
        [0.1, 0.2, 0],
    ]
)
system.set_kpt_list(kpt_list)
system.calculate('shift_integrand', 0, 0)
print(system.kpt_data['shift_integrand'][0][0])
'''
lattice_vec = np.array([
    [1.5000000, -0.8660254, 0.0000000],
    [1.5000000, 0.8660254, 0.0000000],
    [0.0000000, 0.0000000, 10.000000],
]
)
system = Wannier(lattice_vec,
                 {'hr': 'hr.dat', 'rr': 'rr.dat', 'rndegen': 'rndegen.dat'}
                 )
system.read_all()
shift_cond = np.load('shift_integrand[1, 1].npy')
x = np.linspace(0.0, 1.0, 100, endpoint=False)
y = np.linspace(0.0, 1.0, 100, endpoint=False)
z = np.linspace(0.0, 1.0, 1, endpoint=False)
kpt_list = np.zeros((100 ** 2, 3))
cnt = 0
for i in range(100):
    for j in range(100):
        for k in range(1):
            kpt_list[cnt, 0] = x[i]
            kpt_list[cnt, 1] = y[j]
            kpt_list[cnt, 2] = z[k]
            cnt += 1
system.set_kpt_list(kpt_list)
kpt_list = system.kpt_list
kx = kpt_list[:, 0].reshape((100, 100))
ky = kpt_list[:, 1].reshape((100, 100))


#KX, KY = np.meshgrid(kx, ky)

shift = shift_cond[0, 1, :]
shift = shift.reshape((100, 100))
ax = plt.contourf(kx, ky, shift)
plt.colorbar(ax)
plt.show()

