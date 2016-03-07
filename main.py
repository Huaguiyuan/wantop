from wannier import Wannier
from utility import cal_shift_cond
import numpy as np
import multiprocessing
import sys


def worker(data):
    base_wannier = data['base_wannier']
    kpt_list = data['kpt_list']
    omega_list = data['omega_list']
    alpha = data['alpha']
    beta = data['beta']
    wannier = base_wannier.copy()
    wannier.set_kpt_list(kpt_list)
    sigma_list = []
    for omega in omega_list:
        sigma = cal_shift_cond(wannier, omega, alpha, beta)
        sigma_list.append(sigma)
    return np.array(sigma_list)


if __name__ == '__main__':
    N_CORE = int(sys.argv[1])
    lattice_vec = np.array([
        [3.999800000000001, 0.000000000000000, 0.000000000000000],
        [0.000000000000000, 3.999800000000001, 0.000000000000000],
        [0.000000000000000, 0.000000000000000, 4.018000000000000],
    ]
    )
    N = 200
    fermi_energy = 2.4115
    omega_list = np.linspace(2, 6, 40)
    alpha = 2
    beta = 2
    system = Wannier(lattice_vec, {'hr': 'hr.dat', 'rr': 'rr.dat', 'rndegen': 'rndegen.dat'})
    system.read_all()
    system.fermi_energy = fermi_energy
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    y = np.linspace(0.0, 1.0, N, endpoint=False)
    z = np.linspace(0.0, 1.0, N, endpoint=False)
    kpt_list = np.zeros((N ** 3, 3))
    cnt = 0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                kpt_list[cnt, 0] = x[i]
                kpt_list[cnt, 1] = y[j]
                kpt_list[cnt, 2] = z[k]
                cnt += 1
    #KPTLISTMOD
    nkpts = kpt_list.shape[0]
    pkpts = nkpts // N_CORE
    data_list = []
    for i in range(N_CORE):
        data = {'base_wannier': system, 'omega_list': omega_list, 'alpha': alpha, 'beta': beta}
        if i == N_CORE - 1:
            data['kpt_list'] = kpt_list[i * pkpts:, :]
        else:
            data['kpt_list'] = kpt_list[i * pkpts:(i + 1) * pkpts, :]
        data_list.append(data)
    pool = multiprocessing.Pool(processes=N_CORE)
    result = list(pool.map(worker, data_list))
    for i in range(N_CORE):
        result[i] *= data_list[i]['kpt_list'].shape[0]/nkpts
    sigma_list = sum(result)
    file = open('sigma.dat', 'w')
    for i in range(len(omega_list)):
        file.write(str(omega_list[i]))
        file.write('    ')
        file.write(str(sigma_list[i]))
        file.write('\n')
        file.flush()
    file.close()
