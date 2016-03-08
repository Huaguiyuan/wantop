from wannier import Wannier
from utility import cal_shift_cond
import numpy as np
import multiprocessing
import yaml


def worker(data):
    base_wannier = data['base_wannier']
    kpt_list = data['kpt_list']
    omega_list = data['omega_list']
    alpha = data['alpha']
    beta = data['beta']
    epsilon = data['epsilon']
    wannier = base_wannier.copy()
    wannier.set_kpt_list(kpt_list)
    sigma_list = []
    for omega in omega_list:
        sigma = cal_shift_cond(wannier, omega, alpha, beta, epsilon)
        sigma_list.append(sigma)
    return np.array(sigma_list)


if __name__ == '__main__':
    with open('wantop.in') as file:
        config = file.read()
    config = yaml.load(config)
    process_num = config['process_num']
    lattice_vec = np.array(config['lattice_vec'])
    k_ndiv = config['k_ndiv']
    fermi_energy = config['fermi_energy']
    omega_list = np.linspace(config['omega_min'], config['omega_max'], config['omega_ndiv'])
    alpha = config['alpha']
    beta = config['beta']
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
    #KPTLISTMOD
    nkpts = kpt_list.shape[0]
    pkpts = nkpts // process_num
    data_list = []
    for i in range(process_num):
        data = {'base_wannier': system, 'omega_list': omega_list, 'alpha': alpha,
                'beta': beta, 'epsilon': config['delta_epsilon']}
        if i == process_num - 1:
            data['kpt_list'] = kpt_list[i * pkpts:, :]
        else:
            data['kpt_list'] = kpt_list[i * pkpts:(i + 1) * pkpts, :]
        data_list.append(data)
    pool = multiprocessing.Pool(processes=process_num)
    result = list(pool.map(worker, data_list))
    for i in range(process_num):
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
