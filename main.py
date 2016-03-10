from wannier import Wannier
from utility import cal_shift_cond
import numpy as np
from multiprocessing import Process, Queue
import yaml


def worker(system, kpt_list, config, queue):
    # set kpt list and result
    system.set_kpt_list(kpt_list)
    result = {}
    # calculate required values
    if config['cal_shift_cond']:
        omega_list = np.linspace(config['omega_min'], config['omega_max'], config['omega_ndiv'])
        alpha = config['alpha']
        beta = config['beta']
        cond_list = []
 #       for omega in omega_list:
 #           cond = cal_shift_cond(system, omega, alpha, beta, config['delta_epsilon'])
 #           cond_list.append(cond)
        # cond_list would be a list of conductance
        result.update({'shift_cond': cond_list})
    # return the result
    queue.put(system)


if __name__ == '__main__':
    # read config and set up base wannier objects
    with open('wantop.in') as file:
        config = file.read()
    config = yaml.load(config)
    process_num = config['process_num']
    lattice_vec = np.array(config['lattice_vec'])
    fermi_energy = config['fermi_energy']
    system = Wannier(lattice_vec, {'hr': 'hr.dat', 'rr': 'rr.dat', 'rndegen': 'rndegen.dat'})
    system.read_all()
    # set up kpt_list
    if 'kpt_list' in config:
        kpt_list = np.array(config['kpt_list'])
    elif 'k_ndiv' in config:
        k_ndiv = config['k_ndiv']
        system.set_fermi_energy(fermi_energy)
        kx = np.linspace(0.0, 1.0, k_ndiv[0], endpoint=False)
        ky = np.linspace(0.0, 1.0, k_ndiv[1], endpoint=False)
        kz = np.linspace(0.0, 1.0, k_ndiv[2], endpoint=False)
        kpt_list = np.zeros((np.prod(k_ndiv), 3))
        cnt = 0
        for i in range(k_ndiv[0]):
            for j in range(k_ndiv[1]):
                for k in range(k_ndiv[2]):
                    kpt_list[cnt, 0] = kx[i]
                    kpt_list[cnt, 1] = ky[j]
                    kpt_list[cnt, 2] = kz[k]
                    cnt += 1
    else:
        raise Exception('kpt_list is not defined')
    #KPTLISTMOD
    nkpts = kpt_list.shape[0]
    pkpts = nkpts // process_num
    # construct all kpt_list
    kpt_list_list = []
    for cnt in range(process_num):
        if cnt == process_num - 1:
            kpt_list_temp = kpt_list[cnt * pkpts:, :]
        else:
            kpt_list_temp = kpt_list[cnt * pkpts:(cnt + 1) * pkpts, :]
        kpt_list_list.append(kpt_list_temp)
    # set up queue for results
    queue = Queue()
    # spawn processes
    jobs = []
    for cnt in range(process_num):
        job = Process(target=worker, args=(system, kpt_list_list[cnt], config, queue,))
        jobs.append(job)
        job.start()
    for job in jobs:
        job.join()
    # get results
    systems = []
    results = []
    for cnt in range(process_num):
        system, result = queue.get()
        systems.append(system)
        results.append(result)
    # combine results
    if config['cal_shift_cond']:
        omega_list = np.linspace(config['omega_min'], config['omega_max'], config['omega_ndiv'])
        shift_cond = [np.array(result['shift_cond']) for result in results]
        # rescale shift_cond for each result
        for cnt in range(process_num):
            shift_cond[cnt] *= systems[cnt].kpt_list.shape[0]/nkpts
        shift_cond = np.sum(np.array(shift_cond), axis=0)
        # save the result
        file = open('shift_cond.dat', 'w')
        for i in range(len(omega_list)):
            file.write(str(omega_list[i]))
            file.write('    ')
            file.write(str(shift_cond[i]))
            file.write('\n')
            file.flush()
        file.close()
    # save other results
    for matrix_name, matrix_ind in config['save_matrix']:
        if len(matrix_ind) == 1:
            for cnt in range(process_num):
                matrix_list = [systems[cnt].kpt_data[matrix_name][matrix_ind[0]]]
        else:
            for cnt in range(process_num):
                matrix_list = [systems[cnt].kpt_data[matrix_name][matrix_ind[0]][matrix_ind[1]]]
        np.save(matrix_name + str(matrix_ind), np.concatenate(matrix_list, axis=-1))