#!/usr/bin/env python
from wannier import Wannier
from utility import cal_shift_cond
import numpy as np
from multiprocessing import Process, Queue, cpu_count
import yaml


def worker(system, kpt_list, config, queue, cnt):
    # set kpt list and result
    system.set_kpt_list(kpt_list)
    result = {'system': system, 'cnt': cnt}
    # calculate shift conductance
    omega_list = np.linspace(config['omega_min'], config['omega_max'], config['omega_ndiv'])
    alpha = config['alpha']
    beta = config['beta']
    gamma = config['gamma']
    cond_list = []
    for omega in omega_list:
        cond = cal_shift_cond(system, omega, alpha, beta, gamma, config['delta_epsilon'])
        cond_list.append(cond)
    # cond_list would be a list of conductance
    result.update({'shift_cond': cond_list})
    # return the result
    queue.put(result)


if __name__ == '__main__':
    # read config and set up base wannier objects
    with open('wantop.in') as file:
        config = file.read()
    config = yaml.load(config)
    process_num = cpu_count()
    lattice_vec = np.array(config['lattice_vec'])
    fermi_energy = config['fermi_energy']
    job_num = config['job_num']
    job_cnt = config['job_cnt']
    system = Wannier(lattice_vec, {'hr': 'hr.dat', 'rr': 'rr.dat', 'rndegen': 'rndegen.dat'})
    system.tech_para.update({'degen_thresh': config['degen_thresh']})
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
    # calculate kpt_list for this job
    all_job_nkpts = kpt_list.shape[0]
    per_job_nkpts = all_job_nkpts // job_num
    if job_cnt == job_num - 1:
        kpt_list = kpt_list[job_cnt * per_job_nkpts:, :]
    else:
        kpt_list = kpt_list[job_cnt * per_job_nkpts:(job_cnt + 1) * per_job_nkpts, :]
    # calculate kpt_list for each process
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
        job = Process(target=worker, args=(system, kpt_list_list[cnt], config, queue, cnt))
        jobs.append(job)
        job.start()
    # get results
    results = []
    for cnt in range(process_num):
        result = queue.get()
        results.append(result)
    # join all processes
    for job in jobs:
        job.join()
    # sort results list
    results = sorted(results, key=lambda result: result['cnt'])
    # combine results
    omega_list = np.linspace(config['omega_min'], config['omega_max'], config['omega_ndiv'])
    shift_cond = [np.array(result['shift_cond']) for result in results]
    # rescale shift_cond for each result
    for cnt in range(process_num):
        shift_cond[cnt] *= results[cnt]['system'].kpt_list.shape[0]/nkpts
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
        if len(matrix_ind) == 0:
            matrix_list = [result['system'].kpt_data[matrix_name] for result in results]
        elif len(matrix_ind) == 1:
            matrix_list = [result['system'].kpt_data[matrix_name][matrix_ind[0]] for result in results]
        elif len(matrix_ind) == 2:
            matrix_list = [result['system'].kpt_data[matrix_name][matrix_ind[0]][matrix_ind[1]] for result in results]
        elif len(matrix_ind) == 2:
            matrix_list = [result['system'].kpt_data[matrix_name][matrix_ind[0]][matrix_ind[1]][matrix_ind[2]]
                           for result in results]
        np.save(matrix_name + str(matrix_ind), np.concatenate(matrix_list, axis=-1))
