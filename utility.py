import numpy as np
from numpy import linalg as LA


def cal_shift_cond(wannier, omega, alpha=0, beta=0, epsilon=1e-2):
    """
    calculate shift conductance
    :param omega: frequency
    :param epsilon: parameter to control spread of delta function
    :param alpha, beta: 0: x, 1: y, 2: z
    :return: shift conductance
    """
    wannier.calculate('shift_integrand', alpha, beta)
    wannier.calculate('eigenvalue')
    nkpts = wannier.nkpts
    eigenvalue = wannier.kpt_data['eigenvalue']
    E = eigenvalue[:, None, :] - eigenvalue[None, ...]
    delta = 1 / np.pi * (epsilon / (epsilon ** 2 + (E - omega) ** 2))
    # volume of brillouin zone
    volume = abs(np.dot(np.cross(wannier.rlattice_vec[0], wannier.rlattice_vec[1]), wannier.rlattice_vec[2]))
    return np.sum(delta * wannier.kpt_data['shift_integrand'][alpha][beta]) * volume / nkpts


def r_r_from_wann_center(wannier, path):
    num_wann = wannier.num_wann
    wann_center = np.loadtxt(path)
    wannier.r_r = np.zeros((num_wann, num_wann, 3, len(wannier.rpt_list)), dtype='complex')
    for i in range(wannier.num_wann):
        for j in range(wannier.num_wann):
            if i == j:
                wannier.r_r[i, j, :, 0] = wann_center[i, :]
    return wannier


def cal_shift_cond_3D(wannier, omega_list, alpha=0, beta=0, ndiv=100, ndiv_inc=5, inc_thr=1000):
    """
    calculate shift conductance for 3D materials
    :param omega_list: frequency list
    :param alpha, beta: 0: x, 1: y, 2: z
    :param ndiv: k mesh number for every axis
    :param ndiv_inc: mesh increase number for value-rapid-changing area
    :param inc_thr: threshold for mesh increasing
    :return: shift conductance
    """
    kx = np.linspace(0.0, 1.0, ndiv, endpoint=False)
    ky = np.linspace(0.0, 1.0, ndiv, endpoint=False)
    kz = np.linspace(0.0, 1.0, ndiv, endpoint=False)
    kpt_list = np.zeros((ndiv ** 3, 3))


    def unflatten(ind):
        return ind // (ndiv ** 2), (ind % (ndiv ** 2)) // ndiv, ind % ndiv

    cnt = 0
    for i in range(ndiv):
        for j in range(ndiv):
            for k in range(ndiv):
                kpt_list[cnt, 0] = kx[i]
                kpt_list[cnt, 1] = ky[j]
                kpt_list[cnt, 2] = kz[k]
                cnt += 1
    wannier.set_kpt_list(kpt_list)
    nkpts = wannier.nkpts
    wannier.calculate('shift_integrand', alpha, beta)
    wannier.calculate('eigenvalue')
    eigenvalue = wannier.kpt_data['eigenvalue']
    shift_integrand = wannier.kpt_data['shift_integrand'][alpha][beta]
    median = np.median(abs(shift_integrand[abs(shift_integrand) > 0].flatten()))
    inc_index = list(set(np.where(np.abs(shift_integrand) > inc_thr * median)[2]))
    shift_integrand[:, :, inc_index] = 0
    for ind in inc_index:
        ind_unflatten = unflatten(ind)
        # the point where shift integrand values are very large
        kpt = np.array([kx[ind_unflatten[0]], ky[ind_unflatten[1]], kz[ind_unflatten[2]]])
        # construct new kpt_list
        kmin = kpt - 1 / (2 * ndiv) + 1 / (2 * ndiv * ndiv_inc)
        kmax = kpt + 1 / (2 * ndiv) + 1 / (2 * ndiv * ndiv_inc)
        kx_inc = np.linspace(kmin[0], kmax[0], ndiv_inc, endpoint=False)
        ky_inc = np.linspace(kmin[1], kmax[1], ndiv_inc, endpoint=False)
        kz_inc = np.linspace(kmin[2], kmax[2], ndiv_inc, endpoint=False)
        kpt_list_inc = np.zeros((ndiv_inc ** 3, 3))
        cnt = 0
        for i in range(ndiv_inc):
            for j in range(ndiv_inc):
                for k in range(ndiv_inc):
                    kpt_list_inc[cnt, 0] = kx_inc[i]
                    kpt_list_inc[cnt, 1] = ky_inc[j]
                    kpt_list_inc[cnt, 2] = kz_inc[k]
                    cnt += 1
        wannier.set_kpt_list(kpt_list_inc)
        wannier.calculate('eigenvalue')
        wannier.calculate('shift_integrand', alpha, beta)
        # concatenate all the matrices
        shift_integrand = np.concatenate(
            (shift_integrand, wannier.kpt_data['shift_integrand'][alpha][beta] / (ndiv_inc ** 3)), axis=-1)
        kpt_list = np.concatenate((kpt_list, kpt_list_inc), axis=0)
        eigenvalue = np.concatenate((eigenvalue, wannier.kpt_data['eigenvalue']), axis=-1)
    # new number of points
    nkpts_new = kpt_list.shape[0]
    cond_list = []
    for omega in omega_list:
        delta = np.zeros((wannier.num_wann, wannier.num_wann, nkpts_new), dtype='float')
        epsilon = wannier.tech_para['epsilon']
        for i in range(nkpts_new):
            E = eigenvalue[:, i][:, None] - eigenvalue[:, i][None, :]
            delta[:, :, i] = 1 / np.pi * (epsilon / (epsilon ** 2 + (E - omega) ** 2))
        volume = abs(np.dot(np.cross(wannier.rlattice_vec[0], wannier.rlattice_vec[1]), wannier.rlattice_vec[2]))
        cond_list.append(np.sum(delta * shift_integrand) * volume / nkpts)
    return cond_list


def cal_berry_curv(wannier, alpha=0, beta=0):
    """
    calculate accumulated berry curvature
    :param alpha, beta: 0: x, 1: y, 2: z
    :return: berry_curv: berry curvature corresponding to kpt list
    """
    wannier.calculate('omega', alpha, beta)
    wannier.calculate('eigenvalue')
    omega_diag = np.diagonal(wannier.kpt_data['omega'][alpha][beta][:, :, :])
    omega_diag = np.rollaxis(omega_diag, 1)
    fermi = np.zeros((wannier.num_wann, wannier.nkpts), dtype='float')
    fermi[wannier.kpt_data['eigenvalue'] < wannier.fermi_energy] = 1
    return np.real(np.sum(fermi * omega_diag, axis=0))


def plot_band(wannier, kpt_list, ndiv):
    """
    plot band structure of the system
    :param kpt_list: ndarray containing list of kpoints, example: [[0,0,0],[0.5,0.5,0.5]...]
    :param ndiv: number of kpoints in each line
    :return (kpt_flatten, eig): kpt_flatten: flattened kpt distance from the first kpt list
    eig: eigenvalues corresponding to kpt_flatten
    """

    # a list of kpt to be calculated
    def vec_linspace(vec_1, vec_2, num):
        delta = (vec_2 - vec_1) / (num - 1)
        return np.array([vec_1 + delta * i for i in range(num)])

    kpt_plot = np.concatenate(
        tuple([vec_linspace(kpt_list[i, :], kpt_list[i + 1, :], ndiv) for i in range(len(kpt_list) - 1)]))
    wannier.set_kpt_list(kpt_plot)
    # wannier.kpt_list is scaled against reciprocal lattice vector
    kpt_plot = wannier.scale(kpt_plot, 'k')
    wannier.calculate('eigenvalue')
    # calculate k axis
    kpt_flatten = [0.0]
    kpt_distance = 0.0
    for i in range(len(kpt_plot) - 1):
        kpt_distance += LA.norm(kpt_plot[i + 1] - kpt_plot[i])
        kpt_flatten += [kpt_distance]
    return kpt_flatten, wannier.kpt_data['eigenvalue'].T
