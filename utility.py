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
