import numpy as np
import numexpr as ne
from numpy import linalg as LA


class Wannier:
    def __init__(self, lattice_vec, path=None):
        """
        :param lattice_vec: lattice vector, ndarray, example: [[first vector], [second vector]...]
        :param path: a dict of wannier outputs paths,
        current state: {'hr': 'hr.dat', 'rndegen': 'rndegen.dat', 'wann_center': 'wann_center.dat'}
        None means all the needed information will be offered by hand
        """
        self.path = path
        # lattice vector
        self.lattice_vec = lattice_vec
        # wannier function number
        self.num_wann = 0
        # wannnier center
        self.wann_center = np.zeros((0, 3))
        # rpt number
        self.nrpts = 0
        # rpt list in unit of lattice_vec, ndarray, example: [[-5,5,5],[5,4,3]...]
        self.rpt_list = np.zeros((0, 3))
        # unscaled rpt list
        self.unscaled_rpt_list = np.zeros((0, 3))
        # rpt degenerate number list corresponding to rpt list, ndarray, example: [4,1,1,1,2...]
        self.r_ndegen = np.zeros(0)
        # kpt number
        self.nkpts = 0
        # kpt list in unit of rlattice_vec, ndarray, example: [[-0.5,0.5,0.5],[0.5,0.4,0.3]...]
        self.kpt_list = np.zeros((0, 3))
        # unscaled kpt list
        self.unscaled_kpt_list = np.zeros((0, 3))
        # a container for program to check whether some quantities have been calculated
        self.kpt_done = {}
        # a dictionary to store data corresponding kpt_list
        self.kpt_data = {}
        # fermi energy
        self.fermi_energy = 0
        # technical parameters
        self.tech_para = {'degen_thresh': 1e-6}
        # basic naming convention
        # O_r is matrix of <0n|O|Rm>, O_h is matrix of <u^(H)_m||u^(H)_n>, O_w is matrix of <u^(W)_m||u^(W)_n>
        # hamiltonian matrix element in real space, ndarray of dimension (num_wann, num_wann, nrpts)
        self.H_r = np.zeros((self.num_wann, self.num_wann, 0))
        # generate reciprocal lattice vector
        [a1, a2, a3] = self.lattice_vec
        b1 = 2 * np.pi * (np.cross(a2, a3) / np.dot(a1, np.cross(a2, a3)))
        b2 = 2 * np.pi * (np.cross(a3, a1) / np.dot(a2, np.cross(a3, a1)))
        b3 = 2 * np.pi * (np.cross(a1, a2) / np.dot(a3, np.cross(a1, a2)))
        self.rlattice_vec = np.array([b1, b2, b3])

    ##################################################################################################################
    # read files
    ##################################################################################################################
    def read_all(self):
        """
        read all possible output files from wannier
        """
        self.read_rndegen()
        self.read_hr()
        self.read_wann_center()

    def read_rndegen(self):
        """
        read wannier ndegen output file
        """
        with open(self.path['rndegen'], 'r') as file:
            buffer = file.readline().split()
            self.set_r_ndegen(np.array([int(ndegen) for ndegen in buffer], dtype='float'))

    def read_hr(self):
        """
        read wannier hr output file
        """
        with open(self.path['hr'], 'r') as file:
            # read num_wann and nrpts
            num_wann = int(file.readline().split()[0])
            nrpts = int(file.readline().split()[0])
            # read hamiltonian matrix elements
            rpt_list = []
            H_r = np.zeros((num_wann, num_wann, nrpts), dtype='complex')
            for i in range(nrpts):
                for j in range(num_wann):
                    for k in range(num_wann):
                        buffer = file.readline().split()
                        # first index: band k, second index: band j, third index: rpt i
                        H_r[k, j, i] = float(buffer[5]) + 1j * float(buffer[6])
                rpt_list = rpt_list + [buffer[0:3]]
            rpt_list = np.array(rpt_list, dtype='float')
        # save every thing
        self.set_rpt_list(rpt_list)
        self.set_H_r(H_r)
        self.set_num_wann(num_wann)

    def read_wann_center(self):
        """
        read wannier rcenter output file
        """
        with open(self.path['wann_center'], 'r') as file:
            wann_center = np.zeros((self.num_wann, 3))
            for cnt in range(self.num_wann):
                buffer = file.readline()
                buffer = buffer.split()
                wann_center[cnt] = np.array([float(buffer[0]), float(buffer[1]), float(buffer[0])])
            self.set_wann_center(wann_center)

    ##################################################################################################################
    # utilities
    ##################################################################################################################
    def scale(self, v, flag):
        """
        :param v: single point v or v list, v list should be like [[vector 1], [vector 2] ...]
        :param flag: 'k' or 'r'
        :return: scaled v
        """
        # decide scale type
        if flag == 'k':
            scale_vec = self.rlattice_vec
        elif flag == 'r':
            scale_vec = self.lattice_vec
        else:
            raise Exception('flag should be k or r')
        # scale
        return np.dot(v, scale_vec)

    def copy(self):
        """
        return a new wannier object with the same H_r, r_r, rpt_list ... information as self.
        Only kpt related data is not preserved.
        :return: the copy
        """
        new_wannier = Wannier(self.lattice_vec)
        new_wannier.set_rpt_list(self.unscaled_rpt_list)
        new_wannier.set_r_ndegen(self.r_ndegen)
        new_wannier.set_H_r(self.H_r)
        new_wannier.set_fermi_energy(self.fermi_energy)
        new_wannier.set_num_wann(self.num_wann)
        new_wannier.set_wann_center(self.wann_center)
        return new_wannier

    ##################################################################################################################
    #  set input data
    ##################################################################################################################
    def set_rpt_list(self, rpt_list):
        """
        set rpt list
        rpt lists are automatically scaled and nrpts are automatically set
        """
        self.rpt_list = self.scale(rpt_list, 'r')
        self.nrpts = rpt_list.shape[0]
        self.unscaled_rpt_list = rpt_list

    def set_wann_center(self, wann_center):
        """
        set wann_center
        wann_center is automatically scaled
        """
        self.wann_center = self.scale(wann_center, 'r')

    def set_r_ndegen(self, r_ndegen):
        """
        set r_ndegen list
        """
        self.r_ndegen = r_ndegen

    def set_kpt_list(self, kpt_list):
        """
        set kpt list
        kpt lists are automatically scaled and nkpts are automatically set
        """
        self.kpt_list = self.scale(kpt_list, 'k')
        self.nkpts = kpt_list.shape[0]
        self.unscaled_kpt_list = kpt_list
        self.kpt_data = {}
        self.kpt_done = {}

    def set_fermi_energy(self, fermi_energy):
        """
        set fermi energy
        """
        self.fermi_energy = fermi_energy

    def set_H_r(self, H_r):
        """
        set H_r
        """
        self.H_r = H_r

    def set_num_wann(self, num_wann):
        """
        set num_wann
        """
        self.num_wann = num_wann

    ##################################################################################################################
    # private calculation methods
    ##################################################################################################################
    def __cal_H_w(self, alpha=0, beta=0, flag=0):
        """
        calculate H^(W)(k), H^(W)_\alpha(k) or H^(W)_\alpha\beta(k) and store it in 'H_w' or 'H_w_ind' or 'H_w_ind_ind'.
        :param alpha, beta: 0: x, 1: y, 2: z
        :param flag: 0: H^(W)(k), 1: H^(W)_\alpha(k), 2:  H^(W)_\alpha\beta(k)
        """
        phase = 1j * np.tensordot(self.rpt_list[:, None, None, :] + self.wann_center[None, None, :, :] -
                                  self.wann_center[None, :, None, :], self.kpt_list.T, axes=1)
        phase = ne.evaluate("exp(phase)") / self.r_ndegen[:, None, None, None]
        if flag == 0:
            self.kpt_data['H_w'] = np.einsum('ijkli->jkl', self.H_r[None, :, :, None, :] * phase[..., None])
        elif flag == 1:
            self.kpt_data['H_w_ind'][alpha] = \
                np.einsum('ijkli->jkl',
                          1j * (self.H_r *
                                (self.rpt_list[None, None, :, alpha] + self.wann_center[None, :, None, alpha] -
                                 self.wann_center[:, None, None, alpha]))[None, :, :, None, :] * phase[..., None])
        elif flag == 2:
            self.kpt_data['H_w_ind_ind'][alpha][beta] = \
                np.einsum('ijkli->jkl',
                          -(self.H_r *
                                (self.rpt_list[None, None, :, alpha] + self.wann_center[None, :, None, alpha] -
                                 self.wann_center[:, None, None, alpha]) *
                            (self.rpt_list[None, None, :, beta] + self.wann_center[None, :, None, beta] -
                                 self.wann_center[:, None, None, beta]))[None, :, :, None, :] * phase[..., None])
        else:
            raise Exception('flag should be 0, 1 or 2')

    def __cal_eig(self):
        """
        calculate sorted (small to large) eigenvalue and eigenstate and store it in 'eigenvalue' and 'U'
        """
        self.calculate('H_w')
        for i in range(self.nkpts):
            (w, v) = LA.eig(self.kpt_data['H_w'][:, :, i])
            idx = w.argsort()
            w = np.real(w[idx])
            v = v[:, idx]
            self.kpt_data['eigenvalue'][:, i] = np.real(w)
            self.kpt_data['U'][:, :, i] = v

    def __cal_D(self, alpha=0):
        """
        calculate D matrix and store it in 'D_ind' or 'D_ind_ind'
        D is defined as U^\dagger\partial_\alpha U
        If any of the bands are degenerate, zero matrix is returned
        for D_ind_ind, only off_diagonal terms are trusted
        :param alpha: 0: x, 1: y, 2: z
        """
        self.calculate('eigenvalue')
        self.calculate('H_w_ind', alpha)
        for i in range(self.nkpts):
            E = self.kpt_data['eigenvalue'][:, i][:, None] - self.kpt_data['eigenvalue'][:, i][None, :]
            E_mod = np.copy(E)
            np.fill_diagonal(E_mod, 1)
            # return zero matrix if any bands are degenerate
            if (np.abs(E_mod) < self.tech_para['degen_thresh']).any():
                continue
            U = self.kpt_data['U'][:, :, i]
            U_deg = U.conj().T
            H_hbar_alpha = U_deg.dot(self.kpt_data['H_w_ind'][alpha][:, :, i]).dot(U)
            H_hbar_alpha_mod = np.copy(H_hbar_alpha)
            np.fill_diagonal(H_hbar_alpha_mod, 0)
            self.kpt_data['D_ind'][alpha][:, :, i] = - H_hbar_alpha_mod / E_mod

    def __cal_shift_integrand(self, alpha=0, beta=0):
        """
        calculate shift current integrand and store it in 'shift_integrand'
        all parameters in this function are in hamiltonian gauge
        :param alpha, beta: 0: x, 1: y, 2: z
        """
        fermi_energy = self.fermi_energy
        nkpts = self.nkpts
        self.calculate('eigenvalue')
        self.calculate('H_w_ind', beta)
        self.calculate('H_w_ind_ind', beta, alpha)
        self.calculate('D_ind', alpha)
        for i in range(nkpts):
            U = self.kpt_data['U'][:, :, i]
            U_deg = U.conj().T
            D_alpha = self.kpt_data['D_ind'][alpha][:, :, i]
            v_h_beta = self.kpt_data['H_w_ind'][beta][:, :, i]
            v_h_beta_alpha = D_alpha.conj().T.dot(v_h_beta) + \
                             U_deg.dot(self.kpt_data['H_w_ind_ind'][beta][alpha][:, :, i]) + v_h_beta.dot(D_alpha)
            fermi = np.zeros(self.num_wann, dtype='float')
            fermi[self.kpt_data['eigenvalue'][:, i] > fermi_energy] = 0
            fermi[self.kpt_data['eigenvalue'][:, i] < fermi_energy] = 1
            fermi = fermi[:, None] - fermi[None, :]
            ki = (np.diagonal(D_alpha)[:, None] - np.diagonal(D_alpha)[None, :]) * 1j
            self.kpt_data['shift_integrand'][alpha][beta][:, :, i] = \
                np.real(fermi * v_h_beta * v_h_beta.T * (np.imag(v_h_beta_alpha / v_h_beta) - ki))

    ##################################################################################################################
    # public calculation method
    ##################################################################################################################
    def calculate(self, matrix_name, *matrix_ind):
        """
        a wrapper to prevent re-evaluating any matrices.
        :param matrix_name: the needed matrix name
        :param matrix_ind: the needed matrix indices
        """
        num_wann = self.num_wann
        nkpts = self.nkpts

        cal_dict = {
            'H_w': {'func': self.__cal_H_w, 'kwargs': {}, 'dtype': 'complex'},
            'H_w_ind': {'func': self.__cal_H_w, 'kwargs': {'flag': 1}, 'dtype': 'complex'},
            'H_w_ind_ind': {'func': self.__cal_H_w, 'kwargs': {'flag': 2}, 'dtype': 'complex'},
            'shift_integrand': {'func': self.__cal_shift_integrand, 'kwargs': {}, 'dtype': 'float'},
        }
        if matrix_name in cal_dict:
            if matrix_name in self.kpt_done:
                if len(matrix_ind) == 0:
                    pass
                elif len(matrix_ind) == 1:
                    if self.kpt_done[matrix_name][matrix_ind[0]]:
                        pass
                    else:
                        self.kpt_data[matrix_name][matrix_ind[0]] = \
                            np.zeros((num_wann, num_wann, nkpts), dtype=cal_dict[matrix_name]['dtype'])
                        cal_dict[matrix_name]['func'](*matrix_ind, **cal_dict[matrix_name]['kwargs'])
                        self.kpt_done[matrix_name][matrix_ind[0]] = True
                elif len(matrix_ind) == 2:
                    if self.kpt_done[matrix_name][matrix_ind[0], matrix_ind[1]]:
                        pass
                    else:
                        self.kpt_data[matrix_name][matrix_ind[0]][matrix_ind[1]] = \
                            np.zeros((num_wann, num_wann, nkpts), dtype=cal_dict[matrix_name]['dtype'])
                        cal_dict[matrix_name]['func'](*matrix_ind, **cal_dict[matrix_name]['kwargs'])
                        self.kpt_done[matrix_name][matrix_ind[0], matrix_ind[1]] = True
            else:
                if len(matrix_ind) == 0:
                    self.kpt_data.update(
                        {matrix_name: np.zeros((num_wann, num_wann, nkpts), dtype=cal_dict[matrix_name]['dtype'])})
                    cal_dict[matrix_name]['func'](**cal_dict[matrix_name]['kwargs'])
                    self.kpt_done.update({matrix_name: True})
                elif len(matrix_ind) == 1:
                    self.kpt_data.update(
                        {matrix_name: [0, 0, 0]})
                    self.kpt_data[matrix_name][matrix_ind[0]] = \
                        np.zeros((num_wann, num_wann, nkpts), dtype=cal_dict[matrix_name]['dtype'])
                    cal_dict[matrix_name]['func'](*matrix_ind, **cal_dict[matrix_name]['kwargs'])
                    self.kpt_done.update({matrix_name: np.zeros(3, dtype='bool')})
                    self.kpt_done[matrix_name][matrix_ind[0]] = True
                elif len(matrix_ind) == 2:
                    self.kpt_data.update(
                        {matrix_name: [[0, 0, 0]] * 3})
                    self.kpt_data[matrix_name][matrix_ind[0]][matrix_ind[1]] = \
                        np.zeros((num_wann, num_wann, nkpts), dtype=cal_dict[matrix_name]['dtype'])
                    cal_dict[matrix_name]['func'](*matrix_ind, **cal_dict[matrix_name]['kwargs'])
                    self.kpt_done.update({matrix_name: np.zeros((3, 3), dtype='bool')})
                    self.kpt_done[matrix_name][matrix_ind[0], matrix_ind[1]] = True
        else:
            if matrix_name == 'eigenvalue' or matrix_name == 'U':
                if 'eigenvalue' in self.kpt_done:
                    pass
                else:
                    self.kpt_data.update({'eigenvalue': np.zeros((num_wann, nkpts), dtype='float')})
                    self.kpt_data.update({'U': np.zeros((num_wann, num_wann, nkpts), dtype='complex')})
                    self.__cal_eig()
                    self.kpt_done.update({'eigenvalue': True})
            else:
                raise Exception('No such matrix')

    ##################################################################################################################
    # data import method
    ##################################################################################################################
    def import_data(self, file, matrix_name, *matrix_ind):
        """
        import previous saved data to a matrix
        :param file name string or file object created using np.save
        :param matrix_name: matrix name
        :param matrix_ind: matrix indices
        :return:
        """
        '''
        num_wann = self.num_wann
        nkpts = self.nkpts
        if matrix_name == 'shift_integrand':
            data = np.load(file)
            if data.shape[-1] != nkpts:
                raise Exception('The data is not compatible with current kpt_list')
            if 'shift_integrand' in self.kpt_done:
                self.kpt_data['shift_integrand'][:, :, matrix_ind[0], matrix_ind[1], :] = data
                self.kpt_done['shift_integrand'][matrix_ind[0], matrix_ind[1]] = True
            else:
                self.kpt_data.update({'shift_integrand': np.zeros((num_wann, num_wann, 3, 3, nkpts), dtype='float')})
                self.kpt_done.update({'shift_integrand': np.zeros((3, 3), dtype='bool')})
                self.kpt_data['shift_integrand'][:, :, matrix_ind[0], matrix_ind[1], :] = data
                self.kpt_done['shift_integrand'][matrix_ind[0], matrix_ind[1]] = True
        elif matrix_name == 'H_w':
            data = np.load(file)
            if data.shape[-1] != nkpts:
                raise Exception('The data is not compatible with current kpt_list')
            if 'H_w' in self.kpt_done:
                self.kpt_data['H_w'] = data
            else:
                self.kpt_data.update({'H_w': data})
                self.kpt_done.update({'H_w': True})
        else:
            raise Exception('matrix not supported')
        '''
