import numpy as np
import numexpr as ne
from numpy import linalg as LA


class Wannier():
    def __init__(self, path, lattice_vec):
        """
        :param path: a dict of wannier outputs paths,
        current state: {'hr': 'hr.dat', 'rr': 'rr.dat', 'rndegen': 'rndegen.dat'}
        :param lattice_vec: lattice vector, ndarray, example: [[first vector], [second vector]...]
        """
        # wannier outputs paths
        self.path = path
        # lattice vector
        self.lattice_vec = lattice_vec
        # rpt number
        self.nrpts = None
        # wannier function number
        self.num_wann = None
        # rpt list in unit of lattice_vec, ndarray, example: [[-5,5,5],[5,4,3]...]
        self.rpt_list = None
        # rpt degenerate number list corresponding to rpt list, ndarray, example: [4,1,1,1,2...]
        self.r_ndegen = None
        # kpt number
        self.nkpts = None
        # kpt list in unit of rlattice_vec, ndarray, example: [[-0.5,0.5,0.5],[0.5,0.4,0.3]...]
        self.kpt_list = None
        # a container for program to check whether some quantities have been calculated
        self.kpt_done = {}
        # kpt integrate weight list corresponding to kpt list, ndarray, example: [1,1,0.5,0.5], NOT IMPLEMENTED YET!
        self.k_weight_list = None
        # a dictionary to store data corresponding kpt_list
        self.kpt_data = {}
        # fermi energy
        self.fermi_energy = 0
        # technical parameters
        self.tech_para = {'degen_thresh': 1e-7, 'epsilon': 1e-3}
        # basic naming convention
        # O_r is matrix of <0n|O|Rm>, O_h is matrix of <u^(H)_m||u^(H)_n>, O_w is matrix of <u^(W)_m||u^(W)_n>
        # hamiltonian matrix element in real space, ndarray of dimension (num_wann, num_wann, nrpts)
        self.H_r = None
        # r matrix element in real space, ndarray of dimension (num_wann, num_wann, 3, nrpts)
        self.r_r = None
        # generate reciprocal lattice vector
        [a1, a2, a3] = self.lattice_vec
        b1 = 2 * np.pi * (np.cross(a2, a3) / np.dot(a1, np.cross(a2, a3)))
        b2 = 2 * np.pi * (np.cross(a3, a1) / np.dot(a2, np.cross(a3, a1)))
        b3 = 2 * np.pi * (np.cross(a1, a2) / np.dot(a3, np.cross(a1, a2)))
        self.rlattice_vec = np.array([b1, b2, b3])

    def read_all(self):
        """
        read all possible output files from wannier
        """
        self.read_ndegen()
        self.read_hr()
        self.read_rr()

    def read_ndegen(self):
        """
        read wannier ndegen output file
        """
        with open(self.path['rndegen'], 'r') as file:
            buffer = file.readline().split()
            self.r_ndegen = np.array([int(ndegen) for ndegen in buffer], dtype='float')

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
        self.nrpts = nrpts
        self.rpt_list = rpt_list
        self.H_r = H_r
        self.num_wann = num_wann

    def read_rr(self):
        """
        read wannier rr output file
        """
        with open(self.path['rr'], 'r') as file:
            # skip first two lines
            file.readline()
            file.readline()
            self.r_r = np.zeros((self.num_wann, self.num_wann, 3, self.nrpts), dtype='complex')
            for cnt_rpt in range(self.nrpts):
                for i in range(self.num_wann):
                    for j in range(self.num_wann):
                        buffer = file.readline()
                        buffer = buffer.split()
                        self.r_r[j, i, 0, cnt_rpt] = float(buffer[5]) + 1j * float(buffer[6])
                        self.r_r[j, i, 1, cnt_rpt] = float(buffer[7]) + 1j * float(buffer[8])
                        self.r_r[j, i, 2, cnt_rpt] = float(buffer[9]) + 1j * float(buffer[10])

    def __setattr__(self, key, value):
        """
        1. if kpt_list changes, delete all cached kpt related result and change nkpts
        2. scale kpt_list and rpt_list if set
        """
        if key == 'kpt_list':
            self.kpt_data = {}
            self.kpt_done = {}
            if value is not None:
                super(Wannier, self).__setattr__(key, np.dot(value, self.rlattice_vec))
                self.nkpts = value.shape[0]
            else:
                super(Wannier, self).__setattr__(key, value)
        elif key == 'rpt_list':
            if value is not None:
                super(Wannier, self).__setattr__(key, np.dot(value, self.lattice_vec))
            else:
                super(Wannier, self).__setattr__(key, value)
        else:
            super(Wannier, self).__setattr__(key, value)

    def __cal_H_w(self, alpha=0, beta=0, flag=0):
        """
        calculate H^(W)(k), H^(W)_\alpha(k) or H^(W)_\alpha\beta(k) and store it in 'H_w' or 'H_w_ind' or 'H_w_ind_ind'.
        :param alpha, beta: 0: x, 1: y, 2: z
        :param flag: 0: H^(W)(k), 1: H^(W)_\alpha(k), 2:  H^(W)_\alpha\beta(k)
        """
        phase = 1j * np.dot(self.rpt_list, self.kpt_list.T)
        phase = ne.evaluate("exp(phase)") / self.r_ndegen[:, None]
        if flag == 0:
            self.kpt_data['H_w'] = np.tensordot(self.H_r, phase, axes=1)
        elif flag == 1:
            self.kpt_data['H_w_ind'][:, :, alpha, :] = \
                np.tensordot(self.H_r, 1j * self.rpt_list[:, alpha][:, None] * phase, axes=1)
        elif flag == 2:
            self.kpt_data['H_w_ind_ind'][:, :, alpha, beta, :] = np.tensordot(
                self.H_r, - self.rpt_list[:, alpha][:, None] * self.rpt_list[:, beta][:, None] * phase, axes=1
            )
        else:
            raise Exception('flag should be 0, 1 or 2')

    def __cal_A_w(self, alpha=0, beta=0, flag=1):
        """
        calculate A^(W)_\alpha(k), A^(W)_\alpha\beta(k) and store it in 'A_w_ind' or 'A_w_ind_ind'
        :param alpha, beta: 0: x, 1: y, 2: z
        :param flag: 1: A^(W)_\alpha(k), 2:  A^(W)_\alpha\beta(k)
        """

        phase = 1j * np.dot(self.rpt_list, self.kpt_list.T)
        phase = ne.evaluate("exp(phase)") / self.r_ndegen[:, None]
        r_r_alpha = self.r_r[:, :, alpha, :]
        if flag == 1:
            self.kpt_data['A_w_ind'][:, :, alpha, :] = np.tensordot(r_r_alpha, phase, axes=1)
        elif flag == 2:
            self.kpt_data['A_w_ind_ind'][:, :, alpha, beta, :] = np.tensordot(
                r_r_alpha, 1j * self.rpt_list[:, beta][:, None] * phase, axes=1
            )
        else:
            raise Exception('flag should be 1 or 2')

    def __cal_eig(self):
        """
        calculate sorted (small to large) eigenvalue and eigenstate and store it in 'eigenvalue' and 'U'
        :param kpt: kpt, unscaled
        """
        self.calculate('H_w')
        for i in range(self.nkpts):
            (w, v) = LA.eig(self.kpt_data['H_w'][:, :, i])
            idx = w.argsort()
            w = np.real(w[idx])
            v = v[:, idx]
            self.kpt_data['eigenvalue'][:, i] = np.real(w)
            self.kpt_data['U'][:, :, i] = v

    def __cal_D(self, alpha=0, beta=0, flag=1):
        """
        calculate D matrix and store it in 'D_ind' or 'D_ind_ind'
        If any of the bands are degenerate, zero matrix is returned
        :param alpha, beta: 0: x, 1: y, 2: z
        :param flag: 0: D_\alpha(k), 1: D_\alpha\beta(k)
        """
        self.calculate('eigenvalue')
        self.calculate('H_w_ind', alpha)
        if flag == 2:
            self.calculate('H_w_ind', beta)
            self.calculate('H_w_ind_ind', alpha, beta)
        for i in range(self.nkpts):
            E = self.kpt_data['eigenvalue'][:, i][:, None] - self.kpt_data['eigenvalue'][:, i][None, :]
            E_mod = np.copy(E)
            np.fill_diagonal(E_mod, 1)
            # return zero matrix if any bands are degenerate
            if (np.abs(E_mod) < self.tech_para['degen_thresh']).any():
                return None
            U = self.kpt_data['U'][:, :, i]
            U_deg = U.conj().T
            H_hbar_alpha = U_deg.dot(self.kpt_data['H_w_ind'][:, :, alpha, i]).dot(U)
            H_hbar_alpha_mod = np.copy(H_hbar_alpha)
            np.fill_diagonal(H_hbar_alpha_mod, 0)
            if flag == 1:
                self.kpt_data['D_ind'][:, :, alpha, i] = - H_hbar_alpha_mod / E_mod
            if flag == 2:
                H_hbar_beta = U_deg.dot(self.kpt_data['H_w_ind'][:, :, beta, i]).dot(U)
                H_hbar_beta_mod = np.copy(H_hbar_beta)
                np.fill_diagonal(H_hbar_beta_mod, 0)
                D_beta = -H_hbar_beta_mod / E_mod
                H_hbar_alpha_beta = U_deg.dot(self.kpt_data['H_w_ind_ind'][:, :, alpha, beta, i]).dot(U)
                H_hbar_beta_diag = np.diagonal(H_hbar_beta)
                self.kpt_data['D_ind_ind'][:, :, alpha, beta, i] = 1 / E_mod ** 2 * (
                    (H_hbar_beta_diag[:, None] - H_hbar_beta_diag[None, :]) * H_hbar_alpha -
                    E * (D_beta.conj().T.dot(H_hbar_alpha) + H_hbar_alpha_beta + H_hbar_alpha * D_beta)
                )

    def __cal_A_h(self, alpha=0, beta=0, flag=1):
        """
        calculate A^(H)_\alpha(k) or A^(H)_\alpha\beta(k) and store it in 'A_h_ind' or 'A_h_ind_ind'
        If any of the bands are degenerate, zero matrix is returned
        :param alpha, beta: 0: x, 1: y, 2: z
        :param flag: 0: A^(H)_\alpha(k), 1: A^(H)_\alpha\beta(k)
        """
        self.calculate('eigenvalue')
        self.calculate('A_w_ind', alpha)
        self.calculate('D_ind', alpha)
        if flag == 2:
            self.calculate('A_w_ind_ind', alpha, beta)
            self.calculate('D_ind', beta)
            self.calculate('D_ind_ind', alpha, beta)
        for i in range(self.nkpts):
            # E[i,j] would be eigenvalue[i] - eigenvalue[j]
            E = self.kpt_data['eigenvalue'][:, i][:, None] - self.kpt_data['eigenvalue'][:, i][None, :]
            E_mod = np.copy(E)
            np.fill_diagonal(E_mod, 1)
            U = self.kpt_data['U'][:, :, i]
            U_deg = U.conj().T
            # return zero matrix if any bands are degenerate
            if (np.abs(E_mod) < self.tech_para['degen_thresh']).any():
                return None
            A_hbar_alpha = U_deg.dot(self.kpt_data['A_w_ind'][:, :, alpha, i]).dot(U)
            if flag == 1:
                self.kpt_data['A_h_ind'][:, :, alpha, i] = A_hbar_alpha + 1j * self.kpt_data['D_ind'][:, :, alpha, i]
            elif flag == 2:
                D_beta = self.kpt_data['D_ind'][:, :, beta, i]
                D_alpha_beta = self.kpt_data['D_ind_ind'][:, :, alpha, beta, i]
                A_hbar_alpha_beta = U_deg.dot(self.kpt_data['A_w_ind_ind'][:, :, alpha, beta, i]).dot(U)
                self.kpt_data['A_h_ind_ind'][:, :, alpha, beta, i] = \
                    D_beta.conj().T.dot(A_hbar_alpha) + A_hbar_alpha_beta + A_hbar_alpha * D_beta + 1j * D_alpha_beta
            else:
                raise Exception('flag should be 1 or 2')

    def __cal_omega(self, alpha, beta):
        """
        calculate berry curvature and store it in 'berry_curv'
        'berry_curv' is of dimension (num_wann, num_wann, alpha, beta, nkpts)
        :param alpha, beta: 0: x, 1: y, 2: z
        """
        self.calculate('eigenvalue')
        self.calculate('A_w_ind_ind', alpha, beta)
        self.calculate('A_w_ind_ind', beta, alpha)
        self.calculate('D_ind', alpha)
        self.calculate('D_ind', beta)
        self.calculate('A_w_ind', alpha)
        self.calculate('A_w_ind', beta)
        data = self.kpt_data
        for i in range(self.nkpts):
            U = self.kpt_data['U'][:, :, i]
            U_deg = U.conj().T
            omega_hbar_alpha_beta = U_deg.dot(data['A_w_ind_ind'][:, :, beta, alpha, i] -
                                              data['A_w_ind_ind'][:, :, alpha, beta, i]).dot(U)
            A_hbar_alpha = U_deg.dot(data['A_w_ind'][:, :, alpha, i]).dot(U)
            A_hbar_beta = U_deg.dot(data['A_w_ind'][:, :, beta, i]).dot(U)
            D_alpha = self.kpt_data['D_ind'][:, :, alpha, i]
            D_beta = self.kpt_data['D_ind'][:, :, beta, i]
            data['omega'][:, :, alpha, beta, i] = omega_hbar_alpha_beta - \
                                                  (D_alpha.dot(A_hbar_beta) - A_hbar_beta.dot(D_alpha)) + \
                                                  (D_beta.dot(A_hbar_alpha) - A_hbar_alpha.dot(D_beta)) - \
                                                  1j * (D_alpha.dot(D_beta) - D_beta.dot(D_alpha))

    def __cal_shift_integrand(self, alpha=0, beta=0):
        """
        calculate shift current integrand and store it in 'shift_integrand'
        all parameters in this function are in hamiltonian gauge
        :param alpha, beta: 0: x, 1: y, 2: z
        """
        fermi_energy = self.fermi_energy
        nkpts = self.nkpts
        self.calculate('eigenvalue')
        self.calculate('A_h_ind', alpha)
        self.calculate('A_h_ind', beta)
        self.calculate('A_h_ind_ind', beta, alpha)
        for i in range(nkpts):
            A_alpha = self.kpt_data['A_h_ind'][:, :, alpha, i]
            A_beta = self.kpt_data['A_h_ind'][:, :, beta, i]
            A_beta_alpha = self.kpt_data['A_h_ind_ind'][:, :, beta, alpha, i]
            fermi = np.zeros(self.num_wann, dtype='float')
            fermi[self.kpt_data['eigenvalue'][:, i] > fermi_energy] = 0
            fermi[self.kpt_data['eigenvalue'][:, i] < fermi_energy] = 1
            fermi = fermi[:, None] - fermi[None, :]
            ki = np.diagonal(A_alpha)[:, None] - np.diagonal(A_alpha)[None, :]
            self.kpt_data['shift_integrand'][:, :, alpha, beta, i] = \
                np.real(fermi * np.imag(A_beta.T * (A_beta_alpha - 1j * ki * A_beta)))

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
            'A_w_ind': {'func': self.__cal_A_w, 'kwargs': {'flag': 1}, 'dtype': 'complex'},
            'A_w_ind_ind': {'func': self.__cal_A_w, 'kwargs': {'flag': 2}, 'dtype': 'complex'},
            'D_ind': {'func': self.__cal_D, 'kwargs': {'flag': 1}, 'dtype': 'complex'},
            'D_ind_ind': {'func': self.__cal_D, 'kwargs': {'flag': 2}, 'dtype': 'complex'},
            'A_h_ind': {'func': self.__cal_A_h, 'kwargs': {'flag': 1}, 'dtype': 'complex'},
            'A_h_ind_ind': {'func': self.__cal_A_h, 'kwargs': {'flag': 2}, 'dtype': 'complex'},
            'omega': {'func': self.__cal_omega, 'kwargs': {}, 'dtype': 'complex'},
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
                        cal_dict[matrix_name]['func'](*matrix_ind, **cal_dict[matrix_name]['kwargs'])
                        self.kpt_done[matrix_name][matrix_ind[0]] = True
                elif len(matrix_ind) == 2:
                    if self.kpt_done[matrix_name][matrix_ind[0], matrix_ind[1]]:
                        pass
                    else:
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
                        {matrix_name: np.zeros((num_wann, num_wann, 3, nkpts), dtype=cal_dict[matrix_name]['dtype'])})
                    cal_dict[matrix_name]['func'](*matrix_ind, **cal_dict[matrix_name]['kwargs'])
                    self.kpt_done.update({matrix_name: np.zeros(3, dtype='bool')})
                    self.kpt_done[matrix_name][matrix_ind[0]] = True
                elif len(matrix_ind) == 2:
                    self.kpt_data.update(
                        {matrix_name: np.zeros((num_wann, num_wann, 3, 3, nkpts),
                                               dtype=cal_dict[matrix_name]['dtype'])})
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

    def import_data(self, file, matrix_name, *matrix_ind):
        """
        import previous saved data to a matrix
        :param file name string or file object created using np.save
        :param matrix_name: matrix name
        :param matrix_ind: matrix indices
        :return:
        """
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

    def cal_shift_cond(self, omega, alpha=0, beta=0):
        """
        calculate shift conductance
        :param omega: frequency
        :param epsilon: parameter to control spread of delta function
        :param alpha, beta: 0: x, 1: y, 2: z
        :return: shift conductance
        """
        self.calculate('shift_integrand', alpha, beta)
        self.calculate('eigenvalue')
        epsilon = self.tech_para['epsilon']
        nkpts = self.nkpts
        # delta[i, j] = DiracDelta[omega[i] - omega[j] - omega]
        delta = np.zeros((self.num_wann, self.num_wann, nkpts), dtype='float')
        for i in range(nkpts):
            E = self.kpt_data['eigenvalue'][:, i][:, None] - self.kpt_data['eigenvalue'][:, i][None, :]
            delta[:, :, i] = 1 / np.pi * (epsilon / (epsilon ** 2 + (E - omega) ** 2))
        # volume of brillouin zone
        volume = abs(np.dot(np.cross(self.rlattice_vec[0], self.rlattice_vec[1]), self.rlattice_vec[2]))
        return np.sum(delta * self.kpt_data['shift_integrand'][:, :, alpha, beta, :]) * volume / nkpts

    def cal_berry_curv(self, alpha=0, beta=0):
        """
        calculate accumulated berry curvature
        :param alpha, beta: 0: x, 1: y, 2: z
        :return: berry_curv: berry curvature corresponding to kpt list
        """
        self.calculate('omega', alpha, beta)
        self.calculate('eigenvalue')
        omega_diag = np.diagonal(self.kpt_data['omega'][:, :, alpha, beta, :])
        omega_diag = np.rollaxis(omega_diag, 1)
        fermi = np.zeros((self.num_wann, self.nkpts), dtype='float')
        fermi[self.kpt_data['eigenvalue'] < self.fermi_energy] = 1
        return np.real(np.sum(fermi*omega_diag, axis=0)).astype(np.float32)


    def plot_band(self, kpt_list, ndiv):
        """
        plot band structure of the system
        :param kpt_list: ndarray containing list of kpoints, example: [[0,0,0],[0.5,0.5,0.5]...]
        :param ndiv: number of kpoints in each line
        :return (kpt_flatten, eig): kpt_flatten: flattened kpt distance from the first kpt list
        eig: eigenvalues corresponding to kpt_flatten
        """

        # a list of kpt to be calculated
        def vec_linspace(vec_1, vec_2, num):
            delta = (vec_2 - vec_1) / num
            return np.array([vec_1 + delta * i for i in range(num)])

        kpt_plot = np.concatenate(
            tuple([vec_linspace(kpt_list[i, :], kpt_list[i + 1, :], ndiv) for i in range(len(kpt_list) - 1)]))
        self.kpt_list = kpt_plot
        self.calculate('eigenvalue')
        # calculate k axis
        kpt_flatten = [0.0]
        kpt_distance = 0.0
        for i in range(len(kpt_plot) - 1):
            kpt_distance += LA.norm(kpt_plot[i + 1] - kpt_plot[i])
            kpt_flatten += [kpt_distance]
        return kpt_flatten, self.kpt_data['eigenvalue'].T
