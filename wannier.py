import numpy as np
import numexpr as ne
from numpy import linalg as LA


class Wannier():
    def __init__(self, lattice_vec, path=None):
        """
        :param path: a dict of wannier outputs paths,
        current state: {'hr': 'hr.dat', 'rr': 'rr.dat', 'rndegen': 'rndegen.dat'}
        :param lattice_vec: lattice vector, ndarray, example: [[first vector], [second vector]...]
        """
        self.path = path
        # lattice vector
        self.lattice_vec = lattice_vec
        # wannier function number
        self.num_wann = None
        # rpt number
        self.nrpts = None
        # rpt list in unit of lattice_vec, ndarray, example: [[-5,5,5],[5,4,3]...]
        self.rpt_list = None
        # unscaled rpt list
        self.unscaled_rpt_list = None
        # rpt degenerate number list corresponding to rpt list, ndarray, example: [4,1,1,1,2...]
        self.r_ndegen = None
        # kpt number
        self.nkpts = None
        # kpt list in unit of rlattice_vec, ndarray, example: [[-0.5,0.5,0.5],[0.5,0.4,0.3]...]
        self.kpt_list = None
        # unscaled kpt list
        self.unscaled_kpt_list = None
        # a container for program to check whether some quantities have been calculated
        self.kpt_done = {}
        # a dictionary to store data corresponding kpt_list
        self.kpt_data = {}
        # fermi energy
        self.fermi_energy = 0
        # technical parameters
        self.tech_para = {'degen_thresh': 1e-6, 'epsilon': 1e-3}
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

    ##################################################################################################################
    # read files
    ##################################################################################################################
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

    def read_rr(self):
        """
        read wannier rr output file
        """
        with open(self.path['rr'], 'r') as file:
            # skip first two lines
            file.readline()
            file.readline()
            r_r = np.zeros((self.num_wann, self.num_wann, 3, self.nrpts), dtype='complex')
            for cnt_rpt in range(self.nrpts):
                for i in range(self.num_wann):
                    for j in range(self.num_wann):
                        buffer = file.readline()
                        buffer = buffer.split()
                        r_r[j, i, 0, cnt_rpt] = float(buffer[5]) + 1j * float(buffer[6])
                        r_r[j, i, 1, cnt_rpt] = float(buffer[7]) + 1j * float(buffer[8])
                        r_r[j, i, 2, cnt_rpt] = float(buffer[9]) + 1j * float(buffer[10])
            self.set_r_r(r_r)

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

    ##################################################################################################################
    #  set input data
    ##################################################################################################################
    def set_rpt_list(self, rpt_list):
        """
        set rpt list
        :param rpt_list: rpt list in unit of lattice_vec, ndarray, example: [[-5,5,5],[5,4,3]...]
        :param r_ndegen: rpt degenerate number list corresponding to rpt list, ndarray, example: [4,1,1,1,2...].
        If r_ndegen is None, all degenerate number is set to 1
        """
        self.rpt_list = self.scale(rpt_list, 'r')
        self.unscaled_rpt_list = rpt_list
        self.nrpts = rpt_list.shape[0]

    def set_r_ndegen(self, r_ndegen):
        self.r_ndegen = r_ndegen

    def set_kpt_list(self, kpt_list):
        self.kpt_list = self.scale(kpt_list, 'k')
        self.unscaled_kpt_list = kpt_list
        self.nkpts = kpt_list.shape[0]
        self.kpt_data = {}
        self.kpt_done = {}

    def set_fermi_energy(self, fermi_energy):
        self.fermi_energy = fermi_energy

    def set_H_r(self, H_r):
        self.H_r = H_r

    def set_r_r(self, r_r):
        self.r_r = r_r

    def set_num_wann(self, num_wann):
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
        phase = 1j * np.dot(self.rpt_list, self.kpt_list.T)
        phase = ne.evaluate("exp(phase)") / self.r_ndegen[:, None]
        if flag == 0:
            self.kpt_data['H_w'] = np.tensordot(self.H_r, phase, axes=1)
        elif flag == 1:
            self.kpt_data['H_w_ind'][alpha] = \
                np.tensordot(self.H_r, 1j * self.rpt_list[:, alpha][:, None] * phase, axes=1)
        elif flag == 2:
            self.kpt_data['H_w_ind_ind'][alpha][beta] = np.tensordot(
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
            self.kpt_data['A_w_ind'][alpha] = np.tensordot(r_r_alpha, phase, axes=1)
        elif flag == 2:
            self.kpt_data['A_w_ind_ind'][alpha][beta] = np.tensordot(
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
        for D_ind_ind, only off_diagonal terms are trusted
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
                continue
            U = self.kpt_data['U'][:, :, i]
            U_deg = U.conj().T
            H_hbar_alpha = U_deg.dot(self.kpt_data['H_w_ind'][alpha][:, :, i]).dot(U)
            H_hbar_alpha_mod = np.copy(H_hbar_alpha)
            np.fill_diagonal(H_hbar_alpha_mod, 0)
            if flag == 1:
                self.kpt_data['D_ind'][alpha][:, :, i] = - H_hbar_alpha_mod / E_mod
            if flag == 2:
                H_hbar_beta = U_deg.dot(self.kpt_data['H_w_ind'][beta][:, :, i]).dot(U)
                H_hbar_beta_mod = np.copy(H_hbar_beta)
                np.fill_diagonal(H_hbar_beta_mod, 0)
                D_beta = -H_hbar_beta_mod / E_mod
                H_hbar_alpha_beta = U_deg.dot(self.kpt_data['H_w_ind_ind'][alpha][beta][:, :, i]).dot(U)
                H_hbar_beta_diag = np.diagonal(H_hbar_beta)
                self.kpt_data['D_ind_ind'][alpha][beta][:, :, i] = (1 / E_mod ** 2) * (
                    (H_hbar_beta_diag[:, None] - H_hbar_beta_diag[None, :]) * H_hbar_alpha -
                    E * (D_beta.conj().T.dot(H_hbar_alpha) + H_hbar_alpha_beta + H_hbar_alpha.dot(D_beta))
                )

    def __cal_A_h(self, alpha=0, beta=0, flag=1):
        """
        calculate A^(H)_\alpha(k) or A^(H)_\alpha\beta(k) and store it in 'A_h_ind' or 'A_h_ind_ind'
        If any of the bands are degenerate, zero matrix is returned
        for A_ind_ind, only off_diagonal terms are trusted
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
                continue
            A_hbar_alpha = U_deg.dot(self.kpt_data['A_w_ind'][alpha][:, :, i]).dot(U)
            if flag == 1:
                self.kpt_data['A_h_ind'][alpha][:, :, i] = A_hbar_alpha + 1j * self.kpt_data['D_ind'][alpha][:, :, i]
            elif flag == 2:
                D_beta = self.kpt_data['D_ind'][beta][:, :, i]
                D_alpha_beta = self.kpt_data['D_ind_ind'][alpha][beta][:, :, i]
                A_hbar_alpha_beta = U_deg.dot(self.kpt_data['A_w_ind_ind'][alpha][beta][:, :, i]).dot(U)
                self.kpt_data['A_h_ind_ind'][alpha][beta][:, :, i] = \
                   D_beta.conj().T.dot(A_hbar_alpha) + A_hbar_alpha_beta + A_hbar_alpha.dot(D_beta) + 1j * D_alpha_beta
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
            omega_hbar_alpha_beta = U_deg.dot(data['A_w_ind_ind'][beta][alpha][:, :, i] -
                                              data['A_w_ind_ind'][alpha][beta][:, :, i]).dot(U)
            A_hbar_alpha = U_deg.dot(data['A_w_ind'][alpha][:, :, i]).dot(U)
            A_hbar_beta = U_deg.dot(data['A_w_ind'][beta][:, :, i]).dot(U)
            D_alpha = self.kpt_data['D_ind'][alpha][:, :, i]
            D_beta = self.kpt_data['D_ind'][beta][:, :, i]
            data['omega'][alpha][beta][:, :, i] = omega_hbar_alpha_beta - \
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
            A_alpha = self.kpt_data['A_h_ind'][alpha][:, :, i]
            A_beta = self.kpt_data['A_h_ind'][beta][:, :, i]
            A_beta_alpha = self.kpt_data['A_h_ind_ind'][beta][alpha][:, :, i]
            fermi = np.zeros(self.num_wann, dtype='float')
            fermi[self.kpt_data['eigenvalue'][:, i] > fermi_energy] = 0
            fermi[self.kpt_data['eigenvalue'][:, i] < fermi_energy] = 1
            fermi = fermi[:, None] - fermi[None, :]
            ki = np.diagonal(A_alpha)[:, None] - np.diagonal(A_alpha)[None, :]
            self.kpt_data['shift_integrand'][alpha][beta][:, :, i] = \
                np.real(fermi * np.imag(A_beta.T * (A_beta_alpha - 1j * ki * A_beta)))

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
