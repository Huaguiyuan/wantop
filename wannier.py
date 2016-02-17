import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt


class Wannier():
    def __init__(self, path, lattice_vec):
        """
        :param path: a dict of wannier outputs paths, currently: {'hr': 'hr.dat', 'rr': 'rr.dat'}
        :param lattice_vec: lattice vector, ndarray, example: [[fir_rt vector], [second vector]...]
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
        # weight list corresponding to rpt list, ndarray, example: [4,1,1,1,2...]
        self.weight_list = None
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

    def read_hr(self):
        """
        read wannier hr output file
        """
        with open(self.path['hr'], 'r') as file:
            # skip the fir_rt line
            file.readline()
            # read num_wann and nrpts
            num_wann = int(file.readline().split()[0])
            nrpts = int(file.readline().split()[0])
            # read weight_list
            weight_list = []
            for i in range(int(np.ceil(nrpts / 15.0))):
                buffer = file.readline().split()
                weight_list = weight_list + buffer
            weight_list = np.array(weight_list, dtype='float')
            # read hamiltonian matrix elements
            rpt_list = []
            H_r = np.zeros((num_wann, num_wann, nrpts), dtype='complex')
            for i in range(nrpts):
                for j in range(num_wann):
                    for k in range(num_wann):
                        buffer = file.readline().split()
                        # fir_rt index: band k, second index: band j, third index: rpt i
                        H_r[k, j, i] = float(buffer[5]) + 1j * float(buffer[6])
                rpt_list = rpt_list + [buffer[0:3]]
            rpt_list = np.array(rpt_list, dtype='float')
        # save every thing
        self.nrpts = nrpts
        self.rpt_list = rpt_list
        self.weight_list = weight_list
        self.H_r = H_r
        self.num_wann = num_wann

    def read_rr(self):
        """
        read wannier rr output file
        """
        with open(self.path['rr'], 'r') as file:
            # skip fir_rt two lines
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
        if len(v.shape) == 1:
            return np.dot(v, scale_vec)
        elif len(v.shape) == 2:
            return np.array([np.dot(kpt, scale_vec) for kpt in v])
        else:
            raise Exception('v should be an array of dimension 1 or 2')

    def cal_H_w(self, kpt, flag, alpha=0, beta=0):
        """
        calculate H^(W)(k), H^(W)_\alpha(k) or H^(W)_\alpha\beta(k)
        :param kpt: kpt, unscaled
        :param flag: 0: H^(W)(k), 1: H^(W)_\alpha(k), 2:  H^(W)_\alpha\beta(k)
        :param alpha, beta: 0: x, 1: y, 2: z
        :return: H^(W)(k), H^(W)_\alpha(k) or H^(W)_\alpha\beta(k) according to flag
        """
        # scale kpt and rpt
        kpt = self.scale(kpt, 'k')
        rpt_list = self.scale(self.rpt_list, 'r')
        # initialize
        H_w = np.zeros((self.num_wann, self.num_wann), dtype='complex')
        H_w_alpha = np.zeros((self.num_wann, self.num_wann), dtype='complex')
        H_w_alpha_beta = np.zeros((self.num_wann, self.num_wann), dtype='complex')
        # fourier transform
        for i in range(self.nrpts):
            rpt = rpt_list[i, :]
            if flag == 0:
                H_w += self.H_r[:, :, i] * np.exp(1j * np.dot(kpt, rpt)) / self.weight_list[i]
            elif flag == 1:
                H_w_alpha += 1j * rpt[alpha] * self.H_r[:, :, i] * np.exp(1j * np.dot(kpt, rpt)) / self.weight_list[i]
            elif flag == 2:
                H_w_alpha_beta += - rpt[alpha] * rpt[beta] * self.H_r[:, :, i] * \
                                np.exp(1j * np.dot(kpt, rpt)) / self.weight_list[i]
            else:
                raise Exception('flag should be 0, 1 or 2')
        if flag == 0:
            return H_w
        elif flag == 1:
            return H_w_alpha
        elif flag ==2:
            return H_w_alpha_beta
        else:
            raise Exception('flag should be 0, 1 or 2')

    def cal_A_w(self, kpt, flag, alpha=0, beta=0):
        """
        calculate A^(W)_\alpha(k), A^(W)_\alpha\beta(k)
        :param kpt: kpt, unscaled
        :param flag: 1: A^(W)_\alpha(k), 2:  A^(W)_\alpha\beta(k)
        :param alpha, beta: 0: x, 1: y, 2: z
        :return: A^(W)_\alpha(k) or A^(W)_\alpha\beta(k) according to flag
        """
        # scale kpt and rpt
        kpt = self.scale(kpt, 'k')
        rpt_list = self.scale(self.rpt_list, 'r')
        # initialize
        A_w_alpha = np.zeros((self.num_wann, self.num_wann), dtype='complex')
        A_w_alpha_beta = np.zeros((self.num_wann, self.num_wann), dtype='complex')
        # fourier transform
        for i in range(self.nrpts):
            rpt = rpt_list[i, :]
            if flag == 1:
                A_w_alpha += self.r_r[:, :, alpha, i] * np.exp(1j * np.dot(kpt, rpt)) / self.weight_list[i]
            elif flag == 2:
                A_w_alpha_beta += 1j * rpt[beta] * self.r_r[:, :, i] * \
                                np.exp(1j * np.dot(kpt, rpt)) / self.weight_list[i]
            else:
                raise Exception('flag should be 1 or 2')
        if flag == 1:
            return A_w_alpha
        elif flag == 2:
            return A_w_alpha_beta
        else:
            raise Exception('flag should be 1 or 2')

    def cal_eig(self, kpt):
        """
        calculate sorted (large to small) eigenvalue and eigenstate at kpt
        :param kpt: kpt, unscaled
        :return: a tuple (w, v), w is 1-D array of sorted eigenvalues and v is corresponding eigenvectors.
        Each eigenvector is something like v[:, i]
        """
        (w, v) = LA.eig(self.cal_H_w(kpt, flag=0))
        idx = w.argsort()
        w = np.real(w[idx])
        v = v[:, idx]
        return w, v

    def cal_A_h(self, kpt, U, flag, alpha=0):
        """
        calculate A^(H)_\alpha(k) or A^(H)_\alpha\alpha(k)
        :param kpt: kpt, unscaled
        :param U: ndarray of dimension (num_wann, num_wann), matrix that can diagonalize H^(W)(k).
        notice that a global phase factor is included in this matrix
        :param flag: 0: A^(H)_\alpha(k), 1: A^(H)_\alpha\alpha(k)
        :param alpha, beta: 0: x, 1: y, 2: z
        :return: A^(H)_\alpha(k) or A^(H)_\alpha\alpha(k)
        """
        H_w = self.cal_H_w(kpt, 0)
        V = U.conj().T.dot(self.cal_H_w(kpt, 1, alpha)).dot(U)
        # E is now [[eig_value_1, eig_value_1, ...], [eig_value_2, eig_value_2, ...], [eig_value_3, eig_value_3, ...]]
        E = np.tile(np.diagonal(U.conj().T.dot(H_w).dot(U)).reshape((self.num_wann, 1)), self.num_wann)
        # E[i,j] would be eigenvalue[i] - eigenvalue[j]
        E = E - E.T
        V_diag_0 = V
        np.fill_diagonal(V_diag_0, 0)
        E_diag_1 = E
        np.fill_diagonal(E_diag_1, 1)
        if flag == 1:
            D = -V_diag_0 / E_diag_1
            return U.conj().T.dot(self.cal_A_w(kpt, 1, alpha)).dot(U) + 1j * D
        elif flag == 2:
            D = -V_diag_0 / E_diag_1
            A_w_alpha = self.cal_A_w(kpt, 1, alpha)
            A_w_alpha_alpha = self.cal_A_w(kpt, 2, alpha, alpha)
            W = U.conj().T.dot(self.cal_H_w(kpt, 1, alpha, alpha)).dot(U)
            F = -V_diag_0 / E_diag_1
            F = V.dot(F) / E_diag_1
            F -= V / (E_diag_1)**2 * np.tile(np.diagonal(V), (self.num_wann, 1))
            F -= W / E_diag_1
            np.fill_diagonal(F, 0)
            F += np.diag(np.sum((np.norm(V_diag_0) / E_diag_1)**2, axis=1))
            return D.conj().T.dot(U.conj().T).dot(A_w_alpha).dot(U) + \
                   U.conj().T.dot(A_w_alpha_alpha).U + U.conj.T.dot(A_w_alpha).dot(U).dot(D) + \
                   1j * D.conj().T.dot(D) + 1j * F
        else:
            raise Exception('flag should be 1 or 2')

    def plot_band(self, kpt_list, ndiv):
        """
        plot band structure of the system
        :param kpt_list: ndarray containing list of kpoints, example: [[0,0,0],[0.5,0.5,0.5]...]
        :param ndiv: number of kpoints in each line
        """

        # a list of kpt to be calculated
        def vec_linspace(vec_1, vec_2, num):
            delta = (vec_2 - vec_1) / num
            return np.array([vec_1 + delta * i for i in range(num)])

        kpt_plot = np.concatenate(
            tuple([vec_linspace(kpt_list[i, :], kpt_list[i + 1, :], ndiv) for i in range(len(kpt_list) - 1)]))
        # calculate k axis
        kpt_flatten = [0.0]
        kpt_distance = 0.0
        for i in range(len(kpt_plot) - 1):
            kpt_distance += LA.norm(kpt_plot[i + 1] - kpt_plot[i])
            kpt_flatten += [kpt_distance]
        # calculate eigenvalue
        eig = np.zeros((0, self.num_wann))
        for kpt in kpt_plot:
            w = self.cal_eig(kpt)[0].reshape((1, self.num_wann))
            eig = np.concatenate((eig, w))
        plt.plot(kpt_flatten, eig, 'k-')
        plt.show()
