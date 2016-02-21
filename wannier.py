import numpy as np
from numpy import linalg as LA


class Wannier():
    def __init__(self, path, lattice_vec):
        """
        :param path: a dict of wannier outputs paths, currently: {'hr': 'hr.dat', 'rr': 'rr.dat'}
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
        # weight list corresponding to rpt list, ndarray, example: [4,1,1,1,2...]
        self.r_weight_list = None
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
            # skip the first line
            file.readline()
            # read num_wann and nrpts
            num_wann = int(file.readline().split()[0])
            nrpts = int(file.readline().split()[0])
            # read r_weight_list
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
                        # first index: band k, second index: band j, third index: rpt i
                        H_r[k, j, i] = float(buffer[5]) + 1j * float(buffer[6])
                rpt_list = rpt_list + [buffer[0:3]]
            rpt_list = np.array(rpt_list, dtype='float')
        # save every thing
        self.nrpts = nrpts
        self.rpt_list = rpt_list
        self.r_weight_list = weight_list
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

    def cal_H_w(self, kpt, flag=0, alpha=0, beta=0):
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
        # fourier transform
        phase = np.exp(1j * np.dot(kpt, rpt_list.T))/self.r_weight_list
        if flag == 0:
            return np.einsum('k,ijk->ij', phase, self.H_r)
        elif flag == 1:
            return np.einsum('k,ijk->ij', 1j * rpt_list[:, alpha] * phase, self.H_r)
        elif flag == 2:
            return np.einsum('k,ijk->ij', -rpt_list[:, alpha] * rpt_list[:, beta] * phase, self.H_r)
        else:
            raise Exception('flag should be 0, 1 or 2')

    def cal_A_w(self, kpt, flag=1, alpha=0, beta=0):
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
        # fourier transform
        phase = np.exp(1j * np.dot(kpt, rpt_list.T))/self.r_weight_list
        r_alpha = self.r_r[:, :, alpha, :]
        if flag == 1:
            return np.einsum('k,ijk->ij', phase, r_alpha)
        elif flag == 2:
            return np.einsum('k,ijk->ij', 1j * rpt_list[:, beta] * phase, r_alpha)
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

    def cal_A_h(self, kpt, U, flag=1, alpha=0, beta=0, delta=1e-7):
        """
        calculate A^(H)_\alpha(k) or A^(H)_\alpha\beta(k)
        If any of the bands are degenerate, zero matrix is returned
        :param kpt: kpt, unscaled
        :param U: ndarray of dimension (num_wann, num_wann), matrix that can diagonalize H^(W)(k).
        notice that a global phase factor is included in this matrix
        :param flag: 0: A^(H)_\alpha(k), 1: A^(H)_\alpha\beta(k)
        :param alpha, beta: 0: x, 1: y, 2: z
        :param delta: threshold of degenerate bands
        :return: A^(H)_\alpha(k) or A^(H)_\alpha\beta(k)
        """
        U_deg = U.conj().T
        H_w = self.cal_H_w(kpt)
        # E[i, j] = eigenvalue[j]
        E = np.tile(np.diagonal(U_deg.dot(H_w).dot(U)), (self.num_wann, 1))
        # E[i,j] would be eigenvalue[i] - eigenvalue[j]
        E = np.real(E.T - E)
        E_mod = np.copy(E)
        np.fill_diagonal(E_mod, 1)
        # return zero matrix if any bands are degenerate
        if (np.abs(E_mod) < delta).any():
            return np.zeros((self.num_wann, self.num_wann), dtype='complex')
        H_hbar_alpha = U_deg.dot(self.cal_H_w(kpt, flag=1, alpha=alpha)).dot(U)

        H_hbar_alpha_mod = np.copy(H_hbar_alpha)
        np.fill_diagonal(H_hbar_alpha_mod, 0)
        D_alpha = - H_hbar_alpha_mod / E_mod
        A_hbar_alpha = U_deg.dot(self.cal_A_w(kpt, 1, alpha)).dot(U)

        if flag == 1:
            return A_hbar_alpha + 1j * D_alpha
        elif flag == 2:
            H_hbar_beta = U_deg.dot(self.cal_H_w(kpt, flag=1, alpha=beta)).dot(U)
            H_hbar_beta_mod = np.copy(H_hbar_beta)
            np.fill_diagonal(H_hbar_beta_mod, 0)
            D_beta = -H_hbar_beta_mod / E_mod
            H_hbar_alpha_beta = U_deg.dot(self.cal_H_w(kpt, flag=2, alpha=alpha, beta=beta)).dot(U)
            A_hbar_alpha_beta = U_deg.dot(self.cal_A_w(kpt, flag=2, alpha=alpha, beta=beta)).dot(U)
            # H_hbar_beta_diag_copy[i, j] = H_hbar_beta[i, i]
            H_hbar_beta_diag_copy = np.tile(np.diagonal(H_hbar_beta), (self.num_wann, 1))
            D_alpha_beta = 1 / E_mod**2 * (
                (H_hbar_beta_diag_copy - H_hbar_beta_diag_copy.T) * H_hbar_alpha -
                E * (D_beta.conj().T.dot(H_hbar_alpha) + H_hbar_alpha_beta + H_hbar_alpha * D_beta)
            )
            return D_beta.conj().T.dot(A_hbar_alpha) + A_hbar_alpha_beta + A_hbar_alpha * D_beta + 1j * D_alpha_beta
        else:
            raise Exception('flag should be 1 or 2')

    def cal_shift_integrand(self, kpt_list, fermi_energy, alpha=0, beta=0):
        """
        calculate shift current integrand in kpt of kpt_list
        :param kpt_list: ndarray, like [[kpt1], [kpt2], [kpt3] ...]
        :param fermi_energy: fermi energy
        :param alpha, beta: 0: x, 1: y, 2: z
        :return: the integrand of dimension (num_wann, num_wann, len(kpt_list))
        """
        nkpts = kpt_list.shape[0]
        integrand_list = np.zeros((self.num_wann, self.num_wann, nkpts), dtype='complex')
        for i in range(nkpts):
            kpt = kpt_list[i, :]
            (w, v) = self.cal_eig(kpt)
            A_alpha = self.cal_A_h(kpt, v, flag=1, alpha=alpha)
            A_beta = self.cal_A_h(kpt, v, flag=1, alpha=beta)
            A_beta_alpha = self.cal_A_h(kpt, v, flag=2, alpha=beta, beta=alpha)
            # E[i, j] = eigenvalue[j]
            E = np.real(np.tile(w, (self.num_wann, 1)))
            fermi = np.zeros((self.num_wann, self.num_wann), dtype='float')
            fermi[E > fermi_energy] = 0
            fermi[E <= fermi_energy] = 1
            # fermi[i, j] = f[eigenvalue[i]] - f[eigenvalue[j]]
            fermi = fermi.T - fermi
            ki = np.tile(np.diagonal(A_alpha), (self.num_wann, 1))
            # ki[i, j] = berry_connection[i] - berry_connection[j]
            ki = ki.T - ki
            integrand_list[:, :, i] = fermi * np.imag(A_beta.T * (A_beta_alpha - 1j * ki * A_beta))
        return integrand_list

    def cal_shift_cond(self, omega, kpt_list, integrand_list, epsilon=1e-3):
        """
        calculation shift conductance
        :param omega: frequency
        :param kpt_list: a list of kpt. ndarray like [[kpt1], [kpt2], [kpt3] ...]
        :param integrand_list: integrand list corresponding to kpt list
        :param epsilon: parameter to control spread of delta function
        :return: shift conductance
        """
        epsilon = 1e-2
        nkpts = kpt_list.shape[0]
        # delta[i, j] = DiracDelta[omega[i] - omega[j] - omega]
        delta = np.zeros((self.num_wann, self.num_wann, nkpts), dtype='float')
        for i in range(nkpts):
            kpt = kpt_list[i, :]
            (w, v) = self.cal_eig(kpt)
            E = np.real(np.tile(w, (self.num_wann, 1)))
            # E[i,j] would be eigenvalue[i] - eigenvalue[j]
            E = E.T - E
            delta[:, :, i] = 1/np.pi * (epsilon / (epsilon**2 + (E - omega)**2))
        # volume of brillouin zone
        volume = abs(np.dot(np.cross(self.rlattice_vec[0], self.rlattice_vec[1]), self.rlattice_vec[2]))
        return np.sum(delta * integrand_list) * volume / nkpts

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
        return kpt_flatten, eig
