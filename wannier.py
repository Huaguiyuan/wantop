import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt


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
        self.weight_list = None
        # hamiltonian matrix element in real space, ndarray of dimension (num_wann, num_wann, nrpts)
        self.hams = None
        # r matrix element in real space, ndarray of dimension (num_wann, num_wann, 3, nrpts)
        self.rs = None
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
            # read weight_list
            weight_list = []
            for i in range(int(np.ceil(nrpts / 15.0))):
                buffer = file.readline().split()
                weight_list = weight_list + buffer
            weight_list = np.array(weight_list, dtype='float')
            # read hamiltonian matrix elements
            rpt_list = []
            hams_r = np.zeros((num_wann, num_wann, nrpts), dtype='float')
            hams_i = np.zeros((num_wann, num_wann, nrpts), dtype='float')
            for i in range(nrpts):
                for j in range(num_wann):
                    for k in range(num_wann):
                        buffer = file.readline().split()
                        # first index: band k, second index: band j, third index: rpt i
                        hams_r[k, j, i] = float(buffer[5])
                        hams_i[k, j, i] = float(buffer[6])
                rpt_list = rpt_list + [buffer[0:3]]
            hams = hams_r + 1j * hams_i
            rpt_list = np.array(rpt_list, dtype='float')
        # save every thing
        self.nrpts = nrpts
        self.rpt_list = rpt_list
        self.weight_list = weight_list
        self.hams = hams
        self.num_wann = num_wann

    def read_rr(self):
        """
        read wannier rr output file
        """
        with open(self.path['rr'], 'r') as file:
            # skip first two lines
            file.readline()
            file.readline()
            self.rs = np.zeros((self.num_wann, self.num_wann, 3, self.nrpts), dtype='complex')
            for cnt_rpt in range(self.nrpts):
                for i in range(self.num_wann):
                    for j in range(self.num_wann):
                        buffer = file.readline()
                        buffer = buffer.split()
                        self.rs[j, i, 0, cnt_rpt] = float(buffer[5]) + 1j * float(buffer[6])
                        self.rs[j, i, 1, cnt_rpt] = float(buffer[7]) + 1j * float(buffer[8])
                        self.rs[j, i, 2, cnt_rpt] = float(buffer[9]) + 1j * float(buffer[10])

    def scale(self, v, flag):
        """
        :param v: single point v or v list
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

    def cal_hamk(self, kpt, flag, alpha=0, beta=0):
        """
        calculate H^(W)(k), H^(W)_\alpha(k), H^(W)_\alpha\beta(k)
        :param kpt: kpt, unscaled
        :param flag: a list of integers, for example [0, 2] instructs this method to calculate
        H^(W)(k) and H^(W)_\alpha\beta(k)
        :param alpha, beta: 0: x, 1: y, 2: z
        :return: [H^(W)(k), H^(W)_\alpha(k), H^(W)_\alpha\beta(k)], only items corresponding to what is present in
        flag is returned
        """
        # scale kpt and rpt
        kpt = self.scale(kpt, 'k')
        rpt_list = self.scale(self.rpt_list, 'r')
        # initialize
        hamk = np.zeros((self.num_wann, self.num_wann), dtype='complex')
        del_hamk = np.zeros((self.num_wann, self.num_wann), dtype='complex')
        del_del_hamk = np.zeros((self.num_wann, self.num_wann), dtype='complex')
        # fourier transform
        for i in range(self.nrpts):
            rpt = rpt_list[i, :]
            if 0 in flag:
                hamk += self.hams[:, :, i] * np.exp(1j * np.dot(kpt, rpt)) / self.weight_list[i]
            if 1 in flag:
                del_hamk += 1j * rpt[alpha] * self.hams[:, :, i] * np.exp(1j * np.dot(kpt, rpt)) / self.weight_list[i]
            if 2 in flag:
                del_del_hamk += - rpt[alpha] * rpt[beta] * self.hams[:, :, i] * \
                                np.exp(1j * np.dot(kpt, rpt)) / self.weight_list[i]
        return [hamk, del_hamk, del_del_hamk]

    def cal_A(self, kpt, flag, alpha=0, beta=0):
        """
        calculate A^(W)(k), A^(W)_\alpha(k), A^(W)_\alpha\beta(k)
        :param kpt: kpt, unscaled
        :param flag: a list of integers, for example [0, 2] instructs this method to calculate
        A^(W)(k) and A^(W)_\alpha\beta(k)
        :param alpha, beta: 0: x, 1: y, 2: z
        :return: [A^(W)(k), A^(W)_\alpha(k), A^(W)_\alpha\beta(k)], only items corresponding to what is present in
        flag is returned
        """
        # scale kpt and rpt
        kpt = self.scale(kpt, 'k')
        rpt_list = self.scale(self.rpt_list, 'r')
        # initialize
        A = np.zeros((self.num_wann, self.num_wann), dtype='complex')
        del_A = np.zeros((self.num_wann, self.num_wann), dtype='complex')
        del_del_A = np.zeros((self.num_wann, self.num_wann), dtype='complex')
        # fourier transform
        for i in range(self.nrpts):
            rpt = rpt_list[i, :]
            if 0 in flag:
                A += self.rs[:, :, i] * np.exp(1j * np.dot(kpt, rpt)) / self.weight_list[i]
            if 1 in flag:
                del_A += 1j * rpt[alpha] * self.rs[:, :, i] * np.exp(1j * np.dot(kpt, rpt)) / self.weight_list[i]
            if 2 in flag:
                del_del_A += - rpt[alpha] * rpt[beta] * self.rs[:, :, i] * \
                                np.exp(1j * np.dot(kpt, rpt)) / self.weight_list[i]
        return [A, del_A, del_del_A]

    def cal_eig(self, kpt):
        """
        calculate sorted eigenvalue and eigenstate at kpt
        :param kpt: kpt, unscaled
        :return: a sorted list of tuples containing eigenvalue and eigenstate
        """
        (w, v) = LA.eig(self.cal_hamk(kpt, flag=[0])[0])
        eigen_system = [(np.real(w[i]), v[:, i]) for i in range(len(w))]
        return list(sorted(eigen_system))

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
            eigen_system = self.cal_eig(kpt)
            w = np.array([eigen[0] for eigen in eigen_system]).reshape((1, self.num_wann))
            eig = np.concatenate((eig, w))
        plt.plot(kpt_flatten, eig, 'k-')
        plt.show()
