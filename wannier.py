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
        # kpt list in unit of rlattice_vec, ndarray, example: [[0,0,0],[0.1,0.1,0.1],[]]
        self.kpt_list = None
        # kpt number
        self.nkpts = None
        # u matrix in corresponding order with kpt_list, ndarray of dimension (num_wann, num_wann, nkpts)
        self.u_list = None
        # a matrix in corresponding order with kpt_list, ndarray of dimension (num_wann, num_wann, 3, nkpts)
        self.a_list = None
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

    def read_au(self):
        """
        read wannier A output and U output
        """
        with open(self.path['a']) as a_file, open(self.path['u']) as u_file:
            kpt_list = []
            a_list = np.zeros((self.num_wann, self.num_wann, 3, 0), dtype='complex')
            u_list = np.zeros((self.num_wann, self.num_wann, 0), dtype='complex')
            # read a
            while True:
                a_buffer = a_file.readline()
                if a_buffer:
                    kpt_list.append(
                        [float(a_buffer.split()[1]), float(a_buffer.split()[2]), float(a_buffer.split()[3])])
                    a_temp = np.zeros((self.num_wann, self.num_wann, 3, 1), dtype='complex')
                    for i in range(self.num_wann ** 2):
                        a_buffer = a_file.readline()
                        a_buffer = [float(item) for item in a_buffer.split()]
                        a_temp[i // self.num_wann, i % self.num_wann, 0, 0] = a_buffer[0] + 1j * a_buffer[1]
                        a_temp[i // self.num_wann, i % self.num_wann, 1, 0] = a_buffer[2] + 1j * a_buffer[3]
                        a_temp[i // self.num_wann, i % self.num_wann, 2, 0] = a_buffer[4] + 1j * a_buffer[5]
                    a_list = np.concatenate((a_list, a_temp), axis=3)
                else:
                    break
            # read u
            while True:
                u_buffer = u_file.readline()
                if u_buffer:
                    u_temp = np.zeros((self.num_wann, self.num_wann, 1), dtype='complex')
                    for i in range(self.num_wann ** 2):
                        u_buffer = u_file.readline()
                        u_buffer = [float(item) for item in u_buffer.split()]
                        u_temp[i // self.num_wann, i % self.num_wann, 0] = u_buffer[0] + 1j * u_buffer[1]
                    u_list = np.concatenate((u_list, u_temp), axis=2)
                else:
                    break
            self.kpt_list = np.array(kpt_list)
            self.a_list = a_list
            self.u_list = u_list
            self.nkpts = len(kpt_list)

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

    def cal_hamk(self, kpt):
        """
        calculate H(k)
        :param kpt: kpt, unscaled
        :return: H(k)
        """
        # scale kpt and rpt
        kpt = self.scale(kpt, 'k')
        rpt_list = self.scale(self.rpt_list, 'r')
        # initialize
        hamk = np.zeros((self.num_wann, self.num_wann), dtype='complex')
        # fourier transform
        for i in range(self.nrpts):
            hamk += self.hams[:, :, i] * np.exp(1j * np.dot(kpt, rpt_list[i, :])) / self.weight_list[i]
        return hamk

    def cal_eig(self, kpt):
        """
        calculate sorted eigenvalue and eigenstate at kpt
        :param kpt: kpt, unscaled
        :return: a sorted list of tuples containing eigenvalue and eigenstate
        """
        (w, v) = LA.eig(self.cal_hamk(kpt))
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
