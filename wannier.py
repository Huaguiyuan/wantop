import numpy as np


class Wannier():
    def __init__(self, path, lattice_vec):
        """
        :param path: wannier output path, example: '\home\user\wanner.output'
        :param lattice_vec: lattice vector, ndarray, example: [[first vector], [second vector]...]
        """
        # wannier output path
        self.path = path
        # lattice vector
        self.lattice_vec = lattice_vec
        # rpt number
        self.nrpts = None
        # wannier function number
        self.num_wann = None
        # rpt list, ndarray, example: [[-5,5,5],[5,4,3]...]
        self.rpt_list = None
        # weight list corresponding to rpt list, ndarray, example: [4,1,1,1,2...]
        self.weight_list = None
        # hamiltonian matrix element in real space, ndarray of dimension (num_wann, num_wanner, nrpts)
        self.hams = None
        # generate reciprocal lattice vector
        [a1, a2, a3] = self.lattice_vec
        b1 = 2*np.pi*(np.cross(a2, a3)/np.dot(a1, np.cross(a2, a3)))
        b2 = 2*np.pi*(np.cross(a3, a1)/np.dot(a2, np.cross(a3, a1)))
        b3 = 2*np.pi*(np.cross(a1, a1)/np.dot(a3, np.cross(a1, a2)))
        self.rlattice_vec = np.array([b1, b2, b3])


    def read_hr(self):
        """
        read wannier output file
        """
        with open(self.path, 'r') as file:
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
        elif len(v.shape) == 1:
            return np.array([np.dot(kpt, scale_vec) for kpt in v])
        else:
            raise Exception('k should be an array of dimension 2 or 3')

    def get_hamk(self, kpt):
        """
        calculate H(k)
        :param kpt: kpt
        :return: H(k)
        """
        # scale kpt and rpt
        kpt = self.scale(kpt, 'k')
        rpt_list = self.scale(self.rpt_list, 'r')
        hamk = np.zeros((self.num_wann, self.num_wann), dtype='complex')
        for i in range(self.nrpts):
            hamk += self.hams[:,:,i] * np.exp(1j * np.dot(kpt, rpt_list[i, :])) / self.weight[i]
        return hamk
