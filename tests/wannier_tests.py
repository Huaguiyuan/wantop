import unittest
import numpy as np
from wannier import Wannier


class WannierTestBTO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lattice_vec = np.array([
            [3.999800000000001, 0.000000000000000, 0.000000000000000],
            [0.000000000000000, 3.999800000000001, 0.000000000000000],
            [0.000000000000000, 0.000000000000000, 4.018000000000000],
        ]
        )
        system = Wannier(
            {'hr': '../data/hr_BTO.dat', 'rr': '../data/rr_BTO.dat', 'rndegen': '../data/rndegen_BTO.dat'},
            lattice_vec)
        system.read_all()
        cls.system = system

    def test_remove_kpt_list(self):
        kpt_list = np.array([0, 0, 0])
        system = self.system
        system.kpt_list = kpt_list
        self.assertTrue(system.kpt_data == {} and system.kpt_done == {})

    def test_eigenvalue(self):
        kpt_list = np.array([[0.1, 0.2, 0.3]])
        std_value = np.array(
            [-1.25968556, -1.25297927, -0.74183812, -0.73171052,
             -0.37429321, -0.35364465, -0.12091341, -0.06685028,
             0.15536264, 0.18227365, 0.78840589, 0.79268875,
             1.34587565, 1.35604473, 1.51059615, 1.5292992,
             1.73312867, 1.74372568, 4.88864594, 4.88996063,
             5.38465549, 5.39111039, 5.44450065, 5.45198766,
             7.0100683, 7.01533062, 8.18990437, 8.19181835,
             9.2071381, 9.21069766, 9.65986908, 9.67214067,
             11.03796193, 11.04893674, 11.35945704, 11.36537974,
             11.58429844, 11.59385674]
        )
        system = self.system
        system.kpt_list = kpt_list
        system.calculate('eigenvalue')
        self.assertTrue((system.kpt_data['eigenvalue'][:, 0] - std_value < 1e-5).all())

    def test_A_h(self):
        kpt_list = np.array([[0.1, 0.2, 0.3]])
        system = self.system
        system.kpt_list = kpt_list
        system.calculate('A_h_ind', 0)
        self.assertTrue(abs(system.kpt_data['A_h_ind'][0, 1, 0, 0]) - abs((0.9941102 + 1j * -0.3169803)) < 1e-5)
        self.assertTrue(abs(system.kpt_data['A_h_ind'][0, 0, 0, 0] - 1.7101943) < 1e-5)


class WannierTestFe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lattice_vec = np.array([
            [2.71175, 2.71175, 2.71175],
            [-2.71175, 2.71175, 2.71175],
            [-2.71175, -2.71175, 2.71175]
        ]
        ) * 0.5293
        system = Wannier(
            {'hr': '../data/hr_Fe.dat', 'rr': '../data/rr_Fe.dat', 'rndegen': '../data/rndegen_Fe.dat'},
            lattice_vec)
        system.read_all()
        cls.system = system

    def test_berry_curv(self):
        system = self.system
        b1 = np.array([0.5, -0.5, -0.5])
        b2 = np.array([0.5, 0.5, 0.5])
        kpt = 91 / 200 * b1 + 65 / 200 * b2
        system.kpt_list = kpt.reshape((1, 3))
        system.fermi_energy = 12.627900
        self.assertTrue(abs(system.cal_berry_curv(0, 1)[0] + 4569.66796875) < 1.0)
        kpt = 29 / 200 * b1 + 81 / 200 * b2
        system.kpt_list = kpt.reshape((1, 3))
        system.fermi_energy = 12.627900
        self.assertTrue(abs(system.cal_berry_curv(0, 1)[0] + 0.34455416) < 1e-5)
