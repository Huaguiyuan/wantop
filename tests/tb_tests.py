from wannier import Wannier
import unittest
import numpy as np


class TBTestBN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lattice_vec = np.array([
            [2.5124280453, 0.0000000000, 0.0000000000],
            [-1.2562139713, 2.1758270926, 0.0000000000],
            [0.0000000000, 0.0000000000, 15.0000000000]
        ]
        )
        system = Wannier(
            lattice_vec,
            {'hr': '../data/hr_BN.dat', 'wann_center': '../data/wann_center_BN.dat', 'rndegen': '../data/rndegen_BN.dat'}
        )
        system.read_all()
        cls.system = system

    def test_shift_integrand(self):
        kpt_list = np.array(
            [
                [0.1, 0.2, 0.3],
            ]
        )
        system = self.system
        system.set_fermi_energy(-3.9060)
        system.set_kpt_list(kpt_list)
        system.calculate('shift_integrand', 1, 1)
        self.assertTrue(np.abs(system.kpt_data['shift_integrand'][1][1][0, 7, 0] - -0.0786306489972) < 1e-6)