from wannier import Wannier
import numpy as np

lattice_vec = np.array(
        [[4.0771999359, 0.0000000000, 0.0000000000],
         [0.0214194091, 4.0771436725, 0.0000000000],
         [0.0214194091, 0.0213071771, 4.0770879964]]

)
system = Wannier('wannier90_hr.dat', lattice_vec)
system.read_hr()
kpt_list = np.array(
    [
        [0.5, 0.5, 0],
        [0, 0, 0],
        [0.5, 0.5, 0.5]
    ]
)
#system.plot_band(kpt_list, 150)
delta_k_1 = 0.001
delta_k_2 = 0.001
m = 5
n = 10
kpt = np.array([0.0, 0.0, 0.0])
khat_1 = np.array([0.0,0.0,1.0])
khat_2 = np.array([0.0,0.0,1.0])
eigen_system_k = system.cal_eig(kpt)
eigen_system_delta_1 = system.cal_eig(kpt + delta_k_1*khat_1)
eigen_system_delta_12 = system.cal_eig(kpt + delta_k_1*khat_1 + delta_k_2*khat_2)
R = 1/(delta_k_1*delta_k_2)*np.imag(
        np.log((np.dot(eigen_system_k[m][1], eigen_system_delta_1[n][1])*np.dot(eigen_system_delta_1[m][1], eigen_system_delta_12[n][1]))/np.dot(eigen_system_k[m][1], eigen_system_delta_1[n][1]))
        +
        np.log()
)
