from wannier import Wannier
import numpy as np

lattice_vec = np.array(
        [[4.0771999359, 0.0000000000, 0.0000000000],
         [0.0214194091, 4.0771436725, 0.0000000000],
         [0.0214194091, 0.0213071771, 4.0770879964]]

)
system = Wannier({'hr': 'wannier90_hr.dat', 'a': 'A.dat', 'u': 'U.dat'}, lattice_vec)
system.read_hr()
system.read_au()
final = np.matrix(system.u_list[:,:,0]).H * np.matrix(system.cal_hamk(np.array([0.1,0.1,0.1])))*np.matrix(system.u_list[:,:,0])
for i in range(system.num_wann):
    for j in range(system.num_wann):
        if i == j:
            print(final[i, j])
        else:
            if final[i, j] > 1e-4:
                print('fail')
'''
kpt_list = np.array(
    [
        [0.5, 0.5, 0],
        [0, 0, 0],
        [0.5, 0.5, 0.5]
    ]
)
system.plot_band(kpt_list, 150)
'''
print('done')