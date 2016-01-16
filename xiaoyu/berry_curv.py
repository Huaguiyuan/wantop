import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axis3d
import copy


def kpoint_scale(klist):
    b = np.array([[1.807071, 0.000000, 0.000000],
                  [0.000000, 1.005471, 0.000000],
                  [0.000000, 0.000000, 0.448223]])
    n = np.shape(klist)

    kpt = []
    for i in range(n[0]):
        kpt = kpt + [b[0, :] * klist[i, 0] + b[1, :] * klist[i, 1] + b[2, :] * klist[i, 2]]
    kpt = np.array(kpt, dtype='float')
    return kpt


def kpoint_reverse_scale(klist):
    b = np.array([[1.807071, 0.000000, 0.000000],
                  [0.000000, 1.005471, 0.000000],
                  [0.000000, 0.000000, 0.448223]])
    n = np.shape(klist)

    kpt = []
    for i in range(n[0]):
        kpt = kpt + [LA.solve(b, klist[i, :])]
    kpt = np.array(kpt, dtype='float')
    return kpt


def rpoint_scale(rlist):
    a = np.array([[3.477000, 0.000000, 0.000000],
                  [0.000000, 6.249000, 0.000000],
                  [0.000000, 0.000000, 14.018000]])
    n = np.shape(rlist)
    rpt = []
    for i in range(n[0]):
        rpt = rpt + [a[0, :] * rlist[i, 0] + a[1, :] * rlist[i, 1] + a[2, :] * rlist[i, 2]]
    rpt = np.array(rpt, dtype='float')
    return rpt


def read_hr(file, band):
    # band input has been disabled
    next(file)
    global num_wann
    num_wann = int(file.readline().strip().split()[0])
    global nrpts
    nrpts = int(file.readline().strip().split()[0])
    global weight
    weight = []
    for i in range(int(np.ceil(nrpts / 15.0))):
        buffer = file.readline().strip().split()
        weight = weight + buffer
    weight = np.array(weight, dtype='int')
    global rpt
    rpt = []
    global hamr
    hamr = np.zeros((len(band), len(band), nrpts))
    global hami
    hami = np.zeros((len(band), len(band), nrpts))

    for i in range(nrpts):
        for j in range(num_wann):
            for k in range(num_wann):
                buffer = file.readline().strip().split()
                hamr[k, j, i] = float(buffer[5])
                hami[k, j, i] = float(buffer[6])
        rpt = rpt + [buffer[0:3]]

    hamr = np.array(hamr, dtype='float')
    rpt = np.array(rpt, dtype='int')

    return weight, rpt, hamr, hami  #


def check_hermitian(hami):  # generally its not a hermitian
    a = []
    for i in range(nrpts):
        temp = np.matrix(hami[:, :, i])
        differ = abs(temp - temp.getH())
        a = a + [np.max(differ)]
    return a


def fourier(rpt, klist, hamr, hami, krel=True):  ##klist must be an array
    if krel == True:
        klist = kpoint_scale(klist)
    rpt = rpoint_scale(rpt)

    [n, i] = np.shape(klist)
    hamk = []
    for i in range(n):
        temp = 0
        for j in range(nrpts):
            temp = temp + (hamr[:, :, j] + 1j * hami[:, :, j]) * np.exp(1j * np.dot(klist[i, :], rpt[j, :])) / weight[j]
        hamk = hamk + [temp]
    hamk = np.array(hamk, dtype='complex')
    return hamk


def curvature(kpt0, band_num, b=0.00001, krel=True):
    b = b / 2
    if krel == True:
        kpt = kpoint_scale(np.array([kpt0], dtype='float'))[0, :]

    else:
        kpt = kpt0
        kpt0 = kpoint_reverse_scale(np.array([kpt], dtype='float'))[0, :]

    klist = [[kpt[0] + b, kpt[1] + b, kpt[2] + b],
             [kpt[0] - b, kpt[1] + b, kpt[2] + b],
             [kpt[0] + b, kpt[1] - b, kpt[2] + b],
             [kpt[0] + b, kpt[1] + b, kpt[2] - b],
             [kpt[0] - b, kpt[1] - b, kpt[2] + b],
             [kpt[0] + b, kpt[1] - b, kpt[2] - b],
             [kpt[0] - b, kpt[1] + b, kpt[2] - b],
             [kpt[0] - b, kpt[1] - b, kpt[2] - b]]
    klist = np.array(klist, dtype='float')
    # print(klist)
    # x,y,z, 3 directon, use cubic method
    vec = []

    kpoint = np.array(klist)
    hamk = fourier(rpt, kpoint, hamr, hami, krel=False)
    w, v = LA.eig(hamk)
    # print(np.shape(w),np.shape(v))
    for i in range(8):
        vec = vec + [v[i, :, np.argsort(w[i, :])[band_num - 1]]]
    vec = np.array(vec, dtype='complex')
    # print(np.shape(vec))
    curv_z1 = np.log(
        np.vdot(vec[0], vec[1]) * np.vdot(vec[1], vec[4]) * np.vdot(vec[4], vec[2]) * np.vdot(vec[2], vec[0]))
    curv_z2 = np.log(
        np.vdot(vec[3], vec[6]) * np.vdot(vec[6], vec[7]) * np.vdot(vec[7], vec[5]) * np.vdot(vec[5], vec[3]))
    if np.sign(np.imag(curv_z1)) * np.sign(np.imag(curv_z2)) == -1:
        print('WARNING: berry curv z change sign, please reduce b or check. KPT_rel=', kpt0)

    curv_y1 = np.log(
        np.vdot(vec[0], vec[3]) * np.vdot(vec[3], vec[6]) * np.vdot(vec[6], vec[1]) * np.vdot(vec[1], vec[0]))
    curv_y2 = np.log(
        np.vdot(vec[2], vec[5]) * np.vdot(vec[5], vec[7]) * np.vdot(vec[7], vec[4]) * np.vdot(vec[4], vec[2]))
    if np.sign(np.imag(curv_y1)) * np.sign(np.imag(curv_y2)) == -1:
        print('WARNING: berry curv y change sign, please reduce b or check. KPT_rel=', kpt0)

    curv_x1 = np.log(
        np.vdot(vec[0], vec[2]) * np.vdot(vec[2], vec[5]) * np.vdot(vec[5], vec[3]) * np.vdot(vec[3], vec[0]))
    curv_x2 = np.log(
        np.vdot(vec[1], vec[4]) * np.vdot(vec[4], vec[7]) * np.vdot(vec[7], vec[6]) * np.vdot(vec[6], vec[1]))
    if np.sign(np.imag(curv_x1)) * np.sign(np.imag(curv_x2)) == -1:
        print('WARNING: berry curv x change sign, please reduce b or check. KPT_rel=', kpt0)

    # print(np.exp(curv_x1),np.exp(curv_x2),np.exp(curv_y1),np.exp(curv_y2),np.exp(curv_z1),np.exp(curv_z2))
    # print(curv_x1,curv_x2,curv_y1,curv_y2,curv_z1,curv_z2)
    curv = np.array([(curv_x1 + curv_x2) / 2, (curv_y1 + curv_y2) / 2, (curv_z1 + curv_z2) / 2], dtype='complex')
    curv = np.imag(curv) / ((2 * b) ** 2)

    ch = np.imag(curv_x1) - np.imag(curv_x2) + np.imag(curv_y1) - np.imag(curv_y2) + np.imag(curv_z1) - np.imag(curv_z2)
    print(ch / (2 * np.pi))
    print(np.imag(curv_x1), np.imag(curv_x2), np.imag(curv_y1), np.imag(curv_y2), np.imag(curv_z1), np.imag(curv_z2))

    return curv


def hedgehog(weylpt, band_num, weyl_rel=True, nkpt=100, radius=0.001):  # assue we consider 1 weyl point at a time
    num_theta = int(np.ceil(np.sqrt(nkpt) / 3 * 2))
    num_phi = int(np.floor(nkpt / num_theta))
    print(num_theta, num_phi)
    if weyl_rel == True:
        weylpt = kpoint_scale(np.array([weylpt], dtype='float'))[0, :]
    print('weyl point in cartisan coordinate is ', weylpt)

    klist = []
    klist = klist + [[0, 0, radius]]
    for i in range(1, num_theta):
        theta = i * np.pi / num_theta
        for j in range(num_phi):
            phi = j * np.pi * 2 / num_phi
            k = [radius * np.sin(theta) * np.cos(phi), radius * np.sin(theta) * np.sin(phi), radius * np.cos(theta)]
            klist = klist + [k]
    klist = klist + [[0, 0, -radius]]
    klist = np.array(klist, dtype='float')
    # print(np.shape(klist),klist)
    klist = klist + weylpt
    [m, n] = np.shape(klist)

    # fig = plt.figure()
    #
    # ax = fig.add_subplot(111,projection = '3d')
    # c=['b']
    # for i in range(m):
    #     cax = ax.scatter(klist[i,0],klist[i,1],klist[i,2],c=c)
    # plt.show()

    curv = []
    for i in range(m):
        curv = curv + [curvature(klist[i], band_num, krel=False)]
    curv = np.array(curv, dtype='float')

    # print(curv)

    @np.vectorize
    def sign_log(x):
        if x < 0:
            return -np.log(-x)
        else:
            return np.log(x)

    fig = plt.figure()
    cmhot = plt.cm.get_cmap("hot")
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(klist[:, 0], klist[:, 1], klist[:, 2], curv[:, 0], curv[:, 1], curv[:, 2], length=0.001)
    u = sign_log(curv)
    # ax.scatter(k[:,0],k[:,1],k[:,2],c=plt.cm.get_cmap('hot')(np.sqrt((u[:,0]**2+u[:,1]**2+u[:,2]**2))), depthshade=False)
    ax.scatter(klist[:, 0], klist[:, 1], klist[:, 2], c=LA.norm(curv, axis=1), cmap=cmhot, depthshade=False)
    plt.show()

    return klist


def chern(kpath, band_num, interval=0.01, krel=True):
    kpath = np.array(kpath, dtype='float')
    if krel == True:
        kpath = kpoint_scale(kpath)
    # print(kpath)
    [m, n] = np.shape(kpath)
    klist = [kpath[0]]
    for i in range(m - 1):
        distance = LA.norm(kpath[i + 1] - kpath[i])
        num = np.ceil(distance / interval)
        klist = klist + list(interpkpt(kpath[i:i + 2], num)[1:])
    klist = np.array(klist, dtype='float')

    # klist = kpoint_reverse_scale(klist)
    # print(np.shape(klist))
    # print(kpoint_reverse_scale( klist))

    [x, y] = np.shape(klist)
    # print(klist[0],klist[x-1])
    hamk = fourier(rpt, klist, hamr, hami, krel=False)
    w, v = LA.eig(hamk)

    # print(w)
    # print(np.argsort(w))

    overlap = []
    for i in range(x - 1):
        # bandn = np.where(np.argsort(w[i,:])==band_num-1)
        overlap = overlap + [
            np.vdot(v[i, :, np.argsort(w[i, :])[band_num - 1]], v[i + 1, :, np.argsort(w[i + 1, :])[band_num - 1]])]
    #    overlap = overlap + [np.vdot(v[x-1,:,np.argsort(w[x-1,:])[band_num-1]],v[0,:,np.argsort(w[0,:])[band_num-1]])]
    # print(i)
    print(x, len(overlap))
    # print(overlap)

    temp = 0
    for i in range(len(overlap)):
        temp = temp + np.imag(np.log(overlap[i]))
    chern_num = temp  # np.imag(np.log(temp))
    print(chern_num)  # ,np.log(temp))

    # overlap = 1
    # for i in range(x-1):
    #     overlap = overlap * np.vdot(v[i,:,np.argsort(w[i,:])[band_num-1]], v[i+1,:,np.argsort(w[i+1,:])[band_num-1]])
    # overlap = overlap * np.vdot(v[i,:,np.argsort(w[i,:])[band_num-1]],v[0,:,np.argsort(w[0,:])[band_num-1]])
    # chern_num = np.imag(np.log(overlap))
    # print(chern_num)

    return temp


def flux(kplane, band_num, interval=0.001, krel=True):
    kplane = np.array(kplane, dtype='float')
    if krel == True:
        kplane = kpoint_scale(kplane)

    [m, n] = np.shape(kplane)
    if m != 3:
        print('ERROR: kplane input has to be origin, a-vector, b-vector')

    disa = LA.norm(kplane[1])
    disb = LA.norm(kplane[2])
    m = int(np.ceil(disa / interval))
    n = int(np.ceil(disb / interval))
    da = kplane[1] / m
    db = kplane[2] / n
    flux = 0
    for i in range(m):
        for j in range(n):
            klist = [kplane[0] + da * i + db * j, kplane[0] + da * (i + 1) + db * j,
                     kplane[0] + da * (i + 1) + db * (j + 1), kplane[0] + da * i + db * (j + 1)]
            # kplane[0] + da * i + db * j]
            klist = np.array(klist, dtype='float')
            print(klist)

            hamk = fourier(rpt, klist, hamr, hami, krel=False)
            w, v = LA.eig(hamk)
            overlap = np.vdot(v[0, :, np.argsort(w[0, :])[band_num - 1]], v[1, :, np.argsort(w[1, :])[band_num - 1]]) * \
                      np.vdot(v[1, :, np.argsort(w[1, :])[band_num - 1]], v[2, :, np.argsort(w[2, :])[band_num - 1]]) * \
                      np.vdot(v[2, :, np.argsort(w[2, :])[band_num - 1]], v[3, :, np.argsort(w[3, :])[band_num - 1]]) * \
                      np.vdot(v[3, :, np.argsort(w[3, :])[band_num - 1]], v[0, :, np.argsort(w[0, :])[band_num - 1]])
            # np.vdot(v[4, :, np.argsort(w[4, :])[band_num-1]], v[0, :, np.argsort(w[0, :])[band_num-1]])
            phase = np.imag(np.log(overlap))
            flux = flux + phase

    return flux


def define(kpath, band_num, interval=0.002, krel=True):
    kpath = np.array(kpath, dtype='float')
    if krel == True:
        kpath = kpoint_scale(kpath)
    # print(kpath)
    [m, n] = np.shape(kpath)
    klist = [kpath[0]]
    for i in range(m - 1):
        distance = LA.norm(kpath[i + 1] - kpath[i])
        num = np.ceil(distance / interval)
        klist = klist + list(interpkpt(kpath[i:i + 2], num)[1:])
    klist = np.array(klist, dtype='float')

    # klist = kpoint_reverse_scale(klist)
    # print(np.shape(klist))
    # print(kpoint_reverse_scale( klist))

    [x, y] = np.shape(klist)
    # print(klist[0],klist[x-1])
    hamk = fourier(rpt, klist, hamr, hami, krel=False)
    w, v = LA.eig(hamk)

    # print(w)
    # print(np.argsort(w))

    integral = []
    for i in range(x - 1):
        # bandn = np.where(np.argsort(w[i,:])==band_num-1)
        integral = integral + [
            np.vdot(v[i, :, np.argsort(w[i, :])[band_num - 1]], v[i + 1, :, np.argsort(w[i + 1, :])[band_num - 1]]) - 1]
    #    overlap = overlap + [np.vdot(v[x-1,:,np.argsort(w[x-1,:])[band_num-1]],v[0,:,np.argsort(w[0,:])[band_num-1]])]
    # print(i)
    print(x, len(integral))
    print(integral)

    temp = 0
    for i in range(len(integral)):
        temp = temp + integral[i]
    chern_num = np.imag(temp)  # np.imag(np.log(temp))
    print(chern_num)  # ,np.log(temp))

    # overlap = 1
    # for i in range(x-1):
    #     overlap = overlap * np.vdot(v[i,:,np.argsort(w[i,:])[band_num-1]], v[i+1,:,np.argsort(w[i+1,:])[band_num-1]])
    # overlap = overlap * np.vdot(v[i,:,np.argsort(w[i,:])[band_num-1]],v[0,:,np.argsort(w[0,:])[band_num-1]])
    # chern_num = np.imag(np.log(overlap))
    # print(chern_num)

    return chern_num


def interpkpt(klist, num):  # num must be a number or a array
    num = num + 1
    [m, n] = np.shape(klist)
    # print(m)
    kpt = []
    if m == 1:
        print('Please type in at least 2 kpts')
    n = []
    if not np.shape(num):
        for i in range(m - 1):
            n = n + [num]
    else:
        n = num
    kpt = list([klist[0, :]])
    for i in range(m - 1):
        kpt = kpt + list(np.transpose(
                [np.linspace(klist[i, 0], klist[i + 1, 0], n[i]), np.linspace(klist[i, 1], klist[i + 1, 1], n[i]),
                 np.linspace(klist[i, 2], klist[i + 1, 2], n[i])]))[1:]
        # print(type(kpt))
    kpt = np.array(kpt, dtype='float')
    # print(np.shape(kpt))
    # print(kpt)
    return kpt


file = open('wannier90_hr.dat', 'r')
band = range(88)
[weight, rpt, hamr, hami] = read_hr(file, band)

# b=[]
# klist = np.array([[0.2193783999  ,    0.4564836181E-01  , 0.000000000],[0.2201012282 ,     0.3840897403E-01  , 0.000000000]])
# print(kpoint_reverse_scale(klist))

# klist = np.array([[0.1214,0.0454,0],[0.1218, 0.0382, 0]])
# hamk = fourier(rpt,klist,hamr,hami)
# w,v = LA.eig(hamk)
# print(np.vdot(v[0,:,72],v[0,:,72]))

# print(w)
# print(np.shape(v))
# print(np.sort(np.real(w)))
# print(np.argsort(np.real(w)))

# curv = curvature(np.array([0.1214,0.0454,0]),72,b=0.002)
# print(curv)

# klist = hedgehog([0.1214,0.0454,0],72)
# [  1.22059097e-01   3.83651914e-02   2.62165931e-05]
# [  1.21597736e-01   4.60043541e-02   4.36862950e-05]
klist = hedgehog([1.21597736e-01, 4.60043541e-02, 4.36862950e-05], 72)
file.close()
# kplane = [[0.22037842,0.04664838,0.001],[0, -0.002,0],[0,0,-0.002]]

# flux = flux(kplane, 72, interval = 0.002, krel = False)
# print(flux)

# klist = np.array([[0,0,0],[1,1,1]])
# kpt = interpkpt(klist,1)
# print(kpt)
# scan_kx(20)

# kpath= np.array([[1,-1,-1],[1,-1,1],[1,1,1],[1,1,-1],[1,-1,-1]])/2
# kpath = [[1,1,0.5],[0.5,1,0.5],[0.5,0.0444,0.5],[0,0.0444,0.5],[0,0,0.5],[0,0,-0.5],[0,0.0444,-0.5],[0.5,0.0444,-0.5],[0.5,1,-0.5],[1,1,-0.5],[1,1,0.5]]
# kpath = [[1,0,0.5],[0.5,0,0.5],[0.5,0.041,0.5],[0,0.041,0.5],[0,0,0.5],[0,0,-0.5],[0,0.041,-0.5],[0.5,0.041,-0.5],[0.5,0,-0.5],[1,0,-0.5],[1,0,0.5]]
# kpath = [[0.5,0.0444,0.5],[0,0.0444,0.5],[0,0.0444,-0.5],[0.5,0.0444,-0.5]]
# kpath = [[ 0.22037842,  0.04664838,  0.001     ],[ 0.22037842,  0.04464838,  0.001     ],[ 0.22037842,  0.04464838, -0.001     ],[ 0.22037842,  0.04664838, -0.001     ],[ 0.22037842,  0.04664838,  0.001     ]]
# kpath = [[0.13,0.041,0.1],[0.11,0.041,0.1],[0.11,0.041,-0.1],[0.13,0.041,-0.1],[0.13,0.041,0.1]]
# chern(kpath,72,interval = 0.001,krel = False)
# define(kpath,72,interval = 0.001,krel = False)

'''
#[ 0.12499999  0.02499999  0.09999993]
k0 = [0.11,0.020,-0.001]#[0.11,0.035,-0.001]#[0.11,0.035,-0.001]#[0,0,-0.5]#[0.11,0.041,-0.001]#[0, 0, 0]  [ 0.21837842 , 0.04464838, -0.001     ]
k1 = [0.13,0.028, 0.001]#[0.13,0.041,+0.001]#[0.13,0.041, 0.001]#[0.5,0.041,0.5]#[0.13,0.046, 0.001]#[0.5, 0.041, 0.5]  [ 0.22037842 , 0.04664838  ,0.001     ]
def gen_kpath_2D(k0, k1):
    return [[k0[0], k0[1]], [k1[0], k0[1]], [k1[0], k1[1]], [k0[0], k1[1]], [k0[0], k0[1]]]
def insert_const(kpath, cnt, const_k):
    for point in kpath:
        point.insert(cnt, const_k)
    return kpath
def gen_kpath_3D(k0, k1):
    kpath = []
    kpath.append(insert_const(gen_kpath_2D([k0[0], k0[2]], [k1[0], k1[2]]), 1, k0[1]))
    kpath.append(insert_const(gen_kpath_2D([k1[0], k0[2]], [k0[0], k1[2]]), 1, k1[1]))
    kpath.append(insert_const(gen_kpath_2D([k1[1], k0[2]], [k0[1], k1[2]]), 0, k0[0]))
    kpath.append(insert_const(gen_kpath_2D([k0[1], k0[2]], [k1[1], k1[2]]), 0, k1[0]))
    kpath.append(insert_const(gen_kpath_2D([k0[0], k1[1]], [k1[0], k0[1]]), 2, k0[2]))
    kpath.append(insert_const(gen_kpath_2D([k0[0], k0[1]], [k1[0], k1[1]]), 2, k1[2]))
    return kpath
kpath_list = gen_kpath_3D(k0, k1)

#cn = 1
#for kpath in kpath_list:
#print(sum([chern(kpath, 72, interval = 0.01) for kpath in kpath_list]))
#print(sum([define(kpath, 72, interval = 0.01) for kpath in kpath_list]))
kpath_list = np.array(kpath_list)
print(sum([flux([kpath[0],kpath[1]-kpath[0],kpath[3]-kpath[0]], 72, interval = 0.002, krel = True) for kpath in kpath_list]))
'''
