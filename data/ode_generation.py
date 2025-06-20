import os
os.environ['DDE_BACKEND'] = 'pytorch'
print(os.environ.get('DDE_BACKEND'))

# MY EDIT
os.makedirs('data_ode_simple', exist_ok=True)

import deepxde as dde
from deepxde.data.function_spaces import GRF
import numpy as np
import matplotlib.pyplot as plt

# LOADING BAR
from tqdm import tqdm

# PARALLELIZATION
from joblib import Parallel, delayed

# here gen_data_GRF can generate u and v with respect to different x

# solve pde using finite difference method
def compute_numerical_solution(x, v, N):
    h = x[1]-x[0]
    #Kx=b
    K = 1 * np.eye(N-1) - np.eye(N-1,k=-1)
    b = h* v[1:]
    u = np.linalg.solve(K,b)
    return np.concatenate(([0], u))


def generate_sample(M):
    """PARALLEL CODE"""
    # VARY LENGTH SCALE
    l_rand = np.random.uniform(0.2, 1.3)
    # VERY AMPLITUDE
    a_rand = np.random.uniform(0.5, 1.5)
    space = GRF(1, kernel='RBF', length_scale=l_rand, N=M, interp='cubic')
    x = np.ravel(space.x)
    v = np.ravel(a_rand * space.random(1))
    u = compute_numerical_solution(x, v, M)
    return v, u

def gen_data_GRF(M, Nv,Nu, a=1,l=1,train = 150, test = 100):
    """Generate functions"""
    # generate v and compute u in a dense grid with size M
    # x in [0,1] GRF.x.shape = [1, M]
    # l: lenth scale, a is the amplitude

    # INITIALIZE X
    space = GRF(1, kernel = 'RBF', length_scale = l, N = M, interp = 'cubic')
    x = np.ravel(space.x)

    # VERY SMALL NOISE
    noise_level = 1e-4

    # NO NEED FOR INITIALIZATION
    # v_full = []
    # u_full = []

    # COMMENT THIS OUT, USE PARALLEL CODE INSTEAD
    """
    for _ in tqdm(range(train)):

        v = np.ravel(a_rand * space.random(1)) # space.random(1) shape = [1, M], v.shape = [M,1]
        u = compute_numerical_solution(x,v,M) #u.shape = [M]
        v_full.append(v)
        u_full.append(u)
        # plt.figure()
        # plt.plot(x,v,label='v')
        # plt.plot(x,u,label='u')
        # plt.legned()
        # plt.show()
    """

    # PARALLEL LOOP
    results = Parallel(n_jobs=-1)(delayed(generate_sample)(M) for _ in tqdm(range(train)))
    v_full, u_full = zip(*results)
    v_full = np.array(v_full, dtype=np.float32)
    u_full = np.array(u_full, dtype=np.float32)

    np.savez_compressed('data_ode_simple/full_aligned_train.npz',X0 = v_full, X1 = x.reshape(-1,1) , y = u_full)

    # COMMENT OUT, INCREASE SAMPLE DIVERSITY
    """
    # down sampling and take N points
    index_v = [round(M/Nv)*i for i in range(Nv-1)]+[M-1]
    index_u = [round(M/Nu)*i for i in range(Nu-1)]+[M-1]
    """

    # INCREASE SAMPLE DIVERSITY
    index_v = np.sort(np.random.choice(M, Nv, replace=False))
    index_u = np.sort(np.random.choice(M, Nu, replace=False))

    v = v_full[:,index_v]
    xv = x[index_v]
    xu = x[index_u]
    u = u_full[:,index_u]

    # ADD VERY SMALL AMOUNT OF NOISE
    v += np.random.normal(0, noise_level, size=v.shape)
    u += np.random.normal(0, noise_level, size=u.shape)

    np.savez_compressed('data_ode_simple/picked_aligned_train.npz',X0 = v, X1 = xu.reshape(-1,1) , y = u,  X0_p = xv.reshape(-1,1))

    #test data
    # RAVEL INSIDE LOOP INSTEAD
    # x = np.ravel(space.x)
    v_full = []
    u_full = []

    # COMMENT THIS OUT, USE PARALLEL CODE INSTEAD
    """
    for _ in range(test):

        v = np.ravel(a_rand * space.random(1)) # space.random(1) shape = [1, M], v.shape = [M,1]
        u = compute_numerical_solution(x,v,M) #u.shape = [M]
        v_full.append(v)
        u_full.append(u)
        # plt.figure()
        # plt.plot(x,v,label='v')
        # plt.plot(x,u,label='u')
        # plt.legned()
        # plt.show()
    """

    # PARALLEL LOOP
    results = Parallel(n_jobs=-1)(delayed(generate_sample)(M) for _ in tqdm(range(test)))
    v_full, u_full = zip(*results)
    v_full = np.array(v_full, dtype=np.float32)
    u_full = np.array(u_full, dtype=np.float32)

    v_full = np.array(v_full,dtype=np.float32)
    u_full = np.array(u_full,dtype=np.float32)
    np.savez_compressed('data_ode_simple/full_aligned_test.npz',X0 = v_full, X1 = x.reshape(-1,1) , y = u_full)

    # COMMENT OUT, INCREASE SAMPLE DIVERSITY
    """
    # down sampling and take N points
    index_v = [round(M/Nv)*i for i in range(Nv-1)]+[M-1]
    index_u = [round(M/Nu)*i for i in range(Nu-1)]+[M-1]
    """

    # INCREASE SAMPLE DIVERSITY
    index_v = np.sort(np.random.choice(M, Nv, replace=False))
    index_u = np.sort(np.random.choice(M, Nu, replace=False))

    v = v_full[:,index_v]
    xv = x[index_v]
    xu = x[index_u]
    u = u_full[:,index_u]

    # ADD VERY SMALL AMOUNT OF NOISE
    v += np.random.normal(0, noise_level, size=v.shape)
    u += np.random.normal(0, noise_level, size=u.shape)

    np.savez_compressed('data_ode_simple/picked_aligned_test.npz',X0 = v, X1 = xu.reshape(-1,1) , y = u,  X0_p = xv.reshape(-1,1))

# CHANGE L HERE
if __name__ == '__main__':
    # gen_data_GRF(1000,Nv = 10,Nu = 30,l=1,train = 200, test = 100)

    # MORE DIVERSE TRAIN AND TEST SET
    gen_data_GRF(2000, Nv=15, Nu=40, l=1, train=100, test=50)