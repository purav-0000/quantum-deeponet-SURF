import numpy as np
import deepxde as dde

# MAKE DIR BEFORE SAVING
import os

# SET SEEDS OF THESE LIBRARIES
import random
import torch

# REMOVING RELATIVE IMPORT
from src.classical_orthogonal_deeponet import OrthoONetCartesianProd

# RANDOM SEED FOR INITIALIZATION
import secrets
seed = secrets.randbits(32)
print(f"Random hardware-backed seed: {seed}")

dde.config.set_random_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

print("Torch initial seed: ", torch.initial_seed())


def trunk_transform(x, trunk_min, trunk_max):
    d = x.shape[1]
    x = 2 * (x - trunk_min) / (trunk_max - trunk_min) - 1  # Rescale to [-1, 1]
    x = x / np.sqrt(d)
    x_d1 = np.sqrt(1 - np.sum(x**2, axis=1, keepdims=True))
    return np.concatenate((x, x_d1), axis=1)

def branch_transform(x, branch_min, branch_max):
    # For 2-dimensional input
    d = x.shape[1]
    x = 2 * (x - branch_min) / (branch_max - branch_min) - 1  # Rescale to [-1, 1]
    x = x / np.sqrt(d)
    x_d1 = np.sqrt(1 - np.sum(x**2, axis=1, keepdims=True))
    return np.concatenate((x, x_d1), axis=1)

def main():
    # ADD ../../../data/data_ode_simple/ TO PATH
    d1 = np.load(r'../../../data/data_ode_simple/picked_aligned_train.npz',allow_pickle=True)
    # x_train,y_train = (d1['X0'].astype(np.float32),d1['X1'].astype(np.float32)),d1['y'].astype(np.float32)
    # FOR BOOTSTRAPPING
    x_train_full, y_train_full = (d1['X0'].astype(np.float32), d1['X1'].astype(np.float32)), d1['y'].astype(np.float32)

    # ADD ../../../data/data_ode_simple/ TO PATH
    d2 = np.load(r'../../../data/data_ode_simple/picked_aligned_test.npz',allow_pickle=True)
    x_test,y_test = (d2['X0'].astype(np.float32),d2['X1'].astype(np.float32)),d2['y'].astype(np.float32)

    # PASS _FULL SUFFIX VARIABLES WHEN BOOTSTRAPPING
    trunk_min = np.min( np.stack((np.min(x_train_full[1], axis=0),np.min(x_test[1], axis=0)),axis=0),axis=0)
    trunk_max = np.max( np.stack((np.max(x_train_full[1], axis=0),np.max(x_test[1], axis=0)),axis=0),axis=0)
    branch_min = np.min(np.stack((np.min(x_train_full[0], axis=0),np.min(x_test[0], axis=0)),axis=0),axis=0)
    branch_max = np.max( np.stack((np.max(x_train_full[0], axis=0),np.max(x_test[0], axis=0)),axis=0),axis=0)

    # CREATE BOOTSTRAP SAMPLES
    n_train = y_train_full.shape[0]
    bootstrap_indices = np.random.choice(n_train, n_train, replace=True)
    print(len(np.setdiff1d(np.array(range(0, 1001)), bootstrap_indices)))
    x_train_bootstrap = (x_train_full[0][bootstrap_indices], x_train_full[1])
    y_train_bootstrap = y_train_full[bootstrap_indices]

    print((x_train_full[0] < 0).sum())

    # COMMENT OUT WHEN BOOTSTRAPPING
    """
    # MOVING TRUNK_MIN AND TRUNK_MAX INTO TRUNK_TRANSFORM AS ARGUMENTS
    x_train = (branch_transform(x_train[0], branch_min, branch_max),
               trunk_transform(x_train[1], trunk_min, trunk_max))
    """

    # TRANSFORM BOOTSTRAP SAMPLES
    x_train_bootstrap = (branch_transform(x_train_bootstrap[0], branch_min, branch_max),
                         trunk_transform(x_train_bootstrap[1], trunk_min, trunk_max))

    # MOVING BRANCH_MIN AND BRANCH_MAX INTO BRANCH_TRANSFORM AS ARGUMENTS
    # AND MOVING TRUNK_MIN AND TRUNK_MAX INTO TRUNK_TRANSFORM AS ARGUMENTS
    x_test = (branch_transform(x_test[0], branch_min, branch_max),
              trunk_transform(x_test[1], trunk_min, trunk_max))

    # PASS BOOTSTRAP SAMPLES
    # import pdb;pdb.set_trace()
    data = dde.data.TripleCartesianProd(
        X_train = x_train_bootstrap, y_train = y_train_bootstrap, X_test = x_test, y_test = y_test
    )
    #choose network
    # m = 10
    dim_x = 1

    # NEW m DIM
    m = 15

    # CHANGING ACTIVATION FUNCTION FROM RELU TO SILU
    net = OrthoONetCartesianProd(
        [m+1,20,20],
        [dim_x+1,20,20],
        'silu'
    )

    model = dde.Model(data,net)
    model.compile('adam',lr=0.001,metrics=['mean l2 relative error'])
    losshistory, train_state = model.train(iterations=3000,disregard_previous_best = True)

    # SET DIR NAME RELATED TO SEED
    save_dir = f'classical_training_seed{seed}'

    # MAKE DIRECTORY BEFORE SAVING
    os.makedirs(save_dir, exist_ok=True)

    dde.utils.external.save_loss_history(losshistory, f'{save_dir}/loss_history.txt')
    dde_model = model.net
    model.save(f'{save_dir}/model_checkpoint')

    for name, param in dde_model.named_parameters():
        np.savetxt(f'{save_dir}/{name}.txt', param.cpu().detach().numpy())


if __name__ == '__main__':
    main()