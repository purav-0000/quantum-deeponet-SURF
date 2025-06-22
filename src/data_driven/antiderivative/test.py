import numpy as np
# ADD OS IMPORT
import os

# SET PACKAGE TO QISKIT_AER INSTEAD OF QISKIT.PROVIDERS.AER
from qiskit import transpile
from qiskit_aer import AerSimulator

# REMOVE RELATIVE IMPORT
from src.quantum_layer_ideal import tomo_output

# PARALLELISM AND PROGRESS BAR
from joblib import Parallel, delayed
from tqdm import tqdm

# MAKE IMPORTS ABSOLUTE
# Define input directory relative to current file
# Load desired seed parameters
seed_number = 420
input_dir = os.path.join(os.path.dirname(__file__), f"classical_training_seed{seed_number}")

branch_hidden0_bias = np.loadtxt(os.path.join(input_dir, "branch.hidden_layers.0.bias.txt"))
branch_hidden0_thetas = np.loadtxt(os.path.join(input_dir, "branch.hidden_layers.0.thetas.txt"))
branch_output_bias = np.loadtxt(os.path.join(input_dir, "branch.output_layer.bias.txt"))
branch_output_weight = np.loadtxt(os.path.join(input_dir, "branch.output_layer.weight.txt"))

trunk_hidden0_bias = np.loadtxt(os.path.join(input_dir, "trunk.hidden_layers.0.bias.txt"))
trunk_hidden0_thetas = np.loadtxt(os.path.join(input_dir, "trunk.hidden_layers.0.thetas.txt"))
trunk_output_bias = np.loadtxt(os.path.join(input_dir, "trunk.output_layer.bias.txt"))
trunk_output_weight = np.loadtxt(os.path.join(input_dir, "trunk.output_layer.weight.txt"))
b = np.loadtxt(os.path.join(input_dir, "b.txt"))

# SET SIMULATOR TO CPU
simulator = AerSimulator(device='CPU')


def silu(x):
    return x / (1 + np.exp(-x))


# load test data, make sure we use the same input transformation as training

# ADD ../../../data/data_ode_simple/ TO PATH
d1 = np.load(r'../../../data/data_ode_simple/picked_aligned_train.npz', allow_pickle=True)
x_train, y_train = (d1['X0'].astype(np.float32), d1['X1'].astype(np.float32)), d1['y'].astype(np.float32)

# ADD ../../../data/data_ode_simple/ TO PATH
d2 = np.load(r'../../../data/data_ode_simple/picked_aligned_test.npz', allow_pickle=True)
x_test, y_test = (d2['X0'].astype(np.float32), d2['X1'].astype(np.float32)), d2['y'].astype(np.float32)

trunk_min = np.min(np.stack((np.min(x_train[1], axis=0), np.min(x_test[1], axis=0)), axis=0), axis=0)
trunk_max = np.max(np.stack((np.max(x_train[1], axis=0), np.max(x_test[1], axis=0)), axis=0), axis=0)
branch_min = np.min(np.stack((np.min(x_train[0], axis=0), np.min(x_test[0], axis=0)), axis=0), axis=0)
branch_max = np.max(np.stack((np.max(x_train[0], axis=0), np.max(x_test[0], axis=0)), axis=0), axis=0)


def trunk_transform(x):
    d = x.shape[1]
    x = 2 * (x - trunk_min) / (trunk_max - trunk_min) - 1  # Rescale to [-1, 1]
    x = x / np.sqrt(d)
    x_d1 = np.sqrt(1 - np.sum(x ** 2, axis=1, keepdims=True))
    return np.concatenate((x, x_d1), axis=1)


def branch_transform(x):
    # For 1-dimensional input
    d = x.shape[1]
    x = 2 * (x - branch_min) / (branch_max - branch_min) - 1  # Rescale to [-1, 1]
    x = x / np.sqrt(d)
    x_d1 = np.sqrt(1 - np.sum(x ** 2, axis=1, keepdims=True))
    return np.concatenate((x, x_d1), axis=1)


x_train = (branch_transform(x_train[0]), trunk_transform(x_train[1]))
x_test = (branch_transform(x_test[0]), trunk_transform(x_test[1]))

# COMMENTED OUT, NO NEED FOR INITIALIZATION
branch_outputs, trunk_outputs = [], []

for x_branch0 in tqdm(x_test[0]):
    x_branch = x_branch0

    # LAYER SIZE HERE
    x_branch = silu(tomo_output(10 + 1, 10, x_branch, branch_hidden0_thetas, simulator) + branch_hidden0_bias)
    x_branch = np.dot(x_branch, branch_output_weight.T) + branch_output_bias

    branch_outputs.append(x_branch.copy())

for x_trunk in tqdm(x_test[1]):
    # LAYER SIZE HERE
    x_trunk = silu(tomo_output(1 + 1, 10, x_trunk, trunk_hidden0_thetas, simulator) + trunk_hidden0_bias)
    x_trunk = np.dot(x_trunk, trunk_output_weight.T) + trunk_output_bias
    x_trunk = silu(x_trunk)

    trunk_outputs.append(x_trunk.copy())

x = (np.einsum('bi,ni->bn', np.array(branch_outputs), np.array(trunk_outputs)) + b)

# COMMENTED THIS OUT
# x = x * x_test[1].reshape(1,-1)

mean_l2_error = np.mean(np.linalg.norm(x - y_test, axis=1) / np.linalg.norm(y_test, axis=1))

np.savetxt(f'simulation_error_seed{seed_number}.txt', [mean_l2_error])

# SAVE OUTPUT
np.savetxt(f'simulation_output_seed{seed_number}.txt', x)
