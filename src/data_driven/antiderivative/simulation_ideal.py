import numpy as np
# ADD OS IMPORT
import os

# SET PACKAGE TO QISKIT_AER INSTEAD OF QISKIT.PROVIDERS.AER
from qiskit import transpile
from qiskit_aer import AerSimulator

# REMOVE RELATIVE IMPORT
from src.quantum_layer_ideal import custom_tomo_fast, data_loader, tomo_output, tomo_output_fast, W

# PARALLELISM AND PROGRESS BAR
from joblib import Parallel, delayed
from tqdm import tqdm

# MAKE IMPORTS ABSOLUTE
# Define input directory relative to current file
# Load desired seed parameters
seed_number = 3570396786
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
simulator = AerSimulator(device='GPU')

def silu(x):
    return x / (1 + np.exp(-x))

#load test data, make sure we use the same input transformation as training

# ADD ../../../data/data_ode_simple/ TO PATH
d1 = np.load(r'data/data_ode_simple/picked_aligned_train.npz',allow_pickle=True)
x_train,y_train = (d1['X0'].astype(np.float32),d1['X1'].astype(np.float32)),d1['y'].astype(np.float32)

# ADD ../../../data/data_ode_simple/ TO PATH
d2 = np.load(r'data/data_ode_simple/picked_aligned_test.npz',allow_pickle=True)
x_test,y_test = (d2['X0'].astype(np.float32),d2['X1'].astype(np.float32)),d2['y'].astype(np.float32)

trunk_min = np.min( np.stack((np.min(x_train[1], axis=0),np.min(x_test[1], axis=0)),axis=0),axis=0)
trunk_max = np.max( np.stack((np.max(x_train[1], axis=0),np.max(x_test[1], axis=0)),axis=0),axis=0)
branch_min = np.min(np.stack((np.min(x_train[0], axis=0),np.min(x_test[0], axis=0)),axis=0),axis=0)
branch_max = np.max( np.stack((np.max(x_train[0], axis=0),np.max(x_test[0], axis=0)),axis=0),axis=0)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
def trunk_transform(x):
    d = x.shape[1]
    x = 2 * (x - trunk_min) / (trunk_max - trunk_min) - 1  # Rescale to [-1, 1]
    x = x / np.sqrt(d)
    x_d1 = np.sqrt(1 - np.sum(x**2, axis=1, keepdims=True))
    return np.concatenate((x, x_d1), axis=1)

def branch_transform(x):
    # For 1-dimensional input
    d = x.shape[1]
    x = 2 * (x - branch_min) / (branch_max - branch_min) - 1  # Rescale to [-1, 1]
    x = x / np.sqrt(d)
    x_d1 = np.sqrt(1 - np.sum(x**2, axis=1, keepdims=True))
    return np.concatenate((x, x_d1), axis=1)


x_train = (branch_transform(x_train[0]),trunk_transform(x_train[1]))
x_test = (branch_transform(x_test[0]),trunk_transform(x_test[1]))

# COMMENTED OUT, NO NEED FOR INITIALIZATION
# branch_outputs, trunk_outputs = [], []

# Setup (same as before)
n_in = 16
n_out = 20
num_qubits = max(n_in, n_out)
sqrt_norm = np.sqrt(num_qubits)

special_arr = np.full(num_qubits, 1 / sqrt_norm)
W_gate = W(n_in, n_out, branch_hidden0_thetas)
loader_special_gate = data_loader(special_arr)
loader_inv_gate = loader_special_gate.inverse()

# Define safe number of parallel workers
n_jobs = 1

# Helper function to build and transpile a single circuit
def build_circuit(x_branch0):
    x_branch = x_branch0.copy()
    x_branch += (np.abs(x_branch) < 1e-7) * 1e-7
    circ = custom_tomo_fast(n_in, n_out, x_branch, W_gate, loader_special_gate, loader_inv_gate)
    circ.save_statevector('state')
    return transpile(circ, simulator)

# Run parallel circuit construction
print("Building circuits (parallel)...")
circuits = Parallel(n_jobs=n_jobs, backend="loky")(
    delayed(build_circuit)(x) for x in tqdm(x_test[0], desc="Building circuits")
)

valid_inputs = list(x_test[0])  # save for later use

print("Running on GPU...")
job = simulator.run(circuits, shots=1)
results = job.result()

# Post-process statevectors
print("Post-processing...")

def process_state(idx, results, n_in, n_out, sqrt_norm,
                  branch_hidden0_bias, branch_output_weight, branch_output_bias):
    state = np.real(results.data(idx)['state'].data)

    output = []
    for i in range(n_out):
        pos = ['0'] * n_out
        pos[i] = '1'
        pos0 = ['0'] + ['0'] * (n_in - n_out) + pos
        pos1 = ['1'] + ['0'] * (n_in - n_out) + pos
        pos0 = ''.join(pos0)[::-1]
        pos1 = ''.join(pos1)[::-1]
        result0 = state[int(pos0, 2)]
        result1 = state[int(pos1, 2)]
        output.append(np.sqrt(np.maximum(n_in, n_out)) * (result0 ** 2 - result1 ** 2))
    output = np.array(output)

    x_branch = silu(np.array(output) + branch_hidden0_bias)
    return np.dot(x_branch, branch_output_weight.T) + branch_output_bias


print("Post-processing (parallel)...")
branch_outputs = Parallel(n_jobs=n_jobs, backend="loky")(
    delayed(process_state)(
        idx, results, n_in, n_out, sqrt_norm,
        branch_hidden0_bias, branch_output_weight, branch_output_bias
    )
    for idx in tqdm(range(len(circuits)), desc="Processing outputs")
)

branch_outputs = np.array(branch_outputs)


num_qubits = max(1+1, n_out)
sqrt_norm = np.sqrt(num_qubits)

special_arr_2 = np.full(num_qubits, 1 / sqrt_norm)
W_gate_2 = W(1+1, n_out, trunk_hidden0_thetas)
loader_special_gate_2 = data_loader(special_arr)
loader_inv_gate_2 = loader_special_gate.inverse()


def build_circuit_trunk(x_trunk0):
    x_trunk = x_trunk0.copy()
    x_trunk += (np.abs(x_trunk) < 1e-7) * 1e-7
    circ = custom_tomo_fast(1+1, n_out, x_trunk, W_gate_2, loader_special_gate_2, loader_inv_gate_2)
    circ.save_statevector('state')
    return transpile(circ, simulator)

# Run parallel circuit construction
print("Building circuits (parallel)...")
circuits = Parallel(n_jobs=n_jobs, backend="loky")(
    delayed(build_circuit_trunk)(x) for x in tqdm(x_test[1], desc="Building circuits")
)

def process_state_trunk(idx, results, n_in, n_out, sqrt_norm,
                  trunk_hidden0_bias, trunk_output_weight, trunk_output_bias):
    state = np.real(results.data(idx)['state'].data)

    output = []
    for i in range(n_out):
        pos = ['0'] * n_out
        pos[i] = '1'
        pos0 = ['0'] + ['0'] * (n_in - n_out) + pos
        pos1 = ['1'] + ['0'] * (n_in - n_out) + pos
        pos0 = ''.join(pos0)[::-1]
        pos1 = ''.join(pos1)[::-1]
        result0 = state[int(pos0, 2)]
        result1 = state[int(pos1, 2)]
        output.append(np.sqrt(np.maximum(n_in, n_out)) * (result0 ** 2 - result1 ** 2))
    output = np.array(output)

    x_branch = silu(np.array(output) + trunk_hidden0_bias)
    return np.dot(x_branch, trunk_output_weight.T) + trunk_output_bias


print("Post-processing (parallel)...")
trunk_outputs = Parallel(n_jobs=n_jobs, backend="loky")(
    delayed(process_state_trunk)(
        idx, results, 1+1, n_out, sqrt_norm,
        trunk_hidden0_bias, trunk_output_weight, trunk_output_bias
    )
    for idx in tqdm(range(len(circuits)), desc="Processing outputs")
)

trunk_outputs = np.array(trunk_outputs)


x = (np.einsum('bi,ni->bn',branch_outputs,trunk_outputs)+b)

# COMMENTED THIS OUT
# x = x * x_test[1].reshape(1,-1)

mean_l2_error = np.mean(np.linalg.norm(x-y_test,axis=1)/np.linalg.norm(y_test,axis=1))

np.savetxt(f'simulation_error_seed{seed_number}.txt', [mean_l2_error])

# SAVE OUTPUT
np.savetxt(f'simulation_output_seed{seed_number}.txt', x)
