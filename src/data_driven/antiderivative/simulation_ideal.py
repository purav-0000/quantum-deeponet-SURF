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
d1 = np.load(r'../../../data/data_ode_simple/picked_aligned_train.npz',allow_pickle=True)
x_train,y_train = (d1['X0'].astype(np.float32),d1['X1'].astype(np.float32)),d1['y'].astype(np.float32)

# ADD ../../../data/data_ode_simple/ TO PATH
d2 = np.load(r'../../../data/data_ode_simple/picked_aligned_test.npz',allow_pickle=True)
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
branch_outputs, trunk_outputs = [], []

n_in = 16
n_out = 20
num_qubits = max(n_in, n_out)

special_arr = np.full(num_qubits, 1 / np.sqrt(num_qubits))
W_gate = W(n_in, n_out, branch_hidden0_thetas)
loader_special_gate = data_loader(special_arr)
loader_inv_gate = loader_special_gate.inverse()

sqrt_norm = np.sqrt(num_qubits)
circuits = []
valid_inputs = []


for x_branch0 in tqdm(x_test[0], desc="Building circuits"):
    x_branch = x_branch0.copy()
    x_branch += (np.abs(x_branch) < 1e-7) * 1e-7

    # Construct and store the circuit
    circ = custom_tomo_fast(n_in, n_out, x_branch, W_gate, loader_special_gate, loader_inv_gate)
    circ.save_statevector('state')
    circuits.append(transpile(circ, simulator))  # Transpile ahead of time
    valid_inputs.append(x_branch0)  # Save original input for bias layer use

# Run all at once
print("Running on GPU...")
job = simulator.run(circuits, shots=1)
results = job.result()
states = results.data()

# Post-process statevectors
print("Post-processing...")
branch_outputs = []

for idx, state_data in enumerate(states):
    state = np.real(state_data['state'].data)
    output = []
    for i in range(n_out):
        pos = ['0'] * n_out
        pos[i] = '1'
        pos0 = ['0'] + ['0'] * (n_in - n_out) + pos
        pos1 = ['1'] + ['0'] * (n_in - n_out) + pos
        result0 = state[int(''.join(pos0)[::-1], 2)]
        result1 = state[int(''.join(pos1)[::-1], 2)]
        output.append(sqrt_norm * (result0 ** 2 - result1 ** 2))

    # Now do classical NN post-processing
    x_branch = silu(np.array(output) + branch_hidden0_bias)
    x_branch = np.dot(x_branch, branch_output_weight.T) + branch_output_bias
    branch_outputs.append(x_branch)

branch_outputs = np.array(branch_outputs)

"""
for x_branch0 in tqdm(x_test[0], desc="XD"):
    x_branch = x_branch0

    # LAYER SIZE HERE
    x_branch = silu(
        tomo_output_fast(n_in, n_out, x_branch, simulator,
                         W_gate, loader_special_gate, loader_inv_gate)
        + branch_hidden0_bias
    )
    x_branch = np.dot(x_branch, branch_output_weight.T) + branch_output_bias

    branch_outputs.append(x_branch.copy())
"""
"""
for x_branch0 in tqdm(x_test[0]):
    x_branch = x_branch0

    # LAYER SIZE HERE
    x_branch = silu(tomo_output(15+1,20, x_branch, branch_hidden0_thetas,simulator)+branch_hidden0_bias)
    x_branch = np.dot(x_branch,branch_output_weight.T) +branch_output_bias

    branch_outputs.append(x_branch.copy())


# PARALLEL BRANCH OUTPUT WITH PROGRESS BAR
# SET DIM TO 15+1
branch_outputs = Parallel(n_jobs=-1)(
    delayed(lambda xb: (np.dot(
        silu(tomo_output(15+1, 20, xb, branch_hidden0_thetas, simulator) + branch_hidden0_bias),
        branch_output_weight.T
    ) + branch_output_bias).copy())(x_branch)
    for x_branch in tqdm(x_test[0], desc="Computing Branch Outputs")
)
"""

"""
for x_trunk in tqdm(x_test[1]):

    # LAYER SIZE HERE
    x_trunk = silu(tomo_output(1+1,20, x_trunk, trunk_hidden0_thetas,simulator)+trunk_hidden0_bias)
    x_trunk = np.dot(x_trunk,trunk_output_weight.T) +trunk_output_bias
    x_trunk = silu(x_trunk)

    trunk_outputs.append(x_trunk.copy())


trunk_outputs = Parallel(n_jobs=-1)(
    delayed(lambda xt: silu(
        np.dot(
            silu(tomo_output(1+1, 20, xt, trunk_hidden0_thetas, simulator) + trunk_hidden0_bias),
            trunk_output_weight.T
        ) + trunk_output_bias
    ).copy())(x_trunk)
    for x_trunk in tqdm(x_test[1], desc="Computing Trunk Outputs")
)
"""
x = (np.einsum('bi,ni->bn',np.array(branch_outputs),np.array(trunk_outputs))+b)

# COMMENTED THIS OUT
# x = x * x_test[1].reshape(1,-1)

mean_l2_error = np.mean(np.linalg.norm(x-y_test,axis=1)/np.linalg.norm(y_test,axis=1))

np.savetxt(f'simulation_error_seed{seed_number}.txt', [mean_l2_error])

# SAVE OUTPUT
np.savetxt(f'simulation_output_seed{seed_number}.txt', x)
