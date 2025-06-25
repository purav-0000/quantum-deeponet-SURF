import os
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm

from qiskit import transpile
from qiskit_aer import AerSimulator

from src.quantum_layer_ideal import custom_tomo_fast, data_loader, W

# --- Configuration ---
SEED = 777
N_JOBS = 1
SIMULATOR = AerSimulator(device='CPU')
DATA_DIR = "data/data_ode_simple"
INPUT_DIR = os.path.join(os.path.dirname(__file__), f"classical_training_seed{SEED}")

# --- Utility Functions ---
def silu(x):
    return x / (1 + np.exp(-x))


def load_weights(directory):
    return {
        "branch_hidden0_bias": np.loadtxt(os.path.join(directory, "branch.hidden_layers.0.bias.txt")),
        "branch_hidden0_thetas": np.loadtxt(os.path.join(directory, "branch.hidden_layers.0.thetas.txt")),
        "branch_output_bias": np.loadtxt(os.path.join(directory, "branch.output_layer.bias.txt")),
        "branch_output_weight": np.loadtxt(os.path.join(directory, "branch.output_layer.weight.txt")),
        "trunk_hidden0_bias": np.loadtxt(os.path.join(directory, "trunk.hidden_layers.0.bias.txt")),
        "trunk_hidden0_thetas": np.loadtxt(os.path.join(directory, "trunk.hidden_layers.0.thetas.txt")),
        "trunk_output_bias": np.loadtxt(os.path.join(directory, "trunk.output_layer.bias.txt")),
        "trunk_output_weight": np.loadtxt(os.path.join(directory, "trunk.output_layer.weight.txt")),

        "b": np.loadtxt(os.path.join(directory, "b.txt"))
    }


def load_dataset():
    train = np.load(os.path.join(DATA_DIR, 'picked_aligned_train.npz'), allow_pickle=True)
    test = np.load(os.path.join(DATA_DIR, 'picked_aligned_test.npz'), allow_pickle=True)
    return (train['X0'].astype(np.float32), train['X1'].astype(np.float32)), train['y'].astype(np.float32), \
           (test['X0'].astype(np.float32), test['X1'].astype(np.float32)), test['y'].astype(np.float32)


def normalize_bounds(x_train, x_test):
    return {
        "trunk_min": np.min(np.stack((np.min(x_train[1], axis=0), np.min(x_test[1], axis=0)), axis=0), axis=0),
        "trunk_max": np.max(np.stack((np.max(x_train[1], axis=0), np.max(x_test[1], axis=0)), axis=0), axis=0),
        "branch_min": np.min(np.stack((np.min(x_train[0], axis=0), np.min(x_test[0], axis=0)), axis=0), axis=0),
        "branch_max": np.max(np.stack((np.max(x_train[0], axis=0), np.max(x_test[0], axis=0)), axis=0), axis=0),
    }


def transform_input(x, min_val, max_val):
    d = x.shape[1]
    x = 2 * (x - min_val) / (max_val - min_val) - 1
    x = x / np.sqrt(d)
    x_d1 = np.sqrt(1 - np.sum(x**2, axis=1, keepdims=True))
    return np.concatenate((x, x_d1), axis=1)


def build_circuit(x_input0, n_in, n_out, W_gate, loader_special_gate, loader_inv_gate):
    x_input = x_input0.copy()
    x_input += (np.abs(x_input) < 1e-7) * 1e-7
    circ = custom_tomo_fast(n_in, n_out, x_input, W_gate, loader_special_gate, loader_inv_gate)
    circ.save_statevector('state')
    return transpile(circ, SIMULATOR)


def process_state(idx, results, n_in, n_out, hidden0_bias, output_weight, output_bias, trunk=False, final_layer=False):
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
        output.append(np.sqrt(max(n_in, n_out)) * (result0 ** 2 - result1 ** 2))

    ret_val = silu(np.array(output) + hidden0_bias)

    if final_layer:
        ret_val = np.dot(ret_val, output_weight.T) + output_bias
        return silu(ret_val) if trunk else ret_val
    else:
        return ret_val


def run_quantum_layer(inputs, n_in, n_out, hidden0_bias, output_weight, output_bias, thetas, simulator, batch_size=None, trunk=False, final_layer=False, n_jobs=-1):
    sqrt_norm = np.sqrt(max(n_in, n_out))
    W_gate = W(n_in, n_out, thetas)
    loader = data_loader(np.full(max(n_in, n_out), 1 / sqrt_norm))
    loader_inv = loader.inverse()

    if batch_size is None:
        # Run all at once
        circuits = Parallel(n_jobs=n_jobs)(
            delayed(build_circuit)(x, n_in, n_out, W_gate, loader, loader_inv)
            for x in tqdm(inputs, desc="Building circuits")
        )
        job = simulator.run(circuits, shots=1)
        results = job.result()
        return np.array(Parallel(n_jobs=n_jobs)(
            delayed(process_state)(i, results, n_in, n_out,
                                   hidden0_bias, output_weight, output_bias, trunk, final_layer)
            for i in tqdm(range(len(circuits)), desc="Post-processing")
        ))
    else:
        # Use batching
        outputs = []
        for i in tqdm(range(0, len(inputs), batch_size), desc="Running in batches"):
            batch = inputs[i:i + batch_size]
            circuits = Parallel(n_jobs=n_jobs)(
                delayed(build_circuit)(x, n_in, n_out, W_gate, loader, loader_inv)
                for x in batch
            )
            job = simulator.run(circuits, shots=1)
            results = job.result()
            outputs.extend(Parallel(n_jobs=n_jobs)(
                delayed(process_state)(j, results, n_in, n_out,
                                       hidden0_bias, output_weight, output_bias, trunk, final_layer)
                for j in range(len(circuits))
            ))
        return np.array(outputs)


def evaluate_model(branch_out, trunk_out, b, y_true, seed):
    x = (np.einsum('bi,ni->bn', branch_out, np.array(trunk_out)) + b)
    mean_l2_error = np.mean(np.linalg.norm(x - y_true, axis=1) / np.linalg.norm(y_true, axis=1))
    output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    np.savetxt(os.path.join(output_dir, f'simulation_error_seed{seed}.txt'), [mean_l2_error])
    np.savetxt(os.path.join(output_dir, f'simulation_output_seed{seed}.txt'), x)
    print(f"Mean L2 error: {mean_l2_error:.6f}")

# --- Main Entry ---
def main():
    weights = load_weights(INPUT_DIR)
    x_train, y_train, x_test, y_test = load_dataset()
    bounds = normalize_bounds(x_train, x_test)

    x_train = (transform_input(x_train[0], bounds["branch_min"], bounds["branch_max"]),
               transform_input(x_train[1], bounds["trunk_min"], bounds["trunk_max"]))
    x_test = (transform_input(x_test[0], bounds["branch_min"], bounds["branch_max"]),
              transform_input(x_test[1], bounds["trunk_min"], bounds["trunk_max"]))

    branch_outputs = run_quantum_layer(
        x_test[0], 11, 10,
        weights["branch_hidden0_bias"],
        weights["branch_output_weight"],
        weights["branch_output_bias"],
        weights["branch_hidden0_thetas"],
        simulator=SIMULATOR,
        final_layer=True,
        batch_size=None
    )

    trunk_outputs = run_quantum_layer(
        x_test[1], 2, 10,
        weights["trunk_hidden0_bias"],
        weights["trunk_output_weight"],
        weights["trunk_output_bias"],
        weights["trunk_hidden0_thetas"],
        simulator=SIMULATOR,
        batch_size=None,
        final_layer=True,
        trunk=True
    )

    evaluate_model(branch_outputs, trunk_outputs, weights["b"], y_test, SEED)


if __name__ == "__main__":
    main()