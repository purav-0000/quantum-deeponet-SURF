from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
from qiskit import transpile

from src.quantum_layer_ideal import custom_tomo_fast
from src.utils.common import load_dataset

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


def evaluate_model(y_pred, y_true, verbose=False, save_dir=None, ensemble_dir=None, model_name=None):
    def save_evaluation_results(output_dir, y_pred, error, prefix=""):
        """Save evaluation outputs to disk with a timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

        np.savetxt(os.path.join(output_dir, f"{prefix}simulation_error.txt_" + timestamp), [error])
        np.savetxt(os.path.join(output_dir, f"{prefix}simulation_output.txt_" + timestamp), y_pred)

    error = np.mean(np.linalg.norm(y_pred - y_true, axis=1) / np.linalg.norm(y_true, axis=1))

    if verbose:
        print(f"Mean L2 error: {error:.6f}")

    if save_dir:
        save_evaluation_results(save_dir, y_pred, error)

    if ensemble_dir and model_name:
        prefix = f"simulation_{model_name}_"
        save_evaluation_results(ensemble_dir, y_pred, error, prefix=prefix)



    return error


def build_circuit(x_input0, n_in, n_out, W_gate, loader_special_gate, loader_inv_gate, simulator):
    x_input = x_input0.copy()
    x_input += (np.abs(x_input) < 1e-7) * 1e-7
    circ = custom_tomo_fast(n_in, n_out, x_input, W_gate, loader_special_gate, loader_inv_gate)
    circ.save_statevector('state')
    return transpile(circ, simulator)


def plot_pred(x_test, y_test, y_pred, save_path, x_test_plot, q=0.9, confidence=False):

    ensemble = False
    # Check if ensemble or single model
    if len(y_pred.shape) == 3:  # Ensemble will have 3-dimensional output (models, batch index, output)
        ensemble = True

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_path, timestamp)

    indices = np.random.choice(len(y_test), size=15, replace=False)
    fig, axs = plt.subplots(15, 1, figsize=(15, 50), sharex=True, sharey=True)

    # Select trunk inputs
    x_trunk = x_test[1][:, 0]
    for ax, idx in zip(axs, indices):

        # Select input function
        x_branch = x_test_plot[idx, :]
        ax.plot(x_trunk, x_branch, color='orange', alpha=0.8, label="Input")

        y = y_test[idx]
        ax.plot(x_trunk, y, 'r-', label="Ground Truth")

        # Check if ensembles or single model
        if ensemble:  # Ensemble
            samples = y_pred[:, idx, :]

            # Confidence interval
            mean_pred = samples.mean(axis=0)
            std_pred = samples.std(axis=0)

            ax.plot(x_trunk, mean_pred, 'b-', label="Prediction")
            ax.fill_between(x_trunk, mean_pred - q * std_pred, mean_pred + q * std_pred, color='blue', alpha=0.3,
                            label="2σ Interval")
        else:   # Single model
            ax.plot(x_trunk, y_pred[idx, :], 'b-', label="Prediction")

        ax.set_title(f"Sample {idx}")
        ax.grid(True)

    axs[0].legend()
    fig.suptitle("Prediction with 2σ Confidence Interval on 5 Samples")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path)
    plt.close()

