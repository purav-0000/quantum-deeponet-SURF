import numpy as np
import os

# QISKIT_AER INSTEAD OF QISKIT.PROVIDERS.AER
from qiskit_aer import AerSimulator

# COMMENTING OUT QUANTUM_LAYER_INFINITE (?) AND IMPORTING QUANTUM_LAYER_IDEAL
# from quantum_layer_infinite import tomo_output
from src.quantum_layer_ideal import tomo_output

# MAKE PATHS ABSOLUTE
# Define input directory relative to current file
input_dir = os.path.join(os.path.dirname(__file__), "classical_training")

# Branch network weights
branch_hidden0_bias = np.loadtxt(os.path.join(input_dir, "branch.hidden_layers.0.bias.txt"))
branch_hidden0_thetas = np.loadtxt(os.path.join(input_dir, "branch.hidden_layers.0.thetas.txt"))
branch_hidden1_bias = np.loadtxt(os.path.join(input_dir, "branch.hidden_layers.1.bias.txt"))
branch_hidden1_thetas = np.loadtxt(os.path.join(input_dir, "branch.hidden_layers.1.thetas.txt"))
branch_hidden2_bias = np.loadtxt(os.path.join(input_dir, "branch.hidden_layers.2.bias.txt"))
branch_hidden2_thetas = np.loadtxt(os.path.join(input_dir, "branch.hidden_layers.2.thetas.txt"))
branch_hidden3_bias = np.loadtxt(os.path.join(input_dir, "branch.hidden_layers.3.bias.txt"))
branch_hidden3_thetas = np.loadtxt(os.path.join(input_dir, "branch.hidden_layers.3.thetas.txt"))
branch_hidden4_bias = np.loadtxt(os.path.join(input_dir, "branch.hidden_layers.4.bias.txt"))
branch_hidden4_thetas = np.loadtxt(os.path.join(input_dir, "branch.hidden_layers.4.thetas.txt"))
branch_output_bias = np.loadtxt(os.path.join(input_dir, "branch.output_layer.bias.txt"))
branch_output_weight = np.loadtxt(os.path.join(input_dir, "branch.output_layer.weight.txt"))

# Trunk network weights
trunk_hidden0_bias = np.loadtxt(os.path.join(input_dir, "trunk.hidden_layers.0.bias.txt"))
trunk_hidden0_thetas = np.loadtxt(os.path.join(input_dir, "trunk.hidden_layers.0.thetas.txt"))
trunk_hidden1_bias = np.loadtxt(os.path.join(input_dir, "trunk.hidden_layers.1.bias.txt"))
trunk_hidden1_thetas = np.loadtxt(os.path.join(input_dir, "trunk.hidden_layers.1.thetas.txt"))
trunk_hidden2_bias = np.loadtxt(os.path.join(input_dir, "trunk.hidden_layers.2.bias.txt"))
trunk_hidden2_thetas = np.loadtxt(os.path.join(input_dir, "trunk.hidden_layers.2.thetas.txt"))
trunk_hidden3_bias = np.loadtxt(os.path.join(input_dir, "trunk.hidden_layers.3.bias.txt"))
trunk_hidden3_thetas = np.loadtxt(os.path.join(input_dir, "trunk.hidden_layers.3.thetas.txt"))
trunk_hidden4_bias = np.loadtxt(os.path.join(input_dir, "trunk.hidden_layers.4.bias.txt"))
trunk_hidden4_thetas = np.loadtxt(os.path.join(input_dir, "trunk.hidden_layers.4.thetas.txt"))
trunk_output_bias = np.loadtxt(os.path.join(input_dir, "trunk.output_layer.bias.txt"))
trunk_output_weight = np.loadtxt(os.path.join(input_dir, "trunk.output_layer.weight.txt"))

# Final bias
b = np.loadtxt(os.path.join(input_dir, "b.txt"))

simulator = AerSimulator(device='CPU')

def silu(x):
    return x / (1 + np.exp(-x))

#load test data, make sure we use the same input transformation as training
# ADD ../../../data/burgers_data/ TO PATH
d1 = np.load(r'../../../data/burgers_data/Burgers_train.npz',allow_pickle=True)
x_train,y_train = (d1['X_train0'].astype(np.float32),d1['X_train1'].astype(np.float32)),d1['y_train'].astype(np.float32)
d2 = np.load(r'../../../data/burgers_data/Burgers_test.npz',allow_pickle=True)
x_test,y_test = (d2['X_test0'].astype(np.float32),d2['X_test1'].astype(np.float32)),d2['y_test'].astype(np.float32)

def periodic(x):
    return np.concatenate([np.cos(x[:,0:1]*2*np.pi),
                      np.sin(x[:,0:1]*2*np.pi),
                      np.cos(x[:,0:1]*4*np.pi),
                      np.sin(x[:,0:1]*4*np.pi),x[:,1:2]],axis=1)

trunk_min = np.min( np.stack((np.min(periodic(x_train[1]), axis=0),np.min(periodic(x_test[1]), axis=0)),axis=0),axis=0)
trunk_max = np.max( np.stack((np.max(periodic(x_train[1]), axis=0),np.max(periodic(x_test[1]), axis=0)),axis=0),axis=0)
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


x_train = (branch_transform(x_train[0]),trunk_transform(periodic(x_train[1])))
x_test = (branch_transform(x_test[0]),trunk_transform(periodic(x_test[1])))

branch_outputs, trunk_outputs = [], []  

for x_branch0 in x_test[0]:
    x_branch = x_branch0
    x_branch = silu(tomo_output(20+1,20, x_branch, branch_hidden0_thetas,simulator)+branch_hidden0_bias)
    
    x_branch = silu(tomo_output(20,20, x_branch, branch_hidden1_thetas,simulator)+branch_hidden1_bias) + \
        x_branch/np.linalg.norm(x_branch)
    
    x_branch = silu(tomo_output(20,20, x_branch, branch_hidden2_thetas,simulator)+branch_hidden2_bias) + \
        x_branch/np.linalg.norm(x_branch)

    x_branch = silu(tomo_output(20,20, x_branch, branch_hidden3_thetas,simulator)+branch_hidden3_bias) + \
        x_branch/np.linalg.norm(x_branch)
    
    x_branch = silu(tomo_output(20,20, x_branch, branch_hidden4_thetas,simulator)+branch_hidden4_bias) + \
        x_branch/np.linalg.norm(x_branch)
    
    x_branch = np.dot(x_branch,branch_output_weight.T) +branch_output_bias

    branch_outputs.append(x_branch.copy())


for x_trunk in x_test[1]:
    x_trunk = silu(tomo_output(6,20, x_trunk, trunk_hidden0_thetas,simulator)+trunk_hidden0_bias)
    
    x_trunk = silu(tomo_output(20,20, x_trunk, trunk_hidden1_thetas,simulator)+trunk_hidden1_bias) + \
        x_trunk/np.linalg.norm(x_trunk)
    
    x_trunk = silu(tomo_output(20,20, x_trunk, trunk_hidden2_thetas,simulator)+trunk_hidden2_bias) + \
        x_trunk/np.linalg.norm(x_trunk)
    
    x_trunk = silu(tomo_output(20,20, x_trunk, trunk_hidden3_thetas,simulator)+trunk_hidden3_bias) + \
        x_trunk/np.linalg.norm(x_trunk)
    
    x_trunk = silu(tomo_output(20,20, x_trunk, trunk_hidden4_thetas,simulator)+trunk_hidden4_bias) + \
        x_trunk/np.linalg.norm(x_trunk)

    x_trunk = np.dot(x_trunk,trunk_output_weight.T) +trunk_output_bias
    x_trunk = silu(x_trunk)

    trunk_outputs.append(x_trunk.copy())

x = np.einsum('bi,ni->bn',np.array(branch_outputs),np.array(trunk_outputs))+b
# COMMENT THE LINE BELOW OUT
# x = x * x_test[1].reshape(1,-1)


mean_l2_error = np.mean(np.linalg.norm(x-y_test,axis=1)/np.linalg.norm(y_test,axis=1))

np.savetxt(r'simulation_error.txt',[mean_l2_error])