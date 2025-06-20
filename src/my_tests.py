from src.classical_orthogonal_layer import OrthoLayer
import time
import torch

batch_size = 100
in_features = 30
out_features = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.randn(batch_size, in_features).to(device)
layer = OrthoLayer(in_features, out_features).to(device)

start = time.time()
for _ in range(100):
    y = layer(x)
end = time.time()
print(f"Average time per forward: {(end-start)/100:.6f}s")