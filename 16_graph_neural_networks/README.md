# Module 16: Graph Machine Learning & Graph Neural Networks

This module implements spatial and spectral graph neural network architectures from scratch in NumPy, supporting node classification, link prediction, and neighborhood aggregation over non-Euclidean topologies.

---

## Directory Structure

```
16_graph_neural_networks/
├── 01_gnn_catalog.md         # Comprehensive mathematical theory & spectral derivations
├── README.md                 # Usage documentation & examples
├── src/
│   ├── __init__.py           # Public exports
│   ├── gcn.py                # Normalized Laplacian Graph Convolutional Network
│   ├── gat.py                # Multi-head anisotropic Graph Attention Layer
│   └── mpnn.py               # General Message Passing framework (Sum/Mean/Max)
└── tests/
    ├── __init__.py
    ├── test_gcn.py           # Adjacency normalization & classification tests
    ├── test_gat.py           # Multi-head attention concatenation/averaging tests
    └── test_mpnn.py          # Message aggregation invariant tests
```

---

## Quickstart Example

```python
import numpy as np
from src.gcn import GraphConvolutionalNetwork
from src.gat import GraphAttentionLayer

# 1. Semi-Supervised Node Classification with GCN
N, F_in, C = 4, 3, 2
A = np.array([
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0],
])
A_hat = GraphConvolutionalNetwork.normalize_adjacency(A)
X = np.random.randn(N, F_in)
y = np.array([0, 1, 1, 0])
train_mask = np.array([True, True, False, False])

gcn = GraphConvolutionalNetwork(n_features=F_in, n_hidden=4, n_classes=C, learning_rate=0.05)
for _ in range(25):
    loss = gcn.train_step(X, A_hat, y, train_mask)

print("Predicted node labels:", gcn.predict(X, A_hat))

# 2. Multi-Head Attention with GAT
gat = GraphAttentionLayer(in_features=F_in, out_features=4, n_heads=2, concat=True)
node_embeddings = gat.forward(X, A)
print("GAT Embeddings shape (N, heads * out_dim):", node_embeddings.shape)
```

---

## Running Verification Tests

```bash
pytest 16_graph_neural_networks/tests
```
