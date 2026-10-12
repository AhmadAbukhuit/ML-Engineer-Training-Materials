"""Tests for Graph Convolutional Networks (GCN)."""

import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gcn import GraphConvolutionalNetwork


def test_gcn_adjacency_normalization():
    # 3 nodes in a line: 0 - 1 - 2
    A = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ])
    A_hat = GraphConvolutionalNetwork.normalize_adjacency(A)

    assert A_hat.shape == (3, 3)
    # A_hat should be symmetric
    np.testing.assert_allclose(A_hat, A_hat.T)
    # Diagonal should not be zero because self-loops were added
    assert np.all(np.diag(A_hat) > 0)


def test_gcn_forward_pass_and_training():
    N = 6
    F_in = 4
    F_hid = 8
    C = 2

    # Simple cycle graph: 0-1-2-3-4-5-0
    A = np.zeros((N, N))
    for i in range(N):
        A[i, (i + 1) % N] = 1.0
        A[(i + 1) % N, i] = 1.0

    A_hat = GraphConvolutionalNetwork.normalize_adjacency(A)
    X = np.random.randn(N, F_in)
    y = np.array([0, 1, 0, 1, 0, 1])
    train_mask = np.array([True, True, True, True, False, False])

    gcn = GraphConvolutionalNetwork(F_in, F_hid, C, learning_rate=0.05, random_state=42)
    initial_loss = gcn.train_step(X, A_hat, y, train_mask)

    for _ in range(30):
        final_loss = gcn.train_step(X, A_hat, y, train_mask)

    assert final_loss < initial_loss

    preds = gcn.predict(X, A_hat)
    assert preds.shape == (N,)
    assert set(np.unique(preds)).issubset({0, 1})
