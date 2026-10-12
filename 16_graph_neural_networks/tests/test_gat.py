"""Tests for Graph Attention Networks (GAT)."""

import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gat import GraphAttentionLayer


def test_gat_layer_concat_and_shapes():
    N = 5
    F_in = 8
    F_out = 4
    K_heads = 3

    H = np.random.randn(N, F_in)
    A = np.eye(N)  # Self-loops only

    gat = GraphAttentionLayer(in_features=F_in, out_features=F_out, n_heads=K_heads, concat=True, random_state=42)
    out = gat.forward(H, A)

    # Output dimension should be N x (n_heads * out_features) = 5 x 12
    assert out.shape == (N, K_heads * F_out)


def test_gat_layer_average_and_shapes():
    N = 4
    F_in = 6
    F_out = 3
    K_heads = 4

    H = np.random.randn(N, F_in)
    A = np.ones((N, N))  # Fully connected graph

    gat = GraphAttentionLayer(in_features=F_in, out_features=F_out, n_heads=K_heads, concat=False, random_state=42)
    out = gat.forward(H, A)

    # Output dimension should be N x out_features = 4 x 3
    assert out.shape == (N, F_out)
