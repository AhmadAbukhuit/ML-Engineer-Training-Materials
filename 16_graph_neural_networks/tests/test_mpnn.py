"""Tests for Message Passing Neural Networks (MPNN)."""

import sys
from pathlib import Path
import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mpnn import MessagePassingLayer


@pytest.mark.parametrize("aggr", ["sum", "mean", "max"])
def test_mpnn_aggregations(aggr):
    N = 5
    node_dim = 6
    msg_dim = 4
    out_dim = 8

    H = np.random.randn(N, node_dim)
    # Directed star graph: center node 0 receives messages from nodes 1, 2, 3, 4
    adj = np.zeros((N, N))
    adj[0, 1:] = 1.0

    mp = MessagePassingLayer(node_dim=node_dim, msg_dim=msg_dim, out_dim=out_dim, aggr=aggr, random_state=42)
    h_next = mp.forward(H, adj)

    assert h_next.shape == (N, out_dim)
    assert np.all(h_next >= 0.0)  # Post-ReLU activation is non-negative
