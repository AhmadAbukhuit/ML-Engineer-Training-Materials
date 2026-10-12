"""Tests for Ring All-Reduce communication primitive."""

import sys
from pathlib import Path
import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ring_allreduce import ring_allreduce


@pytest.mark.parametrize("P", [2, 3, 4, 7])
@pytest.mark.parametrize("N", [10, 15, 64, 100])
def test_ring_allreduce_correctness(P, N):
    rng = np.random.default_rng(42)
    tensors = [rng.normal(size=N) for _ in range(P)]

    expected_sum = np.sum(tensors, axis=0)
    reduced = ring_allreduce(tensors)

    assert len(reduced) == P
    for r in range(P):
        assert reduced[r].shape == (N,)
        np.testing.assert_allclose(reduced[r], expected_sum, rtol=1e-5, atol=1e-5)


def test_ring_allreduce_single_process():
    tensor = [np.array([1.0, 2.0, 3.0])]
    reduced = ring_allreduce(tensor)
    np.testing.assert_allclose(reduced[0], tensor[0])
