import sys
from pathlib import Path
import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from contrastive_loss import NTXentLoss, compute_alignment_and_uniformity


def test_nt_xent_loss_basic():
    rng = np.random.default_rng(42)
    criterion = NTXentLoss(temperature=0.5)

    batch_size = 16
    dim = 32
    z_i = rng.normal(size=(batch_size, dim))
    # Positives with slight perturbation
    z_j = z_i + 0.05 * rng.normal(size=(batch_size, dim))

    loss = criterion(z_i, z_j)
    assert isinstance(loss, float)
    assert loss > 0.0

    # Identical pairs should have lower loss than random pairs
    loss_identical = criterion(z_i, z_i)
    z_random = rng.normal(size=(batch_size, dim))
    loss_random = criterion(z_i, z_random)
    assert loss_identical < loss_random


def test_alignment_and_uniformity():
    rng = np.random.default_rng(42)
    z_i = rng.normal(size=(10, 16))
    z_j = z_i + 0.01 * rng.normal(size=(10, 16))

    alignment, uniformity = compute_alignment_and_uniformity(z_i, z_j)
    assert alignment >= 0.0
    assert isinstance(uniformity, float)


def test_invalid_temperature_and_batch_size():
    with pytest.raises(ValueError):
        NTXentLoss(temperature=-0.1)

    criterion = NTXentLoss(temperature=0.5)
    with pytest.raises(ValueError):
        criterion(np.zeros((1, 5)), np.zeros((1, 5)))
