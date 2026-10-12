"""Tests for Dynamic Loss Scaler."""

import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from loss_scaler import DynamicLossScaler


def test_loss_scaler_unscale_and_grow():
    scaler = DynamicLossScaler(initial_scale=1024.0, growth_factor=2.0, growth_interval=3)

    grads = [np.array([1024.0, 2048.0])]
    unscaled, is_finite = scaler.unscale_gradients(grads)

    assert is_finite is True
    assert unscaled is not None
    np.testing.assert_allclose(unscaled[0], [1.0, 2.0])

    # Perform 2 more successful steps to trigger growth
    scaler.unscale_gradients(grads)
    scaler.unscale_gradients(grads)

    # Scale should have doubled: 1024 * 2 = 2048
    assert scaler.scale == 2048.0


def test_loss_scaler_overflow_backoff():
    scaler = DynamicLossScaler(initial_scale=1024.0, backoff_factor=0.5)

    inf_grads = [np.array([10.0, np.inf])]
    unscaled, is_finite = scaler.unscale_gradients(inf_grads)

    assert is_finite is False
    assert unscaled is None
    # Scale should be halved
    assert scaler.scale == 512.0
    assert scaler.consecutive_successful_steps == 0
