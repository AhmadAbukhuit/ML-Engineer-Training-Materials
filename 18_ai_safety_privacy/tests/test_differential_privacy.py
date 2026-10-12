"""Tests for Differentially Private SGD (DP-SGD)."""

import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from differential_privacy import DPSGDStep


def test_dpsgd_clipping_and_no_noise():
    # 2 samples with large gradients: norm = 10 each
    grads = np.array([
        [6.0, 8.0],
        [8.0, 6.0],
    ])
    # Clipping norm = 1.0, zero noise
    dp = DPSGDStep(clipping_norm=1.0, noise_multiplier=0.0)
    priv_grad, clip_ratio = dp.privatize_batch_gradients(grads)

    assert clip_ratio == 1.0  # Both samples were clipped
    # Clipped gradients each have norm 1.0
    # grads[0] clipped: [0.6, 0.8]; grads[1] clipped: [0.8, 0.6]
    # average: [0.7, 0.7]
    np.testing.assert_allclose(priv_grad, [0.7, 0.7], atol=1e-5)


def test_dpsgd_noise_injection():
    grads = np.zeros((100, 10))
    dp = DPSGDStep(clipping_norm=1.0, noise_multiplier=2.0, random_state=42)
    priv_grad, clip_ratio = dp.privatize_batch_gradients(grads)

    assert clip_ratio == 0.0
    # Output should not be strictly zero due to injected Gaussian noise
    assert not np.allclose(priv_grad, 0.0)
