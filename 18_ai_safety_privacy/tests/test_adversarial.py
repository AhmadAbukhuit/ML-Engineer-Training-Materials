"""Tests for Adversarial Attacks (FGSM and PGD)."""

import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adversarial import FGSM, PGD


def dummy_grad_fn(x: np.ndarray) -> np.ndarray:
    # Gradient pointing towards positive direction
    return np.ones_like(x)


def test_fgsm_perturbation_bound_and_clipping():
    x = np.array([0.2, 0.5, 0.9])
    eps = 0.15
    fgsm = FGSM(epsilon=eps, clip_min=0.0, clip_max=1.0)
    x_adv = fgsm.generate(x, dummy_grad_fn)

    # Values must remain within [0.0, 1.0]
    assert np.all(x_adv >= 0.0) and np.all(x_adv <= 1.0)
    # L_inf distance cannot exceed epsilon
    max_diff = np.max(np.abs(x_adv - x))
    assert max_diff <= eps + 1e-6
    # For positive gradient, values should increase up to clipping bound
    np.testing.assert_allclose(x_adv, [0.35, 0.65, 1.0], atol=1e-5)


def test_pgd_perturbation_bound():
    x = np.array([[0.5, 0.5]])
    eps = 0.1
    pgd = PGD(epsilon=eps, alpha=0.02, n_steps=5, random_start=True, random_state=42)
    x_adv = pgd.generate(x, dummy_grad_fn)

    assert np.all(x_adv >= 0.0) and np.all(x_adv <= 1.0)
    diff = np.max(np.abs(x_adv - x))
    assert diff <= eps + 1e-6
