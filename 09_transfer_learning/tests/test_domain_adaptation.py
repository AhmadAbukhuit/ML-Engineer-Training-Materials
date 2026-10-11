import sys
from pathlib import Path
import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from domain_adaptation import MaximumMeanDiscrepancy


def test_mmd_identical_distributions():
    rng = np.random.default_rng(42)
    # Samples from the same Gaussian distribution
    X_s = rng.normal(0, 1, size=(50, 4))
    X_t = rng.normal(0, 1, size=(50, 4))

    mmd_calc = MaximumMeanDiscrepancy()
    dist_same = mmd_calc.compute_mmd(X_s, X_t)

    # Samples from a radically shifted distribution
    X_shifted = rng.normal(5, 1, size=(50, 4))
    dist_diff = mmd_calc.compute_mmd(X_s, X_shifted)

    # Discrepancy between shifted distributions should be much higher than identical
    assert dist_diff > dist_same
    assert dist_same >= 0.0


def test_mmd_invalid_sample_count():
    mmd_calc = MaximumMeanDiscrepancy()
    with pytest.raises(ValueError):
        mmd_calc.compute_mmd(np.zeros((1, 2)), np.zeros((10, 2)))
