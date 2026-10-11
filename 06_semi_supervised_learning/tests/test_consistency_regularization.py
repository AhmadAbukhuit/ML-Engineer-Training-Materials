import sys
from pathlib import Path
import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from consistency_regularization import consistency_loss, rampup_weight, sharpen


def test_sharpening_increases_confidence():
    probs = np.array([[0.6, 0.4], [0.55, 0.45]])
    sharpened = sharpen(probs, temperature=0.2)

    # Max probability should be higher after sharpening
    assert sharpened[0, 0] > probs[0, 0]
    np.testing.assert_allclose(np.sum(sharpened, axis=1), 1.0)


def test_rampup_weight_schedule():
    w0 = rampup_weight(0, rampup_length=50)
    w25 = rampup_weight(25, rampup_length=50)
    w50 = rampup_weight(50, rampup_length=50)
    w100 = rampup_weight(100, rampup_length=50)

    assert 0.0 < w0 < w25 < w50 == 1.0 == w100


def test_consistency_loss_metrics():
    p1 = np.array([[0.7, 0.3], [0.2, 0.8]])
    p2 = np.array([[0.65, 0.35], [0.25, 0.75]])

    mse = consistency_loss(p1, p2, metric="mse")
    kl = consistency_loss(p1, p2, metric="kl")

    assert mse > 0.0
    assert kl > 0.0

    with pytest.raises(ValueError):
        consistency_loss(p1, p2, metric="invalid")
