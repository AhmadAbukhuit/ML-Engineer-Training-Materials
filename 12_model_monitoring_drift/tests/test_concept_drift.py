import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from concept_drift import ADWIN, PageHinkley


def test_adwin_detects_abrupt_mean_shift():
    rng = np.random.default_rng(42)
    detector = ADWIN(delta=0.01)

    drift_points = []
    # Stream 100 stationary samples with mean 0.0
    for i in range(100):
        val = rng.normal(0.0, 0.2)
        if detector.update(val):
            drift_points.append(i)

    # Now abruptly jump to mean 5.0
    for i in range(100, 200):
        val = rng.normal(5.0, 0.2)
        if detector.update(val):
            drift_points.append(i)

    # ADWIN should detect drift shortly after index 100
    assert len(drift_points) > 0
    assert any(100 <= dp <= 150 for dp in drift_points)


def test_page_hinkley_drift_detection():
    rng = np.random.default_rng(42)
    ph = PageHinkley(threshold=15.0, alpha=0.05)

    drift_triggered = False
    # Stream stationary error rate of ~0.1
    for _ in range(80):
        val = rng.normal(0.1, 0.05)
        ph.update(val)

    # Sudden surge in error rate to ~2.0
    for _ in range(80):
        val = rng.normal(2.0, 0.1)
        if ph.update(val):
            drift_triggered = True
            break

    assert drift_triggered
