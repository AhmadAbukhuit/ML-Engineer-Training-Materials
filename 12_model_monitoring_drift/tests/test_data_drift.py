import sys
from pathlib import Path
import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_drift import (
    DataDriftDetector,
    ks_two_sample_test,
    population_stability_index,
    wasserstein_distance_1d,
)


def test_psi_identical_distributions():
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, size=1000)
    curr = rng.normal(0, 1, size=1000)

    psi = population_stability_index(ref, curr, num_bins=10)
    # Identical distributions should have PSI well below 0.1
    assert psi < 0.05


def test_psi_detects_severe_drift():
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, size=1000)
    curr = rng.normal(3.0, 1.5, size=1000)  # substantial mean and variance shift

    psi = population_stability_index(ref, curr, num_bins=10)
    # Severe shift should trigger PSI > 0.25
    assert psi > 0.25


def test_ks_two_sample_test():
    rng = np.random.default_rng(42)
    s1 = rng.normal(0, 1, size=200)
    s2 = rng.normal(0, 1, size=200)
    s_shifted = rng.normal(2.5, 1, size=200)

    d_same, _ = ks_two_sample_test(s1, s2)
    d_diff, p_diff = ks_two_sample_test(s1, s_shifted)

    assert d_diff > d_same
    assert p_diff < 0.01  # reject null hypothesis of identical distributions


def test_wasserstein_distance_properties():
    s1 = np.array([0.0, 1.0, 2.0])
    s2 = np.array([1.0, 2.0, 3.0])  # constant shift of +1.0
    w = wasserstein_distance_1d(s1, s2)
    assert w == pytest.approx(1.0, rel=1e-3)


def test_data_drift_detector_audit():
    rng = np.random.default_rng(42)
    ref_matrix = rng.normal(size=(500, 2))
    curr_matrix = ref_matrix.copy()
    curr_matrix[:, 1] += 4.0  # drift only feature 1

    detector = DataDriftDetector(ref_matrix, feature_names=["clean_col", "drifted_col"])
    report = detector.audit_batch(curr_matrix, psi_threshold=0.2)

    assert not report["clean_col"]["drift_detected"]
    assert report["drifted_col"]["drift_detected"]
