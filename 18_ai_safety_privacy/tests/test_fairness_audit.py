"""Tests for Algorithmic Fairness Auditor."""

import sys
from pathlib import Path
import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fairness_audit import FairnessAuditor


def test_fairness_audit_perfect_parity():
    # Both groups have identical predictions and targets
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0])
    sens = np.array([0, 0, 1, 1])

    report = FairnessAuditor.audit(y_true, y_pred, sens)

    assert report.demographic_parity_diff == 0.0
    assert report.disparate_impact_ratio == 1.0
    assert report.equal_opportunity_diff == 0.0
    assert report.equalized_odds_diff == 0.0


def test_fairness_audit_disparate_impact():
    # Group 0: all predicted 1
    # Group 1: all predicted 0
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    sens = np.array([0, 0, 1, 1])

    report = FairnessAuditor.audit(y_true, y_pred, sens)

    assert report.demographic_parity_diff == 1.0
    assert report.disparate_impact_ratio == 0.0
    assert report.equal_opportunity_diff == 1.0


def test_fairness_audit_missing_group_raises():
    y_true = np.array([1, 0])
    y_pred = np.array([1, 0])
    sens = np.array([0, 0])  # No group 1

    with pytest.raises(ValueError):
        FairnessAuditor.audit(y_true, y_pred, sens)
