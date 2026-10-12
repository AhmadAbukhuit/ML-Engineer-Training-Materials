"""Tests for Factorization Machine (FM) CTR model."""

import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from factorization_machine import FactorizationMachine


def test_factorization_machine_init_and_shapes():
    fm = FactorizationMachine(n_factors=4, learning_rate=0.01, random_state=42)
    X = np.random.randn(20, 8)
    y = (X[:, 0] * X[:, 1] > 0).astype(int)

    fm.fit(X, y, epochs=5)
    assert fm.w0 is not None
    assert fm.w is not None and fm.w.shape == (8,)
    assert fm.V is not None and fm.V.shape == (8, 4)


def test_factorization_machine_predictions():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 6))
    y = ((X[:, 0] * X[:, 1]) > 0).astype(int)

    fm = FactorizationMachine(n_factors=4, learning_rate=0.05, reg_w=0.001, reg_v=0.001, random_state=42)
    fm.fit(X, y, epochs=30)

    probs = fm.predict_proba(X)
    preds = fm.predict(X)

    assert probs.shape == (50,)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert set(np.unique(preds)).issubset({0, 1})
    assert np.all((probs >= 0.5).astype(int) == preds)
