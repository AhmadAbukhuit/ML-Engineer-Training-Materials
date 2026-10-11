import sys
from pathlib import Path
import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_regression
from sklearn.metrics import accuracy_score, r2_score

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from knn import KNNClassifier, KNNRegressor


def test_knn_classifier_blobs():
    X, y = make_blobs(n_samples=100, n_features=4, centers=3, random_state=42, cluster_std=1.0)
    train_mask = np.arange(len(X)) < 80

    clf = KNNClassifier(n_neighbors=5, metric="euclidean")
    clf.fit(X[train_mask], y[train_mask])
    preds = clf.predict(X[~train_mask])
    acc = accuracy_score(y[~train_mask], preds)
    assert acc >= 0.90

    probas = clf.predict_proba(X[~train_mask])
    assert probas.shape == (20, 3)
    np.testing.assert_allclose(np.sum(probas, axis=1), 1.0)


def test_knn_classifier_distance_weighting():
    X = np.array([[1.0], [2.0], [10.0]])
    y = np.array([0, 0, 1])

    clf = KNNClassifier(n_neighbors=3, weights="distance")
    clf.fit(X, y)
    pred = clf.predict(np.array([[1.1]]))
    assert pred[0] == 0


def test_knn_regressor():
    X, y = make_regression(n_samples=100, n_features=3, noise=0.1, random_state=42)
    train_mask = np.arange(len(X)) < 80

    reg = KNNRegressor(n_neighbors=5, metric="euclidean")
    reg.fit(X[train_mask], y[train_mask])
    preds = reg.predict(X[~train_mask])
    r2 = r2_score(y[~train_mask], preds)
    assert r2 > 0.60


def test_knn_metrics():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    y = np.array([0, 1, 1])

    for metric in ["euclidean", "manhattan", "minkowski"]:
        clf = KNNClassifier(n_neighbors=1, metric=metric, p=3.0)
        clf.fit(X, y)
        pred = clf.predict(np.array([[0.1, 0.1]]))
        assert pred[0] == 0


def test_invalid_neighbors():
    with pytest.raises(ValueError):
        KNNClassifier(n_neighbors=0)
