import sys
from pathlib import Path
import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_moons

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dbscan import DBSCAN


def test_dbscan_clustering_blobs():
    X, _ = make_blobs(n_samples=150, centers=2, cluster_std=0.4, random_state=42)
    # Add an outlier far away
    X_with_outlier = np.vstack([X, np.array([[50.0, 50.0]])])

    db = DBSCAN(eps=0.8, min_samples=4)
    labels = db.fit_predict(X_with_outlier)

    # The outlier should be identified as noise (-1)
    assert labels[-1] == -1

    # Should find at least 2 non-noise clusters
    unique_clusters = set(labels) - {-1}
    assert len(unique_clusters) >= 2


def test_dbscan_moons():
    X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)
    db = DBSCAN(eps=0.25, min_samples=5)
    labels = db.fit_predict(X)

    unique_clusters = set(labels) - {-1}
    assert len(unique_clusters) == 2


def test_dbscan_invalid_params():
    with pytest.raises(ValueError):
        DBSCAN(eps=-1.0)
    with pytest.raises(ValueError):
        DBSCAN(min_samples=0)
