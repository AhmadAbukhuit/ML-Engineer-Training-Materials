import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kmeans import KMeans, silhouette_score


def test_kmeans_clustering():
    from sklearn.datasets import make_blobs

    # 3 distinct clusters well separated
    X, _y = make_blobs(n_samples=300, centers=3, n_features=2, cluster_std=0.5, random_state=42)

    km = KMeans(n_clusters=3, random_state=42)
    labels = km.fit_predict(X)

    assert km.cluster_centers_.shape == (3, 2)
    assert km.inertia_ > 0.0

    # Silhouette score on well-separated clusters should be > 0.6
    score = silhouette_score(X, labels)
    assert score > 0.5, f"Expected silhouette score > 0.5, got {score:.3f}"
