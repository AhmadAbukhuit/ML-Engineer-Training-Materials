import sys
from pathlib import Path
import numpy as np
import pytest
from sklearn.datasets import make_blobs

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pca import PCA


def test_pca_svd_vs_covariance():
    X, _ = make_blobs(n_samples=100, n_features=5, centers=3, random_state=42)

    pca_svd = PCA(n_components=2, method="svd").fit(X)
    pca_cov = PCA(n_components=2, method="covariance").fit(X)

    # Explained variance should match closely
    np.testing.assert_allclose(pca_svd.explained_variance_, pca_cov.explained_variance_, rtol=1e-3)
    np.testing.assert_allclose(
        pca_svd.explained_variance_ratio_, pca_cov.explained_variance_ratio_, rtol=1e-3
    )

    # Sum of variance ratios is <= 1.0
    assert np.sum(pca_svd.explained_variance_ratio_) <= 1.0


def test_pca_transform_and_reconstruction():
    X, _ = make_blobs(n_samples=80, n_features=4, centers=2, random_state=42)

    pca = PCA(n_components=4, method="svd").fit(X)
    X_proj = pca.transform(X)
    assert X_proj.shape == (80, 4)

    X_recon = pca.inverse_transform(X_proj)
    np.testing.assert_allclose(X, X_recon, atol=1e-5)


def test_pca_invalid_components():
    X = np.random.randn(20, 3)
    with pytest.raises(ValueError):
        PCA(n_components=10).fit(X)
