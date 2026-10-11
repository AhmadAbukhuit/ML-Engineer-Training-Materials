import sys
from pathlib import Path
import numpy as np
from sklearn.datasets import make_blobs

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gmm import GaussianMixtureModel


def test_gmm_convergence_and_clustering():
    X, _ = make_blobs(n_samples=200, centers=2, cluster_std=0.5, random_state=42)

    gmm = GaussianMixtureModel(n_components=2, max_iter=100, tol=1e-4, random_state=42)
    gmm.fit(X)

    assert gmm.converged_
    assert gmm.weights_.shape == (2,)
    np.testing.assert_allclose(np.sum(gmm.weights_), 1.0)
    assert gmm.means_.shape == (2, 2)

    probas = gmm.predict_proba(X)
    assert probas.shape == (200, 2)
    np.testing.assert_allclose(np.sum(probas, axis=1), 1.0, atol=1e-5)

    preds = gmm.predict(X)
    assert len(np.unique(preds)) == 2
