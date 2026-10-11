import sys
from pathlib import Path
import numpy as np
from sklearn.datasets import make_blobs

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from label_propagation import LabelPropagation


def test_label_propagation_accuracy():
    X, y = make_blobs(n_samples=100, centers=2, cluster_std=0.5, random_state=42)

    # Label only 4 samples (2 per class)
    y_semi = np.full_like(y, -1)
    idx_0 = np.where(y == 0)[0][:2]
    idx_1 = np.where(y == 1)[0][:2]
    y_semi[idx_0] = 0
    y_semi[idx_1] = 1

    lp = LabelPropagation(gamma=10.0, max_iter=100)
    lp.fit(X, y_semi)

    preds = lp.predict()
    acc = np.mean(preds == y)
    assert acc >= 0.90

    probas = lp.predict_proba()
    assert probas.shape == (100, 2)
    np.testing.assert_allclose(np.sum(probas, axis=1), 1.0, atol=1e-5)
