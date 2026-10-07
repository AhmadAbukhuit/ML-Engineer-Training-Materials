import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from naive_bayes import GaussianNaiveBayes


def test_gaussian_naive_bayes():
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=200,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        random_state=42
    )

    gnb = GaussianNaiveBayes()
    gnb.fit(X, y)

    preds = gnb.predict(X)
    acc = np.mean(preds == y)
    assert acc >= 0.80, f"Expected Naive Bayes accuracy >= 80%, got {acc:.2f}"

    probs = gnb.predict_proba(X)
    assert probs.shape == (200, 2)
    # Probabilities must sum to 1.0
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)
