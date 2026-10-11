import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from decision_tree import DecisionTreeClassifier


def test_decision_tree_simple_split():
    # 1D threshold dataset: X <= 0 -> class 0; X > 0 -> class 1
    X = np.array([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 0, 1, 1, 1])

    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(X, y)

    preds = tree.predict(X)
    np.testing.assert_array_equal(preds, y)


def test_decision_tree_classification_performance():
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=200,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        random_state=42
    )

    tree = DecisionTreeClassifier(max_depth=5, min_samples_split=4)
    tree.fit(X, y)

    preds = tree.predict(X)
    acc = np.mean(preds == y)
    assert acc >= 0.85, f"Expected tree accuracy >= 85%, got {acc:.2f}"

    probs = tree.predict_proba(X)
    assert probs.shape == (200, 2)
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)
