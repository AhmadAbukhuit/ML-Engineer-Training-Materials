import sys
from pathlib import Path
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pseudo_labeling import SelfTrainingClassifier


def test_self_training_expands_labeled_pool():
    X, y = make_blobs(n_samples=120, centers=2, cluster_std=1.0, random_state=42)

    # Keep only 10 samples labeled, set remaining to -1
    y_semi = np.full_like(y, -1)
    y_semi[:10] = y[:10]

    base_lr = LogisticRegression(random_state=42)
    st = SelfTrainingClassifier(base_estimator=base_lr, threshold=0.75, max_iter=5)
    st.fit(X, y_semi)

    # Ensure pseudo-labeling successfully added confident points
    assert st.n_pseudo_labeled_ > 0

    preds = st.predict(X)
    assert len(preds) == 120
    # Over 80% accuracy on overall blob data
    acc = np.mean(preds == y)
    assert acc > 0.80
