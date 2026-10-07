import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ensemble_benchmark import run_ensemble_benchmark


def test_ensemble_benchmark():
    results = run_ensemble_benchmark(n_samples=200, n_features=6)
    assert len(results) >= 2

    for res in results:
        assert "model" in res
        assert "accuracy" in res
        assert "f1_score" in res
        assert "roc_auc" in res
        assert 0.0 <= res["accuracy"] <= 1.0
