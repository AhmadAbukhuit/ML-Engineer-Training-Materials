import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from simclr_framework import SimCLRFramework


def test_simclr_framework_pipeline():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(24, 10))

    simclr = SimCLRFramework(
        in_dim=10, feature_dim=16, proj_dim=8, temperature=0.5, random_state=42
    )

    loss, z1, z2 = simclr.compute_loss(X)
    assert isinstance(loss, float)
    assert loss > 0.0
    assert z1.shape == (24, 8)
    assert z2.shape == (24, 8)

    # Test representation extraction for downstream evaluation
    representations = simclr.encode(X)
    assert representations.shape == (24, 16)
