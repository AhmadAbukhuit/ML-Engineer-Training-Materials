import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from resnet import ResidualBlock


def test_residual_block_identity_shortcut():
    # In features == out features -> identity shortcut
    block = ResidualBlock(in_features=16, out_features=16, random_state=42)
    assert block.W_s is None

    x = np.random.randn(8, 16)
    out = block.forward(x)

    assert out.shape == (8, 16)
    # Output must be non-negative due to final ReLU
    assert np.all(out >= 0.0)


def test_residual_block_projection_shortcut():
    # In features (16) != out features (32) -> projection shortcut
    block = ResidualBlock(in_features=16, out_features=32, random_state=42)
    assert block.W_s is not None
    assert block.W_s.shape == (16, 32)

    x = np.random.randn(8, 16)
    out = block.forward(x)

    assert out.shape == (8, 32)
    assert np.all(out >= 0.0)
