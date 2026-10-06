import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from attention import scaled_dot_product_attention, MultiHeadAttention


def test_scaled_dot_product_attention():
    # Batch=2, SeqLen=3, d_k=4, d_v=4
    Q = np.random.randn(2, 3, 4)
    K = np.random.randn(2, 3, 4)
    V = np.random.randn(2, 3, 4)

    out, weights = scaled_dot_product_attention(Q, K, V)

    assert out.shape == (2, 3, 4)
    assert weights.shape == (2, 3, 3)

    # Attention weights must sum to 1.0 along the key dimension
    weight_sums = np.sum(weights, axis=-1)
    np.testing.assert_allclose(weight_sums, np.ones((2, 3)), atol=1e-5)


def test_causal_mask_attention():
    # Lower triangular causal mask (can only attend to past and current tokens)
    mask = np.tril(np.ones((3, 3)))
    Q = np.random.randn(1, 3, 4)
    K = np.random.randn(1, 3, 4)
    V = np.random.randn(1, 3, 4)

    _, weights = scaled_dot_product_attention(Q, K, V, mask=mask)

    # First token attending to tokens 1 and 2 must have 0 probability
    assert weights[0, 0, 1] < 1e-6
    assert weights[0, 0, 2] < 1e-6
    # Second token attending to token 2 must have 0 probability
    assert weights[0, 1, 2] < 1e-6


def test_multi_head_attention_shape_preservation():
    mha = MultiHeadAttention(d_model=32, n_heads=4, random_state=42)
    x = np.random.randn(2, 5, 32)  # Batch=2, SeqLen=5, d_model=32

    out, weights = mha.forward(x, x, x)

    # Output representation dimension must match input (2, 5, 32)
    assert out.shape == (2, 5, 32)
    # Weights shape: (Batch, Heads, SeqLen, SeqLen)
    assert weights.shape == (2, 4, 5, 5)
