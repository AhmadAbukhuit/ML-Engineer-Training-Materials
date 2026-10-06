"""
Transformer Self-Attention Mechanisms from First Principles.

This module implements:
1. scaled_dot_product_attention: Vectorized Q, K, V dot-product with 1/sqrt(d_k) scaling and causal masking.
2. MultiHeadAttention: Multi-head linear projections, parallel attention computation, and output projection.
"""

import numpy as np


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes Scaled Dot-Product Attention:
    Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k) + mask) @ V

    Args:
        Q: Queries of shape (..., SeqLen_Q, d_k)
        K: Keys of shape (..., SeqLen_K, d_k)
        V: Values of shape (..., SeqLen_K, d_v)
        mask: Optional binary or float mask where 0 (or large negative) indicates positions to mask out.

    Returns:
        output: Contextualized output representations of shape (..., SeqLen_Q, d_v).
        attention_weights: Normalized attention distributions of shape (..., SeqLen_Q, SeqLen_K).
    """
    d_k = Q.shape[-1]
    # Compute raw attention energy scores
    scores = np.matmul(Q, np.swapaxes(K, -1, -2)) / np.sqrt(d_k)

    # Apply causal or padding mask
    if mask is not None:
        # Mask positions where mask == 0 by setting to large negative value
        scores = np.where(mask == 0, -1e9, scores)

    # Numerically stable softmax along the last dimension (SeqLen_K)
    scores_shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores_shifted)
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Weighted sum of values
    output = np.matmul(attention_weights, V)
    return output, attention_weights


class MultiHeadAttention:
    """
    Multi-Head Attention module from 'Attention Is All You Need'.
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4, random_state: int | None = 42):
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        rng = np.random.default_rng(random_state)
        # Xavier/Glorot normal initialization
        scale = np.sqrt(2.0 / (d_model + self.d_k))
        self.W_q = rng.normal(0, scale, (d_model, d_model))
        self.W_k = rng.normal(0, scale, (d_model, d_model))
        self.W_v = rng.normal(0, scale, (d_model, d_model))
        self.W_o = rng.normal(0, scale, (d_model, d_model))

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        # Transform (Batch, SeqLen, d_model) -> (Batch, n_heads, SeqLen, d_k)
        batch_size, seq_len, _ = x.shape
        x_reshaped = x.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        return np.swapaxes(x_reshaped, 1, 2)

    def _combine_heads(self, x: np.ndarray) -> np.ndarray:
        # Transform (Batch, n_heads, SeqLen, d_k) -> (Batch, SeqLen, d_model)
        batch_size, _, seq_len, _ = x.shape
        x_transposed = np.swapaxes(x, 1, 2)
        return x_transposed.reshape(batch_size, seq_len, self.d_model)

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Runs the full multi-head attention forward pass.

        Args:
            Q, K, V: Arrays of shape (Batch, SeqLen, d_model)
            mask: Optional broadcastable mask array.

        Returns:
            output: Attention output of shape (Batch, SeqLen, d_model)
            weights: Head attention weights of shape (Batch, n_heads, SeqLen, SeqLen)
        """
        # 1. Linear projections
        Q_proj = np.matmul(Q, self.W_q)
        K_proj = np.matmul(K, self.W_k)
        V_proj = np.matmul(V, self.W_v)

        # 2. Split into multiple heads
        Q_heads = self._split_heads(Q_proj)
        K_heads = self._split_heads(K_proj)
        V_heads = self._split_heads(V_proj)

        # 3. Scaled dot-product attention per head
        out_heads, weights = scaled_dot_product_attention(Q_heads, K_heads, V_heads, mask=mask)

        # 4. Concatenate heads and final linear projection
        combined = self._combine_heads(out_heads)
        final_output = np.matmul(combined, self.W_o)

        return final_output, weights
