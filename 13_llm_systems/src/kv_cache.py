"""KV-Cache and Autoregressive Attention Mechanics from First Principles in NumPy."""

import numpy as np


class KVCache:
    """Key-Value Cache for Autoregressive Transformer Decoding.

    Stores historical Key and Value representations in memory across decoding steps,
    eliminating redundant computation of past tokens from O(T^2) to O(T).
    """

    def __init__(self) -> None:
        self.cached_k: np.ndarray | None = None
        self.cached_v: np.ndarray | None = None

    def update(self, key_states: np.ndarray, value_states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Append newly generated key and value token states to the cache.

        Parameters
        ----------
        key_states : np.ndarray of shape (batch, num_heads, seq_len, head_dim)
        value_states : np.ndarray of shape (batch, num_heads, seq_len, head_dim)

        Returns
        -------
        full_k : np.ndarray of shape (batch, num_heads, total_seq_len, head_dim)
        full_v : np.ndarray of shape (batch, num_heads, total_seq_len, head_dim)
        """
        if self.cached_k is None:
            self.cached_k = key_states.copy()
            self.cached_v = value_states.copy()
        else:
            self.cached_k = np.concatenate([self.cached_k, key_states], axis=2)
            self.cached_v = np.concatenate([self.cached_v, value_states], axis=2)

        return self.cached_k, self.cached_v

    def reset(self) -> None:
        """Clear cache for a new prompt."""
        self.cached_k = None
        self.cached_v = None

    @property
    def seq_len(self) -> int:
        return 0 if self.cached_k is None else self.cached_k.shape[2]


class CachedMultiHeadAttention:
    """Multi-Head Attention supporting both prefill and cached single-token decoding.

    Parameters
    ----------
    d_model : int
        Hidden representation dimension.
    num_heads : int
        Number of attention heads.
    """

    def __init__(self, d_model: int = 64, num_heads: int = 4, random_state: int | None = 42) -> None:
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        rng = np.random.default_rng(random_state)
        scale = np.sqrt(2.0 / d_model)
        self.W_q = rng.normal(0, scale, (d_model, d_model))
        self.W_k = rng.normal(0, scale, (d_model, d_model))
        self.W_v = rng.normal(0, scale, (d_model, d_model))
        self.W_o = rng.normal(0, scale, (d_model, d_model))

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        # (batch, seq_len, d_model) -> (batch, num_heads, seq_len, head_dim)
        batch, seq_len, _ = x.shape
        x_reshaped = x.reshape(batch, seq_len, self.num_heads, self.head_dim)
        return np.transpose(x_reshaped, (0, 2, 1, 3))

    def _combine_heads(self, x: np.ndarray) -> np.ndarray:
        # (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, d_model)
        x_trans = np.transpose(x, (0, 2, 1, 3))
        batch, seq_len, _, _ = x_trans.shape
        return x_trans.reshape(batch, seq_len, self.d_model)

    def forward(
        self, x: np.ndarray, kv_cache: KVCache | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Forward pass with optional KV-cache.

        Parameters
        ----------
        x : np.ndarray of shape (batch, seq_len, d_model)
            In prefill mode, seq_len is prompt length.
            In decoding mode with cache, seq_len is 1 (new single token).
        kv_cache : KVCache or None

        Returns
        -------
        output : np.ndarray of shape (batch, seq_len, d_model)
        attention_weights : np.ndarray of shape (batch, num_heads, seq_len, total_seq_len)
        """
        Q = np.dot(x, self.W_q)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)

        q_heads = self._split_heads(Q)
        k_heads = self._split_heads(K)
        v_heads = self._split_heads(V)

        if kv_cache is not None:
            k_heads, v_heads = kv_cache.update(k_heads, v_heads)

        # Scaled dot product attention
        # Q: (B, H, S_q, D), K: (B, H, S_k, D) -> Scores: (B, H, S_q, S_k)
        scores = np.matmul(q_heads, np.swapaxes(k_heads, -1, -2)) / np.sqrt(self.head_dim)

        # Stable Softmax
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        # Context output: (B, H, S_q, D)
        context = np.matmul(attn_weights, v_heads)
        combined = self._combine_heads(context)

        output = np.dot(combined, self.W_o)
        return output, attn_weights
