"""Graph Attention Network (GAT) via Veličković et al. (2018) Formulation."""

import numpy as np


class GraphAttentionLayer:
    """Single Graph Attention Layer with Multi-Head Attention.

    Parameters
    ----------
    in_features : int
        Input feature dimensionality.
    out_features : int
        Output feature dimensionality per attention head.
    n_heads : int
        Number of independent attention heads.
    concat : bool
        If True, concatenate head outputs (e.g. for hidden layers).
        If False, average head outputs (e.g. for final prediction layer).
    negative_slope : float
        LeakyReLU negative slope alpha.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        random_state: int | None = 42,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.concat = concat
        self.negative_slope = negative_slope

        rng = np.random.default_rng(random_state)
        # Weights for each head: shape (n_heads, in_features, out_features)
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.W = rng.uniform(-limit, limit, (n_heads, in_features, out_features))

        # Attention vector a = [a_src || a_dst]: shapes (n_heads, out_features)
        self.a_src = rng.uniform(-limit, limit, (n_heads, out_features))
        self.a_dst = rng.uniform(-limit, limit, (n_heads, out_features))

    def _leaky_relu(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.negative_slope * x)

    def forward(self, H: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Forward pass computing multi-head graph self-attention.

        Parameters
        ----------
        H : np.ndarray of shape (N, in_features)
        A : np.ndarray of shape (N, N)
            Binary or weighted adjacency matrix (self-loops are automatically added).

        Returns
        -------
        H_out : np.ndarray of shape (N, n_heads * out_features) if concat else (N, out_features)
        """
        N = H.shape[0]
        # Ensure self-loops
        adj_mask = (A + np.eye(N)) > 0

        head_outputs = []

        for k in range(self.n_heads):
            # 1. Project node representations: (N, out_features)
            Z = H @ self.W[k]

            # 2. Compute self-attention logits: e_ij = LeakyReLU( a_src^T z_i + a_dst^T z_j )
            score_src = Z @ self.a_src[k]  # shape (N,)
            score_dst = Z @ self.a_dst[k]  # shape (N,)

            # Broadcasted sum: e_ij = score_src[i] + score_dst[j]
            e = self._leaky_relu(score_src[:, np.newaxis] + score_dst[np.newaxis, :])  # (N, N)

            # Masked softmax: set non-neighbors to -inf
            masked_e = np.where(adj_mask, e, -1e9)
            e_max = np.max(masked_e, axis=1, keepdims=True)
            exp_e = np.exp(masked_e - e_max) * adj_mask.astype(float)
            alpha = exp_e / np.maximum(np.sum(exp_e, axis=1, keepdims=True), 1e-12)

            # 3. Aggregate: h_i' = sum_j alpha_ij * z_j
            head_out = alpha @ Z  # (N, out_features)
            head_outputs.append(head_out)

        if self.concat:
            return np.concatenate(head_outputs, axis=-1)
        else:
            return np.mean(head_outputs, axis=0)
