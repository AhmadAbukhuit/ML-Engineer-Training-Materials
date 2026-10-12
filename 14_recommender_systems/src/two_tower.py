"""Two-Tower Dual-Encoder Candidate Retrieval Architecture in NumPy."""

import numpy as np


class TwoTowerModel:
    """Two-Tower Dual Encoder Model for Large-Scale Candidate Retrieval.

    Maps multi-dimensional user features and item features into a shared,
    L2-normalized metric embedding space.

    Parameters
    ----------
    user_dim : int
        Dimension of raw user input features.
    item_dim : int
        Dimension of raw item input features.
    embed_dim : int
        Dimension of the shared latent embedding space.
    temperature : float
        Softmax scaling temperature tau.
    learning_rate : float
        SGD learning rate.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        user_dim: int,
        item_dim: int,
        embed_dim: int = 16,
        temperature: float = 0.1,
        learning_rate: float = 0.05,
        random_state: int | None = 42,
    ) -> None:
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.learning_rate = learning_rate

        rng = np.random.default_rng(random_state)
        # User tower projection weights
        self.W_u = rng.normal(0, np.sqrt(2.0 / user_dim), (user_dim, embed_dim))
        # Item tower projection weights
        self.W_i = rng.normal(0, np.sqrt(2.0 / item_dim), (item_dim, embed_dim))

    def encode_users(self, user_features: np.ndarray) -> np.ndarray:
        """Project user features into normalized latent embedding space."""
        u_raw = np.dot(np.asarray(user_features, dtype=np.float64), self.W_u)
        norms = np.linalg.norm(u_raw, axis=1, keepdims=True)
        return u_raw / np.maximum(norms, 1e-12)

    def encode_items(self, item_features: np.ndarray) -> np.ndarray:
        """Project item features into normalized latent embedding space."""
        i_raw = np.dot(np.asarray(item_features, dtype=np.float64), self.W_i)
        norms = np.linalg.norm(i_raw, axis=1, keepdims=True)
        return i_raw / np.maximum(norms, 1e-12)

    def train_step(self, user_batch: np.ndarray, item_batch: np.ndarray) -> float:
        """Perform one step of in-batch contrastive learning.

        Diagonal entries (user_i, item_i) represent positive engagement pairs;
        off-diagonal entries represent negative in-batch candidate items.
        """
        user_batch = np.asarray(user_batch, dtype=np.float64)
        item_batch = np.asarray(item_batch, dtype=np.float64)
        B = user_batch.shape[0]

        U = self.encode_users(user_batch)  # (B, D)
        V = self.encode_items(item_batch)  # (B, D)

        # In-batch similarity matrix: (B, B)
        sim = (U @ V.T) / self.temperature

        # Softmax cross-entropy on diagonal
        sim_max = np.max(sim, axis=1, keepdims=True)
        exp_sim = np.exp(sim - sim_max)
        probs = exp_sim / np.sum(exp_sim, axis=1, keepdims=True)

        labels = np.arange(B)
        pos_probs = probs[labels, labels]
        loss = float(-np.mean(np.log(pos_probs + 1e-12)))

        # Gradients wrt sim matrix
        d_sim = probs.copy()
        d_sim[labels, labels] -= 1.0
        d_sim = d_sim / (B * self.temperature)

        # d_sim @ V gives dU; d_sim.T @ U gives dV
        dU = d_sim @ V
        dV = d_sim.T @ U

        # Backprop through linear projections
        dW_u = user_batch.T @ dU
        dW_i = item_batch.T @ dV

        # Gradient update
        clip_val = 1.0
        self.W_u -= self.learning_rate * np.clip(dW_u, -clip_val, clip_val)
        self.W_i -= self.learning_rate * np.clip(dW_i, -clip_val, clip_val)

        return loss

    def retrieve_candidates(
        self,
        user_vector: np.ndarray,
        all_item_embeddings: np.ndarray,
        top_k: int = 5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sub-millisecond candidate retrieval via dot-product scoring."""
        if user_vector.ndim == 1:
            user_vector = user_vector[np.newaxis, :]
        u_emb = self.encode_users(user_vector)[0]

        scores = np.dot(all_item_embeddings, u_emb)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return top_indices, scores[top_indices]
