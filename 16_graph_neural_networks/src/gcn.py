"""Graph Convolutional Network (GCN) via Kipf & Welling (2017) Formulation."""

import numpy as np


class GCNLayer:
    """Single Graph Convolutional Layer.

    Propagation: H^(l+1) = sigma( A_hat @ H^(l) @ W )
    where A_hat = D_tilde^(-1/2) @ A_tilde @ D_tilde^(-1/2)
    and A_tilde = A + I_N
    """

    def __init__(self, in_features: int, out_features: int, activation: str = "relu", random_state: int | None = 42) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        rng = np.random.default_rng(random_state)
        # Glorot (Xavier) uniform initialization
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.W = rng.uniform(-limit, limit, (in_features, out_features))
        self.b = np.zeros(out_features)

        self.last_H: np.ndarray | None = None
        self.last_A_hat: np.ndarray | None = None
        self.last_Z: np.ndarray | None = None

    def forward(self, H: np.ndarray, A_hat: np.ndarray) -> np.ndarray:
        """Forward pass across graph topology.

        Parameters
        ----------
        H : np.ndarray of shape (N, in_features)
            Input node feature representations.
        A_hat : np.ndarray of shape (N, N)
            Symmetrically normalized adjacency matrix.
        """
        self.last_H = H
        self.last_A_hat = A_hat

        # Aggregation: A_hat @ H -> (N, in_features)
        agg = A_hat @ H
        # Transformation: agg @ W + b -> (N, out_features)
        Z = agg @ self.W + self.b
        self.last_Z = Z

        if self.activation == "relu":
            return np.maximum(Z, 0.0)
        elif self.activation == "identity":
            return Z
        elif self.activation == "softmax":
            exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
            return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")


class GraphConvolutionalNetwork:
    """Two-layer GCN for Semi-Supervised Node Classification.

    Parameters
    ----------
    n_features : int
        Dimension of input node features.
    n_hidden : int
        Dimension of hidden graph representations.
    n_classes : int
        Number of target node classes.
    learning_rate : float
        SGD learning rate.
    weight_decay : float
        L2 weight regularization penalty.
    """

    def __init__(
        self,
        n_features: int,
        n_hidden: int,
        n_classes: int,
        learning_rate: float = 0.01,
        weight_decay: float = 5e-4,
        random_state: int | None = 42,
    ) -> None:
        self.gc1 = GCNLayer(n_features, n_hidden, activation="relu", random_state=random_state)
        self.gc2 = GCNLayer(n_hidden, n_classes, activation="softmax", random_state=random_state + 1 if random_state else None)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    @staticmethod
    def normalize_adjacency(A: np.ndarray) -> np.ndarray:
        """Compute symmetrically normalized adjacency matrix with self-loops.

        A_tilde = A + I_N
        D_tilde_ii = sum_j A_tilde_ij
        A_hat = D_tilde^(-1/2) @ A_tilde @ D_tilde^(-1/2)
        """
        A = np.asarray(A, dtype=np.float64)
        N = A.shape[0]
        # Add self-loops
        A_tilde = A + np.eye(N)
        # Degree vector
        d = np.sum(A_tilde, axis=1)
        d_inv_sqrt = np.power(d, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0

        D_inv_sqrt = np.diag(d_inv_sqrt)
        return D_inv_sqrt @ A_tilde @ D_inv_sqrt

    def forward(self, X: np.ndarray, A_hat: np.ndarray) -> np.ndarray:
        """Forward pass through 2-layer GCN."""
        h1 = self.gc1.forward(X, A_hat)
        out = self.gc2.forward(h1, A_hat)
        return out

    def train_step(
        self,
        X: np.ndarray,
        A_hat: np.ndarray,
        y: np.ndarray,
        train_mask: np.ndarray,
    ) -> float:
        """Single gradient descent training step on masked labeled nodes."""
        probs = self.forward(X, A_hat)  # shape (N, n_classes)
        N_train = np.sum(train_mask)
        if N_train == 0:
            return 0.0

        # Categorical cross-entropy loss over masked nodes
        labels = np.asarray(y, dtype=int)
        loss = -np.sum(np.log(probs[train_mask, labels[train_mask]] + 1e-12)) / N_train

        # Gradient wrt gc2 output (softmax cross entropy)
        dZ2 = np.zeros_like(probs)
        dZ2[train_mask] = probs[train_mask]
        dZ2[train_mask, labels[train_mask]] -= 1.0
        dZ2 /= N_train

        # Backprop through gc2: Z2 = (A_hat @ H1) @ W2 + b2
        agg1 = A_hat @ self.gc1.forward(X, A_hat)
        dW2 = agg1.T @ dZ2 + self.weight_decay * self.gc2.W
        db2 = np.sum(dZ2, axis=0)
        d_agg1 = dZ2 @ self.gc2.W.T
        dH1 = A_hat.T @ d_agg1

        # Backprop through gc1 ReLU:
        dZ1 = dH1 * (self.gc1.last_Z > 0).astype(float)
        agg0 = A_hat @ X
        dW1 = agg0.T @ dZ1 + self.weight_decay * self.gc1.W
        db1 = np.sum(dZ1, axis=0)

        # Update weights
        self.gc2.W -= self.learning_rate * dW2
        self.gc2.b -= self.learning_rate * db2
        self.gc1.W -= self.learning_rate * dW1
        self.gc1.b -= self.learning_rate * db1

        return float(loss)

    def predict(self, X: np.ndarray, A_hat: np.ndarray) -> np.ndarray:
        """Predict class indices for all nodes."""
        probs = self.forward(X, A_hat)
        return np.argmax(probs, axis=1)
