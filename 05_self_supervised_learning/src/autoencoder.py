"""Deep Autoencoder and Denoising Autoencoder from First Principles in NumPy."""

import numpy as np


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(np.float64)


class DeepAutoencoder:
    """Deep Bottleneck Autoencoder with backpropagation in NumPy.

    Parameters
    ----------
    input_dim : int
        Dimension of input features.
    hidden_dim : int
        Dimension of intermediate encoder/decoder layer.
    latent_dim : int
        Dimension of bottleneck latent representation.
    noise_factor : float
        Noise standard deviation for Denoising Autoencoder (DAE) training.
    learning_rate : float
        Gradient descent learning rate.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        latent_dim: int = 8,
        noise_factor: float = 0.0,
        learning_rate: float = 0.01,
        random_state: int | None = 42,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.noise_factor = noise_factor
        self.learning_rate = learning_rate
        self.random_state = random_state

        rng = np.random.default_rng(random_state)
        # He initialization
        self.W1 = rng.normal(0, np.sqrt(2.0 / input_dim), (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)

        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, latent_dim))
        self.b2 = np.zeros(latent_dim)

        # Decoder weights (tied or independent)
        self.W3 = rng.normal(0, np.sqrt(2.0 / latent_dim), (latent_dim, hidden_dim))
        self.b3 = np.zeros(hidden_dim)

        self.W4 = rng.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, input_dim))
        self.b4 = np.zeros(input_dim)

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Project input into compressed latent space."""
        X = np.asarray(X, dtype=np.float64)
        h1 = _relu(np.dot(X, self.W1) + self.b1)
        z = np.dot(h1, self.W2) + self.b2
        return z

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Reconstruct input from latent representation."""
        z = np.asarray(z, dtype=np.float64)
        h2 = _relu(np.dot(z, self.W3) + self.b3)
        X_recon = np.dot(h2, self.W4) + self.b4
        return X_recon

    def forward(
        self, X: np.ndarray, is_training: bool = False
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Forward pass caching intermediate activations for backprop."""
        X = np.asarray(X, dtype=np.float64)
        X_in = X
        if is_training and self.noise_factor > 0.0:
            rng = np.random.default_rng(self.random_state)
            X_in = X + rng.normal(0.0, self.noise_factor, X.shape)

        a1 = np.dot(X_in, self.W1) + self.b1
        h1 = _relu(a1)

        z = np.dot(h1, self.W2) + self.b2

        a2 = np.dot(z, self.W3) + self.b3
        h2 = _relu(a2)

        X_recon = np.dot(h2, self.W4) + self.b4

        cache = {
            "X_in": X_in,
            "a1": a1,
            "h1": h1,
            "z": z,
            "a2": a2,
            "h2": h2,
        }
        return X_recon, z, cache

    def train_step(self, X: np.ndarray) -> float:
        """Perform one step of gradient descent on reconstruction MSE."""
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]

        X_recon, _, cache = self.forward(X, is_training=True)
        loss = float(np.mean((X_recon - X) ** 2))

        # Backward pass
        # dL/dX_recon = 2 * (X_recon - X) / n
        dX_recon = 2.0 * (X_recon - X) / n

        # Layer 4
        dW4 = np.dot(cache["h2"].T, dX_recon)
        db4 = np.sum(dX_recon, axis=0)
        dh2 = np.dot(dX_recon, self.W4.T)

        # Layer 3
        da2 = dh2 * _relu_grad(cache["a2"])
        dW3 = np.dot(cache["z"].T, da2)
        db3 = np.sum(da2, axis=0)
        dz = np.dot(da2, self.W3.T)

        # Layer 2
        dW2 = np.dot(cache["h1"].T, dz)
        db2 = np.sum(dz, axis=0)
        dh1 = np.dot(dz, self.W2.T)

        # Layer 1
        da1 = dh1 * _relu_grad(cache["a1"])
        dW1 = np.dot(cache["X_in"].T, da1)
        db1 = np.sum(da1, axis=0)

        # Gradient clipping for stability
        clip_val = 1.0
        dW4 = np.clip(dW4, -clip_val, clip_val)
        db4 = np.clip(db4, -clip_val, clip_val)
        dW3 = np.clip(dW3, -clip_val, clip_val)
        db3 = np.clip(db3, -clip_val, clip_val)
        dW2 = np.clip(dW2, -clip_val, clip_val)
        db2 = np.clip(db2, -clip_val, clip_val)
        dW1 = np.clip(dW1, -clip_val, clip_val)
        db1 = np.clip(db1, -clip_val, clip_val)

        # Gradient update
        self.W4 -= self.learning_rate * dW4
        self.b4 -= self.learning_rate * db4
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

        return loss

    def fit(self, X: np.ndarray, epochs: int = 50) -> "DeepAutoencoder":
        """Train autoencoder on data matrix X."""
        for _ in range(epochs):
            self.train_step(X)
        return self
