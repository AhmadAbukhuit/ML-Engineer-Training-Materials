"""SimCLR (Simple Framework for Contrastive Learning) from First Principles in NumPy."""

import numpy as np

try:
    from .contrastive_loss import NTXentLoss
except ImportError:
    from contrastive_loss import NTXentLoss


class ProjectionHead:
    """Non-linear MLP projection head mapping encoder representations to contrastive space."""

    def __init__(
        self, in_dim: int, hidden_dim: int = 32, out_dim: int = 16, random_state: int | None = 42
    ) -> None:
        rng = np.random.default_rng(random_state)
        self.W1 = rng.normal(0, np.sqrt(2.0 / in_dim), (in_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, out_dim))
        self.b2 = np.zeros(out_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0.0, np.dot(x, self.W1) + self.b1)
        z = np.dot(h, self.W2) + self.b2
        return z


class SimCLRFramework:
    """SimCLR Contrastive Learning Pipeline.

    Integrates:
    - Data augmentation operators for generating multiple correlated views.
    - Base feature encoder f(x) -> h.
    - Projection head g(h) -> z.
    - NT-Xent / InfoNCE loss evaluation.
    """

    def __init__(
        self,
        in_dim: int,
        feature_dim: int = 32,
        proj_dim: int = 16,
        temperature: float = 0.5,
        random_state: int | None = 42,
    ) -> None:
        self.temperature = temperature
        self.random_state = random_state

        rng = np.random.default_rng(random_state)
        # Base encoder weights
        self.W_enc = rng.normal(0, np.sqrt(2.0 / in_dim), (in_dim, feature_dim))
        self.b_enc = np.zeros(feature_dim)

        self.projector = ProjectionHead(
            in_dim=feature_dim,
            hidden_dim=feature_dim,
            out_dim=proj_dim,
            random_state=random_state,
        )
        self.loss_fn = NTXentLoss(temperature=temperature)

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Extract high-level features h = f(X) for downstream linear probe."""
        X = np.asarray(X, dtype=np.float64)
        return np.maximum(0.0, np.dot(X, self.W_enc) + self.b_enc)

    def project(self, h: np.ndarray) -> np.ndarray:
        """Project representations into contrastive metric space z = g(h)."""
        return self.projector.forward(h)

    def augment_view(
        self, X: np.ndarray, noise_std: float = 0.05, mask_prob: float = 0.1
    ) -> np.ndarray:
        """Stochastic data augmentation transform."""
        rng = np.random.default_rng(self.random_state)
        # Additive Gaussian jitter
        jittered = X + rng.normal(0.0, noise_std, X.shape)
        # Feature masking dropout
        mask = rng.binomial(1, 1.0 - mask_prob, X.shape)
        return jittered * mask

    def compute_loss(self, X: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        """Generate two augmented views and compute NT-Xent contrastive loss."""
        v1 = self.augment_view(X, noise_std=0.04)
        v2 = self.augment_view(X, noise_std=0.08)

        h1 = self.encode(v1)
        h2 = self.encode(v2)

        z1 = self.project(h1)
        z2 = self.project(h2)

        loss = self.loss_fn(z1, z2)
        return loss, z1, z2
