"""
Residual Block and Skip-Connection Architectures.

This module implements a foundational Residual Block:
y = Activation(F(x) + shortcut(x))
Where shortcut is the identity mapping if dimensions match, or a linear projection if channel dimensions change.
"""

import numpy as np


class ResidualBlock:
    """
    A 2-layer residual block with additive skip connection.
    y = ReLU(Linear2(ReLU(Linear1(x))) + Shortcut(x))
    """

    def __init__(self, in_features: int, out_features: int, random_state: int | None = 42):
        self.in_features = in_features
        self.out_features = out_features

        rng = np.random.default_rng(random_state)
        # He (Kaiming) normal initialization for ReLU activations
        scale1 = np.sqrt(2.0 / in_features)
        self.W1 = rng.normal(0, scale1, (in_features, out_features))
        self.b1 = np.zeros(out_features)

        scale2 = np.sqrt(2.0 / out_features)
        self.W2 = rng.normal(0, scale2, (out_features, out_features))
        self.b2 = np.zeros(out_features)

        # Shortcut projection if dimension changes
        if in_features != out_features:
            scale_s = np.sqrt(1.0 / in_features)
            self.W_s = rng.normal(0, scale_s, (in_features, out_features))
        else:
            self.W_s = None

    @staticmethod
    def _relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with additive residual shortcut.
        """
        # Shortcut pathway
        shortcut = x if self.W_s is None else np.matmul(x, self.W_s)

        # Residual branch: Layer 1
        h1 = self._relu(np.matmul(x, self.W1) + self.b1)

        # Layer 2
        h2 = np.matmul(h1, self.W2) + self.b2

        # Combine residual signal with identity shortcut and apply final activation
        out = self._relu(h2 + shortcut)
        return out
