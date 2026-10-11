"""
Transfer Learning and Feature Extraction Mechanics from First Principles.

This module demonstrates the core principles of transfer learning:
1. Feature Extractor (Backbone): Learns generic semantic representations from inputs.
2. Parameter Freezing & Unfreezing: Prevents gradient updates on pretrained layers.
3. Classification Head Replacement: Replaces task-specific output layers with new target heads.
4. Two-Stage Fine-Tuning:
   - Stage 1: Freeze backbone, train new classification head.
   - Stage 2: Unfreeze backbone with a lower learning rate for end-to-end refinement.
"""

from typing import Any
import numpy as np


class DenseLayer:
    """
    Standard dense linear layer: y = ReLU(X @ W + b) or linear.
    """

    def __init__(self, in_features: int, out_features: int, activation: str = "relu", random_state: int | None = 42):
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.trainable = True

        rng = np.random.default_rng(random_state)
        scale = np.sqrt(2.0 / in_features) if activation == "relu" else np.sqrt(1.0 / in_features)
        self.W = rng.normal(0, scale, (in_features, out_features))
        self.b = np.zeros(out_features)

        # Cached activations for backprop
        self.last_input: np.ndarray | None = None
        self.last_preactivation: np.ndarray | None = None

        # Gradients
        self.grad_W: np.ndarray | None = None
        self.grad_b: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.last_input = x
        z = np.matmul(x, self.W) + self.b
        self.last_preactivation = z
        if self.activation == "relu":
            return np.maximum(0.0, z)
        return z

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Computes gradients w.r.t weights and input.
        """
        assert self.last_input is not None
        assert self.last_preactivation is not None

        if self.activation == "relu":
            grad_z = grad_output * (self.last_preactivation > 0).astype(float)
        else:
            grad_z = grad_output

        if self.trainable:
            self.grad_W = np.matmul(self.last_input.T, grad_z) / self.last_input.shape[0]
            self.grad_b = np.mean(grad_z, axis=0)
        else:
            self.grad_W = np.zeros_like(self.W)
            self.grad_b = np.zeros_like(self.b)

        # Gradient w.r.t input
        grad_input = np.matmul(grad_z, self.W.T)
        return grad_input


class PretrainedBackbone:
    """
    Pretrained sequential feature extractor network.
    """

    def __init__(self, layer_dims: list[int], random_state: int | None = 42):
        self.layers: list[DenseLayer] = []
        for i in range(len(layer_dims) - 1):
            seed = None if random_state is None else random_state + i
            self.layers.append(
                DenseLayer(layer_dims[i], layer_dims[i + 1], activation="relu", random_state=seed)
            )

    @property
    def output_dim(self) -> int:
        return self.layers[-1].out_features

    def freeze(self) -> None:
        """Freezes all layers in the backbone, disabling weight updates."""
        for layer in self.layers:
            layer.trainable = False

    def unfreeze(self) -> None:
        """Unfreezes all layers in the backbone for fine-tuning."""
        for layer in self.layers:
            layer.trainable = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad


class TransferClassifier:
    """
    Modular Transfer Learning Classifier.
    Mounts a custom classification head on top of a pretrained backbone.
    """

    def __init__(
        self,
        backbone: PretrainedBackbone,
        n_classes: int,
        random_state: int | None = 42,
    ):
        self.backbone = backbone
        self.n_classes = n_classes
        # Replace/attach new classification head
        self.head = DenseLayer(
            in_features=backbone.output_dim,
            out_features=n_classes,
            activation="linear",
            random_state=random_state,
        )

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(shifted)
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Extracts features via backbone and computes logits via classification head."""
        features = self.backbone.forward(x)
        logits = self.head.forward(features)
        return logits

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = self.forward(x)
        return self._softmax(logits)

    def predict(self, x: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(x)
        return np.argmax(probs, axis=1)

    def _step(self, x: np.ndarray, y_one_hot: np.ndarray, lr: float, backbone_lr: float | None = None) -> float:
        """
        Executes one forward pass, loss calculation, backprop, and parameter update.
        """
        if backbone_lr is None:
            backbone_lr = lr

        logits = self.forward(x)
        probs = self._softmax(logits)
        eps = 1e-12
        loss = -float(np.mean(np.sum(y_one_hot * np.log(probs + eps), axis=1)))

        # Gradient of Cross-Entropy loss w.r.t logits: (probs - target)
        grad_logits = (probs - y_one_hot)

        # Backpropagation through head
        grad_features = self.head.backward(grad_logits)

        # Backpropagation through backbone
        self.backbone.backward(grad_features)

        # Update head weights
        if self.head.trainable and self.head.grad_W is not None and self.head.grad_b is not None:
            self.head.W -= lr * self.head.grad_W
            self.head.b -= lr * self.head.grad_b

        # Update backbone weights if unfrozen
        for layer in self.backbone.layers:
            if layer.trainable and layer.grad_W is not None and layer.grad_b is not None:
                layer.W -= backbone_lr * layer.grad_W
                layer.b -= backbone_lr * layer.grad_b

        return loss

    def fit_head_only(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        lr: float = 0.05,
        batch_size: int = 32,
    ) -> list[float]:
        """
        Stage 1: Freeze backbone, optimize only the new classification head.
        """
        self.backbone.freeze()
        self.head.trainable = True
        return self._train_loop(X, y, epochs=epochs, lr=lr, backbone_lr=0.0, batch_size=batch_size)

    def fine_tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 30,
        head_lr: float = 0.01,
        backbone_lr: float = 0.001,
        batch_size: int = 32,
    ) -> list[float]:
        """
        Stage 2: Unfreeze backbone, train end-to-end with discriminative learning rates.
        """
        self.backbone.unfreeze()
        self.head.trainable = True
        return self._train_loop(X, y, epochs=epochs, lr=head_lr, backbone_lr=backbone_lr, batch_size=batch_size)

    def _train_loop(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        lr: float,
        backbone_lr: float,
        batch_size: int,
    ) -> list[float]:
        n_samples = X.shape[0]
        y_one_hot = np.zeros((n_samples, self.n_classes))
        y_one_hot[np.arange(n_samples), y] = 1.0

        history: list[float] = []
        rng = np.random.default_rng(42)

        for _ in range(epochs):
            indices = rng.permutation(n_samples)
            epoch_losses = []
            for start_idx in range(0, n_samples, batch_size):
                batch_idx = indices[start_idx : start_idx + batch_size]
                batch_loss = self._step(
                    X[batch_idx],
                    y_one_hot[batch_idx],
                    lr=lr,
                    backbone_lr=backbone_lr,
                )
                epoch_losses.append(batch_loss)
            history.append(float(np.mean(epoch_losses)))

        return history

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Evaluates accuracy and average cross-entropy loss."""
        preds = self.predict(X)
        acc = float(np.mean(preds == y))
        probs = self.predict_proba(X)
        y_one_hot = np.zeros((len(y), self.n_classes))
        y_one_hot[np.arange(len(y)), y] = 1.0
        loss = -float(np.mean(np.sum(y_one_hot * np.log(probs + 1e-12), axis=1)))
        return {"accuracy": acc, "loss": loss}
