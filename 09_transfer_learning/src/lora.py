"""
Low-Rank Adaptation (LoRA) for Parameter-Efficient Fine-Tuning.

This module implements:
1. LoRALinear: Layer that freezes base weight W_0 and adds trainable low-rank matrices A and B:
   y = x @ W_0 + (alpha / r) * (x @ A @ B)
2. Weight Merging: Folds adapter weights into base weights for zero-overhead inference:
   W_merged = W_0 + (alpha / r) * (A @ B)
3. Parameter Efficiency Accounting: Quantifies parameter reduction ratio compared to full fine-tuning.
"""

import numpy as np


class LoRALinear:
    """
    Parameter-Efficient Fine-Tuning (PEFT) Linear Layer via Low-Rank Adaptation.

    Decomposes the weight update matrix Delta W into two low-rank matrices:
    Delta W = (alpha / r) * (A @ B)
    Where:
        A has shape (in_features, r), initialized with Gaussian noise.
        B has shape (r, out_features), initialized to zeros (so Delta W starts at 0).
        r << min(in_features, out_features).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 8.0,
        random_state: int | None = 42,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        rng = np.random.default_rng(random_state)

        # Frozen base weight W_0
        scale_w = np.sqrt(2.0 / in_features)
        self.W_0 = rng.normal(0, scale_w, (in_features, out_features))
        self.bias = np.zeros(out_features)

        # Trainable low-rank adapter matrices
        # Matrix A: initialized with normal distribution
        scale_a = np.sqrt(1.0 / in_features)
        self.lora_A = rng.normal(0, scale_a, (in_features, rank))
        # Matrix B: initialized to zeros so that initially Delta W = 0
        self.lora_B = np.zeros((rank, out_features))

        self.merged = False

        # Cached activations for backprop
        self.last_x: np.ndarray | None = None
        self.grad_A: np.ndarray | None = None
        self.grad_B: np.ndarray | None = None

    def parameter_counts(self) -> dict[str, int]:
        """Calculates base parameters, trainable LoRA parameters, and savings percentage."""
        base_params = self.in_features * self.out_features + self.out_features
        trainable_lora_params = (self.in_features * self.rank) + (self.rank * self.out_features)
        reduction_ratio = base_params / max(1, trainable_lora_params)
        return {
            "base_params": base_params,
            "trainable_lora_params": trainable_lora_params,
            "reduction_ratio": round(reduction_ratio, 2),
        }

    def merge_weights(self) -> None:
        """
        Folds the low-rank delta directly into the base weights for zero-overhead inference:
        W_0 <- W_0 + (alpha / r) * (A @ B)
        """
        if self.merged:
            return
        delta_w = self.scaling * np.matmul(self.lora_A, self.lora_B)
        self.W_0 += delta_w
        self.merged = True

    def unmerge_weights(self) -> None:
        """Subtracts the low-rank delta from base weights to re-enable adapter training."""
        if not self.merged:
            return
        delta_w = self.scaling * np.matmul(self.lora_A, self.lora_B)
        self.W_0 -= delta_w
        self.merged = False

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        If merged: y = x @ W_0 + bias
        If unmerged: y = x @ W_0 + (alpha / r) * ((x @ A) @ B) + bias
        """
        self.last_x = x
        if self.merged:
            return np.matmul(x, self.W_0) + self.bias

        base_out = np.matmul(x, self.W_0)
        # Compute low-rank path: (x @ A) has shape (batch_size, rank)
        lora_h = np.matmul(x, self.lora_A)
        # (lora_h @ B) has shape (batch_size, out_features)
        lora_out = np.matmul(lora_h, self.lora_B) * self.scaling

        return base_out + lora_out + self.bias

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Computes gradients for LoRA matrices A and B while base weights remain frozen.
        """
        if self.merged:
            raise RuntimeError("Cannot compute LoRA gradients when weights are merged. Call unmerge_weights() first.")
        assert self.last_x is not None

        # grad_output shape: (batch_size, out_features)
        # grad_lora_out = grad_output * scaling
        grad_lora = grad_output * self.scaling
        lora_h = np.matmul(self.last_x, self.lora_A)  # (batch_size, rank)

        # Grad w.r.t B: lora_h.T @ grad_lora
        self.grad_B = np.matmul(lora_h.T, grad_lora) / self.last_x.shape[0]

        # Grad w.r.t lora_h: grad_lora @ B.T
        grad_h = np.matmul(grad_lora, self.lora_B.T)

        # Grad w.r.t A: x.T @ grad_h
        self.grad_A = np.matmul(self.last_x.T, grad_h) / self.last_x.shape[0]

        # Grad w.r.t input x (combining base and LoRA pathways)
        delta_w = self.scaling * np.matmul(self.lora_A, self.lora_B)
        grad_input = np.matmul(grad_output, (self.W_0 + delta_w).T)

        return grad_input

    def update_lora(self, lr: float) -> None:
        """Applies gradient descent step to trainable LoRA parameters."""
        if self.grad_A is not None:
            self.lora_A -= lr * self.grad_A
        if self.grad_B is not None:
            self.lora_B -= lr * self.grad_B
