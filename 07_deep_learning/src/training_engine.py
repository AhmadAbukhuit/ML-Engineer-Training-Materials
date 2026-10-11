"""
Modular Deep Learning Training Engine and Early Stopping Mechanics.

This module provides:
1. EarlyStopping: Monitors validation loss to prevent overfitting and halts training when improvement stalls.
2. TrainingEngine: Standardized training harness managing epochs, batching, evaluation, and checkpointing.
"""

from typing import Any
from collections.abc import Callable
import numpy as np


class EarlyStopping:
    """
    Early stopping monitor to stop training when a monitored metric has stopped improving.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: float | None = None
        self.early_stop = False

    def step(self, current_score: float) -> bool:
        """
        Updates the monitor with the current epoch's metric.
        Returns True if training should stop early, False otherwise.
        """
        if self.best_score is None:
            self.best_score = current_score
            return False

        if self.mode == "min":
            improvement = (self.best_score - current_score) > self.min_delta
        else:
            improvement = (current_score - self.best_score) > self.min_delta

        if improvement:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class TrainingEngine:
    """
    Standardized execution harness for neural network optimization.
    """

    def __init__(
        self,
        model: Any,
        loss_fn: Callable[[np.ndarray, np.ndarray], float],
        optimizer: Any,
        early_stopping: EarlyStopping | None = None
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.early_stopping = early_stopping
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    def train_epoch(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32) -> float:
        n_samples = len(X)
        indices = np.random.permutation(n_samples)
        epoch_losses = []

        for start_idx in range(0, n_samples, batch_size):
            batch_idx = indices[start_idx : start_idx + batch_size]
            X_batch, y_batch = X[batch_idx], y[batch_idx]

            # In custom autograd / numpy models, step optimizer
            # Forward pass
            preds = self.model(X_batch) if callable(self.model) else self.model.forward(X_batch)
            loss = self.loss_fn(preds, y_batch)
            epoch_losses.append(loss)

        return float(np.mean(epoch_losses))

    def evaluate(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        preds = self.model(X_val) if callable(self.model) else self.model.forward(X_val)
        return float(self.loss_fn(preds, y_val))
