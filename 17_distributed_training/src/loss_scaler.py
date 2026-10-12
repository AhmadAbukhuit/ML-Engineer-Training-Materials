"""Dynamic Loss Scaler for FP16 Mixed Precision Training."""

import numpy as np


class DynamicLossScaler:
    """Dynamic Loss Scaling for Mixed Precision (Micikevicius et al., 2018).

    Prevents underflow of small gradient values into zero under 16-bit floating point
    representation by scaling the loss during the forward pass and unscaling gradients
    prior to weight updates. Automatically adjusts the scaling factor based on
    numerical overflow/underflow detection.

    Parameters
    ----------
    initial_scale : float
        Initial loss multiplier S.
    growth_factor : float
        Multiplier when no overflow occurs for `growth_interval` steps.
    backoff_factor : float
        Multiplier applied immediately upon detecting inf/NaN gradients.
    growth_interval : int
        Number of consecutive successful steps required to increase scale.
    min_scale : float
        Lower bound for scaling factor.
    max_scale : float
        Upper bound for scaling factor.
    """

    def __init__(
        self,
        initial_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        min_scale: float = 1.0,
        max_scale: float = 2.0**24,
    ) -> None:
        self.scale = float(initial_scale)
        self.growth_factor = float(growth_factor)
        self.backoff_factor = float(backoff_factor)
        self.growth_interval = int(growth_interval)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)

        self.consecutive_successful_steps: int = 0

    def scale_loss(self, loss: float | np.ndarray) -> float | np.ndarray:
        """Multiply loss by current scale factor S prior to backward pass."""
        return loss * self.scale

    def unscale_gradients(
        self, gradients: list[np.ndarray]
    ) -> tuple[list[np.ndarray] | None, bool]:
        """Check for gradient overflow and unscale if finite.

        Parameters
        ----------
        gradients : list[np.ndarray]
            List of raw gradients calculated with scaled loss.

        Returns
        -------
        unscaled_gradients : list[np.ndarray] or None
            Unscaled gradients if finite, else None indicating step should be skipped.
        is_finite : bool
            True if all gradient tensors are finite (no NaN / Inf).
        """
        # Check for overflow
        for g in gradients:
            if not np.all(np.isfinite(g)):
                # Overflow occurred: reduce scale, skip step
                self.scale = max(self.scale * self.backoff_factor, self.min_scale)
                self.consecutive_successful_steps = 0
                return None, False

        # All finite: unscale gradients
        inv_scale = 1.0 / self.scale
        unscaled = [g * inv_scale for g in gradients]

        # Update success counter
        self.consecutive_successful_steps += 1
        if self.consecutive_successful_steps >= self.growth_interval:
            self.scale = min(self.scale * self.growth_factor, self.max_scale)
            self.consecutive_successful_steps = 0

        return unscaled, True
