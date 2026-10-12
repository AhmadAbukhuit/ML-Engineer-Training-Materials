"""Post-Training Quantization (INT8 & NF4) from First Principles in NumPy."""

from typing import Literal
import numpy as np


class Int8Quantizer:
    """8-Bit Integer (INT8) Quantization Engine.

    Parameters
    ----------
    mode : {'symmetric', 'asymmetric'}
        Symmetric keeps zero-point Z=0 (ideal for zero-centered weights).
        Asymmetric computes optimal scale S and zero-point Z (ideal for non-negative ReLU activations).
    """

    def __init__(self, mode: Literal["symmetric", "asymmetric"] = "symmetric") -> None:
        self.mode = mode
        self.scale: float = 1.0
        self.zero_point: int = 0

    def fit_calibrate(self, x: np.ndarray) -> "Int8Quantizer":
        """Compute quantization scale S and zero-point Z."""
        x = np.asarray(x, dtype=np.float64)

        if self.mode == "symmetric":
            max_abs = float(np.max(np.abs(x)))
            if max_abs == 0.0:
                max_abs = 1.0
            # Map [-max_abs, max_abs] to [-127, 127]
            self.scale = max_abs / 127.0
            self.zero_point = 0

        elif self.mode == "asymmetric":
            x_min = float(np.min(x))
            x_max = float(np.max(x))
            if x_max == x_min:
                x_max += 1e-5
            # Map [x_min, x_max] to [0, 255]
            self.scale = (x_max - x_min) / 255.0
            self.zero_point = int(np.round(-x_min / self.scale))
            self.zero_point = int(np.clip(self.zero_point, 0, 255))

        return self

    def quantize(self, x: np.ndarray) -> np.ndarray:
        """Quantize float32 tensor to 8-bit integer representation."""
        x = np.asarray(x, dtype=np.float64)
        if self.mode == "symmetric":
            q = np.round(x / self.scale)
            return np.clip(q, -128, 127).astype(np.int8)
        else:
            q = np.round(x / self.scale) + self.zero_point
            return np.clip(q, 0, 255).astype(np.uint8)

    def dequantize(self, q: np.ndarray) -> np.ndarray:
        """Dequantize 8-bit integers back to floating-point values."""
        q = np.asarray(q, dtype=np.float64)
        return self.scale * (q - self.zero_point)

    def quantization_error(self, x: np.ndarray) -> dict[str, float]:
        """Compute reconstruction RMSE and SNR degradation."""
        q = self.quantize(x)
        x_recon = self.dequantize(q)
        rmse = float(np.sqrt(np.mean((x - x_recon) ** 2)))
        signal_power = float(np.mean(x**2))
        noise_power = float(np.mean((x - x_recon) ** 2)) + 1e-12
        snr_db = float(10.0 * np.log10(signal_power / noise_power))
        return {"rmse": rmse, "snr_db": snr_db}


class NormalFloat4Quantizer:
    """NormalFloat4 (NF4) Quantization Engine (Dettmers et al. - QLoRA).

    An information-theoretically optimal quantile codebook for Gaussian distributed parameters.
    """

    # 16 optimal NF4 quantile points for N(0, 1) normalized to [-1, 1]
    NF4_CODEBOOK = np.array([
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ])

    def quantize(self, weights: np.ndarray) -> tuple[np.ndarray, float]:
        """Quantize weights to 4-bit NF4 indices (0-15).

        Returns
        -------
        indices : np.ndarray of shape weights.shape (dtype uint8)
        absmax : float
        """
        weights = np.asarray(weights, dtype=np.float64)
        absmax = float(np.max(np.abs(weights)))
        if absmax == 0:
            absmax = 1.0

        normalized = weights / absmax
        # Find closest NF4 quantile
        # Shape: (N, 16) distance matrix
        flat = normalized.flatten()
        diff = np.abs(flat[:, np.newaxis] - self.NF4_CODEBOOK[np.newaxis, :])
        indices = np.argmin(diff, axis=1).reshape(weights.shape).astype(np.uint8)

        return indices, absmax

    def dequantize(self, indices: np.ndarray, absmax: float) -> np.ndarray:
        """Dequantize 4-bit NF4 indices back to full-precision weights."""
        return self.NF4_CODEBOOK[indices] * absmax
