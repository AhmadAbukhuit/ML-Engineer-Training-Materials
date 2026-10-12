"""Streaming Concept Drift Detectors: ADWIN and Page-Hinkley."""

import numpy as np


class ADWIN:
    """Adaptive Windowing (ADWIN) streaming change detector (Bifet & Gavaldà).

    Maintains a variable-length window W of recent samples.
    When a statistically significant difference in means between two sub-windows
    W_0 and W_1 is observed, ADWIN drops the older sub-window W_0 and flags drift.

    Parameters
    ----------
    delta : float
        Confidence parameter in (0, 1). Lower delta means fewer false positive alerts.
    max_buckets : int
        Maximum window capacity.
    """

    def __init__(self, delta: float = 0.002, max_buckets: int = 1000) -> None:
        self.delta = delta
        self.max_buckets = max_buckets
        self.window: list[float] = []
        self.drift_detected: bool = False

    def update(self, value: float) -> bool:
        """Add a streaming sample and evaluate whether concept drift occurred."""
        self.window.append(float(value))
        if len(self.window) > self.max_buckets:
            self.window.pop(0)

        self.drift_detected = False
        n = len(self.window)
        if n < 20:
            return False

        window_arr = np.array(self.window)

        # Evaluate potential partition split points
        step = max(1, n // 10)
        for split in range(step, n - step + 1, step):
            w0 = window_arr[:split]
            w1 = window_arr[split:]

            n0 = len(w0)
            n1 = len(w1)
            mu0 = np.mean(w0)
            mu1 = np.mean(w1)

            # Harmonic sample size m
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            delta_prime = self.delta / n

            # Hoeffding-like cut threshold
            eps_cut = np.sqrt((1.0 / (2.0 * m)) * np.log(4.0 / delta_prime))

            if np.abs(mu0 - mu1) >= eps_cut:
                self.drift_detected = True
                # Drop older sub-window W0
                self.window = self.window[split:]
                break

        return self.drift_detected


class PageHinkley:
    """Page-Hinkley Cumulative Sum Change Detector.

    Monitors a streaming scalar metric (e.g. loss or error rate) and triggers
    an alarm when the cumulative positive deviation from the running mean exceeds threshold lambda.

    Parameters
    ----------
    threshold : float
        Detection threshold lambda (> 0).
    alpha : float
        Tolerance / magnitude discount factor delta.
    """

    def __init__(self, threshold: float = 20.0, alpha: float = 0.05) -> None:
        self.threshold = threshold
        self.alpha = alpha

        self.sample_count = 0
        self.mean_val = 0.0
        self.cum_sum = 0.0
        self.min_cum_sum = 0.0
        self.drift_detected = False

    def update(self, value: float) -> bool:
        """Update detector with streaming observation."""
        self.sample_count += 1
        # Update running mean
        self.mean_val += (value - self.mean_val) / self.sample_count

        # Cumulative difference
        self.cum_sum += value - self.mean_val - self.alpha
        self.min_cum_sum = min(self.min_cum_sum, self.cum_sum)

        # Check threshold
        statistic = self.cum_sum - self.min_cum_sum
        if statistic > self.threshold:
            self.drift_detected = True
            # Reset
            self.cum_sum = 0.0
            self.min_cum_sum = 0.0
        else:
            self.drift_detected = False

        return self.drift_detected
