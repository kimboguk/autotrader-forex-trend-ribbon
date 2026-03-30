# -*- coding: utf-8 -*-
"""
1-State Kalman Filter for price noise reduction.

Filters raw close prices to reduce tick noise and spikes
before EMA/grid computation. Designed for bar-by-bar updates
compatible with both backtesting and live trading.

Usage:
    kf = KalmanPriceFilter(Q=1e-5, R=1e-3)
    for close in closes:
        filtered = kf.update(close)
"""

import numpy as np


class KalmanPriceFilter:
    """
    1-dimensional Kalman filter for price smoothing.

    State: estimated "true" price
    Observation: raw close price

    Parameters:
        Q: Process noise variance — how much the true price can change per bar.
           Larger Q → filter follows price more closely (less smoothing).
        R: Observation noise variance — how noisy the raw prices are.
           Larger R → filter trusts observations less (more smoothing).
    """

    def __init__(self, Q: float = 1e-5, R: float = 1e-3):
        self.Q = Q  # process noise
        self.R = R  # observation noise
        self.x = None  # state estimate (filtered price)
        self.P = 1.0   # error covariance (initial uncertainty)

    def update(self, z: float) -> float:
        """
        Update filter with new observation.

        Args:
            z: raw close price

        Returns:
            Filtered (smoothed) close price
        """
        if self.x is None:
            # First observation — initialize state
            self.x = z
            self.P = self.R
            return z

        # Predict
        x_pred = self.x          # price doesn't change (random walk model)
        P_pred = self.P + self.Q

        # Update
        K = P_pred / (P_pred + self.R)  # Kalman gain
        self.x = x_pred + K * (z - x_pred)
        self.P = (1 - K) * P_pred

        return self.x

    def reset(self):
        """Reset filter state."""
        self.x = None
        self.P = 1.0

    def get_state(self) -> dict:
        """Export state for persistence."""
        return {"x": self.x, "P": self.P}

    def restore_state(self, state: dict):
        """Restore state from persistence."""
        self.x = state.get("x")
        self.P = state.get("P", 1.0)


def apply_kalman_to_series(closes: np.ndarray, Q: float = 1e-5, R: float = 1e-3) -> np.ndarray:
    """
    Apply Kalman filter to an array of close prices.

    Args:
        closes: array of raw close prices
        Q: process noise
        R: observation noise

    Returns:
        Array of filtered close prices (same length)
    """
    kf = KalmanPriceFilter(Q=Q, R=R)
    filtered = np.empty_like(closes)
    for i in range(len(closes)):
        if np.isnan(closes[i]):
            filtered[i] = closes[i]
        else:
            filtered[i] = kf.update(closes[i])
    return filtered
