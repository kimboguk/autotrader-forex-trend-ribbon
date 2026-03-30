# -*- coding: utf-8 -*-
"""
Golden Cross Strategy - Signal Generation

Two moving averages (fast/slow). Golden cross = buy, dead cross = sell.

Signal codes (same convention as trade_engine):
   0: no change
   1: long entry (or short exit)
  -1: short entry (or long exit)
   2: short exit + long entry (reverse)
  -2: long exit + short entry (reverse)
"""

import numpy as np
import pandas as pd


def calc_ma(series: pd.Series, period: int, ma_type: str = "ema") -> pd.Series:
    """Calculate moving average on a Series."""
    if ma_type == "ema":
        return series.ewm(span=period, min_periods=period, adjust=False).mean()
    elif ma_type == "sma":
        return series.rolling(window=period, min_periods=period).mean()
    raise ValueError(f"Unknown MA type: {ma_type}")


def compute_grid(df: pd.DataFrame, fast_period: int = 50, slow_period: int = 200,
                 ma_type: str = "ema",
                 use_kalman: bool = False, kalman_qr_ratio: float = 0.1) -> pd.DataFrame:
    """Compute fast and slow MAs."""
    result = df.copy()

    if use_kalman:
        from kalman_price_filter import KalmanPriceFilter
        Q = 1e-3
        R = Q / kalman_qr_ratio
        kf_close = KalmanPriceFilter(Q=Q, R=R)
        kf_open = KalmanPriceFilter(Q=Q, R=R)
        import numpy as np
        result["close"] = np.array([kf_close.update(c) for c in result["close"].values])
        result["open"] = np.array([kf_open.update(o) for o in result["open"].values])

    result[f"ma_{fast_period}"] = calc_ma(result["close"], fast_period, ma_type)
    result[f"ma_{slow_period}"] = calc_ma(result["close"], slow_period, ma_type)
    return result


def generate_signals(df: pd.DataFrame, ma_type: str = "ema",
                     fast_period: int = 50, slow_period: int = 200,
                     use_kalman: bool = False, kalman_qr_ratio: float = 0.1,
                     **kwargs) -> pd.DataFrame:
    """
    Generate golden/dead cross signals.

    Returns DataFrame with 'signal' and 'position' columns added.
    """
    grid = compute_grid(df, fast_period, slow_period, ma_type,
                        use_kalman=use_kalman, kalman_qr_ratio=kalman_qr_ratio)

    n = len(grid)
    signals = np.zeros(n, dtype=int)
    positions = np.zeros(n, dtype=int)

    fast_col = f"ma_{fast_period}"
    slow_col = f"ma_{slow_period}"
    fast_arr = grid[fast_col].values
    slow_arr = grid[slow_col].values

    warmup = slow_period + 1
    position = 0

    for i in range(warmup, n):
        fast = fast_arr[i]
        slow = slow_arr[i]
        prev_fast = fast_arr[i - 1]
        prev_slow = slow_arr[i - 1]

        if np.isnan(fast) or np.isnan(slow) or np.isnan(prev_fast) or np.isnan(prev_slow):
            positions[i] = position
            continue

        # Cross detection
        golden_cross = (prev_fast <= prev_slow) and (fast > slow)
        dead_cross = (prev_fast >= prev_slow) and (fast < slow)

        if position == 0:
            if golden_cross:
                signals[i] = 1
                position = 1
            elif dead_cross:
                signals[i] = -1
                position = -1

        elif position == 1:
            if dead_cross:
                signals[i] = -2  # long exit + short entry
                position = -1

        elif position == -1:
            if golden_cross:
                signals[i] = 2   # short exit + long entry
                position = 1

        positions[i] = position

    grid["signal"] = signals
    grid["position"] = positions
    return grid
