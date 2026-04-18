# -*- coding: utf-8 -*-
"""
DMI Strategy — Indicator Calculations (Wilder 1978)

Computes +DI, -DI, ADX using Wilder smoothing (alpha=1/n, adjust=False).
Also detects Bill Williams fractal swings (5-bar pattern).
"""

import numpy as np
import pandas as pd


# ── DMI / ADX Calculation ─────────────────────────────────────

def compute_dmi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Compute +DI, -DI, DX, ADX using Wilder smoothing.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: DMI period (default 14, Wilder standard)

    Returns:
        DataFrame with 'plus_di', 'minus_di', 'dx', 'adx' columns
    """
    high = df['high']
    low = df['low']
    close = df['close']
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    # True Range
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    # Wilder smoothing (alpha = 1/period)
    alpha = 1.0 / period
    smoothed_tr = tr.ewm(alpha=alpha, adjust=False).mean()
    smoothed_pdm = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    smoothed_ndm = minus_dm.ewm(alpha=alpha, adjust=False).mean()

    # +DI, -DI
    plus_di = 100 * smoothed_pdm / smoothed_tr
    minus_di = 100 * smoothed_ndm / smoothed_tr

    # Handle zero division
    plus_di = plus_di.fillna(0)
    minus_di = minus_di.fillna(0)

    # DX
    di_sum = plus_di + minus_di
    dx = np.where(di_sum > 0, 100 * (plus_di - minus_di).abs() / di_sum, 0.0)
    dx = pd.Series(dx, index=df.index)

    # ADX (Wilder smoothing of DX)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    return pd.DataFrame({
        'plus_di': plus_di,
        'minus_di': minus_di,
        'dx': dx,
        'adx': adx,
    }, index=df.index)


# ── Fractal Swing Detection ───────────────────────────────────

def detect_fractal_swings(df: pd.DataFrame, K: int = 2) -> pd.DataFrame:
    """
    Bill Williams fractal swing detection (5-bar pattern).

    A bar t is a swing low if:
        Low[t] < Low[t-k] and Low[t] < Low[t+k] for k=1..K

    A bar t is a swing high if:
        High[t] > High[t-k] and High[t] > High[t+k] for k=1..K

    Args:
        df: DataFrame with 'high' and 'low' columns
        K: left/right confirmation bars (default 2)

    Returns:
        DataFrame with 'is_swing_low', 'is_swing_high' boolean columns
    """
    low = df['low']
    high = df['high']

    is_swing_low = pd.Series(True, index=df.index)
    is_swing_high = pd.Series(True, index=df.index)

    for k in range(1, K + 1):
        is_swing_low &= (low < low.shift(k))
        is_swing_high &= (high > high.shift(k))
        is_swing_low &= (low < low.shift(-k))
        is_swing_high &= (high > high.shift(-k))

    # First/last K bars cannot be confirmed
    is_swing_low.iloc[:K] = False
    is_swing_low.iloc[-K:] = False
    is_swing_high.iloc[:K] = False
    is_swing_high.iloc[-K:] = False

    return pd.DataFrame({
        'is_swing_low': is_swing_low,
        'is_swing_high': is_swing_high,
    }, index=df.index)


def get_last_confirmed_swing(
    swing_series: pd.Series,
    price_series: pd.Series,
    current_idx: int,
    lookback: int = 15,
    K: int = 2,
) -> float | None:
    """
    Get the price of the most recent confirmed swing within lookback window.

    Confirmed = at least K bars after the swing (so swing at t is
    confirmed at t+K, usable from t+K onward).

    Args:
        swing_series: boolean Series (is_swing_low or is_swing_high)
        price_series: price Series (low for swing_low, high for swing_high)
        current_idx: current bar index (integer position)
        lookback: number of bars to search back
        K: fractal confirmation delay

    Returns:
        Price of the last confirmed swing, or None if not found.
    """
    end_idx = current_idx - K  # latest confirmable
    start_idx = max(0, current_idx - lookback)

    if end_idx < start_idx:
        return None

    window = swing_series.iloc[start_idx:end_idx + 1]
    if not window.any():
        return None

    # Get last True position
    last_pos = window[window].index[-1]
    return float(price_series.loc[last_pos])
