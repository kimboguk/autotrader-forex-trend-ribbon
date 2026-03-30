# -*- coding: utf-8 -*-
"""
Trend Grid Strategy - core signal logic

4 VWMA grid (30, 60, 120, 240) breakout trend-following.

Entry:
  - Long:  prev body_mid <= prev grid_top, curr body_mid > curr grid_top, bullish
  - Short: prev body_mid >= prev grid_bottom, curr body_mid < curr grid_bottom, bearish

Exit (no transition check needed - position state proves we were outside):
  - Long exit:  bearish candle + body_mid < grid_top (price returns into grid)
  - Short exit: bullish candle + body_mid > grid_bottom (price returns into grid)

Price: signal candle close
"""

import numpy as np
import pandas as pd

from config import VWMA_PERIODS, MA_TYPE


# -- MA calculation -------------------------------------------------------

def calc_vwma(df: pd.DataFrame, period: int) -> pd.Series:
    """Volume Weighted Moving Average = SUM(close*vol) / SUM(vol)"""
    pv = df["close"] * df["tick_volume"]
    return (pv.rolling(window=period, min_periods=period).sum() /
            df["tick_volume"].rolling(window=period, min_periods=period).sum())


def calc_sma(df: pd.DataFrame, period: int) -> pd.Series:
    """Simple Moving Average (for comparison)"""
    return df["close"].rolling(window=period, min_periods=period).mean()


def calc_ema(df: pd.DataFrame, period: int) -> pd.Series:
    """Exponential Moving Average"""
    return df["close"].ewm(span=period, min_periods=period, adjust=False).mean()


def calc_wma(df: pd.DataFrame, period: int) -> pd.Series:
    """Weighted Moving Average — linear weights (newest bar gets highest weight)"""
    weights = np.arange(1, period + 1, dtype=np.float64)
    return df["close"].rolling(window=period, min_periods=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def calc_ma(df: pd.DataFrame, period: int, ma_type: str = None) -> pd.Series:
    ma_type = ma_type or MA_TYPE
    if ma_type == "vwma":
        return calc_vwma(df, period)
    elif ma_type == "sma":
        return calc_sma(df, period)
    elif ma_type == "ema":
        return calc_ema(df, period)
    elif ma_type == "wma":
        return calc_wma(df, period)
    raise ValueError(f"Unknown MA type: {ma_type}")


# -- Grid computation -----------------------------------------------------

def compute_grid(df: pd.DataFrame, ma_type: str = None, periods: list = None,
                  use_kalman: bool = False, kalman_qr_ratio: float = 0.1) -> pd.DataFrame:
    result = df.copy()

    # Apply Kalman filter if enabled (body+ema mode)
    # Filtered values used for EMA/grid/signal computation only.
    # Raw open/close preserved for trade execution prices.
    if use_kalman:
        from kalman_price_filter import KalmanPriceFilter
        Q = 1e-3
        R = Q / kalman_qr_ratio
        kf_close = KalmanPriceFilter(Q=Q, R=R)
        kf_open = KalmanPriceFilter(Q=Q, R=R)
        filtered_close = np.array([kf_close.update(c) for c in result["close"].values])
        filtered_open = np.array([kf_open.update(o) for o in result["open"].values])
        # Save raw prices for execution
        result["raw_open"] = result["open"].values.copy()
        result["raw_close"] = result["close"].values.copy()
        # Replace with filtered for EMA/grid computation
        result["close"] = filtered_close
        result["open"] = filtered_open

    # 항상 4개 MA 모두 계산 (chart/alignment용)
    for p in VWMA_PERIODS:
        result[f"ma_{p}"] = calc_ma(result, p, ma_type)

    # grid는 선택된 periods로만 구성
    active_periods = periods if periods else VWMA_PERIODS
    ma_cols = [f"ma_{p}" for p in active_periods]
    result["grid_top"] = result[ma_cols].max(axis=1)
    result["grid_bottom"] = result[ma_cols].min(axis=1)
    result["body_mid"] = (result["open"] + result["close"]) / 2
    result["is_bullish"] = result["close"] >= result["open"]

    # Restore raw prices for execution (open/close columns used by trade_engine)
    if use_kalman:
        result["open"] = result["raw_open"]
        result["close"] = result["raw_close"]

    return result


# -- Signal generation -----------------------------------------------------

def generate_signals(df: pd.DataFrame, ma_type: str = None, periods: list = None, relaxed_entry: bool = True,
                      use_kalman: bool = False, kalman_qr_ratio: float = 0.1) -> pd.DataFrame:
    """
    Signal codes:
       0: no change
       1: long entry (or short exit only)
      -1: short entry (or long exit only)
       2: short exit + long entry
      -2: long exit + short entry
    """
    grid = compute_grid(df, ma_type, periods, use_kalman=use_kalman,
                        kalman_qr_ratio=kalman_qr_ratio)

    n = len(grid)
    signals = np.zeros(n, dtype=int)
    positions = np.zeros(n, dtype=int)

    # to numpy for speed
    bm_arr = grid["body_mid"].values
    gt_arr = grid["grid_top"].values
    gb_arr = grid["grid_bottom"].values
    bull_arr = grid["is_bullish"].values
    open_arr = grid["open"].values
    close_arr = grid["close"].values

    warmup = max(VWMA_PERIODS) + 1
    position = 0

    for i in range(warmup, n):
        gt = gt_arr[i]          # current grid_top
        gb = gb_arr[i]          # current grid_bottom
        prev_gt = gt_arr[i - 1] # previous grid_top
        prev_gb = gb_arr[i - 1] # previous grid_bottom
        bm = bm_arr[i]
        prev_bm = bm_arr[i - 1]
        bull = bull_arr[i]

        if np.isnan(gt) or np.isnan(gb) or np.isnan(prev_gt) or np.isnan(prev_gb):
            positions[i] = position
            continue

        # -- entry checks (transition + direction) --
        # Standard: prev body_mid was on the other side of grid
        prev_below_top = prev_bm <= prev_gt
        prev_above_bot = prev_bm >= prev_gb

        # Relaxed: prev bar straddled the grid boundary (body crosses it)
        if relaxed_entry:
            prev_open = open_arr[i - 1]
            prev_close = close_arr[i - 1]
            prev_bull = bull_arr[i - 1]
            # Bearish candle straddling grid_top: close < grid_top < open
            if not prev_bull and prev_close < prev_gt < prev_open:
                prev_below_top = True
            # Bullish candle straddling grid_bottom: open < grid_bottom < close
            if prev_bull and prev_open < prev_gb < prev_close:
                prev_above_bot = True

        long_entry = bull and (bm > gt) and prev_below_top
        short_entry = (not bull) and (bm < gb) and prev_above_bot

        if position == 0:
            if long_entry:
                signals[i] = 1
                position = 1
            elif short_entry:
                signals[i] = -1
                position = -1

        elif position == 1:
            # long held -> exit when bearish + body_mid < grid_top
            if (not bull) and (bm < gt):
                # also check simultaneous short entry
                if short_entry:
                    signals[i] = -2   # long exit + short entry
                    position = -1
                else:
                    signals[i] = -1   # long exit only
                    position = 0

        elif position == -1:
            # short held -> exit when bullish + body_mid > grid_bottom
            if bull and (bm > gb):
                # also check simultaneous long entry
                if long_entry:
                    signals[i] = 2    # short exit + long entry
                    position = 1
                else:
                    signals[i] = 1    # short exit only
                    position = 0

        positions[i] = position

    grid["signal"] = signals
    grid["position"] = positions
    return grid
