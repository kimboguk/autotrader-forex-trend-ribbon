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

def compute_grid(df: pd.DataFrame, ma_type: str = None, periods: list = None) -> pd.DataFrame:
    result = df.copy()

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

    return result


# -- Signal generation -----------------------------------------------------

def generate_signals(df: pd.DataFrame, ma_type: str = None, periods: list = None) -> pd.DataFrame:
    """
    Signal codes:
       0: no change
       1: long entry (or short exit only)
      -1: short entry (or long exit only)
       2: short exit + long entry
      -2: long exit + short entry
    """
    grid = compute_grid(df, ma_type, periods)

    n = len(grid)
    signals = np.zeros(n, dtype=int)
    positions = np.zeros(n, dtype=int)

    # to numpy for speed
    bm_arr = grid["body_mid"].values
    gt_arr = grid["grid_top"].values
    gb_arr = grid["grid_bottom"].values
    bull_arr = grid["is_bullish"].values

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
        long_entry = bull and (bm > gt) and (prev_bm <= prev_gt)
        short_entry = (not bull) and (bm < gb) and (prev_bm >= prev_gb)

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
