# -*- coding: utf-8 -*-
"""
Golden Cross Strategy - Backtest Runner

Uses common trade engine for simulation and statistics.
Loaded via importlib from engine_adapter, so uses importlib for local imports
to avoid conflicts with trend_grid modules on sys.path.
"""

import sys
import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# ── Load local modules via importlib (avoid sys.path conflicts) ──

_DIR = Path(__file__).resolve().parent
_STRATEGY_ROOT = _DIR.parent

# Ensure common and statarb are on sys.path for trade_engine imports
for d in [_STRATEGY_ROOT / "common", _STRATEGY_ROOT / "statarb", _STRATEGY_ROOT / "trend_grid"]:
    if str(d) not in sys.path:
        sys.path.append(str(d))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Load golden_cross-local modules
_gc_config = _load_module("gc_config", _DIR / "config.py")
_gc_strategy = _load_module("gc_strategy", _DIR / "strategy.py")

# Load shared modules (trend_grid config for SYMBOLS/BACKTEST_CONFIG, common trade_engine)
from config import SYMBOLS, BACKTEST_CONFIG  # trend_grid config (on sys.path)
from trade_engine import (
    load_ohlcv, simulate_trades, compute_stats, clear_m1_cache,
)

generate_signals = _gc_strategy.generate_signals


# ── Higher TF Filter ────────────────────────────────────────

def _build_tf_filter(symbol: str, filter_tf: str, target_index: pd.DatetimeIndex,
                     ma_type: str = "ema",
                     start: str = None, end: str = None,
                     fast_period: int = 50, slow_period: int = 200) -> pd.Series:
    """Build higher TF position filter using golden cross signals."""
    df = load_ohlcv(symbol, filter_tf, start, end)
    grid = generate_signals(df, ma_type, fast_period=fast_period, slow_period=slow_period)
    pos = grid["position"].reindex(target_index, method="ffill").fillna(0).astype(int)
    return pos


# ── Backtest Entry Point ────────────────────────────────────

def run_backtest(
    symbol: str,
    timeframe: str,
    ma_type: str = None,
    start: str = None,
    end: str = None,
    tp_pips: float = None,
    sl_pips: float = None,
    fast_period: int = None,
    slow_period: int = None,
    filter_tfs: list = None,
    alignment_mas: list = None,
    verbose: bool = True,
    progress_callback=None,
    _keep_cache: bool = False,
    # Accept but ignore trend_grid-specific params
    d1_filter: bool = False,
    ribbon_periods: list = None,
    compound: bool = False,
    leverage: int = 1,
    kelly_fraction: float = 0.0,
    use_kalman: bool = False,
    kalman_qr_ratio: float = 0.1,
) -> dict:
    """Run Golden Cross backtest."""
    ma_type = ma_type or _gc_config.MA_TYPE
    fast_period = fast_period or _gc_config.DEFAULT_FAST_PERIOD
    slow_period = slow_period or _gc_config.DEFAULT_SLOW_PERIOD

    # 1) Load data
    if progress_callback:
        progress_callback(f"Loading {symbol} {timeframe}...")
    df = load_ohlcv(symbol, timeframe, start, end)
    if verbose:
        print(f"  [{timeframe}] Loaded {len(df):,} bars "
              f"({df.index[0]} ~ {df.index[-1]})")

    # 2) Generate signals
    if progress_callback:
        progress_callback(f"Generating {timeframe} signals...")
    grid = generate_signals(df, ma_type, fast_period=fast_period, slow_period=slow_period,
                            use_kalman=use_kalman, kalman_qr_ratio=kalman_qr_ratio)

    # Build higher TF filters
    TF_RANK = {"D1": 5, "H4": 4, "H1": 3, "M30": 2, "M15": 1, "M5": 0, "M1": -1}
    active_filters = list(filter_tfs or [])
    my_rank = TF_RANK.get(timeframe, 0)
    active_filters = [tf for tf in active_filters if TF_RANK.get(tf, 0) > my_rank]

    filter_positions = {}
    for ftf in active_filters:
        if progress_callback:
            progress_callback(f"Building {ftf} filter...")
        pos = _build_tf_filter(symbol, ftf, grid.index, ma_type, start, end,
                               fast_period, slow_period)
        filter_positions[ftf] = pos
        if verbose:
            print(f"  [{ftf} filter] active, {(pos != 0).sum():,} bars with position")

    # 3) Simulate trades
    trades, equity_arr = simulate_trades(
        grid=grid,
        symbol=symbol,
        tp_pips=tp_pips,
        sl_pips=sl_pips,
        filter_positions=filter_positions,
        progress_callback=progress_callback,
    )

    # 4) Compute stats
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    equity_df = pd.DataFrame({"time": grid.index, "equity": equity_arr})

    stats = compute_stats(trades_df, equity_df, BACKTEST_CONFIG["initial_capital"],
                          symbol, timeframe, ma_type)

    if not _keep_cache:
        clear_m1_cache()

    return {
        "trades": trades_df,
        "equity": equity_df,
        "grid": grid,
        "stats": stats,
    }
