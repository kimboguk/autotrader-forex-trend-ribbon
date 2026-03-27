# -*- coding: utf-8 -*-
"""
Trend Grid Strategy - Backtest Runner

Uses common trade engine for simulation and statistics.
Strategy-specific: signal generation via generate_signals().
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from config import SYMBOLS, BACKTEST_CONFIG, RESAMPLE_RULES
from strategy import compute_grid, generate_signals
from trade_engine import (
    load_ohlcv, calc_trade_cost, simulate_trades, compute_stats,
    clear_m1_cache, _load_m1_cached,
)


# ── Higher TF Filter ────────────────────────────────────────

def _build_tf_filter(symbol: str, filter_tf: str, target_index: pd.DatetimeIndex,
                     ma_type: str = None,
                     start: str = None, end: str = None,
                     ribbon_periods: list = None) -> pd.Series:
    """Build higher TF position filter aligned to target index."""
    df = load_ohlcv(symbol, filter_tf, start, end)
    grid = generate_signals(df, ma_type, periods=ribbon_periods)
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
    d1_filter: bool = False,
    filter_tfs: list = None,
    alignment_mas: list = None,
    ribbon_periods: list = None,
    verbose: bool = True,
    progress_callback=None,
    _keep_cache: bool = False,
    compound: bool = False,
    next_bar_open: bool = True,
    # Accept but ignore other strategy params
    fast_period: int = None,
    slow_period: int = None,
) -> dict:
    """Run Trend Grid backtest."""
    # 1) Load data
    if progress_callback:
        progress_callback(f"Loading {symbol} {timeframe}...")
    df = load_ohlcv(symbol, timeframe, start, end)
    if verbose:
        print(f"  [{timeframe}] Loaded {len(df):,} bars "
              f"({df.index[0]} ~ {df.index[-1]})")

    # 2) Compute grid (EA-style: signals computed inline during simulation)
    if progress_callback:
        progress_callback(f"Computing {timeframe} grid...")
    grid = compute_grid(df, ma_type, periods=ribbon_periods)

    # Build higher TF filters
    TF_RANK = {"D1": 5, "H4": 4, "H1": 3, "M30": 2, "M15": 1, "M5": 0, "M1": -1}
    active_filters = list(filter_tfs or [])
    if d1_filter and "D1" not in active_filters:
        active_filters.append("D1")
    my_rank = TF_RANK.get(timeframe, 0)
    active_filters = [tf for tf in active_filters if TF_RANK.get(tf, 0) > my_rank]

    filter_positions = {}
    for ftf in active_filters:
        if progress_callback:
            progress_callback(f"Building {ftf} filter...")
        pos = _build_tf_filter(symbol, ftf, grid.index, ma_type, start, end, ribbon_periods)
        filter_positions[ftf] = pos
        if verbose:
            print(f"  [{ftf} filter] active, {(pos != 0).sum():,} bars with position")

    # MA alignment columns
    alignment_col_names = []
    if alignment_mas and len(alignment_mas) >= 2:
        sorted_periods = sorted(alignment_mas)
        alignment_col_names = [f"ma_{p}" for p in sorted_periods]

    # 3) Simulate trades
    trades, equity_arr = simulate_trades(
        grid=grid,
        symbol=symbol,
        tp_pips=tp_pips,
        sl_pips=sl_pips,
        filter_positions=filter_positions,
        alignment_cols=alignment_col_names,
        progress_callback=progress_callback,
        compound=compound,
        next_bar_open=next_bar_open,
    )

    # 4) Compute stats
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    equity_df = pd.DataFrame({"time": grid.index, "equity": equity_arr})

    stats = compute_stats(trades_df, equity_df, BACKTEST_CONFIG["initial_capital"],
                          symbol, timeframe, ma_type, compound=compound)

    if not _keep_cache:
        clear_m1_cache()

    return {
        "trades": trades_df,
        "equity": equity_df,
        "grid": grid,
        "stats": stats,
    }


# ── Report / Save (CLI use) ────────────────────────────────

def print_report(stats: dict):
    """Print backtest report (ASCII only)"""
    print(f"\n{'='*60}")
    print(f"  Trend Grid: {stats['symbol']} {stats['timeframe']} "
          f"(MA: {stats['ma_type']})")
    print(f"{'='*60}")

    if stats["total_trades"] == 0:
        print("  No trades generated.")
        return

    print(f"  Data period:     {stats.get('data_period_days', 'N/A')} days")
    print(f"  Total trades:    {stats['total_trades']} "
          f"(L:{stats['long_trades']} / S:{stats['short_trades']})")
    print(f"  Win rate:        {stats['win_rate']}%")
    print(f"  Profit factor:   {stats['profit_factor']}")
    print(f"  Expectancy:      {stats['expectancy_pips']} pips/trade")
    print(f"  {'-'*40}")
    print(f"  Total P&L:       {stats['total_pnl_pips']:+.1f} pips "
          f"(${stats['total_pnl_usd']:+,.2f})")
    print(f"  Total costs:     {stats['total_cost_pips']:.1f} pips")
    print(f"  {'-'*40}")
    print(f"  Avg win:         ${stats['avg_win_usd']:+,.2f}")
    print(f"  Avg loss:        ${stats['avg_loss_usd']:+,.2f}")
    print(f"  Max drawdown:    {stats['max_drawdown_pct']:.2f}%")
    print(f"  Annual return:   {stats['annual_return_pct']:+.2f}%")
    print(f"  Avg holding:     {stats['avg_holding']}")
    print(f"  {'-'*40}")
    print(f"  Capital:         ${stats['initial_capital']:,.0f} -> "
          f"${stats['final_equity']:,.2f}")
    print(f"{'='*60}\n")


def save_results(result: dict, symbol: str, timeframe: str, ma_type: str = "vwma"):
    """Save trades and equity CSV."""
    out_dir = Path(BACKTEST_CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{symbol}_{timeframe}_{ma_type}"

    if len(result["trades"]) > 0:
        trades_path = out_dir / f"{prefix}_trades.csv"
        result["trades"].to_csv(trades_path, index=False)
        print(f"  Saved: {trades_path}")

    if len(result["equity"]) > 0:
        eq_path = out_dir / f"{prefix}_equity.csv"
        result["equity"].to_csv(eq_path, index=False)
        print(f"  Saved: {eq_path}")
