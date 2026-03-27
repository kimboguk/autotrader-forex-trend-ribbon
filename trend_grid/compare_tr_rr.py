# -*- coding: utf-8 -*-
"""
TR vs RR 성과 분리 측정

TR (Trend Ribbon): 리본 돌파 → 진입, 리본 복귀 → 청산
RR (Range Ribbon): TR 청산 즉시 반대 방향 진입 → 리본 내부 거래

각 트레이드에 type="TR" 또는 "RR" 태그를 부여하여 독립적 성과를 측정합니다.

Usage:
    python compare_tr_rr.py
    python compare_tr_rr.py --timeframe M15 --filter-tf H4
    python compare_tr_rr.py --symbols EURUSD --timeframe M15 --filter-tf H1
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "common"))

from config import SYMBOLS, BACKTEST_CONFIG, VWMA_PERIODS, MA_TYPE
from strategy import compute_grid, generate_signals
from trade_engine import load_ohlcv, calc_trade_cost, compute_stats


def simulate_tr_rr(
    entry_grid: pd.DataFrame,
    filter_positions: pd.Series,
    symbol: str,
) -> tuple[list[dict], np.ndarray]:
    """
    Combined TR+RR simulation with trade type tagging.

    TR trades: breakout entry/exit (standard strategy)
    RR trades: immediate reverse after TR exit (ribbon interior)
    """
    sym_cfg = SYMBOLS[symbol]
    cost_per_side = calc_trade_cost(symbol)
    pip_size = sym_cfg["pip_size"]
    quote_ccy = sym_cfg.get("quote_ccy", "USD")
    lot_size = sym_cfg.get("lot_size", BACKTEST_CONFIG["lot_size"])
    pos_lots = BACKTEST_CONFIG["position_size_lots"]
    units = pos_lots * lot_size
    initial_capital = BACKTEST_CONFIG["initial_capital"]
    equity = initial_capital

    n_bars = len(entry_grid)
    equity_arr = np.full(n_bars, initial_capital, dtype=np.float64)

    close_arr = entry_grid["close"].values
    open_arr = entry_grid["open"].values
    bm_arr = entry_grid["body_mid"].values
    gt_arr = entry_grid["grid_top"].values
    gb_arr = entry_grid["grid_bottom"].values
    bull_arr = entry_grid["is_bullish"].values
    filter_arr = filter_positions.values
    time_idx = entry_grid.index

    trades = []
    entry_price = 0.0
    entry_time = None
    entry_dir = 0
    trade_type = None  # "TR" or "RR"

    warmup = max(VWMA_PERIODS) + 1

    def _record_trade(exit_time, exit_price, direction, reason="signal"):
        nonlocal equity
        if direction == "long":
            pnl_price = exit_price - entry_price
        else:
            pnl_price = entry_price - exit_price

        pnl = pnl_price * units - cost_per_side * units * 2
        total_cost = cost_per_side * units * 2
        if quote_ccy == "JPY":
            pnl = pnl / exit_price
            total_cost = total_cost / exit_price

        equity += pnl
        trades.append({
            "entry_time": entry_time,
            "exit_time": exit_time,
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_pips": pnl_price / pip_size,
            "cost_pips": (cost_per_side * 2) / pip_size,
            "net_pnl_pips": pnl_price / pip_size - (cost_per_side * 2) / pip_size,
            "pnl_usd": pnl,
            "equity_after": equity,
            "exit_reason": reason,
            "type": trade_type,
        })

    for i in range(warmup, n_bars):
        if np.isnan(gt_arr[i]) or np.isnan(gt_arr[i - 1]):
            equity_arr[i] = equity
            continue

        curr_bm = bm_arr[i]
        curr_gt = gt_arr[i]
        curr_gb = gb_arr[i]
        curr_bull = bull_arr[i]
        prev_bm = bm_arr[i - 1]
        prev_gt = gt_arr[i - 1]
        prev_gb = gb_arr[i - 1]
        prev_bull = bull_arr[i - 1]
        prev_open = open_arr[i - 1]
        prev_close = close_arr[i - 1]

        # Entry conditions
        prev_below_top = prev_bm <= prev_gt
        prev_above_bot = prev_bm >= prev_gb

        # Relaxed entry
        if not prev_bull and prev_close < prev_gt < prev_open:
            prev_below_top = True
        if prev_bull and prev_open < prev_gb < prev_close:
            prev_above_bot = True

        long_entry = curr_bull and (curr_bm > curr_gt) and prev_below_top
        short_entry = (not curr_bull) and (curr_bm < curr_gb) and prev_above_bot

        # Exit conditions
        long_exit = (not curr_bull) and (curr_bm < curr_gt)
        short_exit = curr_bull and (curr_bm > curr_gb)

        # Determine action based on actual position and trade type
        action = None

        if entry_dir == 0:
            # Flat → only TR entries
            if long_entry:
                action = "enter_long"
            elif short_entry:
                action = "enter_short"

        elif trade_type == "TR":
            # In a TR trade → check TR exit conditions
            if entry_dir == 1 and long_exit:
                action = "tr_exit_long"
            elif entry_dir == -1 and short_exit:
                action = "tr_exit_short"

        elif trade_type == "RR":
            # In an RR trade → exit when new TR entry in same direction,
            # or when price reaches opposite grid boundary (RR target),
            # or when TR entry in opposite direction (reverse back)
            if entry_dir == 1:
                # RR long: exit on long_exit (same as TR exit condition)
                if long_exit:
                    action = "rr_exit_long"
            elif entry_dir == -1:
                # RR short: exit on short_exit
                if short_exit:
                    action = "rr_exit_short"

        if action is None:
            equity_arr[i] = equity
            continue

        # Execution price = next bar open
        if i + 1 >= n_bars:
            equity_arr[i] = equity
            continue
        exec_price = open_arr[i + 1]
        exec_time = time_idx[i + 1]

        h4_pos = filter_arr[i]

        if action == "enter_long":
            if h4_pos == 1:
                trade_type = "TR"
                entry_price = exec_price
                entry_time = exec_time
                entry_dir = 1

        elif action == "enter_short":
            if h4_pos == -1:
                trade_type = "TR"
                entry_price = exec_price
                entry_time = exec_time
                entry_dir = -1

        elif action == "tr_exit_long":
            _record_trade(exec_time, exec_price, "long")
            # RR: immediately enter short (no H4 filter for RR)
            trade_type = "RR"
            entry_price = exec_price
            entry_time = exec_time
            entry_dir = -1

        elif action == "tr_exit_short":
            _record_trade(exec_time, exec_price, "short")
            # RR: immediately enter long
            trade_type = "RR"
            entry_price = exec_price
            entry_time = exec_time
            entry_dir = 1

        elif action == "rr_exit_long":
            _record_trade(exec_time, exec_price, "long")
            # After RR exit, check if TR entry available
            if short_entry and h4_pos == -1:
                trade_type = "TR"
                entry_price = exec_price
                entry_time = exec_time
                entry_dir = -1
            else:
                entry_dir = 0
                trade_type = None

        elif action == "rr_exit_short":
            _record_trade(exec_time, exec_price, "short")
            if long_entry and h4_pos == 1:
                trade_type = "TR"
                entry_price = exec_price
                entry_time = exec_time
                entry_dir = 1
            else:
                entry_dir = 0
                trade_type = None

        equity_arr[i] = equity

    # Force close
    if entry_dir != 0:
        direction = "long" if entry_dir == 1 else "short"
        _record_trade(time_idx[-1], close_arr[-1], direction, "forced")
        equity_arr[-1] = equity

    return trades, equity_arr


def run_analysis(symbols, timeframe, filter_tf, start=None, end=None):
    """Run TR+RR analysis for all symbols."""
    all_results = {}

    for symbol in symbols:
        print(f"\nProcessing {symbol} {timeframe}+{filter_tf}...")

        entry_df = load_ohlcv(symbol, timeframe, start, end)
        entry_grid = compute_grid(entry_df, MA_TYPE, VWMA_PERIODS)

        filter_df = load_ohlcv(symbol, filter_tf, start, end)
        filter_grid = generate_signals(filter_df, MA_TYPE, periods=VWMA_PERIODS)
        filter_positions = filter_grid["position"].reindex(
            entry_grid.index, method="ffill").fillna(0).astype(int)

        trades_list, equity_arr = simulate_tr_rr(entry_grid, filter_positions, symbol)
        trades_df = pd.DataFrame(trades_list) if trades_list else pd.DataFrame()

        all_results[symbol] = trades_df

    # Summary
    print(f"\n\n{'='*110}")
    print(f"  TR vs RR Performance: {timeframe}+{filter_tf}")
    print(f"{'='*110}")
    print(f"{'Symbol':<10} {'Type':<6} {'Trades':>7} {'WinRate':>8} {'PF':>6} "
          f"{'PnL($)':>12} {'AvgPnL($)':>10} {'AvgPips':>8} {'CostPips':>9}")
    print(f"{'-'*110}")

    for symbol in symbols:
        df = all_results[symbol]
        if df.empty:
            print(f"{symbol:<10} {'N/A':>7}")
            continue

        for ttype in ["TR", "RR", "ALL"]:
            sub = df if ttype == "ALL" else df[df["type"] == ttype]
            if sub.empty:
                continue
            n = len(sub)
            winners = sub[sub["pnl_usd"] > 0]
            wr = len(winners) / n * 100 if n > 0 else 0
            gross_win = winners["pnl_usd"].sum() if len(winners) > 0 else 0
            gross_loss = abs(sub[sub["pnl_usd"] <= 0]["pnl_usd"].sum())
            pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
            total_pnl = sub["pnl_usd"].sum()
            avg_pnl = total_pnl / n if n > 0 else 0
            avg_pips = sub["net_pnl_pips"].mean() if n > 0 else 0
            avg_cost = sub["cost_pips"].mean() if n > 0 else 0

            label = ttype if ttype != "ALL" else "TR+RR"
            print(f"{symbol:<10} {label:<6} {n:>7} {wr:>7.1f}% {pf:>6.2f} "
                  f"{total_pnl:>+12,.2f} {avg_pnl:>+10.2f} {avg_pips:>+8.1f} {avg_cost:>9.1f}")
        print()

    # Aggregate across all symbols
    print(f"{'-'*110}")
    all_df = pd.concat(all_results.values(), ignore_index=True)
    if not all_df.empty:
        for ttype in ["TR", "RR", "TR+RR"]:
            sub = all_df if ttype == "TR+RR" else all_df[all_df["type"] == ttype]
            if sub.empty:
                continue
            n = len(sub)
            winners = sub[sub["pnl_usd"] > 0]
            wr = len(winners) / n * 100 if n > 0 else 0
            gross_win = winners["pnl_usd"].sum() if len(winners) > 0 else 0
            gross_loss = abs(sub[sub["pnl_usd"] <= 0]["pnl_usd"].sum())
            pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
            total_pnl = sub["pnl_usd"].sum()
            avg_pnl = total_pnl / n if n > 0 else 0
            avg_pips = sub["net_pnl_pips"].mean() if n > 0 else 0
            avg_cost = sub["cost_pips"].mean() if n > 0 else 0

            print(f"{'TOTAL':<10} {ttype:<6} {n:>7} {wr:>7.1f}% {pf:>6.2f} "
                  f"{total_pnl:>+12,.2f} {avg_pnl:>+10.2f} {avg_pips:>+8.1f} {avg_cost:>9.1f}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TR vs RR performance analysis")
    parser.add_argument("--symbols", default="EURUSD,USDJPY,EURJPY,XAUUSD,GBPUSD")
    parser.add_argument("--timeframe", default="M30")
    parser.add_argument("--filter-tf", default="H4")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    run_analysis(symbols, args.timeframe, args.filter_tf, args.start, args.end)
