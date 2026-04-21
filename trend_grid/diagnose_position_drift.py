# -*- coding: utf-8 -*-
"""
Position Drift 진단: Python 백테스트 vs EA-style 시뮬레이션

Python 백테스트 문제:
  generate_signals()가 H4 필터 없이 내부 position을 추적하고,
  simulate_trades()가 H4 필터로 진입을 차단하면 두 position 트래커가 분기.
  이후 모든 시그널이 실제 포지션과 불일치 → 팬텀 트레이드 발생.

EA 방식:
  매 바마다 실제 포지션을 기반으로 CheckSignal() 호출.
  H4 필터가 진입을 차단하면 flat 유지, 다음 진입 시그널을 기다림.

이 스크립트는 EA 방식을 Python으로 재현하여 두 결과를 비교합니다.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "common"))

from config import SYMBOLS, BACKTEST_CONFIG, VWMA_PERIODS, MA_TYPE
from strategy import compute_grid, generate_signals
from trade_engine import load_ohlcv, calc_trade_cost, compute_stats


def simulate_ea_style(
    m30_grid: pd.DataFrame,
    h4_positions: pd.Series,
    symbol: str,
) -> tuple[list[dict], np.ndarray]:
    """
    EA-style simulation: compute signals using ACTUAL position, not pre-computed.
    Signal computation and H4 filter are applied together per bar.
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

    n_bars = len(m30_grid)
    equity_arr = np.full(n_bars, initial_capital, dtype=np.float64)

    close_arr = m30_grid["close"].values
    open_arr = m30_grid["open"].values
    bm_arr = m30_grid["body_mid"].values
    gt_arr = m30_grid["grid_top"].values
    gb_arr = m30_grid["grid_bottom"].values
    bull_arr = m30_grid["is_bullish"].values
    h4_pos_arr = h4_positions.values
    time_idx = m30_grid.index

    trades = []
    entry_price = 0.0
    entry_time = None
    entry_dir = 0  # actual position: 1=long, -1=short, 0=flat

    warmup = max(VWMA_PERIODS) + 1

    def _record_trade(exit_time, exit_price, direction, exit_reason="signal"):
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
            "exit_reason": exit_reason,
        })

    for i in range(warmup, n_bars):
        if np.isnan(gt_arr[i]) or np.isnan(gt_arr[i - 1]):
            equity_arr[i] = equity
            continue

        # --- EA-style: compute signal using ACTUAL position ---
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

        # Determine action based on ACTUAL position (like EA's CheckSignal)
        action = None  # None, "enter_long", "enter_short", "exit", "reverse_long", "reverse_short"

        if entry_dir == 0:
            if long_entry:
                action = "enter_long"
            elif short_entry:
                action = "enter_short"
        elif entry_dir == 1:
            if long_exit:
                if short_entry:
                    action = "reverse_short"
                else:
                    action = "exit"
        elif entry_dir == -1:
            if short_exit:
                if long_entry:
                    action = "reverse_long"
                else:
                    action = "exit"

        if action is None:
            equity_arr[i] = equity
            continue

        # --- Apply H4 filter (same as EA) ---
        h4_pos = h4_pos_arr[i]

        if action in ("enter_long", "reverse_long"):
            if h4_pos != 1:
                if action == "reverse_long":
                    action = "exit"  # exit only, block long entry
                else:
                    equity_arr[i] = equity
                    continue  # block entry entirely

        if action in ("enter_short", "reverse_short"):
            if h4_pos != -1:
                if action == "reverse_short":
                    action = "exit"  # exit only, block short entry
                else:
                    equity_arr[i] = equity
                    continue

        # --- Execute at next bar open (like EA) ---
        if i + 1 >= n_bars:
            equity_arr[i] = equity
            continue
        exec_price = open_arr[i + 1]
        exec_time = time_idx[i + 1]

        if action == "exit":
            direction = "long" if entry_dir == 1 else "short"
            _record_trade(exec_time, exec_price, direction)
            entry_dir = 0

        elif action == "enter_long":
            entry_price = exec_price
            entry_time = exec_time
            entry_dir = 1

        elif action == "enter_short":
            entry_price = exec_price
            entry_time = exec_time
            entry_dir = -1

        elif action == "reverse_long":
            _record_trade(exec_time, exec_price, "short")
            entry_price = exec_price
            entry_time = exec_time
            entry_dir = 1

        elif action == "reverse_short":
            _record_trade(exec_time, exec_price, "long")
            entry_price = exec_price
            entry_time = exec_time
            entry_dir = -1

        equity_arr[i] = equity

    # Force close
    if entry_dir != 0:
        direction = "long" if entry_dir == 1 else "short"
        _record_trade(time_idx[-1], close_arr[-1], direction, "forced")
        equity_arr[-1] = equity

    return trades, equity_arr


def run_comparison(symbol: str, timeframe: str = "M30", filter_tf: str = "H4",
                    start=None, end=None):
    """Compare Python backtest (current) vs EA-style simulation."""
    from backtest import run_backtest

    # 1) Current Python backtest (generate_signals + simulate_trades with filter)
    py_result = run_backtest(
        symbol=symbol, timeframe=timeframe, ma_type=MA_TYPE,
        start=start, end=end,
        filter_tfs=[filter_tf], ribbon_periods=VWMA_PERIODS,
        verbose=False, _keep_cache=True, next_bar_open=True,
    )
    py_stats = py_result["stats"]
    py_trades = py_result["trades"]

    # 2) EA-style: compute grid, build filter, simulate with actual position
    entry_df = load_ohlcv(symbol, timeframe, start, end)
    entry_grid = compute_grid(entry_df, MA_TYPE, VWMA_PERIODS)

    filter_df = load_ohlcv(symbol, filter_tf, start, end)
    filter_grid = generate_signals(filter_df, MA_TYPE, periods=VWMA_PERIODS)
    # shift(1): avoid look-ahead — HTF bar close only known at end of bar
    filter_positions = filter_grid["position"].shift(1).reindex(entry_grid.index, method="ffill").fillna(0).astype(int)

    ea_trades_list, ea_equity_arr = simulate_ea_style(entry_grid, filter_positions, symbol)
    ea_trades = pd.DataFrame(ea_trades_list) if ea_trades_list else pd.DataFrame()
    ea_equity = pd.DataFrame({"time": entry_grid.index, "equity": ea_equity_arr})
    ea_stats = compute_stats(ea_trades, ea_equity, BACKTEST_CONFIG["initial_capital"],
                              symbol, timeframe, MA_TYPE)

    return py_stats, ea_stats, py_trades, ea_trades


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="EURUSD,USDJPY,EURJPY,XAUUSD,GBPUSD")
    parser.add_argument("--timeframe", default="M30")
    parser.add_argument("--filter-tf", default="H4")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]

    all_results = {}
    for symbol in symbols:
        print(f"\nProcessing {symbol} {args.timeframe}+{args.filter_tf}...")
        py_stats, ea_stats, py_trades, ea_trades = run_comparison(
            symbol, args.timeframe, args.filter_tf, args.start, args.end)
        all_results[symbol] = (py_stats, ea_stats, py_trades, ea_trades)

    # Summary
    print(f"\n\n{'='*100}")
    print(f"  POSITION DRIFT DIAGNOSIS: Python backtest vs EA-style simulation")
    print(f"{'='*100}")
    print(f"{'Symbol':<10} {'Mode':<18} {'Trades':>7} {'WinRate':>8} {'PF':>6} "
          f"{'PnL($)':>12} {'MaxDD%':>8} {'Annual%':>9} {'Sharpe':>7}")
    print(f"{'-'*100}")

    for symbol in symbols:
        py_stats, ea_stats, _, _ = all_results[symbol]
        for label, s in [("py_backtest", py_stats), ("ea_style", ea_stats)]:
            if s.get("total_trades", 0) == 0:
                print(f"{symbol:<10} {label:<18} {'N/A':>7}")
                continue
            print(f"{symbol:<10} {label:<18} "
                  f"{s['total_trades']:>7} "
                  f"{s['win_rate']:>7.1f}% "
                  f"{s['profit_factor']:>6.2f} "
                  f"{s['total_pnl_usd']:>+12,.2f} "
                  f"{s['max_drawdown_pct']:>7.2f}% "
                  f"{s['annual_return_pct']:>+8.2f}% "
                  f"{s.get('sharpe_ratio', 0):>7.2f}")
        print()

    # Delta
    print(f"\n{'='*100}")
    print(f"  DELTA (ea_style - py_backtest)")
    print(f"{'='*100}")
    print(f"{'Symbol':<10} {'dTrades':>8} {'dWinRate':>9} {'dPnL($)':>14} "
          f"{'dMaxDD%':>9} {'dAnnual%':>10}")
    print(f"{'-'*100}")

    for symbol in symbols:
        py_s, ea_s, py_t, ea_t = all_results[symbol]
        if py_s.get("total_trades", 0) == 0:
            continue

        d_trades = ea_s['total_trades'] - py_s['total_trades']
        d_wr = ea_s['win_rate'] - py_s['win_rate']
        d_pnl = ea_s['total_pnl_usd'] - py_s['total_pnl_usd']
        d_dd = ea_s['max_drawdown_pct'] - py_s['max_drawdown_pct']
        d_ar = ea_s['annual_return_pct'] - py_s['annual_return_pct']

        print(f"{symbol:<10} {d_trades:>+8} {d_wr:>+8.1f}% {d_pnl:>+14,.2f} "
              f"{d_dd:>+8.2f}% {d_ar:>+9.2f}%")

        # Phantom trade analysis
        if not py_t.empty and not ea_t.empty:
            print(f"           Phantom trades: {abs(d_trades)} "
                  f"({'Python takes MORE' if d_trades < 0 else 'EA takes MORE'})")

    print()


if __name__ == "__main__":
    main()
