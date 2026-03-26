# -*- coding: utf-8 -*-
"""
Look-Ahead Bias 비교: signal bar close 체결 vs next bar open 체결

두 모드의 백테스트 결과를 나란히 비교합니다.
Usage:
    python compare_lookahead.py
    python compare_lookahead.py --symbols EURUSD,USDJPY --start 2020-01-01
"""

import sys
import argparse

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from config import SYMBOLS, MA_TYPE, VWMA_PERIODS
from backtest import run_backtest, print_report


COMPARE_SYMBOLS = ["EURUSD", "USDJPY", "EURJPY", "XAUUSD", "GBPUSD"]
TIMEFRAME = "M30"
FILTER_TFS = ["H4"]


def compare(symbols, start=None, end=None):
    """Run both modes and print side-by-side comparison."""
    results = {}

    for symbol in symbols:
        print(f"\n{'='*70}")
        print(f"  {symbol} {TIMEFRAME} (MA: {MA_TYPE})")
        print(f"{'='*70}")

        for mode_name, nbo in [("signal_close", False), ("next_bar_open", True)]:
            print(f"\n  --- Mode: {mode_name} ---")
            res = run_backtest(
                symbol=symbol,
                timeframe=TIMEFRAME,
                ma_type=MA_TYPE,
                start=start,
                end=end,
                filter_tfs=FILTER_TFS,
                ribbon_periods=VWMA_PERIODS,
                verbose=False,
                _keep_cache=True,
                next_bar_open=nbo,
            )
            results[(symbol, mode_name)] = res["stats"]

    # Summary table
    print(f"\n\n{'='*90}")
    print(f"  LOOK-AHEAD BIAS COMPARISON SUMMARY")
    print(f"{'='*90}")
    print(f"{'Symbol':<10} {'Mode':<16} {'Trades':>7} {'WinRate':>8} {'PF':>6} "
          f"{'PnL($)':>12} {'MaxDD%':>8} {'Annual%':>9} {'Sharpe':>7}")
    print(f"{'-'*90}")

    for symbol in symbols:
        for mode_name in ["signal_close", "next_bar_open"]:
            s = results.get((symbol, mode_name))
            if s is None or s.get("total_trades", 0) == 0:
                print(f"{symbol:<10} {mode_name:<16} {'N/A':>7}")
                continue
            print(f"{symbol:<10} {mode_name:<16} "
                  f"{s['total_trades']:>7} "
                  f"{s['win_rate']:>7.1f}% "
                  f"{s['profit_factor']:>6.2f} "
                  f"{s['total_pnl_usd']:>+12,.2f} "
                  f"{s['max_drawdown_pct']:>7.2f}% "
                  f"{s['annual_return_pct']:>+8.2f}% "
                  f"{s.get('sharpe_ratio', 0):>7.2f}")
        print()

    # Delta summary
    print(f"\n{'='*90}")
    print(f"  DELTA (next_bar_open - signal_close)")
    print(f"{'='*90}")
    print(f"{'Symbol':<10} {'dTrades':>8} {'dWinRate':>9} {'dPnL($)':>14} "
          f"{'dMaxDD%':>9} {'dAnnual%':>10}")
    print(f"{'-'*90}")

    for symbol in symbols:
        sc = results.get((symbol, "signal_close"))
        nbo = results.get((symbol, "next_bar_open"))
        if not sc or not nbo or sc.get("total_trades", 0) == 0:
            continue
        print(f"{symbol:<10} "
              f"{nbo['total_trades'] - sc['total_trades']:>+8} "
              f"{nbo['win_rate'] - sc['win_rate']:>+8.1f}% "
              f"{nbo['total_pnl_usd'] - sc['total_pnl_usd']:>+14,.2f} "
              f"{nbo['max_drawdown_pct'] - sc['max_drawdown_pct']:>+8.2f}% "
              f"{nbo['annual_return_pct'] - sc['annual_return_pct']:>+9.2f}%")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Look-ahead bias comparison")
    parser.add_argument("--symbols", default=",".join(COMPARE_SYMBOLS))
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    compare(symbols, args.start, args.end)
