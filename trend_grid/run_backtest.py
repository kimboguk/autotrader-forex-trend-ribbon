# -*- coding: utf-8 -*-
"""
Trend Grid Strategy - 백테스트 실행기

EURUSD 대상 전체 타임프레임 순차 실행.
Usage:
    # 전체 타임프레임 실행
    python run_backtest.py

    # 특정 타임프레임만
    python run_backtest.py --tf D1 H4 H1

    # SMA 비교
    python run_backtest.py --ma sma

    # VWMA vs SMA 비교
    python run_backtest.py --compare
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from config import TIMEFRAMES
from backtest import run_backtest, print_report, save_results


def run_single(symbol: str, timeframe: str, ma_type: str,
               start: str = None, end: str = None, save: bool = True):
    """단일 타임프레임 백테스트"""
    print(f"\n{'-'*60}")
    print(f"  Running: {symbol} {timeframe} (MA: {ma_type})")
    print(f"{'-'*60}")

    try:
        result = run_backtest(symbol, timeframe, ma_type, start, end)
        print_report(result["stats"])

        if save:
            save_results(result, symbol, timeframe, ma_type)

        return result["stats"]

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_all_timeframes(symbol: str, ma_type: str, timeframes: list = None,
                       start: str = None, end: str = None):
    """전체 타임프레임 순차 실행 + 요약 테이블"""
    if timeframes is None:
        timeframes = TIMEFRAMES

    all_stats = []

    for tf in timeframes:
        stats = run_single(symbol, tf, ma_type, start, end)
        if stats:
            all_stats.append(stats)

    if all_stats:
        print_summary_table(all_stats)

    return all_stats


def run_compare(symbol: str, timeframes: list = None,
                start: str = None, end: str = None):
    """VWMA vs SMA 비교 실행"""
    if timeframes is None:
        timeframes = TIMEFRAMES

    print(f"\n{'#'*60}")
    print(f"  VWMA vs SMA Comparison: {symbol}")
    print(f"{'#'*60}")

    vwma_stats = []
    sma_stats = []

    for tf in timeframes:
        print(f"\n{'='*60}")
        print(f"  {tf}: VWMA")
        print(f"{'='*60}")
        v = run_single(symbol, tf, "vwma", start, end)
        if v:
            vwma_stats.append(v)

        print(f"\n{'='*60}")
        print(f"  {tf}: SMA")
        print(f"{'='*60}")
        s = run_single(symbol, tf, "sma", start, end)
        if s:
            sma_stats.append(s)

    if vwma_stats:
        print(f"\n{'#'*60}")
        print(f"  VWMA Results")
        print(f"{'#'*60}")
        print_summary_table(vwma_stats)

    if sma_stats:
        print(f"\n{'#'*60}")
        print(f"  SMA Results")
        print(f"{'#'*60}")
        print_summary_table(sma_stats)


def print_summary_table(stats_list: list):
    """타임프레임별 요약 비교 테이블"""
    print(f"\n{'='*90}")
    print(f"  Summary Comparison")
    print(f"{'='*90}")

    header = (f"  {'TF':<5} {'MA':<5} {'Trades':>6} {'Win%':>6} {'PF':>6} "
              f"{'P&L(pip)':>10} {'Cost(pip)':>10} {'P&L($)':>10} "
              f"{'MaxDD%':>8} {'Ann%':>8}")
    print(header)
    print(f"  {'-'*85}")

    for s in stats_list:
        if s["total_trades"] == 0:
            print(f"  {s['timeframe']:<5} {s['ma_type']:<5} {'no trades':>6}")
            continue

        print(f"  {s['timeframe']:<5} {s['ma_type']:<5} "
              f"{s['total_trades']:>6} "
              f"{s['win_rate']:>5.1f}% "
              f"{s['profit_factor']:>6.2f} "
              f"{s['total_pnl_pips']:>+10.1f} "
              f"{s['total_cost_pips']:>10.1f} "
              f"{s['total_pnl_usd']:>+10.2f} "
              f"{s['max_drawdown_pct']:>7.2f}% "
              f"{s['annual_return_pct']:>+7.2f}%")

    print(f"  {'='*85}\n")


def main():
    parser = argparse.ArgumentParser(description="Trend Grid Backtest Runner")
    parser.add_argument("--symbol", default="EURUSD", help="심볼 (default: EURUSD)")
    parser.add_argument("--tf", nargs="+", default=None,
                        help="타임프레임 (e.g., D1 H4 H1)")
    parser.add_argument("--ma", default="ema", choices=["vwma", "sma", "ema"],
                        help="이동평균 타입 (default: ema)")
    parser.add_argument("--compare", action="store_true",
                        help="VWMA vs SMA 비교 모드")
    parser.add_argument("--start", default=None, help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="종료일 (YYYY-MM-DD)")
    parser.add_argument("--no-save", action="store_true", help="결과 저장 안함")
    args = parser.parse_args()

    timeframes = args.tf if args.tf else TIMEFRAMES

    if args.compare:
        run_compare(args.symbol, timeframes, args.start, args.end)
    else:
        run_all_timeframes(args.symbol, args.ma, timeframes, args.start, args.end)


if __name__ == "__main__":
    main()
