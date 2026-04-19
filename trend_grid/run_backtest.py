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


def print_yearly_breakdown(trades_df: pd.DataFrame, label: str = ""):
    """연도별 trades/WR/PF/PnL 출력"""
    if len(trades_df) == 0:
        print("  (no trades)")
        return
    df = trades_df.copy()
    df["year"] = pd.to_datetime(df["entry_time"]).dt.year
    print(f"\n  === {label} — Yearly Breakdown ===")
    print(f"  {'Year':>6} {'Trades':>7} {'WR%':>7} {'PF':>7} {'Pips':>12} {'PnL($)':>12}")
    print(f"  {'-'*60}")
    for yr in sorted(df["year"].unique()):
        sub = df[df["year"] == yr]
        n = len(sub)
        wr = (sub["pnl_usd"] > 0).mean() * 100
        gw = sub.loc[sub["pnl_usd"] > 0, "pnl_usd"].sum()
        gl = abs(sub.loc[sub["pnl_usd"] <= 0, "pnl_usd"].sum())
        pf = gw / gl if gl > 0 else float("inf")
        pips = sub["net_pnl_pips"].sum()
        pnl = sub["pnl_usd"].sum()
        print(f"  {yr:>6} {n:>7} {wr:>6.1f}% {pf:>7.2f} {pips:>+12.1f} {pnl:>+12.2f}")


def run_single(symbol: str, timeframe: str, ma_type: str,
               start: str = None, end: str = None, save: bool = True,
               filter_tfs: list = None, htf_exit: bool = False,
               yearly: bool = False):
    """단일 타임프레임 백테스트"""
    label = f"{symbol} {timeframe} (MA: {ma_type})"
    if filter_tfs:
        label += f" +filter={','.join(filter_tfs)}"
    if htf_exit:
        label += " htf_exit"
    print(f"\n{'-'*60}")
    print(f"  Running: {label}")
    print(f"{'-'*60}")

    try:
        result = run_backtest(symbol, timeframe, ma_type, start, end,
                              filter_tfs=filter_tfs, htf_exit=htf_exit)
        print_report(result["stats"])

        if yearly:
            print_yearly_breakdown(result["trades"], label)

        if save:
            save_results(result, symbol, timeframe, ma_type)

        return result["stats"]

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_all_timeframes(symbol: str, ma_type: str, timeframes: list = None,
                       start: str = None, end: str = None,
                       filter_tfs: list = None, htf_exit: bool = False,
                       yearly: bool = False):
    """전체 타임프레임 순차 실행 + 요약 테이블"""
    if timeframes is None:
        timeframes = TIMEFRAMES

    all_stats = []

    for tf in timeframes:
        stats = run_single(symbol, tf, ma_type, start, end,
                           filter_tfs=filter_tfs, htf_exit=htf_exit,
                           yearly=yearly)
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
    parser.add_argument("--filter-tf", nargs="+", default=None,
                        help="상위 TF 필터 (e.g., H4 D1)")
    parser.add_argument("--htf-exit", action="store_true",
                        help="청산을 상위 TF 기준으로 (현재 TF exit 신호 무시)")
    parser.add_argument("--yearly", action="store_true",
                        help="연도별 breakdown 출력")
    args = parser.parse_args()

    timeframes = args.tf if args.tf else TIMEFRAMES

    if args.compare:
        run_compare(args.symbol, timeframes, args.start, args.end)
    else:
        run_all_timeframes(args.symbol, args.ma, timeframes,
                           args.start, args.end,
                           filter_tfs=args.filter_tf, htf_exit=args.htf_exit,
                           yearly=args.yearly)


if __name__ == "__main__":
    main()
