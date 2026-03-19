# -*- coding: utf-8 -*-
"""
StatArb - H1 전략 비교 백테스트

4가지 전략을 동일 조건에서 비교:
  1. Ratio Bollinger (가장 단순)
  2. Kalman Filter (연속 적응)
  3. OU-Optimal (이론적 최적)
  4. Adaptive (공적분 기반 이중 조건)

사용법:
    python backtest_compare.py           # 3개 페어 전체
    python backtest_compare.py --pair AUDUSD,NZDUSD  # 단일 페어
"""

import sys
import os
import argparse
import time

import pandas as pd

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from data_loader import DataLoader
from config import (SYMBOLS, RISK_CONFIG,
                    RATIO_BB_CONFIG, KALMAN_H1_CONFIG,
                    OU_OPTIMAL_CONFIG, H1_ADAPTIVE_CONFIG)
from strategy_baselines import (RatioBollingerStrategy,
                                KalmanStrategy,
                                OUOptimalStrategy)
from strategy_adaptive import AdaptiveStatArbStrategy, AdaptiveBacktester
from h1_rolling_coint_scan import load_h1_pair_chunked


TOP_PAIRS = [
    ("EURCHF", "USDJPY"),
    ("AUDUSD", "NZDUSD"),
    ("USDCHF", "EURJPY"),
]

STRATEGY_NAMES = ["RatioBB", "Kalman", "OU-Opt", "Adaptive"]


def make_backtester(sym_y, sym_x):
    """심볼별 거래 비용으로 백테스터 생성"""
    yc = SYMBOLS[sym_y]
    xc = SYMBOLS[sym_x]
    cost_y = yc["pip_size"] * (yc["spread_pips"] + yc["commission_pips"])
    cost_x = xc["pip_size"] * (xc["spread_pips"] + xc["commission_pips"])
    slip_y = yc["pip_size"] * RISK_CONFIG["slippage_pips"]
    slip_x = xc["pip_size"] * RISK_CONFIG["slippage_pips"]
    return AdaptiveBacktester(
        spread_cost_y=cost_y, spread_cost_x=cost_x,
        slippage_y=slip_y, slippage_x=slip_x,
    )


def make_strategies():
    """4가지 전략 인스턴스 생성"""
    return [
        ("RatioBB", RatioBollingerStrategy(**RATIO_BB_CONFIG)),
        ("Kalman", KalmanStrategy(**KALMAN_H1_CONFIG)),
        ("OU-Opt", OUOptimalStrategy(**OU_OPTIMAL_CONFIG)),
        ("Adaptive", AdaptiveStatArbStrategy(
            coint_window=H1_ADAPTIVE_CONFIG["coint_window"],
            coint_recheck=H1_ADAPTIVE_CONFIG["coint_recheck"],
            coint_pvalue=H1_ADAPTIVE_CONFIG["coint_pvalue"],
            z_entry=H1_ADAPTIVE_CONFIG["z_entry"],
            z_exit=H1_ADAPTIVE_CONFIG["z_exit"],
            z_stop=H1_ADAPTIVE_CONFIG["z_stop"],
            max_holding_bars=H1_ADAPTIVE_CONFIG["max_holding_bars"],
            lookback=H1_ADAPTIVE_CONFIG["lookback"],
            degraded_z_exit=H1_ADAPTIVE_CONFIG["degraded_z_exit"],
            degraded_timeout=H1_ADAPTIVE_CONFIG["degraded_timeout"],
        )),
    ]


def run_pair(sym_y, sym_x, output_dir):
    """단일 페어에 대해 4개 전략 실행"""
    pair_name = f"{sym_y}/{sym_x}"
    safe = f"{sym_y}_{sym_x}"

    print(f"\n{'#'*70}")
    print(f"  Loading: {pair_name}")
    print(f"{'#'*70}")

    loader = DataLoader()
    data = load_h1_pair_chunked(loader, sym_y, sym_x)
    y, x = data[sym_y], data[sym_x]
    period = f"{data.index[0].date()} ~ {data.index[-1].date()}"
    print(f"  Period: {period} ({len(data):,} H1 bars)")

    bt = make_backtester(sym_y, sym_x)
    strategies = make_strategies()
    results = []

    for name, strategy in strategies:
        print(f"\n  ── {name} ──")
        t0 = time.time()

        try:
            signals = strategy.generate_signals(y, x, verbose=True)
            result = bt.run(signals)
            elapsed = time.time() - t0
            m = result["metrics"]

            if m["total_trades"] == 0:
                print(f"  No trades ({elapsed:.0f}s)")
                results.append({
                    "strategy": name, "pair": pair_name,
                    "trades": 0, "wr": 0, "pnl": 0, "pf": 0,
                    "sharpe": 0, "avg_hold": 0, "max_dd_pct": 0,
                    "time": elapsed,
                })
                continue

            # Print summary
            print(f"  Trades: {m['total_trades']}  WR: {m['win_rate']:.1f}%  "
                  f"PnL: {m['total_pnl']:+.4f}  PF: {m['profit_factor']:.2f}  "
                  f"Sharpe: {m['sharpe']:.2f}  "
                  f"AvgH: {m['avg_bars_held']:.0f}h  "
                  f"MDD: {m['max_drawdown_pct']:.1f}%  "
                  f"({elapsed:.0f}s)")

            # Exit reason breakdown
            reasons = m.get("exit_reasons", {})
            if reasons:
                parts = [f"{r}={c}" for r, c in
                         sorted(reasons.items(), key=lambda x: -x[1])]
                print(f"  Exits: {', '.join(parts)}")

            # Annual breakdown
            trades = result["trades"]
            if len(trades) > 0 and "entry_time" in trades.columns:
                tc = trades.copy()
                tc["year"] = pd.to_datetime(tc["entry_time"]).dt.year
                yearly = tc.groupby("year").agg(
                    n=("pnl", "count"),
                    pnl=("pnl", "sum"),
                    wr=("pnl", lambda x: (x > 0).mean() * 100),
                )
                print(f"\n  {'Year':>6} {'N':>5} {'PnL':>10} {'WR':>5}")
                print(f"  {'-'*30}")
                for yr, row in yearly.iterrows():
                    print(f"  {yr:>6} {row['n']:>5} {row['pnl']:>+10.4f} "
                          f"{row['wr']:>4.0f}%")

            # Save trades
            csv_path = os.path.join(output_dir, f"compare_{safe}_{name}.csv")
            trades.to_csv(csv_path, index=False, float_format="%.6f")

            results.append({
                "strategy": name, "pair": pair_name,
                "trades": m["total_trades"],
                "wr": m["win_rate"],
                "pnl": m["total_pnl"],
                "pf": m["profit_factor"],
                "sharpe": m["sharpe"],
                "avg_hold": m["avg_bars_held"],
                "max_dd_pct": m["max_drawdown_pct"],
                "time": elapsed,
            })

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ERROR: {e} ({elapsed:.0f}s)")
            import traceback
            traceback.print_exc()
            results.append({
                "strategy": name, "pair": pair_name,
                "trades": 0, "wr": 0, "pnl": 0, "pf": 0,
                "sharpe": 0, "avg_hold": 0, "max_dd_pct": 0,
                "time": elapsed,
            })

    return results


def print_comparison_table(all_results):
    """전략 × 페어 비교 테이블 출력"""
    df = pd.DataFrame(all_results)

    print(f"\n{'='*80}")
    print(f"  Strategy Comparison (H1, 2006-2026)")
    print(f"{'='*80}")
    print(f"\n  {'Strategy':<10} {'Pair':<16} {'Trades':>7} {'WR':>5} "
          f"{'PnL':>10} {'PF':>5} {'Sharpe':>7} {'AvgH':>5} {'MDD%':>6}")
    print(f"  {'-'*72}")

    for _, row in df.iterrows():
        print(f"  {row['strategy']:<10} {row['pair']:<16} "
              f"{row['trades']:>7} {row['wr']:>4.0f}% "
              f"{row['pnl']:>+10.4f} {row['pf']:>5.2f} "
              f"{row['sharpe']:>7.2f} {row['avg_hold']:>4.0f}h "
              f"{row['max_dd_pct']:>5.1f}%")

    # Per-strategy summary (across all pairs)
    print(f"\n  {'─'*40}")
    print(f"  Strategy Totals:")
    print(f"  {'Strategy':<10} {'Trades':>7} {'Total PnL':>12} {'Avg Sharpe':>11}")
    print(f"  {'-'*42}")
    for name in STRATEGY_NAMES:
        sub = df[df["strategy"] == name]
        if len(sub) == 0:
            continue
        print(f"  {name:<10} {sub['trades'].sum():>7} "
              f"{sub['pnl'].sum():>+12.4f} "
              f"{sub['sharpe'].mean():>11.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="H1 StatArb Strategy Comparison")
    parser.add_argument("--pair", type=str, default=None)
    parser.add_argument("--output", type=str, default="outputs")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    pairs = TOP_PAIRS
    if args.pair:
        sy, sx = args.pair.split(",")
        pairs = [(sy, sx)]

    all_results = []
    for sym_y, sym_x in pairs:
        pair_results = run_pair(sym_y, sym_x, args.output)
        all_results.extend(pair_results)

    print_comparison_table(all_results)

    # Save summary
    summary_path = os.path.join(args.output, "compare_summary.csv")
    pd.DataFrame(all_results).to_csv(summary_path, index=False, float_format="%.4f")
    print(f"\n  Summary: {summary_path}")


if __name__ == "__main__":
    main()
