# -*- coding: utf-8 -*-
"""
파라미터 그리드 백테스트

RR(손익비) × SL(타임프레임) × Compositor(on/off) 조합을 자동 실행하여
최적 파라미터를 탐색.

사용법:
    python backtest_grid.py --symbol EURUSD --source db --start 2015-01-01 --end 2025-12-31
    python backtest_grid.py --symbol EURUSD --source csv
"""

import sys
import argparse
import time
import pandas as pd
from datetime import datetime
from pathlib import Path
from itertools import product

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from ob_fvg_strategy import OrderBlockFVGStrategy
from signal_compositor import SignalCompositor
from backtest_ob_fvg import BacktestEngine
from config import BACKTEST_CONFIG

# 그리드 파라미터
PARAM_GRID = {
    "risk_reward_ratio": [1.5, 2.0, 3.0, 5.0, 10.0],
    "sl_timeframe": ["M1", "M15"],
    "use_compositor": [False],
}


def load_data(symbol: str, source: str, start: str = None, end: str = None) -> pd.DataFrame:
    """데이터 로드 (1회만 실행)"""
    if source == "db":
        from db import Database
        db = Database()
        s = datetime.strptime(start, "%Y-%m-%d") if start else None
        e = datetime.strptime(end, "%Y-%m-%d") if end else None
        df = db.query_ohlcv(symbol, start=s, end=e)
    else:
        csv_path = Path(BACKTEST_CONFIG["data_dir"]) / f"{symbol}_M1_data.csv"
        df = pd.read_csv(csv_path, parse_dates=['time'])
        df = df.sort_values('time').reset_index(drop=True)
    return df


def run_grid(symbol: str, m1_df: pd.DataFrame) -> pd.DataFrame:
    """파라미터 그리드 실행"""
    results = []
    combos = list(product(
        PARAM_GRID["risk_reward_ratio"],
        PARAM_GRID["sl_timeframe"],
        PARAM_GRID["use_compositor"],
    ))

    total = len(combos)
    print(f"\nGrid search: {total} combinations")
    print("=" * 80)

    for idx, (rr, sl_tf, use_comp) in enumerate(combos, 1):
        label = f"RR={rr} SL={sl_tf} Comp={'ON' if use_comp else 'OFF'}"
        print(f"\n[{idx}/{total}] {label}")

        t0 = time.time()

        if use_comp:
            strategy = SignalCompositor(
                symbol=symbol,
                risk_reward_ratio=rr,
                sl_timeframe=sl_tf,
            )
        else:
            strategy = OrderBlockFVGStrategy(
                symbol=symbol,
                risk_reward_ratio=rr,
                sl_timeframe=sl_tf,
            )

        engine = BacktestEngine(strategy)
        perf = engine.run_backtest(m1_df)
        elapsed = time.time() - t0

        row = {
            'symbol': symbol,
            'rr': rr,
            'sl_timeframe': sl_tf,
            'compositor': use_comp,
            'elapsed_sec': round(elapsed, 1),
        }
        row.update(perf)
        results.append(row)

        if perf:
            print(f"  -> {perf.get('total_trades', 0)} trades | "
                  f"WR {perf.get('win_rate_%', 0):.1f}% | "
                  f"Net {perf.get('total_net_pips', 0):+.1f}p | "
                  f"DD {perf.get('max_drawdown_pips', 0):.1f}p | "
                  f"{elapsed:.1f}s")
        else:
            print(f"  -> No trades | {elapsed:.1f}s")

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="ICT 파라미터 그리드 백테스트")
    parser.add_argument("--symbol", type=str, default="EURUSD",
                        choices=["EURUSD", "USDJPY", "EURJPY", "XAUUSD"])
    parser.add_argument("--source", type=str, default="db", choices=["csv", "db"])
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    args = parser.parse_args()

    print("=" * 80)
    print(f"  ICT Grid Backtest: {args.symbol}")
    print(f"  Source: {args.source} | Start: {args.start} | End: {args.end}")
    print("=" * 80)

    # 데이터 1회 로드
    print("\nLoading data...")
    m1_df = load_data(args.symbol, args.source, args.start, args.end)
    print(f"Loaded {len(m1_df):,} M1 bars")

    if len(m1_df) == 0:
        print("No data found.")
        return

    # 그리드 실행
    results_df = run_grid(args.symbol, m1_df)

    # 결과 저장
    output_dir = Path(BACKTEST_CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.symbol}_grid_results.csv"
    results_df.to_csv(output_file, index=False)

    # 요약 출력
    print("\n" + "=" * 80)
    print("GRID RESULTS SUMMARY")
    print("=" * 80)

    if len(results_df) > 0 and 'total_net_pips' in results_df.columns:
        summary_cols = ['rr', 'sl_timeframe', 'compositor', 'total_trades',
                        'win_rate_%', 'total_net_pips', 'risk_reward_ratio',
                        'max_drawdown_pips', 'elapsed_sec']
        available_cols = [c for c in summary_cols if c in results_df.columns]
        print(results_df[available_cols].to_string(index=False))

        # 최고 성과
        best = results_df.loc[results_df['total_net_pips'].idxmax()]
        print(f"\nBest: RR={best['rr']} SL={best['sl_timeframe']} "
              f"Comp={best['compositor']} -> {best['total_net_pips']:+.1f}p")

    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()
