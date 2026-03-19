# -*- coding: utf-8 -*-
"""
StatArb - 백테스트 CLI

사용법:
    # 기본 (EURCHF/USDJPY D1, HMM 포함)
    python backtest.py --pair EURCHF,USDJPY

    # HMM 없이
    python backtest.py --pair EURCHF,USDJPY --no-hmm

    # 파라미터 오버라이드
    python backtest.py --pair EURCHF,USDJPY --z-entry 1.5 --z-exit 0.3

    # 다른 페어
    python backtest.py --pair AUDUSD,NZDUSD
    python backtest.py --pair USDCHF,EURJPY

    # 그리드 서치
    python backtest.py --pair EURCHF,USDJPY --grid
"""

import sys
import argparse
import itertools

import pandas as pd

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from data_loader import DataLoader
from config import SYMBOLS, TRADING_CONFIG, RISK_CONFIG
from strategy import StatArbStrategy, StatArbBacktester, print_backtest_report


def run_single(
    sym_y: str, sym_x: str,
    tf: str, start: str, end: str,
    z_entry: float, z_exit: float, z_stop: float,
    max_holding: int, lookback: int,
    use_hmm: bool, mr_threshold: float,
) -> dict:
    """단일 백테스트 실행"""
    loader = DataLoader()
    data = loader.load_pair(sym_y, sym_x, tf, start, end)
    y, x = data[sym_y], data[sym_x]

    # 전략 설정
    strategy = StatArbStrategy(
        z_entry=z_entry,
        z_exit=z_exit,
        z_stop=z_stop,
        max_holding_bars=max_holding,
        lookback=lookback,
        use_hmm=use_hmm,
        mr_threshold=mr_threshold,
    )

    signals = strategy.generate_signals(y, x)

    # 거래 비용 계산 (가격 단위)
    sym_y_cfg = SYMBOLS[sym_y]
    sym_x_cfg = SYMBOLS[sym_x]
    cost_y = sym_y_cfg["pip_size"] * (sym_y_cfg["spread_pips"] + sym_y_cfg["commission_pips"])
    cost_x = sym_x_cfg["pip_size"] * (sym_x_cfg["spread_pips"] + sym_x_cfg["commission_pips"])
    slippage_y = sym_y_cfg["pip_size"] * RISK_CONFIG["slippage_pips"]
    slippage_x = sym_x_cfg["pip_size"] * RISK_CONFIG["slippage_pips"]

    backtester = StatArbBacktester(
        spread_cost_y=cost_y,
        spread_cost_x=cost_x,
        slippage_y=slippage_y,
        slippage_x=slippage_x,
    )

    result = backtester.run(signals)
    result["params"] = {
        "z_entry": z_entry, "z_exit": z_exit, "z_stop": z_stop,
        "max_holding": max_holding, "lookback": lookback,
        "use_hmm": use_hmm, "mr_threshold": mr_threshold,
    }

    return result


def run_grid(
    sym_y: str, sym_x: str,
    tf: str, start: str, end: str,
) -> pd.DataFrame:
    """파라미터 그리드 서치"""
    param_grid = {
        "z_entry": [1.5, 2.0, 2.5],
        "z_exit": [0.0, 0.25, 0.5],
        "use_hmm": [True, False],
        "lookback": [50, 100, 200],
    }

    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    total = len(combos)

    print(f"Grid search: {total} combinations")
    print(f"{'='*70}")

    results = []

    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        print(f"  [{i}/{total}] z_entry={params['z_entry']} z_exit={params['z_exit']} "
              f"hmm={params['use_hmm']} lb={params['lookback']}...", end=" ", flush=True)

        try:
            result = run_single(
                sym_y, sym_x, tf, start, end,
                z_entry=params["z_entry"],
                z_exit=params["z_exit"],
                z_stop=TRADING_CONFIG["z_score_stop"],
                max_holding=TRADING_CONFIG["max_holding_bars"],
                lookback=params["lookback"],
                use_hmm=params["use_hmm"],
                mr_threshold=TRADING_CONFIG["mr_prob_threshold"],
            )

            m = result["metrics"]
            row = {**params, **m}
            results.append(row)

            if m["total_trades"] > 0:
                print(f"trades={m['total_trades']} wr={m['win_rate']:.0f}% "
                      f"pnl={m['total_pnl']:.4f} sharpe={m['sharpe']:.2f}")
            else:
                print("no trades")

        except Exception as e:
            print(f"ERROR: {e}")

    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.sort_values("total_pnl", ascending=False).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="StatArb Backtester")
    parser.add_argument("--pair", type=str, default="EURCHF,USDJPY",
                        help="Pair (y,x)")
    parser.add_argument("--tf", type=str, default="D1")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")

    parser.add_argument("--z-entry", type=float, default=None)
    parser.add_argument("--z-exit", type=float, default=None)
    parser.add_argument("--z-stop", type=float, default=None)
    parser.add_argument("--max-hold", type=int, default=None)
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--no-hmm", action="store_true")
    parser.add_argument("--mr-threshold", type=float, default=None)

    parser.add_argument("--grid", action="store_true", help="Parameter grid search")
    parser.add_argument("--output", type=str, default=None, help="Save grid to CSV")
    args = parser.parse_args()

    sym_y, sym_x = args.pair.split(",")

    if args.grid:
        df = run_grid(sym_y, sym_x, args.tf, args.start, args.end)

        print(f"\n{'='*70}")
        print(f"  Grid Results: {sym_y}/{sym_x} ({args.tf})")
        print(f"  Top 10 by Total P&L:")
        print(f"{'='*70}")

        if len(df) > 0:
            display_cols = ["z_entry", "z_exit", "use_hmm", "lookback",
                           "total_trades", "win_rate", "total_pnl",
                           "profit_factor", "sharpe", "max_drawdown_pct"]
            available = [c for c in display_cols if c in df.columns]
            pd.set_option("display.float_format", lambda x: f"{x:.4f}")
            print(df[available].head(10).to_string(index=False))

        if args.output:
            df.to_csv(args.output, index=False)
            print(f"\nSaved to {args.output}")
    else:
        z_entry = args.z_entry or TRADING_CONFIG["z_score_entry"]
        z_exit = args.z_exit if args.z_exit is not None else TRADING_CONFIG["z_score_exit"]
        z_stop = args.z_stop or TRADING_CONFIG["z_score_stop"]
        max_hold = args.max_hold or TRADING_CONFIG["max_holding_bars"]
        lookback = args.lookback or 100
        use_hmm = not args.no_hmm
        mr_thresh = args.mr_threshold or TRADING_CONFIG["mr_prob_threshold"]

        print(f"Pair: {sym_y}/{sym_x} ({args.tf})")
        print(f"Period: {args.start} ~ {args.end}")
        print(f"Params: z_entry={z_entry} z_exit={z_exit} z_stop={z_stop} "
              f"hold={max_hold} lb={lookback} hmm={use_hmm} mr={mr_thresh}")

        result = run_single(
            sym_y, sym_x, args.tf, args.start, args.end,
            z_entry, z_exit, z_stop, max_hold, lookback,
            use_hmm, mr_thresh,
        )

        print_backtest_report(result, f"{sym_y}/{sym_x} ({args.tf})")


if __name__ == "__main__":
    main()
