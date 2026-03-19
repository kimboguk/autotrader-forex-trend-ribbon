# -*- coding: utf-8 -*-
"""
StatArb - H1 적응형 전략 백테스트

사용법:
    # 단일 페어
    python backtest_adaptive.py --pair EURCHF,USDJPY

    # 모든 상위 3개 페어
    python backtest_adaptive.py --all

    # 파라미터 오버라이드
    python backtest_adaptive.py --pair AUDUSD,NZDUSD --z-entry 1.5 --coint-window 500
"""

import sys
import os
import argparse
import time

import pandas as pd

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from data_loader import DataLoader
from config import SYMBOLS, RISK_CONFIG, H1_ADAPTIVE_CONFIG
from strategy_adaptive import AdaptiveStatArbStrategy, AdaptiveBacktester
from h1_rolling_coint_scan import load_h1_pair_chunked


TOP_PAIRS = [
    ("EURCHF", "USDJPY"),
    ("AUDUSD", "NZDUSD"),
    ("USDCHF", "EURJPY"),
]


def run_adaptive_backtest(sym_y, sym_x, **overrides):
    """단일 페어 적응형 백테스트"""
    loader = DataLoader()

    # H1 데이터 로드 (chunked)
    data = load_h1_pair_chunked(loader, sym_y, sym_x)
    y, x = data[sym_y], data[sym_x]

    # 전략 설정
    cfg = dict(H1_ADAPTIVE_CONFIG)
    cfg.update(overrides)

    strategy = AdaptiveStatArbStrategy(
        coint_window=cfg["coint_window"],
        coint_recheck=cfg["coint_recheck"],
        coint_pvalue=cfg["coint_pvalue"],
        z_entry=cfg["z_entry"],
        z_exit=cfg["z_exit"],
        z_stop=cfg["z_stop"],
        max_holding_bars=cfg["max_holding_bars"],
        lookback=cfg["lookback"],
        degraded_z_exit=cfg["degraded_z_exit"],
        degraded_timeout=cfg["degraded_timeout"],
    )

    t0 = time.time()
    signals = strategy.generate_signals(y, x, verbose=True)
    sig_time = time.time() - t0

    # 거래 비용
    sym_y_cfg = SYMBOLS[sym_y]
    sym_x_cfg = SYMBOLS[sym_x]
    cost_y = sym_y_cfg["pip_size"] * (sym_y_cfg["spread_pips"] + sym_y_cfg["commission_pips"])
    cost_x = sym_x_cfg["pip_size"] * (sym_x_cfg["spread_pips"] + sym_x_cfg["commission_pips"])
    slip_y = sym_y_cfg["pip_size"] * RISK_CONFIG["slippage_pips"]
    slip_x = sym_x_cfg["pip_size"] * RISK_CONFIG["slippage_pips"]

    backtester = AdaptiveBacktester(
        spread_cost_y=cost_y,
        spread_cost_x=cost_x,
        slippage_y=slip_y,
        slippage_x=slip_x,
    )

    result = backtester.run(signals)
    result["signal_time"] = sig_time
    result["params"] = cfg
    result["data_period"] = f"{data.index[0].date()} ~ {data.index[-1].date()}"
    result["n_bars"] = len(data)

    return result


def print_adaptive_report(result, pair_name):
    """적응형 백테스트 결과 출력"""
    m = result["metrics"]
    trades = result["trades"]

    print(f"\n{'='*70}")
    print(f"  H1 Adaptive StatArb: {pair_name}")
    print(f"  Period: {result['data_period']} ({result['n_bars']:,} H1 bars)")
    print(f"{'='*70}")

    if m["total_trades"] == 0:
        print("  No trades generated.")
        return

    p = result["params"]
    print(f"\n  Params: cw={p['coint_window']} cr={p['coint_recheck']} "
          f"z_e={p['z_entry']} z_x={p['z_exit']} z_s={p['z_stop']} "
          f"lb={p['lookback']} hold={p['max_holding_bars']}")
    print(f"  Degraded: z_x={p['degraded_z_exit']} timeout={p['degraded_timeout']}")

    print(f"\n  Trades:        {m['total_trades']}  "
          f"(Long: {m['long_trades']}, Short: {m['short_trades']})")
    print(f"  Win Rate:      {m['win_rate']:.1f}%")
    print(f"  Profit Factor: {m['profit_factor']:.2f}")
    print(f"  Avg P&L:       {m['avg_pnl']:.6f}")
    print(f"  Total P&L:     {m['total_pnl']:.4f}")
    print(f"  Avg Win:       {m['avg_win']:.6f}")
    print(f"  Avg Loss:      {m['avg_loss']:.6f}")
    print(f"  Sharpe:        {m['sharpe']:.2f}")
    print(f"  Sortino:       {m['sortino']:.2f}")
    print(f"  Max DD:        {m['max_drawdown']:.4f} ({m['max_drawdown_pct']:.1f}%)")
    print(f"  Avg Hold:      {m['avg_bars_held']:.0f}h ({m['avg_bars_held']/24:.1f}d)")
    print(f"  Max Hold:      {m['max_bars_held']}h ({m['max_bars_held']/24:.1f}d)")

    # Exit reason breakdown
    reasons = m.get("exit_reasons", {})
    if reasons:
        print(f"\n  Exit reasons:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason:<15} {count:>5} ({count/m['total_trades']*100:.1f}%)")

    print(f"\n  Signal time:   {result['signal_time']:.0f}s")

    # Annual breakdown
    if len(trades) > 0 and "entry_time" in trades.columns:
        tc = trades.copy()
        tc["year"] = pd.to_datetime(tc["entry_time"]).dt.year
        yearly = tc.groupby("year").agg(
            n=("pnl", "count"),
            pnl=("pnl", "sum"),
            wr=("pnl", lambda x: (x > 0).mean() * 100),
            avg_bars=("bars_held", "mean"),
        )
        print(f"\n  {'Year':>6} {'Trades':>7} {'PnL':>10} {'WR':>5} {'AvgH':>6}")
        print(f"  {'-'*42}")
        for yr, row in yearly.iterrows():
            print(f"  {yr:>6} {row['n']:>7} {row['pnl']:>+10.4f} "
                  f"{row['wr']:>4.0f}% {row['avg_bars']:>5.0f}h")


def main():
    parser = argparse.ArgumentParser(description="H1 Adaptive StatArb Backtester")
    parser.add_argument("--pair", type=str, default=None, help="Pair (y,x)")
    parser.add_argument("--all", action="store_true", help="Run all 3 top pairs")

    parser.add_argument("--coint-window", type=int, default=None)
    parser.add_argument("--coint-recheck", type=int, default=None)
    parser.add_argument("--coint-pvalue", type=float, default=None)
    parser.add_argument("--z-entry", type=float, default=None)
    parser.add_argument("--z-exit", type=float, default=None)
    parser.add_argument("--z-stop", type=float, default=None)
    parser.add_argument("--max-hold", type=int, default=None)
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--degraded-z-exit", type=float, default=None)
    parser.add_argument("--degraded-timeout", type=int, default=None)

    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    output_dir = args.output or "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Collect overrides (only non-None)
    overrides = {}
    mapping = {
        "coint_window": args.coint_window,
        "coint_recheck": args.coint_recheck,
        "coint_pvalue": args.coint_pvalue,
        "z_entry": args.z_entry,
        "z_exit": args.z_exit,
        "z_stop": args.z_stop,
        "max_holding_bars": args.max_hold,
        "lookback": args.lookback,
        "degraded_z_exit": args.degraded_z_exit,
        "degraded_timeout": args.degraded_timeout,
    }
    for k, v in mapping.items():
        if v is not None:
            overrides[k] = v

    pairs = TOP_PAIRS if args.all else []
    if args.pair:
        sym_y, sym_x = args.pair.split(",")
        pairs = [(sym_y, sym_x)]

    if not pairs:
        parser.print_help()
        return

    all_trades = []

    for sym_y, sym_x in pairs:
        pair_name = f"{sym_y}/{sym_x}"
        print(f"\n{'#'*70}")
        print(f"  Running: {pair_name}")
        print(f"{'#'*70}")

        try:
            result = run_adaptive_backtest(sym_y, sym_x, **overrides)
            print_adaptive_report(result, pair_name)

            if len(result["trades"]) > 0:
                safe = f"{sym_y}_{sym_x}"
                csv_path = os.path.join(output_dir, f"h1_adaptive_{safe}.csv")
                result["trades"].to_csv(csv_path, index=False, float_format="%.6f")
                print(f"\n  Saved: {csv_path}")

                all_trades.append(result["trades"])
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    if len(all_trades) > 1:
        combined = pd.concat(all_trades, ignore_index=True)
        csv_path = os.path.join(output_dir, "h1_adaptive_ALL.csv")
        combined.to_csv(csv_path, index=False, float_format="%.6f")
        print(f"\nCombined: {csv_path} ({len(combined)} trades)")


if __name__ == "__main__":
    main()
