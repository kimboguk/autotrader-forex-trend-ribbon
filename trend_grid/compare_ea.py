# -*- coding: utf-8 -*-
"""
Compare Python backtest trades vs MT5 EA backtest trades.

Usage:
    python compare_ea.py <ea_report.csv> [--symbol EURUSD] [--start 2025-01-01] [--end 2025-12-31]

Outputs:
    1. python_trades_2025.csv  — Python trade log
    2. comparison_report.txt   — Side-by-side diff
"""

import sys
import os
import re
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Add parent dirs to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "common"))

from backtest import run_backtest


def parse_ea_report(filepath: str) -> pd.DataFrame:
    """Parse MT5 Strategy Tester CSV report to extract trades."""
    # Try different encodings
    for enc in ['utf-16-le', 'utf-16', 'utf-8', 'cp949', 'euc-kr']:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                content = f.read()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        # Read as binary and clean
        with open(filepath, 'rb') as f:
            raw = f.read()
        content = raw.decode('utf-16-le', errors='ignore')

    # Clean up wide-char spacing
    content = content.replace('\x00', '')

    lines = content.strip().split('\n')

    trades = []
    in_trades = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect trade section by looking for "balance" entry or trade data
        parts = [p.strip() for p in line.split(',')]

        # Look for trade entries: time, deal#, symbol, type, direction, volume, price, ...
        if len(parts) >= 10:
            time_str = parts[0].strip()
            # Check if it looks like a datetime
            if re.match(r'\d{4}\.\d{2}\.\d{2}', time_str):
                symbol = parts[1].strip()
                trade_type = parts[2].strip()
                direction = parts[3].strip()

                if trade_type in ('buy', 'sell') and direction in ('in', 'out'):
                    try:
                        volume = float(parts[4].replace(' ', ''))
                        price = float(parts[5].replace(' ', ''))
                        pnl = float(parts[9].replace(' ', '')) if parts[9].strip() else 0
                    except (ValueError, IndexError):
                        continue

                    trades.append({
                        'time': pd.Timestamp(time_str.replace('.', '-')),
                        'symbol': symbol,
                        'type': trade_type,
                        'direction': direction,
                        'volume': volume,
                        'price': price,
                        'pnl': pnl,
                    })

    df = pd.DataFrame(trades)
    if df.empty:
        print("WARNING: No trades found in EA report. Trying alternative parser...")
        df = parse_ea_report_alt(filepath)

    return df


def parse_ea_report_alt(filepath: str) -> pd.DataFrame:
    """Alternative parser for EA report CSV (tab-separated format)."""
    for enc in ['utf-16-le', 'utf-16', 'utf-8', 'cp949']:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                content = f.read()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue

    content = content.replace('\x00', '')
    lines = content.strip().split('\n')

    trades = []
    for line in lines:
        parts = [p.strip() for p in re.split(r'[,\t]', line)]
        # Filter for trade-like lines
        for i, p in enumerate(parts):
            if re.match(r'\d{4}\.\d{2}\.\d{2}\s+\d{2}:\d{2}', p):
                remaining = parts[i:]
                if len(remaining) >= 8:
                    if any(x in remaining for x in ['buy', 'sell']):
                        # Found a trade line
                        break

    return pd.DataFrame(trades)


def run_python_backtest(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Run Python backtest and return trades DataFrame."""
    print(f"\n=== Running Python Backtest: {symbol} M30+H4 ({start} ~ {end}) ===")

    result = run_backtest(
        symbol=symbol,
        timeframe="M30",
        ma_type="ema",
        start=start,
        end=end,
        filter_tfs=["H4"],
        ribbon_periods=[30, 60, 120, 240],
        verbose=True,
        _keep_cache=True,
    )

    trades_df = result["trades"]
    if trades_df.empty:
        print("No trades from Python backtest!")
        return trades_df

    print(f"  Python trades: {len(trades_df)}")
    print(f"  PnL: ${trades_df['pnl_usd'].sum():.2f}")

    return trades_df


def compare_trades(py_trades: pd.DataFrame, ea_trades: pd.DataFrame):
    """Compare Python and EA trades side by side."""
    print("\n" + "=" * 80)
    print("COMPARISON REPORT")
    print("=" * 80)

    # Pair up EA trades (in/out)
    ea_pairs = []
    pending = {}
    for _, row in ea_trades.iterrows():
        sym = row['symbol']
        if row['direction'] == 'in':
            pending[sym] = row
        elif row['direction'] == 'out' and sym in pending:
            entry = pending.pop(sym)
            ea_pairs.append({
                'entry_time': entry['time'],
                'exit_time': row['time'],
                'direction': 1 if entry['type'] == 'buy' else -1,
                'entry_price': entry['price'],
                'exit_price': row['price'],
                'pnl': row['pnl'],
            })
    ea_df = pd.DataFrame(ea_pairs)

    print(f"\nPython trades: {len(py_trades)}")
    print(f"EA trades: {len(ea_df)}")

    if py_trades.empty or ea_df.empty:
        print("Cannot compare — one side has no trades.")
        return

    # Summary stats
    py_pnl = py_trades['pnl_usd'].sum() if 'pnl_usd' in py_trades.columns else 0
    ea_pnl = ea_df['pnl'].sum()
    print(f"\nPython total PnL: ${py_pnl:.2f}")
    print(f"EA total PnL:     ${ea_pnl:.2f}")
    print(f"Difference:       ${py_pnl - ea_pnl:.2f}")

    # Match trades by entry time (within 30 min tolerance)
    matched = 0
    py_only = 0
    ea_only = 0
    price_diffs = []

    py_times = py_trades['entry_time'].values if 'entry_time' in py_trades.columns else []
    ea_used = set()

    report_lines = []
    report_lines.append(f"{'#':>4} | {'Python Entry':>20} {'Dir':>5} {'Price':>10} {'PnL':>8} | "
                        f"{'EA Entry':>20} {'Dir':>5} {'Price':>10} {'PnL':>8} | {'Note'}")
    report_lines.append("-" * 120)

    for i, py_row in py_trades.iterrows():
        py_entry = pd.Timestamp(py_row['entry_time'])
        py_dir = py_row.get('direction', 0)
        py_price = py_row.get('entry_price', 0)
        py_pnl_i = py_row.get('pnl_usd', 0)

        # Find closest EA trade within 60 min
        best_match = None
        best_dt = pd.Timedelta(hours=2)

        for j, ea_row in ea_df.iterrows():
            if j in ea_used:
                continue
            dt = abs(py_entry - ea_row['entry_time'])
            if dt < best_dt:
                best_dt = dt
                best_match = (j, ea_row)

        if best_match and best_dt <= pd.Timedelta(minutes=60):
            j, ea_row = best_match
            ea_used.add(j)
            matched += 1
            pdiff = abs(py_price - ea_row['entry_price'])
            price_diffs.append(pdiff)

            note = "OK" if py_dir == ea_row['direction'] else "DIR MISMATCH"
            if pdiff > 0.001:
                note += f" price_diff={pdiff:.5f}"

            report_lines.append(
                f"{i:4d} | {str(py_entry):>20} {py_dir:>5} {py_price:>10.5f} {py_pnl_i:>8.1f} | "
                f"{str(ea_row['entry_time']):>20} {ea_row['direction']:>5} {ea_row['entry_price']:>10.5f} {ea_row['pnl']:>8.1f} | {note}"
            )
        else:
            py_only += 1
            report_lines.append(
                f"{i:4d} | {str(py_entry):>20} {py_dir:>5} {py_price:>10.5f} {py_pnl_i:>8.1f} | "
                f"{'--- NO MATCH ---':>55} | PY ONLY"
            )

    # EA trades not matched
    for j, ea_row in ea_df.iterrows():
        if j not in ea_used:
            ea_only += 1
            report_lines.append(
                f"{'':>4} | {'--- NO MATCH ---':>46} | "
                f"{str(ea_row['entry_time']):>20} {ea_row['direction']:>5} {ea_row['entry_price']:>10.5f} {ea_row['pnl']:>8.1f} | EA ONLY"
            )

    print(f"\nMatched:    {matched}")
    print(f"Python only: {py_only}")
    print(f"EA only:     {ea_only}")
    if price_diffs:
        print(f"Avg price diff: {np.mean(price_diffs):.5f}")
        print(f"Max price diff: {np.max(price_diffs):.5f}")

    # Write report
    report_path = Path(__file__).parent / "comparison_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"\nDetailed report: {report_path}")

    # Print first 20 lines
    print("\n--- First 20 trades ---")
    for line in report_lines[:22]:
        print(line)


def main():
    parser = argparse.ArgumentParser(description="Compare Python vs EA backtest")
    parser.add_argument("ea_report", nargs="?", help="MT5 EA report CSV file")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--py-only", action="store_true", help="Only run Python backtest and export CSV")
    args = parser.parse_args()

    # Step 1: Run Python backtest
    py_trades = run_python_backtest(args.symbol, args.start, args.end)

    # Save Python trades
    py_csv = Path(__file__).parent / f"python_trades_{args.symbol}_{args.start[:4]}.csv"
    if not py_trades.empty:
        py_trades.to_csv(py_csv, index=False)
        print(f"  Saved: {py_csv}")

    if args.py_only:
        return

    # Step 2: Parse EA report
    if not args.ea_report:
        print("\nNo EA report provided. Use --py-only or provide EA report CSV.")
        return

    print(f"\n=== Parsing EA Report: {args.ea_report} ===")
    ea_trades = parse_ea_report(args.ea_report)
    print(f"  EA trades: {len(ea_trades)}")

    # Step 3: Compare
    compare_trades(py_trades, ea_trades)


if __name__ == "__main__":
    main()
