# -*- coding: utf-8 -*-
"""Detailed yearly comparison for XAUUSD: standard vs relaxed entry."""

import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import strategy
import backtest as bt
from trade_engine import clear_m1_cache

_orig_gen = strategy.generate_signals

def _make_patched(relaxed):
    def _patched(df, ma_type=None, periods=None, relaxed_entry=False):
        return _orig_gen(df, ma_type, periods, relaxed_entry=relaxed)
    return _patched

def run_one(symbol, relaxed):
    patched = _make_patched(relaxed)
    strategy.generate_signals = patched
    bt.generate_signals = patched
    try:
        result = bt.run_backtest(
            symbol=symbol, timeframe="M30", ma_type="ema",
            filter_tfs=["H4"], verbose=False, _keep_cache=True,
        )
    finally:
        strategy.generate_signals = _orig_gen
        bt.generate_signals = _orig_gen
    return result

def yearly_breakdown(trades_df):
    if trades_df is None or len(trades_df) == 0:
        return []
    df = trades_df.copy()
    df["year"] = pd.to_datetime(df["exit_time"]).dt.year
    rows = []
    for year, grp in df.groupby("year"):
        n = len(grp)
        winners = grp[grp["pnl_usd"] > 0]
        losers = grp[grp["pnl_usd"] <= 0]
        gp = winners["pnl_usd"].sum() if len(winners) > 0 else 0
        gl = abs(losers["pnl_usd"].sum()) if len(losers) > 0 else 0
        pf = gp / gl if gl > 0 else float("inf")
        rows.append({
            "year": int(year),
            "trades": n,
            "win_rate": round(len(winners) / n * 100, 1) if n > 0 else 0,
            "pf": round(pf, 2),
            "pips": round(grp["net_pnl_pips"].sum(), 1),
            "usd": round(grp["pnl_usd"].sum(), 0),
        })
    return rows

print("=" * 90)
print("XAUUSD M30+H4 EMA — Yearly Comparison: Standard vs Relaxed")
print("=" * 90)
print(f"{'Year':>6} | {'--- Standard ---':^35} | {'--- Relaxed ---':^35}")
print(f"{'':>6} | {'Trades':>6} {'WR%':>5} {'PF':>5} {'Pips':>9} {'USD':>8} | {'Trades':>6} {'WR%':>5} {'PF':>5} {'Pips':>9} {'USD':>8}")
print("-" * 90)

r1 = run_one("XAUUSD", False)
r2 = run_one("XAUUSD", True)
y1 = yearly_breakdown(r1["trades"])
y2 = yearly_breakdown(r2["trades"])

y2_map = {y["year"]: y for y in y2}
for row in y1:
    yr = row["year"]
    r = y2_map.get(yr, {"trades": 0, "win_rate": 0, "pf": 0, "pips": 0, "usd": 0})
    print(f"{yr:>6} | {row['trades']:>6} {row['win_rate']:>5.1f} {row['pf']:>5.2f} {row['pips']:>+9.1f} {row['usd']:>+8.0f} | "
          f"{r['trades']:>6} {r['win_rate']:>5.1f} {r['pf']:>5.2f} {r['pips']:>+9.1f} {r['usd']:>+8.0f}")

clear_m1_cache()
print()
