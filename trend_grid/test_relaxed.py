# -*- coding: utf-8 -*-
"""Quick comparison: standard vs relaxed entry for TR M30+H4."""

import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import strategy
import backtest as bt
from trade_engine import clear_m1_cache

SYMBOLS = ["EURUSD", "USDJPY", "EURJPY", "XAUUSD", "GBPUSD"]
TF = "M30"
FILTER = ["H4"]
MA = "ema"

_orig_gen = strategy.generate_signals


def _make_patched(relaxed):
    def _patched(df, ma_type=None, periods=None, relaxed_entry=False):
        return _orig_gen(df, ma_type, periods, relaxed_entry=relaxed)
    return _patched


def run_one(symbol, relaxed):
    patched = _make_patched(relaxed)
    # Patch both strategy module and backtest module's reference
    strategy.generate_signals = patched
    bt.generate_signals = patched
    try:
        result = bt.run_backtest(
            symbol=symbol, timeframe=TF, ma_type=MA,
            filter_tfs=FILTER, verbose=False, _keep_cache=True,
        )
    finally:
        strategy.generate_signals = _orig_gen
        bt.generate_signals = _orig_gen
    return result["stats"]


def fmt(s):
    return (f"Trades={s['total_trades']:>5}  "
            f"WR={s['win_rate']:>5.1f}%  "
            f"PF={s['profit_factor']:>5.2f}  "
            f"Pips={s['total_pnl_pips']:>+9.1f}  "
            f"USD={s['total_pnl_usd']:>+10.2f}  "
            f"MDD={s['max_drawdown_pct']:>6.2f}%  "
            f"SR={s.get('sharpe_ratio', 0):>5.2f}")


print("=" * 80)
print("Trend Ribbon M30+H4 EMA — Standard vs Relaxed Entry")
print("=" * 80)

std_totals = {"trades": 0, "pips": 0, "usd": 0}
rel_totals = {"trades": 0, "pips": 0, "usd": 0}

for sym in SYMBOLS:
    print(f"\n{sym}:")
    s1 = run_one(sym, relaxed=False)
    print(f"  [Standard] {fmt(s1)}")
    std_totals["trades"] += s1["total_trades"]
    std_totals["pips"] += s1["total_pnl_pips"]
    std_totals["usd"] += s1["total_pnl_usd"]

    s2 = run_one(sym, relaxed=True)
    print(f"  [Relaxed ] {fmt(s2)}")
    rel_totals["trades"] += s2["total_trades"]
    rel_totals["pips"] += s2["total_pnl_pips"]
    rel_totals["usd"] += s2["total_pnl_usd"]

print(f"\n{'='*80}")
print("PORTFOLIO TOTALS:")
print(f"  [Standard]  Trades={std_totals['trades']:>5}  Pips={std_totals['pips']:>+10.1f}  USD={std_totals['usd']:>+11.2f}")
print(f"  [Relaxed ]  Trades={rel_totals['trades']:>5}  Pips={rel_totals['pips']:>+10.1f}  USD={rel_totals['usd']:>+11.2f}")

clear_m1_cache()
print()
