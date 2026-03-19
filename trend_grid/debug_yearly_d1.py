# -*- coding: utf-8 -*-
import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(1, os.path.join(_here, '..', 'statarb'))

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from backtest import run_backtest

result = run_backtest('EURUSD', 'D1', ma_type='ema')
trades = result['trades']

trades['year'] = trades['entry_time'].dt.year

yearly = trades.groupby('year').agg(
    trades=('pnl_pips', 'count'),
    wins=('pnl_pips', lambda x: (x > 0).sum()),
    win_pct=('pnl_pips', lambda x: f"{(x > 0).mean()*100:.0f}%"),
    pnl_pips=('pnl_pips', 'sum'),
    pnl_usd=('pnl_usd', 'sum'),
    avg_win=('pnl_pips', lambda x: x[x > 0].mean() if (x > 0).any() else 0),
    avg_loss=('pnl_pips', lambda x: x[x <= 0].mean() if (x <= 0).any() else 0),
    best=('pnl_pips', 'max'),
    worst=('pnl_pips', 'min'),
).reset_index()

pd.set_option('display.float_format', lambda x: f'{x:+.1f}' if abs(x) >= 1 else f'{x:+.2f}')
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 140)

print("=" * 120)
print("  EURUSD D1 EMA Trend Grid - Yearly Breakdown")
print("=" * 120)
print(yearly.to_string(index=False))
print("-" * 120)
print(f"  Total: {len(trades)} trades, {trades['pnl_pips'].sum():+.1f} pips, ${trades['pnl_usd'].sum():+.2f}")

# 연속 손실연도 확인
print("\n  Loss years:", yearly[yearly['pnl_pips'] < 0]['year'].tolist())
print("  Win years: ", yearly[yearly['pnl_pips'] > 0]['year'].tolist())
