# -*- coding: utf-8 -*-
import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(1, os.path.join(_here, '..', 'statarb'))

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from backtest import run_backtest

pd.set_option('display.float_format', lambda x: f'{x:+.1f}' if abs(x) >= 1 else f'{x:+.2f}')
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 140)

for tf in ['D1', 'H4', 'H1', 'M30']:
    result = run_backtest('EURUSD', tf, ma_type='ema')
    trades = result['trades']
    trades['year'] = trades['entry_time'].dt.year

    yearly = trades.groupby('year').agg(
        trades=('net_pnl_pips', 'count'),
        wins=('net_pnl_pips', lambda x: (x > 0).sum()),
        win_pct=('net_pnl_pips', lambda x: f"{(x > 0).mean()*100:.0f}%"),
        net_pips=('net_pnl_pips', 'sum'),
        net_usd=('pnl_usd', 'sum'),
        avg_win=('net_pnl_pips', lambda x: x[x > 0].mean() if (x > 0).any() else 0),
        avg_loss=('net_pnl_pips', lambda x: x[x <= 0].mean() if (x <= 0).any() else 0),
        best=('net_pnl_pips', 'max'),
        worst=('net_pnl_pips', 'min'),
    ).reset_index()

    print()
    print("=" * 120)
    print(f"  EURUSD {tf} EMA Trend Grid - Yearly Breakdown (net of costs)")
    print("=" * 120)
    print(yearly.to_string(index=False))
    print("-" * 120)
    total_net_pips = trades['net_pnl_pips'].sum()
    total_net_usd = trades['pnl_usd'].sum()
    print(f"  Total: {len(trades)} trades, {total_net_pips:+.1f} net pips, ${total_net_usd:+.2f}")
    loss_years = yearly[yearly['net_pips'] < 0]['year'].tolist()
    win_years = yearly[yearly['net_pips'] > 0]['year'].tolist()
    print(f"  Win years ({len(win_years)}): {win_years}")
    print(f"  Loss years ({len(loss_years)}): {loss_years}")
