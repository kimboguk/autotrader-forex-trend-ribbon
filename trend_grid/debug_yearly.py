# -*- coding: utf-8 -*-
import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(1, os.path.join(_here, '..', 'statarb'))

import pandas as pd
from strategy import generate_signals
from backtest import load_ohlcv

df = load_ohlcv('EURUSD', 'H4')
grid = generate_signals(df)

# yearly trade count
sigs = grid[grid['signal'] != 0].copy()
sigs['year'] = sigs.index.year
yearly = sigs.groupby('year').agg(
    signals=('signal', 'count'),
    longs=('signal', lambda x: (x > 0).sum()),
    shorts=('signal', lambda x: (x < 0).sum()),
).reset_index()
print("=== H4 Yearly Signal Count ===")
print(yearly.to_string(index=False))
print(f"\nTotal: {len(sigs)} signals")

# 2001년 상세 확인
print("\n=== 2001 H4 Signals ===")
g01 = grid[(grid.index >= '2001-01-01') & (grid.index < '2002-01-01')]
s01 = g01[g01['signal'] != 0][['open','close','body_mid','grid_top','grid_bottom','is_bullish','signal','position']]
print(f'2001 bars: {len(g01)}, signals: {len(s01)}')
print(s01.to_string())

# 2001 grid width
print('\n2001 Grid width (pips):')
gw01 = (g01['grid_top'] - g01['grid_bottom']) / 0.0001
print(gw01.describe())
