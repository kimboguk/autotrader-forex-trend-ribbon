# -*- coding: utf-8 -*-
import sys, os

# trend_grid first, then statarb (for DataLoader only)
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(1, os.path.join(_here, '..', 'statarb'))

import pandas as pd
from strategy import generate_signals
from backtest import load_ohlcv

df = load_ohlcv('EURUSD', 'H4')
grid = generate_signals(df)

# 2025 signals
g25 = grid[grid.index >= '2025-01-01']
sigs = g25[g25['signal'] != 0][['open','close','body_mid','grid_top','grid_bottom','is_bullish','signal','position']]
print(f'2025 H4 bars: {len(g25)}, signals: {len(sigs)}')
print(sigs.to_string())
print()

# Grid width stats
print('Grid width stats (pips):')
gw = (g25['grid_top'] - g25['grid_bottom']) / 0.0001
print(gw.describe())
print()

# Sample bars around grid_top to check near-misses
print('\n--- Sample: body_mid vs grid_top proximity (2025, within 20 pips) ---')
g25c = g25.copy()
g25c['dist_to_gt'] = (g25c['body_mid'] - g25c['grid_top']) / 0.0001
near = g25c[g25c['dist_to_gt'].abs() < 20][['open','close','body_mid','grid_top','grid_bottom','is_bullish','dist_to_gt','signal']]
print(f'Bars within 20 pips of grid_top: {len(near)}')
print(near.head(30).to_string())
