# -*- coding: utf-8 -*-
import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(1, os.path.join(_here, '..', 'statarb'))

import pandas as pd
from backtest import load_ohlcv

df = load_ohlcv('EURUSD', 'H4')

# tick_volume 현황
print(f"Total H4 bars: {len(df)}")
print(f"tick_volume NaN: {df['tick_volume'].isna().sum()}")
print(f"tick_volume == 0: {(df['tick_volume'] == 0).sum()}")
print(f"tick_volume > 0: {(df['tick_volume'] > 0).sum()}")
print(f"\nFirst 5 rows:")
print(df[['open','high','low','close','tick_volume']].head())
print(f"\nFirst row with tick_volume > 0:")
valid = df[df['tick_volume'] > 0]
print(f"  {valid.index[0]}: volume={valid['tick_volume'].iloc[0]}")
print(f"\n2001 tick_volume stats:")
d01 = df[(df.index >= '2001-01-01') & (df.index < '2002-01-01')]
print(d01['tick_volume'].describe())
print(f"\n2025 tick_volume stats:")
d25 = df[df.index >= '2025-01-01']
print(d25['tick_volume'].describe())
