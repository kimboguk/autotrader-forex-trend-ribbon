# -*- coding: utf-8 -*-
import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(1, os.path.join(_here, '..', 'statarb'))

import pandas as pd
from strategy import compute_grid
from backtest import load_ohlcv

df = load_ohlcv('EURUSD', 'H4')
grid = compute_grid(df)

# Check first valid grid_top
first_valid = grid['grid_top'].first_valid_index()
print(f"Data starts:     {grid.index[0]}")
print(f"First grid_top:  {first_valid}")
print(f"Total bars:      {len(grid)}")
print(f"Bars before first grid: {grid.index.get_loc(first_valid)}")
print(f"VWMA-240 on H4 = 240 bars x 4h = {240*4}h = {240*4/24:.0f} days")

# Check NaN counts per MA
for p in [30, 60, 120, 240]:
    col = f'ma_{p}'
    nan_count = grid[col].isna().sum()
    first = grid[col].first_valid_index()
    print(f"  MA-{p}: {nan_count} NaN, first valid: {first}")

# Check which years have valid grid
grid['year'] = grid.index.year
valid_by_year = grid.dropna(subset=['grid_top']).groupby('year').size()
print(f"\nValid bars by year:\n{valid_by_year}")
