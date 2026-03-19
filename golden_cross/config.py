# -*- coding: utf-8 -*-
"""
Golden Cross Strategy - Configuration

Two MA crossover: buy on golden cross (fast > slow), sell on dead cross (fast < slow).
"""

import sys
from pathlib import Path

# Reuse statarb DataLoader
STATARB_DIR = Path(__file__).resolve().parent.parent / "statarb"
if str(STATARB_DIR) not in sys.path:
    sys.path.append(str(STATARB_DIR))

# Also add common directory for trade_engine
COMMON_DIR = Path(__file__).resolve().parent.parent / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.append(str(COMMON_DIR))

# Also add trend_grid for shared config (SYMBOLS, BACKTEST_CONFIG, RESAMPLE_RULES)
TREND_GRID_DIR = Path(__file__).resolve().parent.parent / "trend_grid"
if str(TREND_GRID_DIR) not in sys.path:
    sys.path.append(str(TREND_GRID_DIR))

# ── Strategy defaults ─────────────────────────────────────────

STRATEGY_NAME = "golden_cross"
DEFAULT_FAST_PERIOD = 60
DEFAULT_SLOW_PERIOD = 240
MA_TYPE = "ema"
