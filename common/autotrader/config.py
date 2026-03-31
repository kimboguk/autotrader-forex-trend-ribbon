# -*- coding: utf-8 -*-
"""
Trend Ribbon MT5 Auto-Trader — Configuration

TR M30 + H4 filter, 5 symbols, FTMO $200K account.
"""

import sys
import importlib.util
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────
STRATEGY_ROOT = Path(__file__).resolve().parent.parent.parent
TREND_GRID_DIR = STRATEGY_ROOT / "trend_grid"

for d in [str(STRATEGY_ROOT / "common"), str(TREND_GRID_DIR)]:
    if d not in sys.path:
        sys.path.insert(0, d)

# ── Symbols ─────────────────────────────────────────────────
# Explicitly load trend_grid/config.py to avoid circular import
_spec = importlib.util.spec_from_file_location("trend_grid_config", TREND_GRID_DIR / "config.py")
_tg_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tg_config)
_ALL_SYMBOLS = _tg_config.SYMBOLS

LIVE_SYMBOLS = ["EURUSD", "USDJPY", "EURJPY", "GBPUSD"]  # XAUUSD excluded (spread too high)
SYMBOLS = {s: _ALL_SYMBOLS[s] for s in LIVE_SYMBOLS}

# ── Strategy parameters ────────────────────────────────────
MA_TYPE = "ema"
MA_PERIODS = [30, 60, 120, 240]
VWMA_PERIODS = MA_PERIODS  # alias for trend_grid/strategy.py compatibility

ENTRY_TF = "M30"
FILTER_TF = "H4"

# Bars to fetch (EMA-240 warmup needs 240+ bars)
M30_BAR_COUNT = 500   # ~10 trading days
H4_BAR_COUNT = 500    # ~83 trading days

# ── FTMO account parameters ($200K) ────────────────────────
FTMO = {
    "account_size": 200_000,
    "max_daily_loss_pct": 5.0,      # $10,000
    "max_total_loss_pct": 10.0,     # $20,000
    # Safety margins for new-entry blocking / force-close
    "block_daily_loss_pct": 4.0,    # $8,000 → stop new entries
    "close_daily_loss_pct": 4.5,    # $9,000 → close all positions
    "block_total_dd_pct": 8.0,      # $16,000 → stop new entries
    "close_total_dd_pct": 9.0,      # $18,000 → close all positions
}

# Per-symbol emergency SL budget = daily limit / number of symbols
EMERGENCY_SL_BUDGET_USD = (
    FTMO["account_size"] * FTMO["max_daily_loss_pct"] / 100 / len(LIVE_SYMBOLS)
)  # $2,000 per symbol

# ── Lot sizing ──────────────────────────────────────────────
LOT_SIZES = {
    "EURUSD": 1.0,
    "USDJPY": 1.0,
    "EURJPY": 1.0,
    "XAUUSD": 1.0,
    "GBPUSD": 1.0,
}

# ── Execution ───────────────────────────────────────────────
MAGIC_NUMBER = 20260319
ORDER_COMMENT = "TR_M30H4"
SLIPPAGE_POINTS = 10
POLL_INTERVAL_SEC = 5

# ── News Filter ─────────────────────────────────────────────
NEWS_FILTER = {
    "enabled": True,
    "before_minutes": 2,
    "after_minutes": 2,
    "refresh_interval_hours": 4,
    "impact_levels": ["high"],
}

# ── Logging ─────────────────────────────────────────────────
LOG_DIR = Path(__file__).resolve().parent / "logs"
