# -*- coding: utf-8 -*-
"""
Trend Ribbon multi-symbol auto-trader orchestrator.

Polls MT5 for M30/H4 data, generates signals via SignalEngine,
enforces FTMO risk limits, and executes trades.
"""

import time
import logging
from datetime import datetime, timezone
from typing import Dict, Optional

from config import (
    LIVE_SYMBOLS, SYMBOLS, MA_TYPE, MA_PERIODS,
    ENTRY_TF, FILTER_TF, M30_BAR_COUNT, H4_BAR_COUNT,
    FTMO, EMERGENCY_SL_BUDGET_USD,
    LOT_SIZES, MAGIC_NUMBER, ORDER_COMMENT,
    POLL_INTERVAL_SEC, NEWS_FILTER,
)
from mt5_client import MT5Client
from signal_engine import SignalEngine
from risk_manager import FTMORiskManager
from news_filter import NewsFilter
from state_manager import StateManager

logger = logging.getLogger(__name__)

# Pip value per 1 standard lot in USD (approximate, for emergency SL)
# Forex standard lot = 100,000 units → 1 pip = lot_size × pip_size × (100,000 / pip_size) ... simplified:
# EURUSD/GBPUSD (quote=USD): 1 pip = $10/lot
# USDJPY/EURJPY (quote=JPY): 1 pip ≈ $6.5/lot (varies with USD/JPY rate, use ~7)
# XAUUSD (lot=100oz): 1 pip(=$0.10) = 100 × $0.10 = $10/lot
PIP_VALUE_PER_LOT = {
    "EURUSD": 10.0,
    "GBPUSD": 10.0,
    "USDJPY": 7.0,    # approximate — recalculated at runtime if needed
    "EURJPY": 7.0,
    "XAUUSD": 10.0,   # 100oz × $0.10 per pip
}


class TrendRibbonTrader:
    """Main trading loop for TR M30+H4 strategy."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.mt5 = MT5Client()
        self.signals = SignalEngine(LIVE_SYMBOLS, MA_TYPE, MA_PERIODS,
                                    use_kalman=False)
        self.risk = FTMORiskManager(FTMO)
        self.news = NewsFilter(NEWS_FILTER) if NEWS_FILTER.get("enabled") else None
        self.state_mgr = StateManager()

        self._save_counter = 0

        # Virtual positions for dry-run mode (symbol → direction: 1/-1)
        self._dry_positions: Dict[str, int] = {}

    # ── Main loop ───────────────────────────────────────────

    def run(self):
        """Connect and enter the polling loop. Ctrl+C to stop."""
        if not self.mt5.connect():
            logger.error("Cannot start — MT5 connection failed")
            return

        self._restore_state()
        self._log_startup()

        try:
            while True:
                try:
                    self._tick()
                except Exception as e:
                    logger.error("Loop error: %s", e, exc_info=True)
                    time.sleep(30)
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        finally:
            self._save_state()
            self.mt5.disconnect()

    # ── Per-cycle logic ─────────────────────────────────────

    def _tick(self):
        """One poll cycle."""
        # Skip weekends (Sat=5, Sun=6)
        now_utc = datetime.now(timezone.utc)
        wd = now_utc.weekday()
        if wd == 5 or (wd == 6 and now_utc.hour < 22):
            time.sleep(60)
            return

        # Ensure MT5 connection
        if not self.mt5.ensure_connected():
            time.sleep(10)
            return

        # Account info + risk check
        acct = self.mt5.get_account_info()
        if not acct:
            time.sleep(10)
            return

        self.risk.update(acct)

        # Get current positions for this strategy
        live_positions = self.mt5.get_positions_by_magic(MAGIC_NUMBER)

        if self.risk.should_close_all():
            if live_positions:
                self._close_all_positions()
            else:
                logger.warning(
                    "Risk limit exceeded but no strategy positions to close "
                    "(manual positions may be causing the drawdown). "
                    "New entries blocked."
                )
            time.sleep(POLL_INTERVAL_SEC)
            return

        # Process each symbol
        for symbol in LIVE_SYMBOLS:
            self._process_symbol(symbol, live_positions, acct)

        # Periodic state save (every ~60 seconds)
        self._save_counter += 1
        if self._save_counter >= (60 // POLL_INTERVAL_SEC):
            self._save_state()
            self._save_counter = 0

        time.sleep(POLL_INTERVAL_SEC)

    def _process_symbol(
        self,
        symbol: str,
        live_positions: Dict[str, Dict],
        acct: Dict,
    ):
        """Fetch data, check signal, execute for one symbol."""
        # Fetch M30 + H4 data
        m30 = self.mt5.get_rates(symbol, ENTRY_TF, M30_BAR_COUNT)
        h4 = self.mt5.get_rates(symbol, FILTER_TF, H4_BAR_COUNT)

        if m30 is None or h4 is None:
            return

        # Current position direction (use virtual positions in dry-run)
        if self.dry_run:
            current_dir = self._dry_positions.get(symbol, 0)
            pos = None
        else:
            pos = live_positions.get(symbol)
            current_dir = self._position_direction(pos)

        # Check signal
        action = self.signals.update(symbol, m30, h4, current_dir)
        if action is None:
            return

        # Execute
        act = action["action"]

        if act == "exit":
            self._do_exit(symbol, pos)

        elif act in ("enter_long", "enter_short"):
            if self.risk.can_enter() and self._news_allows_entry(symbol) and self._hour_allows_entry(symbol):
                direction = 1 if act == "enter_long" else -1
                self._do_enter(symbol, direction, acct)

        elif act in ("reverse_long", "reverse_short"):
            # Exit first (always allowed), then enter if permitted
            self._do_exit(symbol, pos)
            if self.risk.can_enter() and self._news_allows_entry(symbol) and self._hour_allows_entry(symbol):
                direction = 1 if act == "reverse_long" else -1
                self._do_enter(symbol, direction, acct)

    # ── Execution helpers ───────────────────────────────────

    def _do_enter(self, symbol: str, direction: int, acct: Dict):
        """Place a market order with emergency SL."""
        # Spread filter
        if not self._spread_allows_entry(symbol):
            return

        lot_size = LOT_SIZES.get(symbol, 0.1)
        sym_cfg = SYMBOLS[symbol]

        # Get current price for SL calculation
        tick = self.mt5.get_tick(symbol)
        if tick is None:
            return

        entry_price = tick.ask if direction == 1 else tick.bid

        # Emergency SL
        pip_val = PIP_VALUE_PER_LOT.get(symbol, 10.0)

        # For JPY pairs, recalculate pip value dynamically
        if sym_cfg.get("quote_ccy") == "JPY":
            usdjpy_tick = self.mt5.get_tick("USDJPY")
            if usdjpy_tick:
                pip_val = 100_000 * 0.01 / usdjpy_tick.bid  # pip in USD

        sl_price = FTMORiskManager.calc_emergency_sl(
            direction=direction,
            entry_price=entry_price,
            lot_size=lot_size,
            budget_usd=EMERGENCY_SL_BUDGET_USD,
            pip_size=sym_cfg["pip_size"],
            pip_value_per_lot=pip_val,
        )

        dir_str = "LONG" if direction == 1 else "SHORT"
        sl_pips = abs(entry_price - sl_price) / sym_cfg["pip_size"]

        if self.dry_run:
            logger.info("[DRY RUN] %s %s %.2f lots @ %.5f, SL=%.5f (%.0f pips)",
                         dir_str, symbol, lot_size, entry_price, sl_price, sl_pips)
            self._dry_positions[symbol] = direction
            return

        ticket = self.mt5.place_market_order(
            symbol=symbol,
            direction=direction,
            lot_size=lot_size,
            magic=MAGIC_NUMBER,
            comment=ORDER_COMMENT,
            sl=sl_price,
        )

        if ticket:
            logger.info("ENTRY %s %s %.2f lots @ %.5f, SL=%.5f (%.0f pips), ticket=%d",
                         dir_str, symbol, lot_size, entry_price, sl_price, sl_pips, ticket)

    def _do_exit(self, symbol: str, pos: Optional[Dict]):
        """Close position for symbol."""
        if self.dry_run:
            if symbol in self._dry_positions:
                dir_str = "LONG" if self._dry_positions[symbol] == 1 else "SHORT"
                logger.info("[DRY RUN] EXIT %s (was %s)", symbol, dir_str)
                del self._dry_positions[symbol]
            return

        if pos is None:
            return

        ok = self.mt5.close_position(pos)
        if ok:
            logger.info("EXIT %s ticket=%d, profit=%.2f",
                         symbol, pos["ticket"], pos.get("profit", 0))

    def _close_all_positions(self):
        """Emergency: close every position owned by this strategy."""
        logger.critical("CLOSING ALL POSITIONS — FTMO risk limit")
        positions = self.mt5.get_positions_by_magic(MAGIC_NUMBER)
        for symbol, pos in positions.items():
            if self.dry_run:
                logger.info("[DRY RUN] FORCE CLOSE %s ticket=%d", symbol, pos["ticket"])
            else:
                self.mt5.close_position(pos)

    # ── Helpers ─────────────────────────────────────────────

    def _news_allows_entry(self, symbol: str) -> bool:
        """Check news filter. Returns True if no filter or entry allowed."""
        if self.news is None:
            return True
        return self.news.can_enter(symbol)

    def _hour_allows_entry(self, symbol: str = None) -> bool:
        """Check if current KST hour allows entry for the given symbol."""
        from datetime import datetime, timezone, timedelta
        kst = datetime.now(timezone(timedelta(hours=9)))
        allowed_by_symbol = {
            "EURUSD": {17, 18, 20, 21, 22},
            "USDJPY": {8, 21, 22},
            "EURJPY": {21, 22},
            "XAUUSD": {2, 15, 22},
            "GBPUSD": {1, 17, 21, 22},
        }
        allowed = allowed_by_symbol.get(symbol, set())
        if not allowed:
            return True  # unknown symbol → allow all
        return kst.hour in allowed

    def _spread_allows_entry(self, symbol: str) -> bool:
        """Check if current spread is acceptable (max 1 pip = 10 points)."""
        max_spread_pts = 10  # 10 points = 1 pip for 5-digit broker
        info = self.mt5.get_symbol_info(symbol)
        if info is None:
            return False
        if info.spread > max_spread_pts:
            logger.warning("Spread too high for %s: %d pts (max %d), skip entry",
                           symbol, info.spread, max_spread_pts)
            return False
        return True

    @staticmethod
    def _position_direction(pos: Optional[Dict]) -> int:
        """Extract direction from MT5 position dict. 0 if flat."""
        if pos is None:
            return 0
        # MT5: POSITION_TYPE_BUY=0, POSITION_TYPE_SELL=1
        return 1 if pos["type"] == 0 else -1

    def _log_startup(self):
        acct = self.mt5.get_account_info()
        mode = "DRY RUN" if self.dry_run else "LIVE"
        logger.info("=" * 60)
        logger.info("  Trend Ribbon M30+H4 Auto-Trader [%s]", mode)
        logger.info("  Symbols: %s", ", ".join(LIVE_SYMBOLS))
        logger.info("  Lots: %s", LOT_SIZES)
        logger.info("  Balance: $%.2f  Equity: $%.2f",
                     acct.get("balance", 0), acct.get("equity", 0))
        logger.info("  Emergency SL budget: $%.0f/symbol", EMERGENCY_SL_BUDGET_USD)
        logger.info("  News filter: %s", "ON" if self.news else "OFF")
        logger.info("=" * 60)

    # ── State persistence ───────────────────────────────────

    def _save_state(self):
        state = {
            "signal_engine": self.signals.get_state(),
            "risk_manager": self.risk.get_state(),
        }
        if self.news:
            state["news_filter"] = self.news.get_state()
        self.state_mgr.save(state)

    def _restore_state(self):
        state = self.state_mgr.load()
        if state is None:
            return

        if "signal_engine" in state:
            self.signals.restore_state(state["signal_engine"])
        if "risk_manager" in state:
            self.risk.restore_state(state["risk_manager"])
        if self.news and "news_filter" in state:
            self.news.restore_state(state["news_filter"])

        # Re-derive H4 positions from fresh data as sanity check
        logger.info("Verifying H4 positions from fresh data...")
        for symbol in LIVE_SYMBOLS:
            h4 = self.mt5.get_rates(symbol, FILTER_TF, H4_BAR_COUNT)
            if h4 is not None and len(h4) > max(MA_PERIODS) + 2:
                h4_completed = h4.iloc[:-1]
                self.signals._update_h4_position(symbol, h4_completed)

        # Reconcile with actual MT5 positions
        live_pos = self.mt5.get_positions_by_magic(MAGIC_NUMBER)
        for symbol in LIVE_SYMBOLS:
            pos = live_pos.get(symbol)
            dir_str = "LONG" if pos and pos["type"] == 0 else "SHORT" if pos else "FLAT"
            h4_str = {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(
                self.signals.h4_position.get(symbol, 0), "?"
            )
            logger.info("  %s: MT5=%s, H4_filter=%s", symbol, dir_str, h4_str)
