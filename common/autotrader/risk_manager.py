# -*- coding: utf-8 -*-
"""
FTMO risk manager for Trend Ribbon auto-trader.

Enforces daily loss limit and max drawdown with safety margins.
Calculates per-symbol emergency SL price for order protection.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class FTMORiskManager:
    """
    Monitors account equity against FTMO rules and decides whether
    new entries are allowed or all positions should be closed.

    Daily loss is tracked from the **higher of balance and equity** at
    the start of each trading day (FTMO rule).
    """

    def __init__(self, config: Dict):
        self.account_size = config["account_size"]
        self.max_daily_loss = self.account_size * config["max_daily_loss_pct"] / 100
        self.max_total_loss = self.account_size * config["max_total_loss_pct"] / 100
        self.block_daily = self.account_size * config["block_daily_loss_pct"] / 100
        self.close_daily = self.account_size * config["close_daily_loss_pct"] / 100
        self.block_total = self.account_size * config["block_total_dd_pct"] / 100
        self.close_total = self.account_size * config["close_total_dd_pct"] / 100

        # Daily tracking
        self.daily_start_equity: Optional[float] = None
        self.current_date: Optional[str] = None

        # Status
        self.entries_blocked = False
        self.force_close = False

    def update(self, account_info: Dict) -> None:
        """
        Update risk state from account info.
        Call on every poll cycle.
        """
        equity = account_info.get("equity", 0)
        balance = account_info.get("balance", 0)

        # Daily reset
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.current_date != today:
            self.daily_start_equity = max(balance, equity)
            self.current_date = today
            self.entries_blocked = False
            self.force_close = False
            logger.info("Daily reset: start_equity=%.2f, date=%s",
                         self.daily_start_equity, today)

        if self.daily_start_equity is None:
            self.daily_start_equity = max(balance, equity)

        daily_loss = self.daily_start_equity - equity
        total_dd = self.account_size - equity

        # Check force-close thresholds
        if daily_loss >= self.close_daily or total_dd >= self.close_total:
            self.force_close = True
            self.entries_blocked = True
            logger.critical(
                "FORCE CLOSE — daily_loss=%.2f (limit=%.2f), total_dd=%.2f (limit=%.2f)",
                daily_loss, self.close_daily, total_dd, self.close_total,
            )
            return

        # Check entry-block thresholds
        if daily_loss >= self.block_daily or total_dd >= self.block_total:
            self.entries_blocked = True
            logger.warning(
                "Entries BLOCKED — daily_loss=%.2f/%.2f, total_dd=%.2f/%.2f",
                daily_loss, self.block_daily, total_dd, self.block_total,
            )
            return

        self.entries_blocked = False
        self.force_close = False

    def can_enter(self) -> bool:
        """True if new entries are allowed."""
        return not self.entries_blocked

    def should_close_all(self) -> bool:
        """True if all positions must be closed immediately."""
        return self.force_close

    # ── Emergency SL calculation ────────────────────────────

    @staticmethod
    def calc_emergency_sl(
        direction: int,
        entry_price: float,
        lot_size: float,
        budget_usd: float,
        pip_size: float,
        pip_value_per_lot: float,
    ) -> float:
        """
        Calculate emergency SL price so max loss ≤ budget_usd.

        Args:
            direction: 1 (long) or -1 (short)
            entry_price: expected fill price
            lot_size: position size in lots
            budget_usd: max loss for this position ($2,000 default)
            pip_size: e.g. 0.0001 for EURUSD
            pip_value_per_lot: USD per pip per lot (e.g. 10 for EURUSD standard lot)

        Returns:
            SL price level.
        """
        max_pips = budget_usd / (lot_size * pip_value_per_lot)
        sl_distance = max_pips * pip_size

        if direction == 1:
            return round(entry_price - sl_distance, 5)
        else:
            return round(entry_price + sl_distance, 5)

    # ── State persistence ───────────────────────────────────

    def get_state(self) -> Dict:
        return {
            "daily_start_equity": self.daily_start_equity,
            "current_date": self.current_date,
        }

    def restore_state(self, state: Dict):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        saved_date = state.get("current_date")
        if saved_date == today:
            self.daily_start_equity = state.get("daily_start_equity")
            self.current_date = saved_date
            logger.info("Risk state restored: date=%s, start_equity=%.2f",
                         self.current_date, self.daily_start_equity or 0)
        else:
            logger.info("Risk state expired (saved=%s, today=%s) — will reset on first update",
                         saved_date, today)
