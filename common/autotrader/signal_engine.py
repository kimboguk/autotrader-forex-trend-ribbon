# -*- coding: utf-8 -*-
"""
Signal engine for Trend Ribbon auto-trader.

Generates M30 entry/exit signals filtered by H4 directional bias.
Uses compute_grid() from trend_grid/strategy.py for grid calculation.
Signal conditions are evaluated against the *actual MT5 position*,
not the strategy's internal position tracker, to avoid state drift.
"""

import logging
from typing import Optional, Dict, List
from datetime import datetime

import numpy as np
import pandas as pd

# trend_grid/strategy.py is importable because config.py adds TREND_GRID_DIR to sys.path
# and our config.py exports VWMA_PERIODS / MA_TYPE aliases for compatibility.
from strategy import compute_grid, generate_signals

logger = logging.getLogger(__name__)


class SignalEngine:
    """
    Per-symbol signal generation with H4 filter.

    On each poll cycle, call ``update()`` with fresh M30/H4 data and the
    current MT5 position direction.  Returns an action dict when a new M30
    bar completes and a tradable signal exists, or None otherwise.
    """

    def __init__(self, symbols: List[str], ma_type: str, periods: List[int],
                 use_kalman: bool = False, kalman_qr_ratio: float = 0.1):
        self.symbols = symbols
        self.ma_type = ma_type
        self.periods = periods
        self.use_kalman = use_kalman
        self.kalman_qr_ratio = kalman_qr_ratio

        # Last-processed bar timestamps (to detect new bar completion)
        self.last_m30_bar_time: Dict[str, datetime] = {}
        self.last_h4_bar_time: Dict[str, datetime] = {}

        # H4 directional filter: 1 = long bias, -1 = short bias, 0 = flat
        self.h4_position: Dict[str, int] = {s: 0 for s in symbols}

    # ── Public API ──────────────────────────────────────────

    def update(
        self,
        symbol: str,
        m30_raw: pd.DataFrame,
        h4_raw: pd.DataFrame,
        current_position: int,
    ) -> Optional[Dict]:
        """
        Process new data for *symbol*.

        Args:
            m30_raw: DataFrame from MT5 including the currently-forming bar.
            h4_raw:  DataFrame from MT5 including the currently-forming bar.
            current_position: 1 (long), -1 (short), 0 (flat) — from MT5.

        Returns:
            Action dict or None.
            Action dict keys:
                symbol, action ("enter_long", "enter_short", "exit",
                "reverse_long", "reverse_short"), bar_time
        """
        # Strip the currently-forming bar → use only completed bars
        m30 = m30_raw.iloc[:-1].copy()
        h4 = h4_raw.iloc[:-1].copy()

        if len(m30) < max(self.periods) + 2 or len(h4) < max(self.periods) + 2:
            return None

        # -- Update H4 filter on new H4 bar -----------------------
        h4_bar_time = h4.iloc[-1]["time"]
        if symbol not in self.last_h4_bar_time or self.last_h4_bar_time[symbol] != h4_bar_time:
            self._update_h4_position(symbol, h4)
            self.last_h4_bar_time[symbol] = h4_bar_time

        # -- Check for new M30 bar ---------------------------------
        m30_bar_time = m30.iloc[-1]["time"]
        if symbol in self.last_m30_bar_time and self.last_m30_bar_time[symbol] == m30_bar_time:
            return None  # no new bar yet

        self.last_m30_bar_time[symbol] = m30_bar_time

        # -- Compute M30 grid and check signal ---------------------
        grid = compute_grid(m30, self.ma_type, self.periods,
                            use_kalman=self.use_kalman, kalman_qr_ratio=self.kalman_qr_ratio)

        signal = self._check_signal(grid, current_position)

        if signal == 0:
            return None

        action = self._signal_to_action(signal, current_position)

        # Apply H4 filter to entries (exits are always allowed)
        if action in ("enter_long", "reverse_long") and self.h4_position[symbol] != 1:
            logger.debug("%s %s blocked by H4 filter (H4 pos=%d)",
                         symbol, action, self.h4_position[symbol])
            # If this was a reverse, we still exit the current position
            if action == "reverse_long":
                action = "exit"
            else:
                return None

        if action in ("enter_short", "reverse_short") and self.h4_position[symbol] != -1:
            logger.debug("%s %s blocked by H4 filter (H4 pos=%d)",
                         symbol, action, self.h4_position[symbol])
            if action == "reverse_short":
                action = "exit"
            else:
                return None

        logger.info("SIGNAL %s: %s  (H4=%+d, M30 bar=%s)",
                     symbol, action, self.h4_position[symbol],
                     m30_bar_time.strftime("%Y-%m-%d %H:%M"))
        return {"symbol": symbol, "action": action, "bar_time": m30_bar_time}

    # ── H4 filter ───────────────────────────────────────────

    def _update_h4_position(self, symbol: str, h4: pd.DataFrame):
        """
        Run generate_signals on H4 data and store the latest position.
        """
        h4_grid = generate_signals(h4, self.ma_type, self.periods)
        last_pos = int(h4_grid["position"].iloc[-1])
        if last_pos != self.h4_position.get(symbol, 0):
            logger.info("H4 filter %s: %+d → %+d",
                         symbol, self.h4_position.get(symbol, 0), last_pos)
        self.h4_position[symbol] = last_pos

    # ── M30 signal logic ────────────────────────────────────

    def _check_signal(self, grid: pd.DataFrame, current_position: int) -> int:
        """
        Evaluate entry/exit conditions on the last completed M30 bar,
        given the actual MT5 position.

        Returns:
            0  = no signal
            1  = long entry
           -1  = short entry
            10 = long exit (no entry)
           -10 = short exit (no entry)
            2  = reverse to long (short exit + long entry)
           -2  = reverse to short (long exit + short entry)
        """
        if len(grid) < 2:
            return 0

        curr = grid.iloc[-1]
        prev = grid.iloc[-2]

        if np.isnan(curr["grid_top"]) or np.isnan(prev["grid_top"]):
            return 0

        # Entry conditions
        long_entry = (
            curr["is_bullish"]
            and curr["body_mid"] > curr["grid_top"]
            and prev["body_mid"] <= prev["grid_top"]
        )
        short_entry = (
            not curr["is_bullish"]
            and curr["body_mid"] < curr["grid_bottom"]
            and prev["body_mid"] >= prev["grid_bottom"]
        )

        # Exit conditions
        long_exit = not curr["is_bullish"] and curr["body_mid"] < curr["grid_top"]
        short_exit = curr["is_bullish"] and curr["body_mid"] > curr["grid_bottom"]

        if current_position == 0:
            if long_entry:
                return 1
            if short_entry:
                return -1

        elif current_position == 1:
            if long_exit:
                if short_entry:
                    return -2   # reverse to short
                return -10      # exit long only

        elif current_position == -1:
            if short_exit:
                if long_entry:
                    return 2    # reverse to long
                return 10       # exit short only

        return 0

    def _signal_to_action(self, signal: int, current_position: int) -> str:
        """Convert numeric signal to action string."""
        return {
            1:   "enter_long",
            -1:  "enter_short",
            10:  "exit",
            -10: "exit",
            2:   "reverse_long",
            -2:  "reverse_short",
        }.get(signal, "none")

    # ── State serialisation ─────────────────────────────────

    def get_state(self) -> Dict:
        """Export state for persistence."""
        return {
            "h4_position": dict(self.h4_position),
            "last_m30_bar_time": {
                s: t.isoformat() for s, t in self.last_m30_bar_time.items()
            },
            "last_h4_bar_time": {
                s: t.isoformat() for s, t in self.last_h4_bar_time.items()
            },
        }

    def restore_state(self, state: Dict):
        """Restore state from persistence."""
        if "h4_position" in state:
            self.h4_position.update(state["h4_position"])
        for key, store in [
            ("last_m30_bar_time", self.last_m30_bar_time),
            ("last_h4_bar_time", self.last_h4_bar_time),
        ]:
            if key in state:
                for s, t in state[key].items():
                    store[s] = pd.Timestamp(t)
