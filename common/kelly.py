# -*- coding: utf-8 -*-
"""
Kelly Criterion position sizing for backtesting.

Calculates optimal bet fraction from rolling trade history,
applies fractional Kelly for risk management.

Usage:
    ks = KellyPositionSizer(kelly_fraction=0.5, lookback=100)
    ks.record_trade(pnl_usd=50.0)  # win
    ks.record_trade(pnl_usd=-20.0) # loss
    scale = ks.get_scale(equity, initial_capital)
"""

from collections import deque


class KellyPositionSizer:
    """
    Rolling Kelly Criterion position sizer.

    Tracks recent trade results and computes Kelly-optimal
    fraction of equity to allocate per trade.

    Parameters:
        kelly_fraction: α — 0.25 (Quarter), 0.5 (Half), 0.75, 1.0 (Full)
        lookback: rolling window size for parameter estimation
        min_trades: minimum trades before Kelly kicks in (use scale=1.0 before)
        max_position_pct: safety cap on position size (fraction of equity)
    """

    def __init__(
        self,
        kelly_fraction: float = 0.5,
        lookback: int = 100,
        min_trades: int = 30,
        max_position_pct: float = 0.25,
    ):
        self.kelly_fraction = kelly_fraction
        self.lookback = lookback
        self.min_trades = min_trades
        self.max_position_pct = max_position_pct
        self.trades = deque(maxlen=lookback)

    def record_trade(self, pnl_usd: float):
        """Record a completed trade result."""
        self.trades.append(pnl_usd)

    def compute_kelly_raw(self) -> float:
        """
        Compute raw Kelly fraction f* from recent trades.

        f* = (b * p - q) / b
        where b = payoff ratio, p = win rate, q = 1 - p

        Returns 0 if Kelly is negative (don't bet).
        """
        if len(self.trades) < self.min_trades:
            return 0.0

        wins = [t for t in self.trades if t > 0]
        losses = [t for t in self.trades if t <= 0]

        if not wins or not losses:
            return 0.0

        p = len(wins) / len(self.trades)
        q = 1 - p
        avg_win = sum(wins) / len(wins)
        avg_loss = sum(abs(t) for t in losses) / len(losses)

        if avg_loss <= 0:
            return 0.0

        b = avg_win / avg_loss  # payoff ratio
        f_star = (b * p - q) / b

        return max(0.0, f_star)

    def get_scale(self, equity: float, initial_capital: float) -> float:
        """
        Get position scale factor for next trade.

        Returns scale multiplier compatible with simulate_trades.
        - Before min_trades: scale = 1.0 (fixed lot, no Kelly)
        - After: scale = kelly_fraction * f* * (equity / initial_capital)
        - Capped at max_position_pct * (equity / initial_capital)

        Args:
            equity: current account equity
            initial_capital: starting capital

        Returns:
            Scale multiplier for position size
        """
        if equity <= 0 or initial_capital <= 0:
            return 0.0

        f_star = self.compute_kelly_raw()

        if f_star <= 0 or len(self.trades) < self.min_trades:
            # Not enough data or negative edge — use base size
            return 1.0

        f_actual = self.kelly_fraction * f_star

        # Cap at max position
        f_actual = min(f_actual, self.max_position_pct)

        # Scale relative to initial capital (like compound mode)
        scale = f_actual * (equity / initial_capital)

        # Floor at base size
        return max(scale, 0.01)

    def get_diagnostics(self) -> dict:
        """Return current Kelly diagnostics."""
        wins = [t for t in self.trades if t > 0]
        losses = [t for t in self.trades if t <= 0]
        n = len(self.trades)

        return {
            "num_trades": n,
            "win_rate": len(wins) / n if n > 0 else 0,
            "payoff_ratio": (sum(wins) / len(wins)) / (sum(abs(t) for t in losses) / len(losses))
                if wins and losses else 0,
            "kelly_raw": self.compute_kelly_raw(),
            "kelly_applied": self.kelly_fraction * self.compute_kelly_raw(),
            "kelly_fraction": self.kelly_fraction,
        }

    def reset(self):
        """Reset trade history."""
        self.trades.clear()
