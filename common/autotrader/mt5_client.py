# -*- coding: utf-8 -*-
"""
MT5 connection wrapper for Trend Ribbon auto-trader.

Handles: connection, data fetching, order placement, position management.
Extracted and generalised from ICT/live_trader_ob_fvg.py.
"""

import os
import logging
from typing import Optional, Dict

import pandas as pd

logger = logging.getLogger(__name__)


def _load_credentials():
    """Load MT5 credentials from .env / environment variables."""
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
    except ImportError:
        pass

    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")

    if not login or not password or not server:
        raise ValueError(
            "MT5 credentials missing. Set MT5_LOGIN, MT5_PASSWORD, MT5_SERVER "
            "in .env or environment variables."
        )
    return int(login), password, server


class MT5Client:
    """Thin wrapper around the MetaTrader5 Python API."""

    # MT5 timeframe mapping (populated after import)
    _TF_MAP: dict = {}

    def __init__(self):
        try:
            import MetaTrader5 as mt5
            self.mt5 = mt5
        except ImportError:
            raise ImportError("pip install MetaTrader5")

        self._TF_MAP = {
            "M1":  mt5.TIMEFRAME_M1,
            "M5":  mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1":  mt5.TIMEFRAME_H1,
            "H4":  mt5.TIMEFRAME_H4,
            "D1":  mt5.TIMEFRAME_D1,
        }

        self.login_id, self.password, self.server = _load_credentials()
        self.connected = False

    # ── Connection ──────────────────────────────────────────

    def connect(self) -> bool:
        if not self.mt5.initialize():
            logger.error("MT5 initialize failed: %s", self.mt5.last_error())
            return False

        if not self.mt5.login(self.login_id, password=self.password, server=self.server):
            logger.error("MT5 login failed: %s", self.mt5.last_error())
            return False

        self.connected = True
        acct = self.mt5.account_info()
        logger.info("MT5 connected — server=%s, account=%s, balance=%.2f",
                     self.server, acct.login, acct.balance)
        return True

    def disconnect(self):
        if self.connected:
            self.mt5.shutdown()
            self.connected = False
            logger.info("MT5 disconnected")

    def ensure_connected(self) -> bool:
        """Reconnect if connection dropped."""
        if self.connected and self.mt5.terminal_info() is not None:
            return True
        logger.warning("MT5 connection lost — reconnecting...")
        self.connected = False
        return self.connect()

    # ── Data ────────────────────────────────────────────────

    def get_rates(self, symbol: str, tf_str: str, count: int) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV bars.  The last bar (index -1) is the *currently forming*
        bar.  Callers should use iloc[:-1] for completed bars.
        """
        tf = self._TF_MAP.get(tf_str)
        if tf is None:
            logger.error("Unsupported timeframe: %s", tf_str)
            return None

        rates = self.mt5.copy_rates_from_pos(symbol, tf, 0, count)
        if rates is None or len(rates) == 0:
            logger.warning("No data for %s %s", symbol, tf_str)
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.sort_values("time", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    # ── Symbol info helpers ─────────────────────────────────

    def get_symbol_info(self, symbol: str):
        """Return mt5 symbol_info object (for filling mode, digits, etc.)."""
        info = self.mt5.symbol_info(symbol)
        if info is None:
            logger.error("Symbol %s not found", symbol)
        return info

    def get_tick(self, symbol: str):
        """Return latest tick (bid/ask/time)."""
        return self.mt5.symbol_info_tick(symbol)

    def get_filling_mode(self, symbol: str) -> int:
        """Determine the correct filling mode for a symbol."""
        info = self.get_symbol_info(symbol)
        if info is None:
            return self.mt5.ORDER_FILLING_IOC
        modes = info.filling_mode
        # filling_mode bitmask: bit0 (1)=FOK, bit1 (2)=IOC
        if modes & 1:
            return self.mt5.ORDER_FILLING_FOK
        if modes & 2:
            return self.mt5.ORDER_FILLING_IOC
        return self.mt5.ORDER_FILLING_RETURN

    # ── Orders ──────────────────────────────────────────────

    def place_market_order(
        self,
        symbol: str,
        direction: int,
        lot_size: float,
        magic: int,
        comment: str = "",
        sl: float = 0.0,
        tp: float = 0.0,
    ) -> Optional[int]:
        """
        Place a market order.

        Args:
            direction: 1 = BUY, -1 = SELL
            sl/tp: absolute price levels (0 = none)

        Returns:
            Order ticket on success, None on failure.
        """
        order_type = self.mt5.ORDER_TYPE_BUY if direction == 1 else self.mt5.ORDER_TYPE_SELL

        tick = self.get_tick(symbol)
        if tick is None:
            logger.error("Cannot get tick for %s", symbol)
            return None

        price = tick.ask if direction == 1 else tick.bid

        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 10,
            "magic": magic,
            "comment": comment,
            "type_filling": self.get_filling_mode(symbol),
        }

        result = self.mt5.order_send(request)
        if result is None or result.retcode != self.mt5.TRADE_RETCODE_DONE:
            retcode = getattr(result, "retcode", "?")
            msg = getattr(result, "comment", "unknown")
            logger.error("Order failed for %s: retcode=%s, %s", symbol, retcode, msg)
            return None

        logger.info("Order filled: %s %s %.2f lots @ %.5f  ticket=%d",
                     "BUY" if direction == 1 else "SELL", symbol, lot_size, price, result.order)
        return result.order

    def close_position(self, position: Dict) -> bool:
        """
        Close a position by its ticket.

        Args:
            position: dict with keys 'ticket', 'volume', 'type', 'symbol'
        """
        ticket = position["ticket"]
        volume = position["volume"]
        pos_type = position["type"]
        symbol = position["symbol"]

        tick = self.get_tick(symbol)
        if tick is None:
            logger.error("Cannot get tick for %s to close ticket %d", symbol, ticket)
            return False

        # Opposite direction to close
        if pos_type == self.mt5.POSITION_TYPE_BUY:
            price = tick.bid
            order_type = self.mt5.ORDER_TYPE_SELL
        else:
            price = tick.ask
            order_type = self.mt5.ORDER_TYPE_BUY

        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "position": ticket,
            "volume": volume,
            "type": order_type,
            "price": price,
            "deviation": 10,
            "magic": position.get("magic", 0),
            "comment": "TR_CLOSE",
            "type_filling": self.get_filling_mode(symbol),
        }

        result = self.mt5.order_send(request)
        if result is None or result.retcode != self.mt5.TRADE_RETCODE_DONE:
            retcode = getattr(result, "retcode", "?")
            msg = getattr(result, "comment", "unknown")
            logger.error("Close failed ticket=%d: retcode=%s, %s", ticket, retcode, msg)
            return False

        logger.info("Position closed: ticket=%d, %s @ %.5f", ticket, symbol, price)
        return True

    # ── Position queries ────────────────────────────────────

    def get_positions_by_magic(self, magic: int) -> Dict[str, Dict]:
        """
        Return open positions filtered by magic number.

        Returns:
            {symbol: position_dict, ...}
        """
        positions = {}
        all_pos = self.mt5.positions_get()
        if all_pos is None:
            return positions

        for pos in all_pos:
            p = pos._asdict()
            if p["magic"] == magic:
                positions[p["symbol"]] = p

        return positions

    # ── Account ─────────────────────────────────────────────

    def get_account_info(self) -> Dict:
        """Return account balance, equity, margin, free margin."""
        info = self.mt5.account_info()
        if info is None:
            return {}
        return {
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "free_margin": info.margin_free,
            "leverage": info.leverage,
            "server_time": self.mt5.symbol_info_tick(
                "EURUSD"
            ).time if self.mt5.symbol_info_tick("EURUSD") else None,
        }
