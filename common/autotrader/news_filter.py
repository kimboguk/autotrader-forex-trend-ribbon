# -*- coding: utf-8 -*-
"""
Economic news filter for Trend Ribbon auto-trader.

Fetches high-impact economic events from JBlanked Forex Factory API
and blocks new trade entries within a configurable window around
each event. Exits are never blocked.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set

import requests

logger = logging.getLogger(__name__)

# Currency → affected symbols mapping
CURRENCY_SYMBOL_MAP = {
    "USD": ["EURUSD", "USDJPY", "GBPUSD", "XAUUSD"],
    "EUR": ["EURUSD", "EURJPY"],
    "JPY": ["USDJPY", "EURJPY"],
    "GBP": ["GBPUSD"],
    "XAU": ["XAUUSD"],
    "CHF": [],
    "AUD": [],
    "NZD": [],
    "CAD": [],
}

# Forex Factory calendar via faireconomy.media (free, no auth)
API_THIS_WEEK = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
API_NEXT_WEEK = "https://nfs.faireconomy.media/ff_calendar_nextweek.json"


def _get_currencies_for_symbol(symbol: str) -> Set[str]:
    """Return the set of currencies in a forex pair."""
    # EURUSD → {EUR, USD}, XAUUSD → {XAU, USD}
    currencies = set()
    for ccy, syms in CURRENCY_SYMBOL_MAP.items():
        if symbol in syms:
            currencies.add(ccy)
    return currencies


class NewsFilter:
    """
    Blocks new entries during high-impact economic news events.

    Fetches the economic calendar periodically and checks whether
    the current time falls within the blackout window of any
    high-impact event affecting the given symbol.
    """

    def __init__(self, config: Dict):
        self._enabled = config.get("enabled", True)
        self._before = timedelta(minutes=config.get("before_minutes", 2))
        self._after = timedelta(minutes=config.get("after_minutes", 2))
        self._refresh_interval = timedelta(hours=config.get("refresh_interval_hours", 4))
        self._impact_levels = set(config.get("impact_levels", ["high"]))

        self._calendar: List[Dict] = []  # [{time, currency, title, impact}]
        self._last_fetch: Optional[datetime] = None

    # ── Public API ──────────────────────────────────────────

    def can_enter(self, symbol: str) -> bool:
        """
        Check if a new entry is allowed for *symbol* right now.
        Always returns True if the filter is disabled.
        """
        if not self._enabled:
            return True

        self._maybe_refresh()

        now = datetime.now(timezone.utc)
        affected_ccys = _get_currencies_for_symbol(symbol)

        for event in self._calendar:
            if event["currency"] not in affected_ccys:
                continue

            window_start = event["time"] - self._before
            window_end = event["time"] + self._after

            if window_start <= now <= window_end:
                logger.warning(
                    "NEWS BLOCK: %s entry blocked — %s %s at %s",
                    symbol, event["currency"], event["title"],
                    event["time"].strftime("%H:%M UTC"),
                )
                return False

        return True

    # ── Calendar fetch ──────────────────────────────────────

    def _maybe_refresh(self):
        """Fetch calendar if not yet loaded or cache expired."""
        now = datetime.now(timezone.utc)
        if self._last_fetch and (now - self._last_fetch) < self._refresh_interval:
            return
        self._fetch_calendar()

    def _fetch_calendar(self):
        """Fetch this week's + next week's high-impact events from Forex Factory."""
        all_items = []
        for url in [API_THIS_WEEK, API_NEXT_WEEK]:
            try:
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                all_items.extend(resp.json())
            except Exception as e:
                logger.warning("News calendar fetch failed (%s): %s", url.split("/")[-1], e)

        if not all_items:
            logger.warning("No calendar data fetched — keeping cached data")
            return  # fail-open

        events = []
        for item in all_items:
            impact = (item.get("impact") or "").lower()
            if impact not in self._impact_levels:
                continue

            currency = (item.get("country") or "").upper()
            title = item.get("title") or ""
            date_str = item.get("date") or ""

            event_time = self._parse_event_time(date_str)
            if event_time is None:
                continue

            events.append({
                "time": event_time,
                "currency": currency,
                "title": title,
                "impact": impact,
            })

        self._calendar = events
        self._last_fetch = datetime.now(timezone.utc)

        now = datetime.now(timezone.utc)
        upcoming = [e for e in events if e["time"] >= now]
        logger.info(
            "News calendar refreshed: %d high-impact events (%d upcoming)",
            len(events), len(upcoming),
        )
        for ev in upcoming[:10]:
            logger.info(
                "  %s %s: %s",
                ev["time"].strftime("%Y-%m-%d %H:%M UTC"),
                ev["currency"],
                ev["title"],
            )

    @staticmethod
    def _parse_event_time(date_str: str) -> Optional[datetime]:
        """Parse event date/time string into UTC datetime."""
        if not date_str:
            return None
        try:
            # ISO 8601 with timezone offset (e.g. "2026-03-15T17:30:00-04:00")
            dt = datetime.fromisoformat(date_str)
            # Convert to UTC
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc)
            else:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            return None

    # ── State persistence ───────────────────────────────────

    def get_state(self) -> Dict:
        return {
            "calendar": [
                {
                    "time": ev["time"].isoformat(),
                    "currency": ev["currency"],
                    "title": ev["title"],
                    "impact": ev["impact"],
                }
                for ev in self._calendar
            ],
            "last_fetch": self._last_fetch.isoformat() if self._last_fetch else None,
        }

    def restore_state(self, state: Dict):
        if state.get("last_fetch"):
            self._last_fetch = datetime.fromisoformat(state["last_fetch"])
        if state.get("calendar"):
            self._calendar = []
            for ev in state["calendar"]:
                try:
                    self._calendar.append({
                        "time": datetime.fromisoformat(ev["time"]),
                        "currency": ev["currency"],
                        "title": ev["title"],
                        "impact": ev["impact"],
                    })
                except Exception:
                    continue
            logger.info("News filter state restored: %d cached events", len(self._calendar))
