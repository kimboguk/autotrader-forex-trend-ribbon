# -*- coding: utf-8 -*-
"""
State persistence for Trend Ribbon auto-trader.

Saves/restores bot state to a JSON file so the system survives restarts.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

STATE_FILE = Path(__file__).resolve().parent / "tr_live_state.json"


class StateManager:
    """Load / save bot state to a JSON file."""

    def __init__(self, path: Path = STATE_FILE):
        self.path = path

    def save(self, state: Dict) -> None:
        """Atomically write state to disk."""
        tmp = self.path.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, default=str)
            tmp.replace(self.path)
            logger.debug("State saved to %s", self.path)
        except Exception as e:
            logger.error("Failed to save state: %s", e)
            if tmp.exists():
                tmp.unlink()

    def load(self) -> Optional[Dict]:
        """Load state from disk.  Returns None if missing or corrupt."""
        if not self.path.exists():
            logger.info("No state file found — starting fresh")
            return None
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                state = json.load(f)
            logger.info("State loaded from %s", self.path)
            return state
        except Exception as e:
            logger.error("Failed to load state: %s", e)
            return None
