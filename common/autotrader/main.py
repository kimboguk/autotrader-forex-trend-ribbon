# -*- coding: utf-8 -*-
"""
Trend Ribbon M30+H4 Auto-Trader — Entry Point

Usage:
    python main.py              # Live trading
    python main.py --dry-run    # Signal logging only, no orders
"""

import sys
import argparse
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Windows console UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from config import LOG_DIR  # noqa: E402


def setup_logging(level: str = "INFO"):
    """Configure console + rotating file logging."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File handler (10MB per file, keep 5 backups)
    fh = RotatingFileHandler(
        LOG_DIR / "tr_autotrader.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)


def main():
    parser = argparse.ArgumentParser(description="Trend Ribbon MT5 Auto-Trader")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log signals without placing orders",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Console log level (default: INFO)",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)

    from trader import TrendRibbonTrader  # noqa: E402

    trader = TrendRibbonTrader(dry_run=args.dry_run)
    trader.run()


if __name__ == "__main__":
    main()
