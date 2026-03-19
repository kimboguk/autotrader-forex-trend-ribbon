# -*- coding: utf-8 -*-
"""
Binance M1 데이터 파이프라인 (BTC/USDT)

Binance REST API → M1 OHLCV → PostgreSQL import

사용법:
    python fetch_binance.py                    # 전체 기간 수집 (2017-08~현재)
    python fetch_binance.py --start 2024-01-01 # 특정 날짜부터
    python fetch_binance.py --dry-run          # DB 저장 없이 테스트
"""

import sys
import time
import argparse
from datetime import datetime, timezone
from io import StringIO

import requests
import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

# ── Config ──

BINANCE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL_BINANCE = "BTCUSDT"
SYMBOL_DB = "BTCUSD"
INTERVAL = "1m"
LIMIT = 1000  # max per request
REQUEST_DELAY = 0.1  # seconds between requests (stay under rate limit)

# BTC on Binance starts 2017-08-17
DEFAULT_START = "2017-08-17"


def fetch_klines(start_ms: int, end_ms: int = None) -> list:
    """Fetch up to 1000 M1 klines from Binance."""
    params = {
        "symbol": SYMBOL_BINANCE,
        "interval": INTERVAL,
        "startTime": start_ms,
        "limit": LIMIT,
    }
    if end_ms:
        params["endTime"] = end_ms

    resp = requests.get(BINANCE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def klines_to_df(klines: list) -> pd.DataFrame:
    """Convert Binance klines to DataFrame matching ohlcv_m1 schema."""
    rows = []
    for k in klines:
        rows.append({
            "time": pd.Timestamp(k[0], unit="ms", tz="UTC").tz_localize(None),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "tick_volume": int(float(k[5])),  # base asset volume
        })
    return pd.DataFrame(rows)


def get_db_connection():
    params = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", 5432)),
        "dbname": os.getenv("DB_NAME", "ict_trading"),
        "user": os.getenv("DB_USER", "postgres"),
    }
    pw = os.getenv("DB_PASSWORD", "")
    if pw:
        params["password"] = pw
    return psycopg2.connect(**params)


def get_symbol_id(conn, symbol: str) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM symbols WHERE name = %s", (symbol,))
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"Symbol '{symbol}' not in DB. Run migration first.")
        return row[0]


def get_last_timestamp(conn, symbol_id: int) -> datetime | None:
    """Get the latest timestamp for this symbol in DB."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT MAX(time) FROM ohlcv_m1 WHERE symbol_id = %s",
            (symbol_id,),
        )
        row = cur.fetchone()
        return row[0] if row and row[0] else None


def bulk_insert(conn, df: pd.DataFrame, symbol_id: int, source: str = "binance") -> int:
    """Insert M1 data using COPY + temp table (same pattern as fetch_histdata.py)."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TEMP TABLE _tmp_ohlcv (
                time TIMESTAMPTZ,
                open NUMERIC(16,6),
                high NUMERIC(16,6),
                low  NUMERIC(16,6),
                close NUMERIC(16,6),
                tick_volume BIGINT,
                source VARCHAR(20)
            ) ON COMMIT DROP
        """)

        buf = StringIO()
        for _, row in df.iterrows():
            buf.write(
                f"{row['time']}\t{row['open']}\t{row['high']}\t"
                f"{row['low']}\t{row['close']}\t"
                f"{int(row['tick_volume'])}\t{source}\n"
            )
        buf.seek(0)

        cur.copy_expert(
            "COPY _tmp_ohlcv (time, open, high, low, close, tick_volume, source) "
            "FROM STDIN WITH (FORMAT text)",
            buf,
        )

        cur.execute(f"""
            INSERT INTO ohlcv_m1 (time, symbol_id, open, high, low, close, tick_volume, source)
            SELECT time, {symbol_id}, open, high, low, close, tick_volume, source
            FROM _tmp_ohlcv
            ON CONFLICT (symbol_id, time) DO NOTHING
        """)
        inserted = cur.rowcount

    conn.commit()
    return inserted


def main():
    parser = argparse.ArgumentParser(description="Binance BTC M1 데이터 수집")
    parser.add_argument("--start", default=None, help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="종료일 (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true", help="DB 저장 없이 테스트")
    args = parser.parse_args()

    conn = get_db_connection()
    symbol_id = get_symbol_id(conn, SYMBOL_DB)

    # Determine start time
    if args.start:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    else:
        # Resume from last DB timestamp, or default start
        last_ts = get_last_timestamp(conn, symbol_id)
        if last_ts:
            start_dt = last_ts.replace(tzinfo=None) + pd.Timedelta(minutes=1)
            print(f"Resuming from {start_dt}")
        else:
            start_dt = datetime.strptime(DEFAULT_START, "%Y-%m-%d")

    end_dt = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.now(timezone.utc).replace(tzinfo=None)

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    print(f"Fetching {SYMBOL_BINANCE} M1: {start_dt.date()} ~ {end_dt.date()}")

    total_bars = 0
    total_inserted = 0
    current_ms = start_ms

    while current_ms < end_ms:
        try:
            klines = fetch_klines(current_ms, end_ms)
        except Exception as e:
            print(f"  Error: {e}, retrying in 5s...")
            time.sleep(5)
            continue

        if not klines:
            break

        df = klines_to_df(klines)
        total_bars += len(df)

        if not args.dry_run:
            inserted = bulk_insert(conn, df, symbol_id)
            total_inserted += inserted

        # Move cursor to after last bar
        last_bar_ms = klines[-1][0]
        current_ms = last_bar_ms + 60000  # +1 minute

        # Progress
        last_date = pd.Timestamp(last_bar_ms, unit="ms").strftime("%Y-%m-%d %H:%M")
        print(f"  {last_date} | bars: {total_bars:,} | inserted: {total_inserted:,}", end="\r")

        time.sleep(REQUEST_DELAY)

    print(f"\n\nDone! Total bars fetched: {total_bars:,}, inserted: {total_inserted:,}")
    conn.close()


if __name__ == "__main__":
    main()
