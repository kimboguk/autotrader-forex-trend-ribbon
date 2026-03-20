# -*- coding: utf-8 -*-
"""
Import Bitstamp BTC/USD M1 CSV into PostgreSQL ohlcv_m1 table.

Data source: https://github.com/ff137/bitstamp-btcusd-minute-data

Usage:
    python import_bitstamp_btc.py path/to/bitstampUSD.csv
    python import_bitstamp_btc.py path/to/bitstampUSD.csv --dry-run
    python import_bitstamp_btc.py path/to/bitstampUSD.csv --start 2017-01-01
"""

import sys
import argparse
from io import StringIO
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

SYMBOL_DB = "BTCUSD"
CHUNK_SIZE = 100_000  # rows per bulk insert


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


def bulk_insert(conn, df: pd.DataFrame, symbol_id: int, source: str = "bitstamp") -> int:
    """Insert M1 data using COPY + temp table."""
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
    parser = argparse.ArgumentParser(description="Import Bitstamp BTC/USD M1 CSV to DB")
    parser.add_argument("csv_path", help="Path to bitstampUSD.csv")
    parser.add_argument("--start", default=None, help="Start date filter (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date filter (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true", help="Parse only, no DB insert")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        sys.exit(1)

    print(f"Reading {csv_path}...")

    # CSV format: timestamp(unix), open, high, low, close, volume
    df = pd.read_csv(
        csv_path,
        names=["timestamp", "open", "high", "low", "close", "volume"],
        header=0,  # skip header if present
    )

    # Convert unix timestamp to datetime
    df["time"] = pd.to_datetime(df["timestamp"], unit="s")
    df["tick_volume"] = df["volume"].fillna(0).astype(int)
    df = df[["time", "open", "high", "low", "close", "tick_volume"]]
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.sort_values("time").reset_index(drop=True)

    # Date filters
    if args.start:
        df = df[df["time"] >= args.start]
    if args.end:
        df = df[df["time"] <= args.end]

    print(f"  Rows: {len(df):,}")
    print(f"  Range: {df['time'].iloc[0]} ~ {df['time'].iloc[-1]}")
    print(f"  Price: ${df['close'].iloc[0]:.2f} ~ ${df['close'].iloc[-1]:.2f}")

    if args.dry_run:
        print("Dry run — no DB insert.")
        return

    conn = get_db_connection()
    symbol_id = get_symbol_id(conn, SYMBOL_DB)

    total_inserted = 0
    n_chunks = (len(df) + CHUNK_SIZE - 1) // CHUNK_SIZE

    for i in range(n_chunks):
        chunk = df.iloc[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
        inserted = bulk_insert(conn, chunk, symbol_id)
        total_inserted += inserted
        last_time = chunk["time"].iloc[-1]
        print(f"  Chunk {i+1}/{n_chunks}: {last_time} | inserted: {total_inserted:,}", end="\r")

    print(f"\n\nDone! Total rows: {len(df):,}, inserted: {total_inserted:,}")
    conn.close()


if __name__ == "__main__":
    main()
