# -*- coding: utf-8 -*-
"""
Index & Daily 데이터 파이프라인

yfinance 일봉 → PostgreSQL import (SP500, KOSPI200 등)
일봉 데이터는 ohlcv_m1 테이블에 D1 bar로 저장 (time = 날짜 00:00 UTC).
DataLoader.resample()에서 D1으로 사용 가능.

사용법:
    python fetch_index.py --symbol SP500                 # SP500 전체 기간
    python fetch_index.py --symbol KOSPI200              # KOSPI200 전체 기간
    python fetch_index.py --symbol SP500 --start 2020-01-01
"""

import sys
import os
import argparse
from io import StringIO
from datetime import datetime

import pandas as pd
import psycopg2
from dotenv import load_dotenv

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

# ── Symbol → yfinance ticker mapping ──

TICKER_MAP = {
    "SP500":    "^GSPC",      # S&P 500 Index
    "KOSPI200": "^KS200",     # KOSPI 200 Index
    "JPN225":   "^N225",      # Nikkei 225 Index
}

DEFAULT_START = "2000-01-01"


def fetch_daily(ticker: str, start: str, end: str = None) -> pd.DataFrame:
    """Fetch daily OHLCV from yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    print(f"Fetching {ticker} daily from {start}...")
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if data.empty:
        print(f"No data returned for {ticker}")
        return pd.DataFrame()

    # yfinance returns MultiIndex columns when downloading single ticker with newer versions
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    df = pd.DataFrame({
        "time": data.index.tz_localize(None) if data.index.tz else data.index,
        "open": data["Open"].values,
        "high": data["High"].values,
        "low": data["Low"].values,
        "close": data["Close"].values,
        "tick_volume": data["Volume"].values.astype(int) if "Volume" in data.columns else 0,
    })

    df = df.dropna(subset=["open"]).reset_index(drop=True)
    print(f"  {len(df):,} daily bars: {df['time'].iloc[0].date()} ~ {df['time'].iloc[-1].date()}")
    return df


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


def bulk_insert(conn, df: pd.DataFrame, symbol_id: int, source: str = "yfinance") -> int:
    """Insert daily data using COPY + temp table."""
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
    parser = argparse.ArgumentParser(description="Index 일봉 데이터 수집 (yfinance)")
    parser.add_argument("--symbol", required=True, choices=list(TICKER_MAP.keys()),
                        help="심볼 (SP500, KOSPI200)")
    parser.add_argument("--start", default=DEFAULT_START, help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="종료일 (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true", help="DB 저장 없이 테스트")
    args = parser.parse_args()

    ticker = TICKER_MAP[args.symbol]
    df = fetch_daily(ticker, args.start, args.end)

    if df.empty:
        print("No data to insert.")
        return

    if args.dry_run:
        print(f"Dry run: {len(df)} bars would be inserted.")
        return

    conn = get_db_connection()
    symbol_id = get_symbol_id(conn, args.symbol)
    inserted = bulk_insert(conn, df, symbol_id)

    # Verify
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*), MIN(time), MAX(time) FROM ohlcv_m1 WHERE symbol_id = %s", (symbol_id,))
        count, min_t, max_t = cur.fetchone()

    print(f"\nInserted: {inserted:,} bars")
    print(f"Total in DB: {count:,}")
    print(f"Range: {min_t} ~ {max_t}")
    conn.close()


if __name__ == "__main__":
    main()
