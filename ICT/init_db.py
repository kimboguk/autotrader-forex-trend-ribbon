# -*- coding: utf-8 -*-
"""
ICT Trading Strategy - 데이터베이스 초기화 스크립트

사용법:
    python init_db.py              # DB 생성 + 스키마 + 심볼 시드
    python init_db.py --import-csv # 기존 MT5 CSV도 DB에 import
"""

import sys
import argparse
from pathlib import Path

import pandas as pd

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from db import Database
from config import SYMBOLS, BACKTEST_CONFIG


def init_database():
    """데이터베이스 생성 + 스키마 초기화 + 심볼 시드"""
    db = Database()

    print("=" * 60)
    print("ICT Trading - Database Initialization")
    print("=" * 60)

    # 1. DB 생성
    print("\n[1/3] Creating database...")
    db.create_database()

    # 2. 스키마 초기화
    print("\n[2/3] Initializing schema...")
    db.init_schema()

    # 3. 심볼 시드
    print("\n[3/3] Seeding symbols...")
    db.seed_symbols()

    # 검증
    print("\n" + "-" * 60)
    print("Verification:")
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT name, pip_size, spread_pips FROM symbols ORDER BY id")
            for row in cur.fetchall():
                print(f"  {row[0]}: pip_size={row[1]}, spread={row[2]}p")

            for table in ["ohlcv_m1", "trades", "backtest_sessions"]:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                count = cur.fetchone()[0]
                print(f"  {table}: {count} rows")

    print("\nDatabase initialization complete!")
    print("=" * 60)


def import_existing_csv():
    """기존 MT5 CSV 파일을 DB에 import"""
    db = Database()
    data_dir = Path(BACKTEST_CONFIG["data_dir"])

    print("\n" + "=" * 60)
    print("Importing existing MT5 CSV data to PostgreSQL...")
    print("=" * 60)

    for symbol in SYMBOLS:
        csv_path = data_dir / f"{symbol}_M1_data.csv"
        if not csv_path.exists():
            print(f"\n[SKIP] {csv_path} not found")
            continue

        print(f"\n[{symbol}] Loading {csv_path}...")
        df = pd.read_csv(csv_path, parse_dates=["time"])
        print(f"  CSV rows: {len(df):,}")

        inserted = db.bulk_insert_ohlcv(df, symbol, source="mt5")
        total = db.count_ohlcv(symbol)
        date_range = db.get_ohlcv_date_range(symbol)

        print(f"  Inserted: {inserted:,} rows")
        print(f"  Total in DB: {total:,} rows")
        print(f"  Range: {date_range[0]} ~ {date_range[1]}")

    print("\nCSV import complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="ICT Trading DB 초기화")
    parser.add_argument(
        "--import-csv",
        action="store_true",
        help="기존 MT5 CSV 데이터를 DB에 import",
    )
    args = parser.parse_args()

    init_database()

    if args.import_csv:
        import_existing_csv()


if __name__ == "__main__":
    main()
