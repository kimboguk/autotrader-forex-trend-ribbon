# -*- coding: utf-8 -*-
"""
HistData.com M1 데이터 파이프라인

HistData ZIP → 압축 해제 → EST→UTC 변환 → CSV 정규화 → PostgreSQL import

사용법:
    python fetch_histdata.py --symbol EURUSD --convert      # ZIP → CSV 변환
    python fetch_histdata.py --symbol EURUSD --import-db    # CSV → PostgreSQL
    python fetch_histdata.py --symbol EURUSD --all          # 전체 파이프라인

데이터 디렉토리:
    data/histdata_raw/EURUSD/   ← ZIP 파일 (수동 다운로드)
    data/histdata_csv/          ← 변환된 CSV

HistData.com 포맷:
    구분자: 세미콜론 (;)
    컬럼: YYYYMMDD HHMMSS;OPEN;HIGH;LOW;CLOSE;VOLUME
    타임존: EST (UTC-5, DST 없음)
"""

import sys
import argparse
import zipfile
from pathlib import Path

import pandas as pd

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from config import BACKTEST_CONFIG

DATA_DIR = Path(BACKTEST_CONFIG["data_dir"])
RAW_DIR = Path(r"E:\Backup\forex\historical_data")  # 심볼별 서브디렉토리
CSV_DIR = DATA_DIR / "histdata_csv"


def parse_histdata_csv(filepath: str) -> pd.DataFrame:
    """
    HistData.com Generic ASCII 포맷 파싱

    Input:  YYYYMMDD HHMMSS;OPEN;HIGH;LOW;CLOSE;VOLUME (EST)
    Output: time,open,high,low,close,tick_volume (UTC)
    """
    df = pd.read_csv(
        filepath,
        sep=';',
        header=None,
        dtype=str,
        on_bad_lines='skip',
    )

    # 빈 컬럼 제거 (trailing semicolon → 빈 마지막 컬럼)
    df = df.dropna(axis=1, how='all')

    # 첫 5~6개 컬럼만 사용 (위치 기반)
    if len(df.columns) >= 6:
        df = df.iloc[:, :6]
        df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    elif len(df.columns) == 5:
        df.columns = ['datetime', 'open', 'high', 'low', 'close']
        df['volume'] = '0'
    else:
        return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'tick_volume'])

    # 비데이터 행 제거 (저작권 행 등)
    df = df[df['datetime'].str.strip().str.match(r'^\d{8}\s+\d{6}$', na=False)]

    # datetime 파싱: "20120201 000000" → datetime
    df['time'] = pd.to_datetime(df['datetime'].str.strip(), format='%Y%m%d %H%M%S')

    # 숫자 컬럼 변환
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)

    # EST → UTC 변환 (EST = UTC-5, HistData는 DST 없음)
    df['time'] = df['time'] + pd.Timedelta(hours=5)

    # 컬럼 정리
    df.rename(columns={'volume': 'tick_volume'}, inplace=True)
    df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]

    return df.sort_values('time').reset_index(drop=True)


def extract_and_convert(symbol: str) -> str:
    """
    심볼의 histdata_raw ZIP 파일들을 추출/변환/병합하여 단일 CSV로 저장

    Returns: 출력 CSV 경로
    """
    symbol_dir = RAW_DIR / symbol
    if not symbol_dir.exists():
        print(f"Directory not found: {symbol_dir}")
        print(f"Please download HistData.com ZIPs and place them in: {symbol_dir}/")
        return None

    # ZIP 파일 검색
    zip_files = sorted(symbol_dir.glob("*.zip"))
    csv_files = sorted(symbol_dir.glob("*.csv"))

    if not zip_files and not csv_files:
        print(f"No ZIP or CSV files found in {symbol_dir}")
        return None

    all_dfs = []

    # ZIP 파일 처리
    for zf_path in zip_files:
        print(f"  Extracting: {zf_path.name}")
        with zipfile.ZipFile(zf_path, 'r') as zf:
            for name in zf.namelist():
                if name.lower().endswith('.csv') or name.lower().endswith('.txt'):
                    with zf.open(name) as f:
                        df = parse_histdata_csv(f)
                        all_dfs.append(df)
                        print(f"    {name}: {len(df):,} bars")

    # 이미 추출된 CSV 파일 처리
    for csv_path in csv_files:
        print(f"  Parsing: {csv_path.name}")
        df = parse_histdata_csv(str(csv_path))
        all_dfs.append(df)
        print(f"    {csv_path.name}: {len(df):,} bars")

    if not all_dfs:
        print("No data extracted")
        return None

    # 병합 + 정렬 + 중복 제거
    merged = pd.concat(all_dfs, ignore_index=True)
    merged = merged.sort_values('time').drop_duplicates(subset=['time']).reset_index(drop=True)

    # 저장
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CSV_DIR / f"{symbol}_M1_histdata.csv"
    merged.to_csv(output_path, index=False)

    print(f"\nMerged: {len(merged):,} bars")
    print(f"Range: {merged['time'].iloc[0]} ~ {merged['time'].iloc[-1]}")
    print(f"Saved: {output_path}")

    return str(output_path)


def import_to_db(symbol: str, csv_path: str = None):
    """변환된 HistData CSV를 PostgreSQL에 import"""
    from db import Database

    if csv_path is None:
        csv_path = str(CSV_DIR / f"{symbol}_M1_histdata.csv")

    if not Path(csv_path).exists():
        print(f"CSV not found: {csv_path}")
        print("Run --convert first.")
        return

    db = Database()

    print(f"\nLoading {csv_path}...")
    df = pd.read_csv(csv_path, parse_dates=['time'])
    print(f"CSV rows: {len(df):,}")

    # 청크 단위로 삽입 (메모리 효율)
    chunk_size = 100_000
    total_inserted = 0

    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size]
        inserted = db.bulk_insert_ohlcv(chunk, symbol, source='histdata')
        total_inserted += inserted
        print(f"  Chunk {i // chunk_size + 1}: {inserted:,} inserted "
              f"({i + len(chunk):,}/{len(df):,})")

    total = db.count_ohlcv(symbol)
    date_range = db.get_ohlcv_date_range(symbol)
    print(f"\nTotal inserted: {total_inserted:,}")
    print(f"Total in DB: {total:,}")
    print(f"Range: {date_range[0]} ~ {date_range[1]}")


def main():
    parser = argparse.ArgumentParser(description="HistData.com M1 데이터 파이프라인")
    parser.add_argument(
        "--symbol",
        required=True,
        choices=["EURUSD", "USDJPY", "EURJPY", "XAUUSD",
                 "GBPUSD", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD", "EURGBP", "EURCHF"],
        help="거래 심볼",
    )
    parser.add_argument("--convert", action="store_true", help="ZIP → CSV 변환")
    parser.add_argument("--import-db", action="store_true", help="CSV → PostgreSQL")
    parser.add_argument("--all", action="store_true", help="전체 파이프라인 (convert + import)")
    args = parser.parse_args()

    if not (args.convert or args.import_db or args.all):
        parser.print_help()
        return

    print("=" * 60)
    print(f"HistData Pipeline: {args.symbol}")
    print("=" * 60)

    csv_path = None

    if args.convert or args.all:
        print(f"\n[Step 1] Converting HistData ZIPs for {args.symbol}...")
        csv_path = extract_and_convert(args.symbol)
        if csv_path is None and args.all:
            print("Convert failed. Skipping DB import.")
            return

    if args.import_db or args.all:
        print(f"\n[Step 2] Importing to PostgreSQL...")
        import_to_db(args.symbol, csv_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
