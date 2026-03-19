# -*- coding: utf-8 -*-
"""
StatArb - HistData.com M1 데이터 파이프라인

후보 7개 심볼의 HistData ZIP → DB import.
ICT/fetch_histdata.py의 파서를 재활용.

사용법:
    # 단일 심볼
    python fetch_histdata.py --symbol GBPUSD --all

    # 전체 7개 심볼 일괄
    python fetch_histdata.py --all-symbols

데이터 디렉토리:
    E:\Backup\forex\historical_data\{SYMBOL}\  ← ZIP 파일
"""

import sys
import argparse
import zipfile
from pathlib import Path

import pandas as pd
import psycopg2
from io import StringIO
from dotenv import load_dotenv
import os

load_dotenv()

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from config import SYMBOLS

# 데이터 경로
HISTDATA_DIR = Path(r"E:\Backup\forex\historical_data")
CSV_DIR = Path(__file__).parent / "data" / "histdata_csv"

# 대상 심볼 (DB에 없는 7개)
TARGET_SYMBOLS = ["GBPUSD", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD", "EURGBP", "EURCHF"]


def parse_histdata_csv(filepath) -> pd.DataFrame:
    """
    HistData.com Generic ASCII 포맷 파싱 (ICT/fetch_histdata.py와 동일)

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

    df = df.dropna(axis=1, how='all')

    if len(df.columns) >= 6:
        df = df.iloc[:, :6]
        df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    elif len(df.columns) == 5:
        df.columns = ['datetime', 'open', 'high', 'low', 'close']
        df['volume'] = '0'
    else:
        return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'tick_volume'])

    df = df[df['datetime'].str.strip().str.match(r'^\d{8}\s+\d{6}$', na=False)]

    df['time'] = pd.to_datetime(df['datetime'].str.strip(), format='%Y%m%d %H%M%S')

    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)

    # EST → UTC (EST = UTC-5)
    df['time'] = df['time'] + pd.Timedelta(hours=5)

    df.rename(columns={'volume': 'tick_volume'}, inplace=True)
    df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]

    return df.sort_values('time').reset_index(drop=True)


def extract_and_convert(symbol: str) -> str:
    """ZIP 파일들을 추출/변환/병합하여 단일 CSV로 저장"""
    symbol_dir = HISTDATA_DIR / symbol

    if not symbol_dir.exists():
        print(f"  Directory not found: {symbol_dir}")
        return None

    zip_files = sorted(symbol_dir.glob("*.zip"))
    csv_files = sorted(symbol_dir.glob("*.csv"))

    if not zip_files and not csv_files:
        print(f"  No ZIP or CSV files found in {symbol_dir}")
        return None

    all_dfs = []

    for zf_path in zip_files:
        print(f"  Extracting: {zf_path.name}", end="")
        with zipfile.ZipFile(zf_path, 'r') as zf:
            for name in zf.namelist():
                if name.lower().endswith(('.csv', '.txt')):
                    with zf.open(name) as f:
                        df = parse_histdata_csv(f)
                        all_dfs.append(df)
                        print(f" → {len(df):,} bars")

    for csv_path in csv_files:
        print(f"  Parsing: {csv_path.name}", end="")
        df = parse_histdata_csv(str(csv_path))
        all_dfs.append(df)
        print(f" → {len(df):,} bars")

    if not all_dfs:
        print("  No data extracted")
        return None

    merged = pd.concat(all_dfs, ignore_index=True)
    merged = merged.sort_values('time').drop_duplicates(subset=['time']).reset_index(drop=True)

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CSV_DIR / f"{symbol}_M1_histdata.csv"
    merged.to_csv(output_path, index=False)

    print(f"  Merged: {len(merged):,} bars "
          f"({merged['time'].iloc[0]} ~ {merged['time'].iloc[-1]})")

    return str(output_path)


class DBImporter:
    """PostgreSQL 데이터 임포터"""

    def __init__(self):
        self.host = os.getenv("DB_HOST", "localhost")
        self.port = int(os.getenv("DB_PORT", 5432))
        self.dbname = os.getenv("DB_NAME", "ict_trading")
        self.user = os.getenv("DB_USER", "postgres")
        self.password = os.getenv("DB_PASSWORD", "")

    def _connect(self):
        params = {
            "host": self.host,
            "port": self.port,
            "dbname": self.dbname,
            "user": self.user,
        }
        if self.password:
            params["password"] = self.password
        return psycopg2.connect(**params)

    def ensure_symbol(self, symbol: str) -> int:
        """심볼이 DB에 없으면 생성, 있으면 ID 반환"""
        sym_cfg = SYMBOLS.get(symbol)
        if sym_cfg is None:
            raise ValueError(f"Symbol '{symbol}' not in config.py SYMBOLS")

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM symbols WHERE name = %s", (symbol,))
                row = cur.fetchone()
                if row:
                    return row[0]

                # 신규 심볼 등록
                cur.execute(
                    """
                    INSERT INTO symbols (name, pip_size, spread_pips, commission_pips, sl_buffer)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        symbol,
                        sym_cfg["pip_size"],
                        sym_cfg["spread_pips"],
                        sym_cfg["commission_pips"],
                        sym_cfg["pip_size"],  # sl_buffer = 1 pip
                    ),
                )
                symbol_id = cur.fetchone()[0]
            conn.commit()
            print(f"  Registered new symbol: {symbol} (id={symbol_id})")
            return symbol_id

    def bulk_insert(self, df: pd.DataFrame, symbol: str, source: str = "histdata") -> int:
        """COPY 프로토콜로 대량 삽입"""
        symbol_id = self.ensure_symbol(symbol)

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TEMP TABLE _tmp_ohlcv (
                        time TIMESTAMPTZ,
                        open NUMERIC(12,6),
                        high NUMERIC(12,6),
                        low  NUMERIC(12,6),
                        close NUMERIC(12,6),
                        tick_volume INTEGER,
                        source VARCHAR(20)
                    ) ON COMMIT DROP
                    """
                )

                buf = StringIO()
                for _, row in df.iterrows():
                    buf.write(
                        f"{row['time']}\t{row['open']}\t{row['high']}\t"
                        f"{row['low']}\t{row['close']}\t"
                        f"{int(row.get('tick_volume', 0))}\t{source}\n"
                    )
                buf.seek(0)

                cur.copy_expert(
                    "COPY _tmp_ohlcv (time, open, high, low, close, tick_volume, source) "
                    "FROM STDIN WITH (FORMAT text)",
                    buf,
                )

                cur.execute(
                    f"""
                    INSERT INTO ohlcv_m1 (time, symbol_id, open, high, low, close, tick_volume, source)
                    SELECT time, {symbol_id}, open, high, low, close, tick_volume, source
                    FROM _tmp_ohlcv
                    ON CONFLICT (symbol_id, time) DO NOTHING
                    """
                )
                inserted = cur.rowcount

            conn.commit()
        return inserted

    def import_csv(self, symbol: str, csv_path: str = None):
        """CSV → PostgreSQL"""
        if csv_path is None:
            csv_path = str(CSV_DIR / f"{symbol}_M1_histdata.csv")

        if not Path(csv_path).exists():
            print(f"  CSV not found: {csv_path}")
            return

        print(f"  Loading {csv_path}...")
        df = pd.read_csv(csv_path, parse_dates=['time'])
        print(f"  CSV rows: {len(df):,}")

        chunk_size = 100_000
        total_inserted = 0

        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size]
            inserted = self.bulk_insert(chunk, symbol)
            total_inserted += inserted
            pct = (i + len(chunk)) / len(df) * 100
            print(f"    Chunk {i // chunk_size + 1}: +{inserted:,} "
                  f"({pct:.0f}%)")

        print(f"  Total inserted: {total_inserted:,}")

    def get_status(self, symbol: str):
        """심볼 데이터 현황"""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM symbols WHERE name = %s", (symbol,))
                row = cur.fetchone()
                if not row:
                    return None
                sid = row[0]
                cur.execute(
                    "SELECT COUNT(*), MIN(time), MAX(time) FROM ohlcv_m1 WHERE symbol_id = %s",
                    (sid,),
                )
                return cur.fetchone()


def process_symbol(symbol: str, convert: bool = True, import_db: bool = True):
    """단일 심볼 처리"""
    print(f"\n{'=' * 60}")
    print(f"  {symbol}")
    print(f"{'=' * 60}")

    csv_path = None

    if convert:
        print(f"\n[Convert] ZIP → CSV")
        csv_path = extract_and_convert(symbol)
        if csv_path is None and import_db:
            print("  Convert failed. Skipping import.")
            return

    if import_db:
        print(f"\n[Import] CSV → PostgreSQL")
        db = DBImporter()
        db.import_csv(symbol, csv_path)

        status = db.get_status(symbol)
        if status:
            count, start, end = status
            print(f"  DB total: {count:,} rows ({start} ~ {end})")


def main():
    parser = argparse.ArgumentParser(description="StatArb HistData 파이프라인")
    parser.add_argument("--symbol", type=str, default=None,
                        help="단일 심볼 처리")
    parser.add_argument("--convert", action="store_true", help="ZIP → CSV")
    parser.add_argument("--import-db", action="store_true", help="CSV → DB")
    parser.add_argument("--all", action="store_true", help="convert + import")
    parser.add_argument("--all-symbols", action="store_true",
                        help="7개 심볼 전체 일괄 처리 (convert + import)")
    args = parser.parse_args()

    if args.all_symbols:
        for sym in TARGET_SYMBOLS:
            process_symbol(sym, convert=True, import_db=True)
    elif args.symbol:
        do_convert = args.convert or args.all
        do_import = args.import_db or args.all
        if not do_convert and not do_import:
            parser.print_help()
            return
        process_symbol(args.symbol, convert=do_convert, import_db=do_import)
    else:
        parser.print_help()
        return

    print("\n" + "=" * 60)
    print("  Final DB Status")
    print("=" * 60)
    db = DBImporter()
    for sym in TARGET_SYMBOLS:
        status = db.get_status(sym)
        if status and status[0] > 0:
            print(f"  {sym}: {status[0]:>12,} rows ({status[1]} ~ {status[2]})")
        else:
            print(f"  {sym}: no data")

    print("\nDone!")


if __name__ == "__main__":
    main()
