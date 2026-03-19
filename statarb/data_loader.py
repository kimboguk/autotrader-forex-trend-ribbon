# -*- coding: utf-8 -*-
"""
StatArb - 데이터 로딩 + 리샘플링 모듈

PostgreSQL에서 M1 OHLCV 데이터를 로드하고,
M15/H1/H4/D1 타임프레임으로 리샘플링.
복수 심볼을 시간 기준으로 정렬하여 페어 분석에 적합한 형태로 반환.
"""

import os
import sys
from datetime import datetime
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


# ── 타임프레임 리샘플 규칙 ─────────────────────────────────

RESAMPLE_RULES = {
    "M1":  "1min",
    "M5":  "5min",
    "M15": "15min",
    "H1":  "1h",
    "H4":  "4h",
    "D1":  "1D",
}


class DataLoader:
    """PostgreSQL OHLCV 데이터 로더"""

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

    def _get_symbol_id(self, symbol: str) -> int:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM symbols WHERE name = %s", (symbol,))
                row = cur.fetchone()
                if row is None:
                    raise ValueError(f"Symbol '{symbol}' not found in DB. "
                                     f"Run data collection first.")
                return row[0]

    def list_symbols(self) -> List[str]:
        """DB에 등록된 심볼 목록 반환"""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT name FROM symbols ORDER BY name")
                return [row[0] for row in cur.fetchall()]

    def get_date_range(self, symbol: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """심볼의 데이터 시작/종료 시간"""
        symbol_id = self._get_symbol_id(symbol)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT MIN(time), MAX(time) FROM ohlcv_m1 WHERE symbol_id = %s",
                    (symbol_id,),
                )
                return cur.fetchone()

    def count_rows(self, symbol: str) -> int:
        symbol_id = self._get_symbol_id(symbol)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM ohlcv_m1 WHERE symbol_id = %s",
                    (symbol_id,),
                )
                return cur.fetchone()[0]

    # ── M1 데이터 로드 ──

    def load_m1(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        M1 OHLCV 데이터를 DB에서 로드.

        Args:
            symbol: 심볼명 (EURUSD, GBPUSD, ...)
            start: 시작일 (YYYY-MM-DD)
            end: 종료일 (YYYY-MM-DD)

        Returns:
            DataFrame(time, open, high, low, close, tick_volume)
            time은 DatetimeIndex로 설정됨
        """
        symbol_id = self._get_symbol_id(symbol)

        query = """
            SELECT time, open, high, low, close, tick_volume
            FROM ohlcv_m1
            WHERE symbol_id = %s
        """
        params: list = [symbol_id]

        if start:
            query += " AND time >= %s"
            params.append(datetime.strptime(start, "%Y-%m-%d"))
        if end:
            query += " AND time <= %s"
            params.append(datetime.strptime(end, "%Y-%m-%d"))

        query += " ORDER BY time"

        with self._connect() as conn:
            df = pd.read_sql(query, conn, params=params)

        if len(df) == 0:
            return df

        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)
        df["tick_volume"] = df["tick_volume"].astype(int)
        df = df.set_index("time")
        return df

    # ── 리샘플링 ──

    @staticmethod
    def resample(m1_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        M1 데이터를 상위 타임프레임으로 리샘플링.

        Args:
            m1_df: M1 OHLCV (DatetimeIndex)
            timeframe: "M5", "M15", "H1", "H4", "D1"

        Returns:
            리샘플된 OHLCV DataFrame
        """
        if timeframe == "M1":
            return m1_df.copy()

        rule = RESAMPLE_RULES.get(timeframe)
        if rule is None:
            raise ValueError(f"Unknown timeframe: {timeframe}. "
                             f"Supported: {list(RESAMPLE_RULES.keys())}")

        resampled = m1_df.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "tick_volume": "sum",
        }).dropna(subset=["open"])

        return resampled

    # ── 페어 데이터 로드 ──

    def load_pair(
        self,
        symbol_y: str,
        symbol_x: str,
        timeframe: str = "H1",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        두 심볼의 close 가격을 시간 정렬하여 반환.

        Args:
            symbol_y: 종속 변수 심볼 (예: EURUSD)
            symbol_x: 독립 변수 심볼 (예: GBPUSD)
            timeframe: 분석 타임프레임 (M1/M15/H1/H4/D1)
            start, end: 기간 필터

        Returns:
            DataFrame(symbol_y, symbol_x) - inner join on time, NaN 제거
        """
        m1_y = self.load_m1(symbol_y, start, end)
        m1_x = self.load_m1(symbol_x, start, end)

        if len(m1_y) == 0 or len(m1_x) == 0:
            raise ValueError(f"No data: {symbol_y}={len(m1_y)}, {symbol_x}={len(m1_x)}")

        df_y = self.resample(m1_y, timeframe)[["close"]].rename(columns={"close": symbol_y})
        df_x = self.resample(m1_x, timeframe)[["close"]].rename(columns={"close": symbol_x})

        merged = df_y.join(df_x, how="inner").dropna()

        if len(merged) == 0:
            raise ValueError(f"No overlapping data between {symbol_y} and {symbol_x}")

        return merged

    def load_multi(
        self,
        symbols: List[str],
        timeframe: str = "H1",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        복수 심볼의 close 가격을 시간 정렬하여 반환.

        Returns:
            DataFrame(sym1, sym2, ...) - inner join on time
        """
        dfs = []
        for sym in symbols:
            m1 = self.load_m1(sym, start, end)
            if len(m1) == 0:
                raise ValueError(f"No data for {sym}")
            resampled = self.resample(m1, timeframe)[["close"]].rename(columns={"close": sym})
            dfs.append(resampled)

        merged = dfs[0]
        for df in dfs[1:]:
            merged = merged.join(df, how="inner")

        return merged.dropna()

    # ── DB 상태 조회 ──

    def print_status(self):
        """DB에 있는 모든 심볼의 데이터 현황 출력"""
        symbols = self.list_symbols()
        print(f"\n{'Symbol':<10} {'Rows':>12} {'Start':>12} {'End':>12}")
        print("-" * 50)
        for sym in symbols:
            try:
                count = self.count_rows(sym)
                start, end = self.get_date_range(sym)
                start_str = start.strftime("%Y-%m-%d") if start else "N/A"
                end_str = end.strftime("%Y-%m-%d") if end else "N/A"
                print(f"{sym:<10} {count:>12,} {start_str:>12} {end_str:>12}")
            except Exception as e:
                print(f"{sym:<10} Error: {e}")


# ── CLI ──

if __name__ == "__main__":
    loader = DataLoader()
    loader.print_status()
