# -*- coding: utf-8 -*-
"""
ICT Trading Strategy - PostgreSQL 데이터베이스 모듈

OHLCV 시계열 데이터 저장/조회, 거래 기록 관리
"""

import os
import sys
from io import StringIO
from datetime import datetime
from typing import Optional, Tuple, List, Dict

import pandas as pd
import psycopg2
from psycopg2 import extras
from dotenv import load_dotenv

from config import SYMBOLS

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()


# ──────────────────────────────────────────────
# SQL: 스키마 정의
# ──────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS symbols (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(20) NOT NULL UNIQUE,
    pip_size        NUMERIC(10,6) NOT NULL,
    spread_pips     NUMERIC(6,2) DEFAULT 0.4,
    commission_pips NUMERIC(6,2) DEFAULT 0.3,
    sl_buffer       NUMERIC(10,6) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ohlcv_m1 (
    time            TIMESTAMPTZ NOT NULL,
    symbol_id       INTEGER NOT NULL REFERENCES symbols(id),
    open            NUMERIC(12,6) NOT NULL,
    high            NUMERIC(12,6) NOT NULL,
    low             NUMERIC(12,6) NOT NULL,
    close           NUMERIC(12,6) NOT NULL,
    tick_volume     INTEGER DEFAULT 0,
    source          VARCHAR(20) DEFAULT 'mt5',
    PRIMARY KEY (symbol_id, time)
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_m1_time
    ON ohlcv_m1 (time);

CREATE TABLE IF NOT EXISTS trades (
    id              SERIAL PRIMARY KEY,
    symbol_id       INTEGER NOT NULL REFERENCES symbols(id),
    session_id      VARCHAR(50) NOT NULL,
    session_type    VARCHAR(10) NOT NULL,
    entry_time      TIMESTAMPTZ NOT NULL,
    entry_price     NUMERIC(12,6) NOT NULL,
    direction       VARCHAR(4) NOT NULL,
    stop_loss       NUMERIC(12,6) NOT NULL,
    take_profit     NUMERIC(12,6) NOT NULL,
    risk_pips       NUMERIC(10,2) NOT NULL,
    exit_time       TIMESTAMPTZ,
    exit_price      NUMERIC(12,6),
    exit_reason     VARCHAR(10),
    bars_held       INTEGER,
    gross_pips      NUMERIC(10,2),
    net_pips        NUMERIC(10,2),
    profit_loss     VARCHAR(5),
    strategy_params JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trades_session
    ON trades (symbol_id, session_id);

CREATE TABLE IF NOT EXISTS backtest_sessions (
    session_id          VARCHAR(50) PRIMARY KEY,
    symbol_id           INTEGER NOT NULL REFERENCES symbols(id),
    start_time          TIMESTAMPTZ NOT NULL,
    end_time            TIMESTAMPTZ NOT NULL,
    data_source         VARCHAR(20) NOT NULL,
    total_trades        INTEGER,
    wins                INTEGER,
    losses              INTEGER,
    win_rate_pct        NUMERIC(6,2),
    total_net_pips      NUMERIC(10,2),
    risk_reward_ratio   NUMERIC(6,2),
    max_drawdown_pips   NUMERIC(10,2),
    strategy_params     JSONB NOT NULL,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);
"""


class Database:
    """PostgreSQL 데이터베이스 연결 및 CRUD"""

    def __init__(self):
        self.host = os.getenv("DB_HOST", "localhost")
        self.port = int(os.getenv("DB_PORT", 5432))
        self.dbname = os.getenv("DB_NAME", "ict_trading")
        self.user = os.getenv("DB_USER", "postgres")
        self.password = os.getenv("DB_PASSWORD", "")

    # ── 연결 ──

    def connect(self, dbname: str = None):
        """PostgreSQL 연결 반환 (trust/peer 인증 시 비밀번호 불필요)"""
        params = {
            "host": self.host,
            "port": self.port,
            "dbname": dbname or self.dbname,
            "user": self.user,
        }
        if self.password:
            params["password"] = self.password
        return psycopg2.connect(**params)

    # ── 스키마 초기화 ──

    def create_database(self):
        """ict_trading 데이터베이스 생성 (없으면)"""
        conn = self.connect(dbname="postgres")
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (self.dbname,)
        )
        if not cur.fetchone():
            cur.execute(f'CREATE DATABASE "{self.dbname}"')
            print(f"Database '{self.dbname}' created.")
        else:
            print(f"Database '{self.dbname}' already exists.")
        cur.close()
        conn.close()

    def init_schema(self):
        """테이블 생성"""
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(SCHEMA_SQL)
            conn.commit()
        print("Schema initialized (4 tables).")

    def seed_symbols(self):
        """config.py의 SYMBOLS를 DB에 삽입"""
        with self.connect() as conn:
            with conn.cursor() as cur:
                for name, cfg in SYMBOLS.items():
                    cur.execute(
                        """
                        INSERT INTO symbols (name, pip_size, spread_pips, commission_pips, sl_buffer)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (name) DO UPDATE SET
                            pip_size = EXCLUDED.pip_size,
                            spread_pips = EXCLUDED.spread_pips,
                            commission_pips = EXCLUDED.commission_pips,
                            sl_buffer = EXCLUDED.sl_buffer
                        """,
                        (
                            name,
                            cfg["pip_size"],
                            cfg["spread_pips"],
                            cfg["commission_pips"],
                            cfg["sl_buffer_pips"],
                        ),
                    )
            conn.commit()
        print(f"Symbols seeded: {list(SYMBOLS.keys())}")

    # ── 심볼 조회 ──

    def get_symbol_id(self, symbol: str) -> int:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM symbols WHERE name = %s", (symbol,))
                row = cur.fetchone()
                if row is None:
                    raise ValueError(f"Symbol '{symbol}' not found in DB")
                return row[0]

    # ── OHLCV 데이터 ──

    def bulk_insert_ohlcv(
        self, df: pd.DataFrame, symbol: str, source: str = "mt5"
    ) -> int:
        """
        COPY 프로토콜로 OHLCV 대량 삽입.
        중복 키는 무시 (ON CONFLICT DO NOTHING via temp table).

        Returns: 삽입된 행 수
        """
        symbol_id = self.get_symbol_id(symbol)

        # 임시 테이블 → COPY → INSERT ... ON CONFLICT
        with self.connect() as conn:
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

                # DataFrame → TSV buffer
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

    def query_ohlcv(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """OHLCV 데이터 조회 → DataFrame"""
        symbol_id = self.get_symbol_id(symbol)

        query = """
            SELECT time, open, high, low, close, tick_volume
            FROM ohlcv_m1
            WHERE symbol_id = %s
        """
        params: list = [symbol_id]

        if start:
            query += " AND time >= %s"
            params.append(start)
        if end:
            query += " AND time <= %s"
            params.append(end)

        query += " ORDER BY time"

        with self.connect() as conn:
            df = pd.read_sql(query, conn, params=params)

        # 컬럼 타입 보정
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)
        df["tick_volume"] = df["tick_volume"].astype(int)

        return df

    def get_ohlcv_date_range(self, symbol: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """심볼의 데이터 시작/종료 시간"""
        symbol_id = self.get_symbol_id(symbol)
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT MIN(time), MAX(time) FROM ohlcv_m1 WHERE symbol_id = %s",
                    (symbol_id,),
                )
                return cur.fetchone()

    def count_ohlcv(self, symbol: str) -> int:
        symbol_id = self.get_symbol_id(symbol)
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM ohlcv_m1 WHERE symbol_id = %s",
                    (symbol_id,),
                )
                return cur.fetchone()[0]

    # ── 거래 기록 ──

    def save_trades(
        self,
        trades: List[Dict],
        session_id: str,
        symbol: str,
        session_type: str = "backtest",
        strategy_params: Optional[Dict] = None,
    ):
        """거래 목록을 DB에 저장"""
        symbol_id = self.get_symbol_id(symbol)
        import json

        params_json = json.dumps(strategy_params) if strategy_params else None

        with self.connect() as conn:
            with conn.cursor() as cur:
                for t in trades:
                    cur.execute(
                        """
                        INSERT INTO trades (
                            symbol_id, session_id, session_type,
                            entry_time, entry_price, direction,
                            stop_loss, take_profit, risk_pips,
                            exit_time, exit_price, exit_reason,
                            bars_held, gross_pips, net_pips,
                            profit_loss, strategy_params
                        ) VALUES (
                            %s, %s, %s,
                            %s, %s, %s,
                            %s, %s, %s,
                            %s, %s, %s,
                            %s, %s, %s,
                            %s, %s
                        )
                        """,
                        (
                            symbol_id, session_id, session_type,
                            t.get("entry_time"), t.get("entry_price"), t.get("direction"),
                            t.get("stop_loss"), t.get("take_profit"), t.get("risk_pips"),
                            t.get("exit_time"), t.get("exit_price"), t.get("exit_reason"),
                            t.get("bars_held"), t.get("gross_pips"), t.get("net_pips"),
                            t.get("profit_loss"), params_json,
                        ),
                    )
            conn.commit()

    def save_backtest_session(
        self,
        session_id: str,
        symbol: str,
        results: Dict,
        strategy_params: Dict,
        start_time: datetime,
        end_time: datetime,
        data_source: str = "mt5",
    ):
        """백테스트 세션 요약 저장"""
        import json

        symbol_id = self.get_symbol_id(symbol)

        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO backtest_sessions (
                        session_id, symbol_id, start_time, end_time, data_source,
                        total_trades, wins, losses, win_rate_pct,
                        total_net_pips, risk_reward_ratio, max_drawdown_pips,
                        strategy_params
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (session_id) DO UPDATE SET
                        total_trades = EXCLUDED.total_trades,
                        wins = EXCLUDED.wins,
                        losses = EXCLUDED.losses,
                        win_rate_pct = EXCLUDED.win_rate_pct,
                        total_net_pips = EXCLUDED.total_net_pips,
                        risk_reward_ratio = EXCLUDED.risk_reward_ratio,
                        max_drawdown_pips = EXCLUDED.max_drawdown_pips,
                        strategy_params = EXCLUDED.strategy_params
                    """,
                    (
                        session_id, symbol_id, start_time, end_time, data_source,
                        results.get("total_trades"),
                        results.get("wins"),
                        results.get("losses"),
                        results.get("win_rate_%"),
                        results.get("total_net_pips"),
                        results.get("risk_reward_ratio"),
                        results.get("max_drawdown_pips"),
                        json.dumps(strategy_params),
                    ),
                )
            conn.commit()
