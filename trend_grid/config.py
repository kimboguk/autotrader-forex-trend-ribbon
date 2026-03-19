# -*- coding: utf-8 -*-
"""
Trend Grid Strategy - 설정

VWMA 기반 그물망 돌파 추세추종 전략.
4개의 VWMA(30,60,120,240)를 그리드로 사용하여
가격이 그리드를 완전히 돌파할 때 진입, 되돌아올 때 청산.
"""

import sys
from pathlib import Path

# statarb DataLoader + common trade_engine 재활용
STATARB_DIR = Path(__file__).resolve().parent.parent / "statarb"
COMMON_DIR = Path(__file__).resolve().parent.parent / "common"
ICT_DIR = Path(__file__).resolve().parent.parent / "ICT"
for d in [STATARB_DIR, COMMON_DIR, ICT_DIR]:
    if str(d) not in sys.path:
        sys.path.append(str(d))

# ── 심볼 설정 ────────────────────────────────────────────────

SYMBOLS = {
    # ── Forex ──
    "EURUSD": {"pip_size": 0.0001, "spread_pips": 0.4, "commission_pips": 0.3, "category": "forex", "quote_ccy": "USD"},
    "USDJPY": {"pip_size": 0.01,   "spread_pips": 0.5, "commission_pips": 0.3, "category": "forex", "quote_ccy": "JPY"},
    "GBPUSD": {"pip_size": 0.0001, "spread_pips": 0.5, "commission_pips": 0.3, "category": "forex", "quote_ccy": "USD"},
    "EURJPY": {"pip_size": 0.01,   "spread_pips": 0.6, "commission_pips": 0.3, "category": "forex", "quote_ccy": "JPY"},
    "XAUUSD": {"pip_size": 0.10,   "spread_pips": 3.0, "commission_pips": 0.7, "category": "forex", "quote_ccy": "USD", "lot_size": 100},
    "USDCHF": {"pip_size": 0.0001, "spread_pips": 0.5, "commission_pips": 0.3, "category": "forex", "quote_ccy": "CHF"},
    "AUDUSD": {"pip_size": 0.0001, "spread_pips": 0.5, "commission_pips": 0.3, "category": "forex", "quote_ccy": "USD"},
    "NZDUSD": {"pip_size": 0.0001, "spread_pips": 0.6, "commission_pips": 0.3, "category": "forex", "quote_ccy": "USD"},
    "USDCAD": {"pip_size": 0.0001, "spread_pips": 0.6, "commission_pips": 0.3, "category": "forex", "quote_ccy": "CAD"},
    "EURGBP": {"pip_size": 0.0001, "spread_pips": 0.6, "commission_pips": 0.3, "category": "forex", "quote_ccy": "GBP"},
    "EURCHF": {"pip_size": 0.0001, "spread_pips": 0.6, "commission_pips": 0.3, "category": "forex", "quote_ccy": "CHF"},
    # ── Index ──
    "SP500":    {"pip_size": 0.25, "spread_pips": 0.5, "commission_pips": 0, "category": "index",
                 "point_value": 50, "quote_ccy": "USD", "lot_size": 1},
    "KOSPI200": {"pip_size": 0.05, "spread_pips": 1.0, "commission_pips": 0, "category": "index",
                 "point_value": 250000, "quote_ccy": "KRW", "lot_size": 1},
    "JPN225":   {"pip_size": 1.0, "spread_pips": 7.0, "commission_pips": 0, "category": "index",
                 "point_value": 500, "quote_ccy": "JPY", "lot_size": 1},
    # ── Crypto ──
    "BTCUSD":   {"pip_size": 0.01, "spread_pips": 0, "commission_pips": 0, "category": "crypto",
                 "point_value": 1, "quote_ccy": "USD", "lot_size": 1, "fee_rate": 0.001},
}

# ── VWMA 그리드 설정 ─────────────────────────────────────────

VWMA_PERIODS = [30, 60, 120, 240]

# 캔들 몸통 돌파 비율 (0.5 = 50%)
BODY_PENETRATION_RATIO = 0.5

# ── 타임프레임 설정 ──────────────────────────────────────────

TIMEFRAMES = ["D1", "H4", "H1", "M30", "M15", "M5", "M1"]

# pandas resample rule 매핑
RESAMPLE_RULES = {
    "M1":  "1min",
    "M5":  "5min",
    "M15": "15min",
    "M30": "30min",
    "H1":  "1h",
    "H4":  "4h",
    "D1":  "1D",
}

# ── 이동평균 타입 ────────────────────────────────────────────

# "vwma" = Volume Weighted MA (tick_volume 사용 - 2025 이전 데이터는 volume=0으로 사용 불가)
# "sma"  = Simple MA (거래량 무관)
# "ema"  = Exponential MA (기본값)
MA_TYPE = "ema"

# ── 거래 비용 ────────────────────────────────────────────────
# 진입 시: spread (반영)
# 왕복: spread + commission * 2
# spread_pips + commission_pips = 한쪽 비용 (pips)

# ── 백테스트 설정 ────────────────────────────────────────────

BACKTEST_CONFIG = {
    "initial_capital": 10_000,      # USD
    "lot_size": 100_000,            # 1 standard lot
    "position_size_lots": 0.1,      # 거래 단위 (mini lot)
    "output_dir": str(Path(__file__).resolve().parent / "outputs"),
}
