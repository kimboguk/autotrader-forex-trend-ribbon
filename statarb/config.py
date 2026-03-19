# -*- coding: utf-8 -*-
"""
StatArb - Cointegration 기반 통계적 차익거래 설정

심볼별 설정, cointegration 파라미터, 트레이딩/리스크 설정 관리.
ICT 프로젝트와 독립적으로 운용되며, DB 인프라만 공유.
"""

import sys
from pathlib import Path

# ICT 프로젝트의 DB 모듈 재활용
ICT_DIR = Path(__file__).resolve().parent.parent / "ICT"
if str(ICT_DIR) not in sys.path:
    sys.path.insert(0, str(ICT_DIR))

# ── 심볼별 설정 ──────────────────────────────────────────────

SYMBOLS = {
    # 기존 DB 보유
    "EURUSD": {"pip_size": 0.0001, "spread_pips": 0.4, "commission_pips": 0.3},
    "USDJPY": {"pip_size": 0.01,   "spread_pips": 0.5, "commission_pips": 0.3},
    "EURJPY": {"pip_size": 0.01,   "spread_pips": 0.6, "commission_pips": 0.3},
    "XAUUSD": {"pip_size": 0.10,   "spread_pips": 3.0, "commission_pips": 0.7},
    # 신규 후보 (데이터 수집 필요)
    "GBPUSD": {"pip_size": 0.0001, "spread_pips": 0.5, "commission_pips": 0.3},
    "USDCHF": {"pip_size": 0.0001, "spread_pips": 0.5, "commission_pips": 0.3},
    "AUDUSD": {"pip_size": 0.0001, "spread_pips": 0.5, "commission_pips": 0.3},
    "NZDUSD": {"pip_size": 0.0001, "spread_pips": 0.6, "commission_pips": 0.3},
    "USDCAD": {"pip_size": 0.0001, "spread_pips": 0.6, "commission_pips": 0.3},
    "EURGBP": {"pip_size": 0.0001, "spread_pips": 0.6, "commission_pips": 0.3},
    "EURCHF": {"pip_size": 0.0001, "spread_pips": 0.6, "commission_pips": 0.3},
}

# ── 후보 페어 (D1 10년 Cointegration 검정 결과 기반) ─────────
# 2015-2025 D1 Engle-Granger 검정에서 p < 0.05인 16/45 페어

# Tier 1: p < 0.01, HL < 80일 — 최우선 트레이딩 후보
TIER1_PAIRS = [
    ("EURCHF", "USDJPY"),   # p=0.0000, HL=47d, corr=-0.877 ★ 최강
    ("USDCHF", "EURJPY"),   # p=0.0003, HL=54d, corr=-0.792
    ("USDCAD", "EURJPY"),   # p=0.0007, HL=71d, corr=+0.466
    ("EURCHF", "EURJPY"),   # p=0.0007, HL=63d, corr=-0.795
    ("AUDUSD", "NZDUSD"),   # p=0.0019, HL=54d, corr=+0.955
    ("NZDUSD", "EURJPY"),   # p=0.0026, HL=77d, corr=-0.691
    ("USDCAD", "EURCHF"),   # p=0.0031, HL=73d, corr=-0.561
    ("USDCAD", "USDJPY"),   # p=0.0036, HL=71d, corr=+0.566
    ("NZDUSD", "EURCHF"),   # p=0.0057, HL=74d, corr=+0.752
    ("NZDUSD", "USDJPY"),   # p=0.0064, HL=81d, corr=-0.778
    ("AUDUSD", "EURCHF"),   # p=0.0080, HL=74d, corr=+0.730
]

# Tier 2: p < 0.05 — 보조 후보
TIER2_PAIRS = [
    ("USDCHF", "USDJPY"),   # p=0.0104, HL=77d, corr=-0.641
    ("AUDUSD", "EURJPY"),   # p=0.0127, HL=89d, corr=-0.594
    ("AUDUSD", "USDJPY"),   # p=0.0151, HL=89d, corr=-0.701
    ("USDCAD", "EURGBP"),   # p=0.0166, HL=104d, corr=+0.134
    ("USDCHF", "NZDUSD"),   # p=0.0361, HL=92d, corr=+0.484
]

CANDIDATE_PAIRS = TIER1_PAIRS + TIER2_PAIRS

# ── Cointegration 설정 ───────────────────────────────────────

COINT_CONFIG = {
    "test_method": "engle_granger",     # "engle_granger" | "johansen"
    "significance_level": 0.05,
    "rolling_window": 252,              # D1 바 기준 (1년 영업일)
    "rolling_step": 21,                 # 롤링 스텝 (D1 약 1개월)
    "min_half_life": 10,                # 최소 반감기 (D1 바)
    "max_half_life": 120,               # 최대 반감기 (D1 바)
}

# ── Kalman Filter 설정 ───────────────────────────────────────

KALMAN_CONFIG = {
    "delta": 1e-4,          # 상태 전이 노이즈 (헤지비율 변화 속도)
    "Ve": 1e-3,             # 관측 노이즈
    "optimize": True,       # 파라미터 자동 최적화
    "delta_range": (1e-6, 1e-2, 20),   # logspace(start, end, n)
    "Ve_range": (1e-4, 1e-1, 20),
}

# ── HMM 설정 ─────────────────────────────────────────────────

HMM_CONFIG = {
    "n_states": 2,                  # 2: MR vs Trending, 3: + High-vol
    "n_iter": 100,                  # EM 최대 반복
    "covariance_type": "full",
    "feature_lookback": 20,         # 특성 계산 롤링 윈도우
    "retrain_frequency": "weekly",  # 재학습 주기
}

# ── GARCH 설정 ────────────────────────────────────────────────

GARCH_CONFIG = {
    "p": 1,
    "q": 1,
    "dist": "t",           # Student-t (fat tail 대응)
    "vol_lookback": 100,   # 평균 계산용 롤링 윈도우
}

# ── 트레이딩 설정 ─────────────────────────────────────────────

TRADING_CONFIG = {
    "z_score_entry": 2.0,          # 진입 임계값
    "z_score_exit": 0.5,           # 청산 임계값
    "z_score_stop": 4.0,           # 강제 손절 임계값
    "max_holding_bars": 500,       # 최대 보유 기간 (바)
    "min_correlation": 0.6,        # 최소 상관계수
    "mr_prob_threshold": 0.7,      # HMM 평균회귀 레짐 확률 임계값
}

# ── 리스크 관리 ───────────────────────────────────────────────

RISK_CONFIG = {
    "max_concurrent_pairs": 3,     # 동시 보유 페어 수
    "risk_per_trade_pct": 1.0,     # 거래당 리스크 (계좌 %)
    "daily_max_loss_pips": 50,
    "weekly_max_loss_pips": 150,
    "max_drawdown_pct": 5.0,       # 최대 허용 DD (계좌 %)
    "slippage_pips": 0.5,          # 슬리피지 모델링
}

# ── 타임프레임별 역할 ─────────────────────────────────────────

TIMEFRAME_ROLES = {
    "D1": ["hmm_regime"],                   # 레짐 감지
    "H4": ["hmm_regime", "gpr_hedge"],      # 레짐 + 비선형 헤지
    "H1": ["kalman", "garch", "kde", "rolling_coint"],  # 핵심 시그널
    "M15": ["signal_generation", "position_sizing"],     # 진입/청산
    "M1":  ["execution"],                   # 실행 타이밍
}

# ── 백테스트 설정 ─────────────────────────────────────────────

BACKTEST_CONFIG = {
    "output_dir": "./outputs",
    "default_timeframe": "H1",
    "initial_capital": 10000,       # USD
}

# ── Walk-Forward 설정 ─────────────────────────────────────────

WFO_CONFIG = {
    "in_sample_bars": 4380,     # IS: 6개월 H1 (730*6)
    "out_sample_bars": 730,     # OOS: 1개월 H1
    "roll_step_bars": 730,      # 롤 스텝: 1개월
    "cpcv_n_groups": 6,         # CPCV 그룹 수
    "cpcv_n_test": 2,           # 테스트 그룹 수
    "purge_length": 10,         # purge 바 수
}

# ── H1 적응형 전략 설정 ──────────────────────────────────────

H1_ADAPTIVE_CONFIG = {
    "coint_window": 240,        # 공적분 검정 윈도우 (H1 바, ~10일)
    "coint_recheck": 24,        # 재검정 간격 (H1 바, ~1일)
    "coint_pvalue": 0.05,       # 진입 허용 p-value
    "z_entry": 2.0,             # 진입 z-score
    "z_exit": 0.5,              # 청산 z-score
    "z_stop": 4.0,              # 손절 z-score
    "max_holding_bars": 240,    # 최대 보유 (~10일)
    "lookback": 100,            # z-score 롤링 윈도우
    "degraded_z_exit": 0.25,    # 공적분 붕괴 시 tight TP
    "degraded_timeout": 72,     # 공적분 붕괴 시 3일 내 청산
}

# ── Baseline 전략 설정 ───────────────────────────────────────

RATIO_BB_CONFIG = {
    "lookback": 100,
    "z_entry": 2.0,
    "z_exit": 0.5,
    "z_stop": 4.0,
    "max_holding_bars": 240,
}

KALMAN_H1_CONFIG = {
    "delta": 1e-4,
    "Ve": 1e-3,
    "lookback": 100,
    "z_entry": 2.0,
    "z_exit": 0.5,
    "z_stop": 4.0,
    "max_holding_bars": 240,
}

OU_OPTIMAL_CONFIG = {
    "lookback": 240,
    "recheck": 24,
    "min_half_life": 5,
    "max_half_life": 120,
    "z_exit": 0.0,
    "z_stop": 4.0,
    "holding_mult": 2.0,
}
