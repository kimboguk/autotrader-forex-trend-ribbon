# -*- coding: utf-8 -*-
"""
ICT Trading Strategy - 전략 설정

심볼별 pip_size, 스프레드, 커미션 및 전략 기본 파라미터 관리
"""

# 심볼별 설정
SYMBOLS = {
    "EURUSD": {
        "pip_size": 0.0001,
        "spread_pips": 0.4,
        "commission_pips": 0.3,
        "sl_buffer_pips": 0.0001,   # 0.1 pip
    },
    "USDJPY": {
        "pip_size": 0.01,
        "spread_pips": 0.5,
        "commission_pips": 0.3,
        "sl_buffer_pips": 0.01,     # 0.1 pip (JPY 단위)
    },
    "EURJPY": {
        "pip_size": 0.01,
        "spread_pips": 0.6,
        "commission_pips": 0.3,
        "sl_buffer_pips": 0.01,
    },
    "XAUUSD": {
        "pip_size": 0.10,           # Gold: 1 pip = $0.10
        "spread_pips": 3.0,         # ~$0.30 spread
        "commission_pips": 0.7,
        "sl_buffer_pips": 0.10,
    },
    # StatArb 후보 심볼
    "GBPUSD": {
        "pip_size": 0.0001,
        "spread_pips": 0.5,
        "commission_pips": 0.3,
        "sl_buffer_pips": 0.0001,
    },
    "USDCHF": {
        "pip_size": 0.0001,
        "spread_pips": 0.5,
        "commission_pips": 0.3,
        "sl_buffer_pips": 0.0001,
    },
    "AUDUSD": {
        "pip_size": 0.0001,
        "spread_pips": 0.5,
        "commission_pips": 0.3,
        "sl_buffer_pips": 0.0001,
    },
    "NZDUSD": {
        "pip_size": 0.0001,
        "spread_pips": 0.6,
        "commission_pips": 0.3,
        "sl_buffer_pips": 0.0001,
    },
    "USDCAD": {
        "pip_size": 0.0001,
        "spread_pips": 0.6,
        "commission_pips": 0.3,
        "sl_buffer_pips": 0.0001,
    },
    "EURGBP": {
        "pip_size": 0.0001,
        "spread_pips": 0.6,
        "commission_pips": 0.3,
        "sl_buffer_pips": 0.0001,
    },
    "EURCHF": {
        "pip_size": 0.0001,
        "spread_pips": 0.6,
        "commission_pips": 0.3,
        "sl_buffer_pips": 0.0001,
    },
}

# 전략 기본 파라미터
STRATEGY_DEFAULTS = {
    "risk_reward_ratio": 2.0,       # 손익비 기본값 (1.5~3.0 권장)
    "max_bars_per_trade": 100,      # 최대 홀딩 바 수
    "allowed_hours": {0, 1, 8, 16, 17},     # 런던+미국 세션 (UTC), 9시 제외
    "require_fvg_confirm": True,    # OB+FVG 동시 확인 여부
    "sl_timeframe": "M15",          # SL 기준 타임프레임 ("M1" 또는 "M15")
    # 백테스트 윈도우 크기
    "m15_lookback": 50,             # M15 바 윈도우 (Phase 2 BOS/Liquidity용)
    "m1_lookback": 3,               # M1 바 윈도우 (FVG 3봉)
    # Phase 2 모듈 (기본 비활성)
    "use_compositor": False,        # SignalCompositor 사용 여부
    "bos_swing_lookback": 20,       # BOS 스윙 탐색 범위 (M15 바)
    "bos_swing_strength": 5,        # 스윙 포인트 강도 (양쪽 N바)
    "liquidity_lookback": 50,       # Liquidity 탐색 범위 (M15 바)
    "liquidity_tolerance_pips": 3.0, # Equal High/Low 허용 오차 (핍)
    "compositor_threshold": 0.6,    # Compositor 최소 스코어 (0~1)
    "compositor_weights": {         # 가중치
        "ob_fvg": 0.4,
        "bos": 0.3,
        "choch": 0.2,
        "liquidity": 0.1,
    },
}

# 백테스트 설정
BACKTEST_CONFIG = {
    "data_dir": "./data",
    "output_dir": "./outputs",
}

# 리스크 관리 한도
RISK_LIMITS = {
    "daily_max_loss_pips": 100,
    "weekly_max_loss_pips": 300,
    "monthly_max_loss_pips": 500,
    "max_drawdown_pct": 30.0,
}
