# -*- coding: utf-8 -*-
"""
ICT Signal Compositor

OB+FVG 기본 신호에 BOS/CHoCH/Liquidity 분석을 조합하여
가중 스코어 기반 진입 결정을 수행.

사용법:
    compositor = SignalCompositor(symbol="EURUSD")
    signal = compositor.get_composite_signal(m15_bars, m1_bars)
"""

import pandas as pd
from typing import Optional, Dict

from config import STRATEGY_DEFAULTS
from ob_fvg_strategy import OrderBlockFVGStrategy
from structure_detector import BOSDetector, CHoCHDetector
from liquidity_detector import LiquidityDetector


class SignalCompositor:
    """
    다중 ICT 신호 조합기

    OB+FVG를 기본 조건으로 요구하고, BOS/CHoCH/Liquidity로 스코어를 조정.
    composite_score >= threshold 일 때만 진입 신호 반환.
    """

    def __init__(
        self,
        symbol: str = "EURUSD",
        threshold: float = None,
        weights: Dict[str, float] = None,
        **strategy_kwargs,
    ):
        """
        Args:
            symbol: 거래 심볼
            threshold: 최소 composite score (기본: config.py)
            weights: 가중치 딕셔너리 (기본: config.py)
            **strategy_kwargs: OrderBlockFVGStrategy에 전달할 추가 인자
        """
        self.symbol = symbol
        self.threshold = threshold if threshold is not None else STRATEGY_DEFAULTS.get("compositor_threshold", 0.6)
        self.weights = weights if weights is not None else STRATEGY_DEFAULTS.get("compositor_weights", {
            "ob_fvg": 0.4, "bos": 0.3, "choch": 0.2, "liquidity": 0.1,
        })

        # 하위 모듈 초기화
        self.strategy = OrderBlockFVGStrategy(symbol=symbol, **strategy_kwargs)
        self.bos_detector = BOSDetector()
        self.choch_detector = CHoCHDetector()
        self.liquidity_detector = LiquidityDetector(symbol=symbol)

        # 하위 호환: strategy의 공개 속성 노출
        self.pip_size = self.strategy.pip_size

    def get_composite_signal(
        self,
        m15_bars: pd.DataFrame,
        m1_bars: pd.DataFrame,
    ) -> Optional[Dict]:
        """
        복합 신호 생성

        OB+FVG 기본 신호가 있을 때만 BOS/CHoCH/Liquidity를 추가 평가.

        Args:
            m15_bars: M15 OHLC DataFrame
            m1_bars: M1 OHLC DataFrame

        Returns:
            기존 get_entry_signal 포맷 + composite 부가 정보, 또는 None
        """
        # 1. 기본 OB+FVG 신호 (필수 조건)
        base_signal = self.strategy.get_entry_signal(m15_bars, m1_bars)
        if base_signal is None:
            return None

        direction = base_signal['signal']
        components: Dict[str, float] = {}
        score = 0.0

        # 2. OB+FVG (항상 1.0)
        components['ob_fvg'] = 1.0
        score += self.weights.get('ob_fvg', 0.4) * 1.0

        # 3. BOS 방향 일치
        bos = self.bos_detector.detect(m15_bars)
        if bos is not None and bos == direction:
            components['bos'] = 1.0   # BOS 방향 일치
        elif bos is not None and bos != direction:
            components['bos'] = -0.5  # BOS 반대 방향 (강한 감점)
        else:
            components['bos'] = 0.0   # BOS 없음
        score += self.weights.get('bos', 0.3) * components['bos']

        # 4. CHoCH 전환 확인
        choch = self.choch_detector.detect(m15_bars)
        if choch is not None and choch == direction:
            components['choch'] = 1.0   # 추세 전환이 진입 방향과 일치
        elif choch is not None and choch != direction:
            components['choch'] = -0.3  # 반대 방향 전환 (약한 감점)
        else:
            components['choch'] = 0.0
        score += self.weights.get('choch', 0.2) * components['choch']

        # 5. Liquidity 컨텍스트
        liq = self.liquidity_detector.check_liquidity_context(
            m15_bars, base_signal['entry_price'], direction,
        )
        components['liquidity'] = liq['score_adjustment']
        score += self.weights.get('liquidity', 0.1) * liq['score_adjustment']

        # 6. 임계값 체크
        if score < self.threshold:
            return None

        # 7. 신호에 composite 정보 추가
        base_signal['composite_score'] = round(score, 4)
        base_signal['components'] = components
        base_signal['bos_direction'] = bos
        base_signal['choch_signal'] = choch
        base_signal['liquidity_context'] = {
            'score': liq['score_adjustment'],
            'sweep': liq['sweep_occurred'],
        }

        return base_signal

    # 하위 호환 메서드
    def calculate_pnl(self, *args, **kwargs):
        return self.strategy.calculate_pnl(*args, **kwargs)

    def format_signal(self, sig: Dict) -> str:
        """신호 포맷 (composite 정보 포함)"""
        if sig is None:
            return "None"
        base = self.strategy.format_signal(sig)
        if 'composite_score' in sig:
            return f"{base} | Score {sig['composite_score']:.2f}"
        return base
