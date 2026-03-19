# -*- coding: utf-8 -*-
"""
ICT Market Structure 감지 모듈

- SwingDetector: M15 봉에서 스윙 고점/저점 식별
- BOSDetector: Break of Structure (추세 지속 확인)
- CHoCHDetector: Change of Character (추세 전환 감지)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List

from config import STRATEGY_DEFAULTS


class SwingDetector:
    """
    스윙 고점/저점 식별

    N-bar 룩백 방식: 양쪽 N개 봉보다 높은(낮은) 극값을 스윙 포인트로 식별.
    식별된 스윙을 이전 스윙과 비교하여 HH/HL/LH/LL 라벨 부여.
    """

    def __init__(self, swing_strength: int = 5):
        """
        Args:
            swing_strength: 스윙 포인트 판별 시 양쪽에 필요한 최소 봉 수
        """
        self.swing_strength = swing_strength

    def find_swings(self, bars: pd.DataFrame) -> List[Dict]:
        """
        스윙 고점/저점 리스트 반환

        Returns:
            [{'index': int, 'time': Timestamp, 'price': float,
              'swing_type': 'high'|'low', 'label': 'HH'|'LH'|'HL'|'LL'|None}]
        """
        n = self.swing_strength
        if len(bars) < 2 * n + 1:
            return []

        highs = bars['high'].values
        lows = bars['low'].values
        times = bars['time'].values

        swings: List[Dict] = []
        last_swing_high_price = None
        last_swing_low_price = None

        for i in range(n, len(bars) - n):
            # 스윙 고점: 양쪽 N개 봉보다 high가 높음
            window_highs = highs[i - n:i + n + 1]
            if highs[i] == window_highs.max() and np.sum(window_highs == highs[i]) == 1:
                label = None
                if last_swing_high_price is not None:
                    label = 'HH' if highs[i] > last_swing_high_price else 'LH'
                swings.append({
                    'index': i,
                    'time': times[i],
                    'price': float(highs[i]),
                    'swing_type': 'high',
                    'label': label,
                })
                last_swing_high_price = float(highs[i])

            # 스윙 저점: 양쪽 N개 봉보다 low가 낮음
            window_lows = lows[i - n:i + n + 1]
            if lows[i] == window_lows.min() and np.sum(window_lows == lows[i]) == 1:
                label = None
                if last_swing_low_price is not None:
                    label = 'HL' if lows[i] > last_swing_low_price else 'LL'
                swings.append({
                    'index': i,
                    'time': times[i],
                    'price': float(lows[i]),
                    'swing_type': 'low',
                    'label': label,
                })
                last_swing_low_price = float(lows[i])

        # 시간순 정렬
        swings.sort(key=lambda s: s['index'])
        return swings


class BOSDetector:
    """
    Break of Structure (BOS) 감지

    추세가 지속되는 구간에서 가격이 최근 스윙 포인트를 돌파하는 패턴.
    - 상승추세(HH+HL): 현재 가격이 최근 스윙 고점을 상향 돌파 → 매수 BOS (+1)
    - 하락추세(LH+LL): 현재 가격이 최근 스윙 저점을 하향 돌파 → 매도 BOS (-1)
    """

    def __init__(
        self,
        swing_lookback: int = None,
        swing_strength: int = None,
    ):
        self.swing_lookback = swing_lookback or STRATEGY_DEFAULTS.get("bos_swing_lookback", 20)
        self.swing_strength = swing_strength or STRATEGY_DEFAULTS.get("bos_swing_strength", 5)
        self.swing_detector = SwingDetector(self.swing_strength)

    def _get_trend(self, swings: List[Dict]) -> Optional[int]:
        """
        스윙 패턴으로 추세 판별

        Returns: 1 (상승), -1 (하락), None (판별 불가)
        """
        # 최근 스윙 고점과 저점 라벨 추출
        recent_highs = [s for s in swings if s['swing_type'] == 'high' and s['label'] is not None]
        recent_lows = [s for s in swings if s['swing_type'] == 'low' and s['label'] is not None]

        if not recent_highs or not recent_lows:
            return None

        last_high_label = recent_highs[-1]['label']
        last_low_label = recent_lows[-1]['label']

        # 상승: HH + HL
        if last_high_label == 'HH' and last_low_label == 'HL':
            return 1
        # 하락: LH + LL
        if last_high_label == 'LH' and last_low_label == 'LL':
            return -1

        return None

    def detect(self, m15_bars: pd.DataFrame) -> Optional[int]:
        """
        BOS 감지

        Args:
            m15_bars: M15 OHLC DataFrame (최소 m15_lookback 개)

        Returns: 1 (매수 BOS), -1 (매도 BOS), None
        """
        if len(m15_bars) < 2 * self.swing_strength + 1:
            return None

        swings = self.swing_detector.find_swings(m15_bars)
        if len(swings) < 3:
            return None

        trend = self._get_trend(swings)
        if trend is None:
            return None

        current_close = float(m15_bars.iloc[-1]['close'])

        # 최근 스윙 고점/저점
        swing_highs = [s for s in swings if s['swing_type'] == 'high']
        swing_lows = [s for s in swings if s['swing_type'] == 'low']

        if trend == 1 and swing_highs:
            # 상승추세: 최근 스윙 고점 돌파 → 매수 BOS
            last_high = swing_highs[-1]['price']
            if current_close > last_high:
                return 1

        elif trend == -1 and swing_lows:
            # 하락추세: 최근 스윙 저점 돌파 → 매도 BOS
            last_low = swing_lows[-1]['price']
            if current_close < last_low:
                return -1

        return None


class CHoCHDetector:
    """
    Change of Character (CHoCH) 감지

    기존 추세의 첫 번째 반대방향 스윙 돌파를 감지.
    - 상승추세에서 최근 스윙 저점 하향 돌파 → 매도 전환 (-1)
    - 하락추세에서 최근 스윙 고점 상향 돌파 → 매수 전환 (+1)
    """

    def __init__(
        self,
        swing_lookback: int = None,
        swing_strength: int = None,
    ):
        self.swing_lookback = swing_lookback or STRATEGY_DEFAULTS.get("bos_swing_lookback", 20)
        self.swing_strength = swing_strength or STRATEGY_DEFAULTS.get("bos_swing_strength", 5)
        self.swing_detector = SwingDetector(self.swing_strength)

    def detect(self, m15_bars: pd.DataFrame) -> Optional[int]:
        """
        CHoCH 감지

        Args:
            m15_bars: M15 OHLC DataFrame

        Returns: 1 (매수 전환), -1 (매도 전환), None
        """
        if len(m15_bars) < 2 * self.swing_strength + 1:
            return None

        swings = self.swing_detector.find_swings(m15_bars)
        if len(swings) < 3:
            return None

        # BOS의 _get_trend과 동일한 로직으로 추세 판별
        recent_highs = [s for s in swings if s['swing_type'] == 'high' and s['label'] is not None]
        recent_lows = [s for s in swings if s['swing_type'] == 'low' and s['label'] is not None]

        if not recent_highs or not recent_lows:
            return None

        last_high_label = recent_highs[-1]['label']
        last_low_label = recent_lows[-1]['label']

        current_close = float(m15_bars.iloc[-1]['close'])

        swing_highs = [s for s in swings if s['swing_type'] == 'high']
        swing_lows = [s for s in swings if s['swing_type'] == 'low']

        # 상승추세(HH+HL)에서 스윙 저점 돌파 → 매도 전환
        if last_high_label == 'HH' and last_low_label == 'HL':
            if swing_lows:
                last_low = swing_lows[-1]['price']
                if current_close < last_low:
                    return -1

        # 하락추세(LH+LL)에서 스윙 고점 돌파 → 매수 전환
        elif last_high_label == 'LH' and last_low_label == 'LL':
            if swing_highs:
                last_high = swing_highs[-1]['price']
                if current_close > last_high:
                    return 1

        return None
