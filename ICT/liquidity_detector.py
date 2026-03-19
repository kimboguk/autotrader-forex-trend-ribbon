# -*- coding: utf-8 -*-
"""
ICT Liquidity Pool 감지 모듈

- Equal Highs / Equal Lows 클러스터 → 유동성 풀 식별
- Buy-Side Liquidity (BSL): 스윙 고점 클러스터 위에 매수 스톱 집중
- Sell-Side Liquidity (SSL): 스윙 저점 클러스터 아래에 매도 스톱 집중
- 유동성 스윕(sweep) 감지: 기관이 유동성을 흡수한 후 반전하는 패턴
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List

from config import STRATEGY_DEFAULTS, SYMBOLS
from structure_detector import SwingDetector


class LiquidityDetector:
    """
    유동성 풀 감지 및 진입 컨텍스트 분석

    스윙 고점/저점 중 유사한 가격대에 2개 이상 밀집된 구간을 유동성 풀로 식별.
    """

    def __init__(
        self,
        symbol: str = "EURUSD",
        lookback: int = None,
        tolerance_pips: float = None,
        swing_strength: int = None,
    ):
        """
        Args:
            symbol: 거래 심볼 (pip_size 결정)
            lookback: 유동성 탐색 범위 (M15 바)
            tolerance_pips: Equal High/Low 허용 오차 (핍)
            swing_strength: 스윙 포인트 강도
        """
        sym_cfg = SYMBOLS.get(symbol, SYMBOLS["EURUSD"])
        self.pip_size = sym_cfg["pip_size"]
        self.lookback = lookback or STRATEGY_DEFAULTS.get("liquidity_lookback", 50)
        self.tolerance = (tolerance_pips or STRATEGY_DEFAULTS.get("liquidity_tolerance_pips", 3.0)) * self.pip_size
        self.swing_strength = swing_strength or STRATEGY_DEFAULTS.get("bos_swing_strength", 5)
        self.swing_detector = SwingDetector(self.swing_strength)

    def find_liquidity_pools(self, m15_bars: pd.DataFrame) -> List[Dict]:
        """
        유동성 풀 식별

        Returns:
            [{
                'price': float,          # 풀 중심 가격
                'type': 'BSL' | 'SSL',   # Buy-Side / Sell-Side Liquidity
                'strength': int,         # 구성 스윙 포인트 수
                'last_touch_idx': int,   # 마지막 스윙의 인덱스
                'swept': bool,           # 최근 바에 의해 스윕 여부
            }]
        """
        swings = self.swing_detector.find_swings(m15_bars)
        if not swings:
            return []

        pools: List[Dict] = []

        # 스윙 고점 클러스터 → BSL
        swing_highs = [s for s in swings if s['swing_type'] == 'high']
        bsl_pools = self._cluster_prices(swing_highs, 'BSL', m15_bars)
        pools.extend(bsl_pools)

        # 스윙 저점 클러스터 → SSL
        swing_lows = [s for s in swings if s['swing_type'] == 'low']
        ssl_pools = self._cluster_prices(swing_lows, 'SSL', m15_bars)
        pools.extend(ssl_pools)

        return pools

    def _cluster_prices(
        self,
        swing_points: List[Dict],
        pool_type: str,
        m15_bars: pd.DataFrame,
    ) -> List[Dict]:
        """가격이 유사한 스윙 포인트를 클러스터링"""
        if len(swing_points) < 2:
            return []

        # 가격순 정렬
        sorted_swings = sorted(swing_points, key=lambda s: s['price'])
        pools: List[Dict] = []
        cluster: List[Dict] = [sorted_swings[0]]

        for i in range(1, len(sorted_swings)):
            if sorted_swings[i]['price'] - cluster[0]['price'] <= self.tolerance:
                cluster.append(sorted_swings[i])
            else:
                if len(cluster) >= 2:
                    pools.append(self._make_pool(cluster, pool_type, m15_bars))
                cluster = [sorted_swings[i]]

        if len(cluster) >= 2:
            pools.append(self._make_pool(cluster, pool_type, m15_bars))

        return pools

    def _make_pool(
        self,
        cluster: List[Dict],
        pool_type: str,
        m15_bars: pd.DataFrame,
    ) -> Dict:
        """클러스터로부터 유동성 풀 생성"""
        avg_price = np.mean([s['price'] for s in cluster])
        last_idx = max(s['index'] for s in cluster)

        # 스윕 확인: 최근 바의 high/low가 풀을 관통했는지
        last_bar = m15_bars.iloc[-1]
        if pool_type == 'BSL':
            swept = float(last_bar['high']) > avg_price
        else:  # SSL
            swept = float(last_bar['low']) < avg_price

        return {
            'price': float(avg_price),
            'type': pool_type,
            'strength': len(cluster),
            'last_touch_idx': last_idx,
            'swept': swept,
        }

    def check_liquidity_context(
        self,
        m15_bars: pd.DataFrame,
        entry_price: float,
        direction: int,
    ) -> Dict:
        """
        진입 시점의 유동성 컨텍스트 분석

        Args:
            m15_bars: M15 OHLC DataFrame
            entry_price: 진입 가격
            direction: 1 (BUY) 또는 -1 (SELL)

        Returns:
            {
                'score_adjustment': float,   # -1.0 ~ +1.0 (compositor 점수 조정)
                'pools_above': List[Dict],   # 진입가 위의 풀
                'pools_below': List[Dict],   # 진입가 아래의 풀
                'nearest_bsl': Optional[Dict],
                'nearest_ssl': Optional[Dict],
                'sweep_occurred': bool,      # 최근 유동성 스윕 발생
            }
        """
        pools = self.find_liquidity_pools(m15_bars)

        pools_above = [p for p in pools if p['price'] > entry_price]
        pools_below = [p for p in pools if p['price'] <= entry_price]

        nearest_bsl = min(pools_above, key=lambda p: p['price'] - entry_price) if pools_above else None
        nearest_ssl = min(pools_below, key=lambda p: entry_price - p['price']) if pools_below else None

        sweep_occurred = any(p['swept'] for p in pools)

        # 스코어 조정 로직
        score = 0.0

        if direction == 1:  # BUY
            # 아래쪽 SSL 스윕 발생 → 매수 유리 (기관 매집 후 상승)
            ssl_swept = any(p['swept'] for p in pools_below if p['type'] == 'SSL')
            if ssl_swept:
                score += 0.5

            # 가까운 BSL 있으면 → TP 타겟으로 유리
            if nearest_bsl and not nearest_bsl['swept']:
                dist_pips = (nearest_bsl['price'] - entry_price) / self.pip_size
                if dist_pips > 5:
                    score += 0.3

            # BSL 바로 앞에서 매수 진입 → 불리 (유동성 함정)
            if nearest_bsl and not nearest_bsl['swept']:
                dist_pips = (nearest_bsl['price'] - entry_price) / self.pip_size
                if dist_pips < 3:
                    score -= 0.5

        else:  # SELL
            # 위쪽 BSL 스윕 발생 → 매도 유리
            bsl_swept = any(p['swept'] for p in pools_above if p['type'] == 'BSL')
            if bsl_swept:
                score += 0.5

            # 가까운 SSL 있으면 → TP 타겟으로 유리
            if nearest_ssl and not nearest_ssl['swept']:
                dist_pips = (entry_price - nearest_ssl['price']) / self.pip_size
                if dist_pips > 5:
                    score += 0.3

            # SSL 바로 앞에서 매도 진입 → 불리
            if nearest_ssl and not nearest_ssl['swept']:
                dist_pips = (entry_price - nearest_ssl['price']) / self.pip_size
                if dist_pips < 3:
                    score -= 0.5

        # 클램프
        score = max(-1.0, min(1.0, score))

        return {
            'score_adjustment': score,
            'pools_above': pools_above,
            'pools_below': pools_below,
            'nearest_bsl': nearest_bsl,
            'nearest_ssl': nearest_ssl,
            'sweep_occurred': sweep_occurred,
        }
