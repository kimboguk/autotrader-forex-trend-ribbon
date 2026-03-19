# -*- coding: utf-8 -*-
"""
Order Block + Fair Value Gap (FVG) 거래 전략
- Timeframe: M15 (방향 결정) + M1 (진입 타이밍)
- SL: 직전 봉의 저/고가 ± sl_buffer_pips
- TP: SL 기반 손익비 배수
- 거래 비용: 스프레드 + 수수료 (심볼별 설정)
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List

from config import SYMBOLS, STRATEGY_DEFAULTS


class OrderBlockFVGStrategy:
    """Order Block + FVG 전략 클래스"""

    def __init__(
        self,
        symbol: str = "EURUSD",
        risk_reward_ratio: float = None,
        spread_pips: float = None,
        commission_pips: float = None,
        sl_buffer_pips: float = None,
        allowed_hours: set = None,
        require_fvg_confirm: bool = None,
        sl_timeframe: str = None,
    ):
        """
        전략 초기화

        Args:
            symbol: 거래 심볼 (EURUSD / USDJPY / EURJPY / XAUUSD)
            risk_reward_ratio: 손익비 (기본: config.py의 STRATEGY_DEFAULTS)
            spread_pips: 스프레드 (기본: config.py의 SYMBOLS 값)
            commission_pips: 거래 비용 (기본: config.py의 SYMBOLS 값)
            sl_buffer_pips: SL 버퍼 (기본: config.py의 SYMBOLS 값)
            allowed_hours: 거래 허용 시간 (기본: {0,1,8,9,16,17})
            require_fvg_confirm: OB+FVG 동시 확인 여부 (기본: True)
            sl_timeframe: SL 기준 타임프레임 ("M1" 또는 "M15", 기본: config.py)
        """
        sym_cfg = SYMBOLS.get(symbol, SYMBOLS["EURUSD"])

        self.symbol = symbol
        self.pip_size = sym_cfg["pip_size"]
        self.risk_reward_ratio = risk_reward_ratio if risk_reward_ratio is not None else STRATEGY_DEFAULTS["risk_reward_ratio"]
        self.spread_pips = spread_pips if spread_pips is not None else sym_cfg["spread_pips"]
        self.commission_pips = commission_pips if commission_pips is not None else sym_cfg["commission_pips"]
        self.total_cost_pips = self.spread_pips + self.commission_pips  # 왕복 비용 (스프레드 + 커미션)
        self.sl_buffer_pips = sl_buffer_pips if sl_buffer_pips is not None else sym_cfg["sl_buffer_pips"]
        self.allowed_hours = allowed_hours if allowed_hours is not None else STRATEGY_DEFAULTS["allowed_hours"]
        self.require_fvg_confirm = require_fvg_confirm if require_fvg_confirm is not None else STRATEGY_DEFAULTS["require_fvg_confirm"]
        self.sl_timeframe = sl_timeframe if sl_timeframe is not None else STRATEGY_DEFAULTS.get("sl_timeframe", "M1")
    
    def is_trading_hour(self, timestamp: pd.Timestamp) -> bool:
        """거래 허용 시간인지 확인"""
        return timestamp.hour in self.allowed_hours
    
    def detect_order_block(
        self, 
        prev_bar: Dict, 
        curr_bar: Dict
    ) -> Optional[int]:
        """
        Order Block 감지
        
        직전 봉과 현재 봉의 방향 전환 + 극값 돌파
        
        Returns:
            1: BUY signal (상승 전환), -1: SELL signal (하강 전환), None: no signal
        """
        prev_close = prev_bar['close']
        prev_open = prev_bar['open']
        prev_high = prev_bar['high']
        prev_low = prev_bar['low']
        
        curr_close = curr_bar['close']
        curr_open = curr_bar['open']
        curr_high = curr_bar['high']
        curr_low = curr_bar['low']
        
        # 직전 봉이 하강, 현재 봉이 상승 → BUY signal
        # 조건: 현재 봉의 저가 < 직전 봉의 저가 (돌파)
        if prev_close < prev_open and curr_close > curr_open:
            if curr_low < prev_low:
                return 1  # BUY
        
        # 직전 봉이 상승, 현재 봉이 하강 → SELL signal
        # 조건: 현재 봉의 고가 > 직전 봉의 고가 (돌파)
        if prev_close > prev_open and curr_close < curr_open:
            if curr_high > prev_high:
                return -1  # SELL
        
        return None
    
    def detect_fvg(
        self, 
        bars: pd.DataFrame, 
        index: int
    ) -> Optional[int]:
        """
        Fair Value Gap (FVG) 감지
        
        3개 봉 갭 확인:
        - N과 N+2가 반대 방향
        - N+1에서 갭이 생김
        
        Args:
            bars: OHLC 데이터프레임
            index: 현재 봉 인덱스
        
        Returns:
            1: UP FVG (매수 기회), -1: DOWN FVG (매도 기회), None: no FVG
        """
        if index < 2 or index >= len(bars):
            return None
        
        n = bars.iloc[index - 2]      # N
        n1 = bars.iloc[index - 1]     # N+1
        n2 = bars.iloc[index]         # N+2
        
        n_close = n['close']
        n_open = n['open']
        n1_close = n1['close']
        n1_open = n1['open']
        n2_close = n2['close']
        n2_open = n2['open']

        # 각 봉의 방향
        n_is_up = n_close > n_open
        n1_is_up = n1_close > n1_open   # N+1 상승 여부 (n1_is_down과 분리)
        n2_is_up = n2_close > n2_open

        # DOWN FVG (매도 기회): N(상승), N+1(하락), N+2(상승) + N.high < N+2.low (상방 갭)
        if n_is_up and (not n1_is_up) and n2_is_up:
            if n['high'] < n2['low']:
                return -1  # DOWN FVG

        # UP FVG (매수 기회): N(하락), N+1(상승), N+2(하락) + N.low > N+2.high (하방 갭)
        elif (not n_is_up) and n1_is_up and (not n2_is_up):
            if n['low'] > n2['high']:
                return 1  # UP FVG

        return None
    
    def get_entry_signal(
        self, 
        m15_bars: pd.DataFrame, 
        m1_bars: pd.DataFrame
    ) -> Optional[Dict]:
        """
        진입 신호 생성
        
        M15에서 방향 결정 → M1에서 타이밍 확인
        
        Args:
            m15_bars: 15분봉 OHLC 데이터
            m1_bars: 1분봉 OHLC 데이터
        
        Returns:
            {
                'signal': 1 (BUY) or -1 (SELL),
                'entry_time': datetime,
                'entry_price': float,
                'stop_loss': float,
                'take_profit': float,
                'risk_pips': float
            }
        """
        if len(m15_bars) < 2 or len(m1_bars) < 2:
            return None
        
        # M15 최신 봉
        m15_curr = m15_bars.iloc[-1]
        m15_prev = m15_bars.iloc[-2]
        m15_time = m15_curr['time']
        
        # 거래 허용 시간 확인
        if not self.is_trading_hour(m15_time):
            return None
        
        # Order Block 감지 (M15)
        ob_signal = self.detect_order_block(
            m15_prev.to_dict(),
            m15_curr.to_dict()
        )
        
        if ob_signal is None:
            return None
        
        # M1 최신 봉에서 FVG 확인
        m1_fvg = self.detect_fvg(m1_bars, len(m1_bars) - 1)
        
        # OB+FVG 신호 연동
        if self.require_fvg_confirm:
            # FVG 확인 필수: OB와 FVG 방향이 일치해야 진입
            if m1_fvg is None or ob_signal != m1_fvg:
                return None
            signal = ob_signal
        else:
            # FVG 확인 불필요: OB 신호만으로 진입 (허위 신호 증가 주의)
            signal = ob_signal
        entry_time = m1_bars.iloc[-1]['time']
        entry_price = m1_bars.iloc[-1]['close']
        
        # SL 계산: sl_timeframe에 따라 M1 또는 M15 직전 봉 기준
        if self.sl_timeframe == "M15" and len(m15_bars) >= 2:
            sl_ref_bar = m15_bars.iloc[-2]  # M15 직전 봉 (넓은 SL, 노이즈 방지)
        else:
            sl_ref_bar = m1_bars.iloc[-2]   # M1 직전 봉 (기존 동작)

        if signal == 1:  # BUY
            stop_loss = sl_ref_bar['low'] - self.sl_buffer_pips
            risk_pips = (entry_price - stop_loss) / self.pip_size
        else:  # SELL
            stop_loss = sl_ref_bar['high'] + self.sl_buffer_pips
            risk_pips = (stop_loss - entry_price) / self.pip_size

        # SL이 진입가 반대편에 있는 invalid 설정 거부
        if risk_pips <= 0:
            return None

        # 스프레드 인식 필터: risk_pips가 총 비용의 2배 미만이면 거부
        # (스프레드+커미션 대비 리스크가 너무 작으면 실질 RR이 왜곡됨)
        if risk_pips < self.total_cost_pips * 2:
            return None

        # TP 계산: 손익비 기반
        profit_pips = risk_pips * self.risk_reward_ratio
        
        if signal == 1:  # BUY
            take_profit = entry_price + (profit_pips * self.pip_size)
        else:  # SELL
            take_profit = entry_price - (profit_pips * self.pip_size)
        
        return {
            'signal': signal,
            'entry_time': entry_time,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_pips': risk_pips,
            'profit_pips': profit_pips
        }
    
    def calculate_pnl(
        self,
        entry_price: float,
        exit_price: float,
        signal: int,
        risk_pips: float
    ) -> Tuple[float, float, str]:
        """
        손익 계산
        
        Returns:
            (gross_pips, net_pips, exit_type)
        """
        if signal == 1:  # BUY
            gross_pips = (exit_price - entry_price) / self.pip_size
        else:  # SELL
            gross_pips = (entry_price - exit_price) / self.pip_size
        
        # 거래 비용 차감 (편도)
        net_pips = gross_pips - self.total_cost_pips
        
        # 손익 판정
        if net_pips > 0:
            exit_type = 'TP'
        elif net_pips < 0:
            exit_type = 'SL'
        else:
            exit_type = 'BREAK'
        
        return gross_pips, net_pips, exit_type
    
    def format_signal(self, sig: Dict) -> str:
        """신호를 문자열로 포맷"""
        if sig is None:
            return "None"
        
        direction = "BUY" if sig['signal'] == 1 else "SELL"
        return (
            f"{direction} @ {sig['entry_price']:.5f} | "
            f"SL {sig['stop_loss']:.5f} | "
            f"TP {sig['take_profit']:.5f} | "
            f"Risk {sig['risk_pips']:.2f}p"
        )
