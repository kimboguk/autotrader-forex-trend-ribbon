# -*- coding: utf-8 -*-
"""
Order Block + FVG MT5 실시간 트레이더

M15 + M1 멀티스케일 실시간 거래:
- MT5 API를 통해 실시간 OHLC 수신
- M15 데이터로 방향 결정
- M1 데이터로 진입 타이밍 확인
- 자동 주문 및 포지션 관리

계정 정보는 .env 파일에 설정:
    MT5_LOGIN=계좌번호
    MT5_PASSWORD=비밀번호
    MT5_SERVER=브로커서버명
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Optional, Dict, List
import logging

# Windows 콘솔 UTF-8 출력 설정
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from ob_fvg_strategy import OrderBlockFVGStrategy
from config import SYMBOLS

# Logging 설정
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_mt5_credentials():
    """
    .env 파일 또는 환경변수에서 MT5 계정 정보 로드

    Returns:
        (login, password, server)
    Raises:
        ValueError: 필수 환경변수 누락 시
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv 없어도 환경변수에서 직접 읽기 시도

    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")

    if not login or not password or not server:
        raise ValueError(
            "MT5 계정 정보가 누락되었습니다.\n"
            ".env 파일에 다음 항목을 설정하세요:\n"
            "  MT5_LOGIN=계좌번호\n"
            "  MT5_PASSWORD=비밀번호\n"
            "  MT5_SERVER=브로커서버명"
        )

    return int(login), password, server


class MT5LiveTrader:
    """Order Block + FVG MT5 실시간 트레이더"""

    def __init__(
        self,
        symbol: str = "EURUSD",
        lot_size: float = 0.1,
        slippage: int = 5,
        poll_interval: int = 1,
        login: int = None,
        password: str = None,
        server: str = None,
    ):
        """
        MT5 트레이더 초기화

        계정 정보는 .env 파일에서 자동 로드됩니다.
        (login/password/server 직접 전달도 가능하나 권장하지 않음)

        Args:
            symbol: 거래 심볼 (기본: EURUSD)
            lot_size: 거래량 (기본: 0.1)
            slippage: 슬리피지 (포인트)
            poll_interval: 폴 간격 (초)
            login: MT5 계정 (생략 시 .env에서 로드)
            password: MT5 비밀번호 (생략 시 .env에서 로드)
            server: MT5 브로커 서버 (생략 시 .env에서 로드)
        """
        try:
            import MetaTrader5 as mt5
            self.mt5 = mt5
        except ImportError:
            raise ImportError(
                "MetaTrader5 라이브러리가 설치되어 있지 않습니다.\n"
                "pip install MetaTrader5"
            )

        # 계정 정보: 직접 전달 > .env 파일
        if login and password and server:
            self.login = login
            self.password = password
            self.server = server
        else:
            self.login, self.password, self.server = load_mt5_credentials()

        if symbol not in SYMBOLS:
            raise ValueError(f"지원하지 않는 심볼: {symbol}. 지원: {list(SYMBOLS.keys())}")

        self.symbol = symbol
        self.lot_size = lot_size
        self.slippage = slippage
        self.poll_interval = poll_interval

        # 전략 초기화 (config.py 기본값 사용)
        self.strategy = OrderBlockFVGStrategy(symbol=symbol)

        # 상태
        self.connected = False
        self.active_trade = None
        self.trade_history: List[Dict] = []
        self.last_m15_bar_time = None
        self.last_m1_bar_time = None
    
    def connect(self) -> bool:
        """MT5 연결"""
        try:
            if not self.mt5.initialize():
                logger.error(f"MT5 초기화 실패: {self.mt5.last_error()}")
                return False
            
            if not self.mt5.login(self.login, password=self.password, server=self.server):
                logger.error(f"MT5 로그인 실패: {self.mt5.last_error()}")
                return False
            
            self.connected = True
            logger.info(f"✓ MT5 연결 성공 ({self.server})")
            
            # 심볼 확인
            symbol_info = self.mt5.symbol_info(self.symbol)
            if symbol_info is None:
                logger.error(f"심볼 {self.symbol}을(를) 찾을 수 없습니다.")
                return False
            
            logger.info(f"✓ 심볼 {self.symbol} 확인됨")
            return True
        
        except Exception as e:
            logger.error(f"연결 실패: {e}")
            return False
    
    def disconnect(self):
        """MT5 연결 해제"""
        if self.connected:
            self.mt5.shutdown()
            self.connected = False
            logger.info("✓ MT5 연결 해제")
    
    def get_rates(
        self, 
        timeframe_str: str, 
        count: int = 100
    ) -> Optional[pd.DataFrame]:
        """
        실시간 OHLC 데이터 수신
        
        Args:
            timeframe_str: "M1" 또는 "M15"
            count: 필요한 바 수
        
        Returns:
            OHLC DataFrame
        """
        try:
            # Timeframe 변환
            timeframe_map = {
                'M1': self.mt5.TIMEFRAME_M1,
                'M15': self.mt5.TIMEFRAME_M15,
                'H1': self.mt5.TIMEFRAME_H1,
                'H4': self.mt5.TIMEFRAME_H4
            }
            
            if timeframe_str not in timeframe_map:
                logger.error(f"지원하지 않는 시간프레임: {timeframe_str}")
                return None
            
            timeframe = timeframe_map[timeframe_str]
            
            # 데이터 수신
            rates = self.mt5.copy_rates_from_pos(self.symbol, timeframe, 0, count)
            
            if rates is None or len(rates) == 0:
                return None
            
            # DataFrame 변환
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            return df.sort_values('time').reset_index(drop=True)
        
        except Exception as e:
            logger.error(f"데이터 수신 실패: {e}")
            return None
    
    def place_order(
        self, 
        direction: int, 
        entry_price: float, 
        sl: float, 
        tp: float
    ) -> bool:
        """
        주문 발주
        
        Args:
            direction: 1 (BUY) 또는 -1 (SELL)
            entry_price: 진입 가격
            sl: 손절가
            tp: 익절가
        
        Returns:
            성공 여부
        """
        try:
            order_type = self.mt5.ORDER_TYPE_BUY if direction == 1 else self.mt5.ORDER_TYPE_SELL
            
            request = {
                "action": self.mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": self.lot_size,
                "type": order_type,
                "price": entry_price,
                "sl": sl,
                "tp": tp,
                "deviation": self.slippage,
                "type_filling": self.mt5.ORDER_FILLING_IOC,
                "comment": "OB_FVG_STRATEGY"
            }
            
            result = self.mt5.order_send(request)
            
            if result is None or result.retcode != self.mt5.TRADE_RETCODE_DONE:
                logger.error(f"주문 실패: {getattr(result, 'comment', 'Unknown error')}")
                return False
            
            logger.info(f"✓ 주문 성공 (Ticket: {result.order})")
            return True
        
        except Exception as e:
            logger.error(f"주문 발주 실패: {e}")
            return False
    
    def get_open_position(self) -> Optional[Dict]:
        """
        현재 포지션 조회
        
        Returns:
            포지션 정보 또는 None
        """
        try:
            pos_total = self.mt5.positions_total()
            
            if pos_total <= 0:
                return None
            
            for pos in self.mt5.positions_get(symbol=self.symbol):
                return pos._asdict()
            
            return None
        
        except Exception as e:
            logger.error(f"포지션 조회 실패: {e}")
            return None
    
    def close_position(self, pos: Dict) -> bool:
        """
        포지션 청산
        
        Args:
            pos: 포지션 정보
        
        Returns:
            성공 여부
        """
        try:
            ticket = pos['ticket']
            volume = pos['volume']
            pos_type = pos['type']
            
            # 현재 틱 정보 조회
            tick = self.mt5.symbol_info_tick(self.symbol)
            if tick is None:
                logger.error("틱 정보 조회 실패")
                return False
            
            # 청산 가격 (반대 방향)
            price = tick.bid if pos_type == self.mt5.POSITION_TYPE_BUY else tick.ask
            order_type = self.mt5.ORDER_TYPE_SELL if pos_type == self.mt5.POSITION_TYPE_BUY else self.mt5.ORDER_TYPE_BUY
            
            request = {
                "action": self.mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "position": ticket,
                "volume": volume,
                "type": order_type,
                "price": price,
                "deviation": self.slippage,
                "type_filling": self.mt5.ORDER_FILLING_IOC,
                "comment": "OB_FVG_CLOSE"
            }
            
            result = self.mt5.order_send(request)
            
            if result is None or result.retcode != self.mt5.TRADE_RETCODE_DONE:
                logger.error(f"청산 실패: {getattr(result, 'comment', 'Unknown error')}")
                return False
            
            logger.info(f"✓ 포지션 청산 (Profit: {pos['profit']:.2f})")
            return True
        
        except Exception as e:
            logger.error(f"포지션 청산 실패: {e}")
            return False
    
    def update_active_trade(self, pos: Dict) -> bool:
        """활성 포지션 상태 업데이트"""
        if pos is None:
            self.active_trade = None
            return False
        
        self.active_trade = {
            'ticket': pos['ticket'],
            'direction': 1 if pos['type'] == self.mt5.POSITION_TYPE_BUY else -1,
            'entry_price': pos['price_open'],
            'entry_time': datetime.fromtimestamp(pos['time']),
            'current_price': pos['price_current'],
            'profit': pos['profit'],
            'sl': pos['sl'],
            'tp': pos['tp']
        }
        return True
    
    def run(self):
        """
        실시간 트레이딩 루프
        
        Ctrl+C로 중단 가능
        """
        if not self.connect():
            logger.error("MT5 연결 실패")
            return
        
        try:
            logger.info("🚀 트레이딩 루프 시작 (Ctrl+C로 중단)")
            logger.info("=" * 80)
            
            while True:
                try:
                    # M1, M15 데이터 수신
                    m1_df = self.get_rates('M1', 100)
                    m15_df = self.get_rates('M15', 10)
                    
                    if m1_df is None or m15_df is None or len(m1_df) < 2 or len(m15_df) < 2:
                        time.sleep(self.poll_interval)
                        continue
                    
                    # 최신 M1 바 시간
                    m1_current_time = m1_df.iloc[-1]['time']
                    
                    # 새로운 M1 바가 완성되었는지 확인
                    if self.last_m1_bar_time is not None and self.last_m1_bar_time == m1_current_time:
                        time.sleep(self.poll_interval)
                        continue
                    
                    self.last_m1_bar_time = m1_current_time
                    
                    logger.info(f"\n📊 Bar @ {m1_current_time.strftime('%Y-%m-%d %H:%M')} | "
                              f"M1: {m1_df.iloc[-1]['close']:.5f}")
                    
                    # 1. 활성 포지션 확인
                    pos = self.get_open_position()
                    self.update_active_trade(pos)
                    
                    if self.active_trade is None:
                        # 신호 탐지
                        signal = self.strategy.get_entry_signal(m15_df, m1_df)
                        
                        if signal is not None:
                            logger.info(f"📈 진입 신호: {self.strategy.format_signal(signal)}")
                            
                            # 주문 발주
                            ok = self.place_order(
                                signal['signal'],
                                signal['entry_price'],
                                signal['stop_loss'],
                                signal['take_profit']
                            )
                            
                            if ok:
                                self.active_trade = {
                                    'signal': signal['signal'],
                                    'entry_price': signal['entry_price'],
                                    'sl': signal['stop_loss'],
                                    'tp': signal['take_profit']
                                }
                    
                    else:
                        # 포지션 모니터링
                        tick = self.mt5.symbol_info_tick(self.symbol)
                        if tick is None:
                            time.sleep(self.poll_interval)
                            continue
                        
                        price = tick.bid if self.active_trade['direction'] == 1 else tick.ask
                        
                        # 프로피트 모니터링 (정보용)
                        current_pips = (
                            (price - self.active_trade['entry_price']) / self.strategy.pip_size
                            if self.active_trade['direction'] == 1
                            else (self.active_trade['entry_price'] - price) / self.strategy.pip_size
                        )
                        
                        logger.info(f"📍 포지션 모니터링 | 현재 P&L: {current_pips:+.2f}p")
                    
                    time.sleep(self.poll_interval)
                
                except Exception as e:
                    logger.error(f"루프 에러: {e}")
                    time.sleep(self.poll_interval)
        
        except KeyboardInterrupt:
            logger.info("\n⏹️  사용자 중단")
        
        finally:
            self.disconnect()


def main():
    """
    메인 실행

    계정 정보는 .env 파일에서 자동 로드됩니다:
        MT5_LOGIN=계좌번호
        MT5_PASSWORD=비밀번호
        MT5_SERVER=브로커서버명
    """
    # 심볼 설정 (EURUSD / USDJPY / EURJPY)
    SYMBOL = "EURUSD"
    LOT_SIZE = 0.01  # 실계좌 첫 실행 시 최소 로트 권장

    print("=" * 60)
    print(f"  ICT OB+FVG 실시간 트레이더")
    print(f"  심볼: {SYMBOL} | 로트: {LOT_SIZE}")
    print("=" * 60)
    print("⚠️  주의: 실계좌 자동 거래입니다. 신중하게 사용하세요.")
    print()

    trader = MT5LiveTrader(
        symbol=SYMBOL,
        lot_size=LOT_SIZE,
        slippage=5,
    )

    # 실시간 거래 시작
    trader.run()


if __name__ == '__main__':
    main()
