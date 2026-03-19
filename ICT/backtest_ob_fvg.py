# -*- coding: utf-8 -*-
"""
Order Block + FVG 백테스팅 엔진

M15 + M1 데이터를 사용한 완전 백테스팅:
- 1분봉 M1 데이터에서 신호 생성 (M15는 방향 결정)
- 실시간 청산 규칙 적용
- 성과 분석
"""

import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

# Windows 콘솔 UTF-8 출력 설정
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from ob_fvg_strategy import OrderBlockFVGStrategy
from config import BACKTEST_CONFIG, STRATEGY_DEFAULTS

warnings.filterwarnings('ignore')

# 디렉토리 자동 생성
Path(BACKTEST_CONFIG["data_dir"]).mkdir(parents=True, exist_ok=True)
Path(BACKTEST_CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)


class BacktestEngine:
    """Order Block + FVG 백테스팅 엔진"""

    def __init__(
        self,
        strategy: OrderBlockFVGStrategy,
        max_bars_per_trade: int = None,
        m15_lookback: int = None,
        m1_lookback: int = None,
    ):
        """
        Args:
            strategy: OrderBlockFVGStrategy 인스턴스
            max_bars_per_trade: 최대 홀딩 바 수 (기본: config.py의 STRATEGY_DEFAULTS)
            m15_lookback: M15 윈도우 크기 (기본: config.py)
            m1_lookback: M1 윈도우 크기 (기본: config.py)
        """
        self.strategy = strategy
        self.max_bars_per_trade = max_bars_per_trade if max_bars_per_trade is not None else STRATEGY_DEFAULTS["max_bars_per_trade"]
        self.m15_lookback = m15_lookback if m15_lookback is not None else STRATEGY_DEFAULTS.get("m15_lookback", 50)
        self.m1_lookback = m1_lookback if m1_lookback is not None else STRATEGY_DEFAULTS.get("m1_lookback", 3)
        self.trades: List[Dict] = []
        self.pip_size = strategy.pip_size  # 심볼별 pip_size 사용
    
    def load_data(self, csv_file: str, timeframe: str = "M1") -> pd.DataFrame:
        """
        CSV 데이터 로드
        
        예상 컬럼: time, open, high, low, close, tick_volume
        """
        df = pd.read_csv(csv_file, parse_dates=['time'])
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
        return df
    
    def aggregate_to_m15(self, m1_df: pd.DataFrame) -> pd.DataFrame:
        """M1 데이터를 M15로 변환"""
        m1_df = m1_df.copy()
        
        # 15분 단위로 그룹화
        m1_df['m15_time'] = m1_df['time'].dt.floor('15min')
        
        m15_df = m1_df.groupby('m15_time').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'tick_volume': 'sum'
        }).reset_index()
        
        m15_df.rename(columns={'m15_time': 'time'}, inplace=True)
        m15_df['time'] = pd.to_datetime(m15_df['time'])
        
        return m15_df
    
    def run_backtest(self, m1_data) -> Dict:
        """
        백테스팅 실행

        Args:
            m1_data: M1 데이터 CSV 파일 경로 (str) 또는 DataFrame

        Returns:
            성과 분석 결과
        """
        if isinstance(m1_data, pd.DataFrame):
            m1_df = m1_data.copy()
            m1_df['time'] = pd.to_datetime(m1_df['time'])
            m1_df = m1_df.sort_values('time').reset_index(drop=True)
            print(f"✓ Loaded {len(m1_df):,} M1 bars (DataFrame)")
        else:
            print(f"Loading data from {m1_data}...")
            m1_df = self.load_data(m1_data)
            print(f"✓ Loaded {len(m1_df):,} M1 bars")

        # M15 생성
        m15_df = self.aggregate_to_m15(m1_df)
        print(f"✓ Generated {len(m15_df):,} M15 bars")

        self.trades = []
        active_trade = None

        # M15 시간 인덱스 (searchsorted용 numpy 배열)
        m15_times = m15_df['time'].values  # numpy datetime64 배열

        total_bars = len(m1_df)
        progress_step = max(1, total_bars // 20)  # 5% 단위 진행률

        print(f"\nStarting backtest ({total_bars:,} bars)...")
        print("=" * 80)

        # 각 M1 봉마다 실행 (고정 윈도우, O(1) per iteration)
        for i in range(self.m1_lookback, total_bars):
            m1_current = m1_df.iloc[i]
            m1_time = m1_current['time']

            # 진행률 표시 (5% 단위)
            if i % progress_step == 0:
                pct = i / total_bars * 100
                print(f"  [{pct:5.1f}%] {m1_time.strftime('%Y-%m-%d')} | trades: {len(self.trades)}")

            # M15 인덱스 탐색: O(log m) binary search
            m15_floor = np.datetime64(m1_time.floor('15min'))
            m15_idx = np.searchsorted(m15_times, m15_floor, side='right') - 1

            if m15_idx < 1:  # 최소 2개 M15 바 필요
                continue

            # M15 고정 윈도우 슬라이스 (복사 없음, O(1))
            m15_start = max(0, m15_idx - self.m15_lookback + 1)
            m15_bars = m15_df.iloc[m15_start:m15_idx + 1]

            # M1 고정 윈도우 슬라이스 (복사 없음, O(1))
            m1_start = max(0, i - self.m1_lookback + 1)
            m1_bars = m1_df.iloc[m1_start:i + 1]

            # 1. 활성 포지션이 없으면 신호 탐지
            if active_trade is None:
                # compositor가 있으면 get_composite_signal, 없으면 get_entry_signal
                if hasattr(self.strategy, 'get_composite_signal'):
                    signal = self.strategy.get_composite_signal(m15_bars, m1_bars)
                else:
                    signal = self.strategy.get_entry_signal(m15_bars, m1_bars)

                if signal is not None:
                    active_trade = {
                        'entry_time': signal['entry_time'],
                        'entry_price': signal['entry_price'],
                        'signal': signal['signal'],
                        'stop_loss': signal['stop_loss'],
                        'take_profit': signal['take_profit'],
                        'risk_pips': signal['risk_pips'],
                        'entry_bar_idx': i,
                        'bars_held': 0
                    }
                    print(f"\n  ENTRY @ {m1_time.strftime('%Y-%m-%d %H:%M')} | "
                          f"{self.strategy.format_signal(signal)}")

            # 2. 활성 포지션이 있으면 청산 규칙 확인
            if active_trade is not None:
                active_trade['bars_held'] = i - active_trade['entry_bar_idx']
                exit_price = m1_current['close']
                high = m1_current['high']
                low = m1_current['low']

                exit_signal = None

                # SL 확인 (우선순위 1)
                if active_trade['signal'] == 1:  # BUY
                    if low <= active_trade['stop_loss']:
                        exit_price = active_trade['stop_loss']
                        exit_signal = 'SL'
                else:  # SELL
                    if high >= active_trade['stop_loss']:
                        exit_price = active_trade['stop_loss']
                        exit_signal = 'SL'

                # TP 확인 (우선순위 2)
                if exit_signal is None:
                    if active_trade['signal'] == 1:  # BUY
                        if high >= active_trade['take_profit']:
                            exit_price = active_trade['take_profit']
                            exit_signal = 'TP'
                    else:  # SELL
                        if low <= active_trade['take_profit']:
                            exit_price = active_trade['take_profit']
                            exit_signal = 'TP'

                # 최대 홀딩 바 초과 (우선순위 3)
                if exit_signal is None and active_trade['bars_held'] >= self.max_bars_per_trade:
                    exit_signal = 'TIMEOUT'

                # 청산 실행
                if exit_signal is not None:
                    gross_pips, net_pips, _ = self.strategy.calculate_pnl(
                        active_trade['entry_price'],
                        exit_price,
                        active_trade['signal'],
                        active_trade['risk_pips']
                    )

                    trade_record = {
                        'entry_time': active_trade['entry_time'],
                        'exit_time': m1_time,
                        'direction': 'BUY' if active_trade['signal'] == 1 else 'SELL',
                        'entry_price': active_trade['entry_price'],
                        'exit_price': exit_price,
                        'stop_loss': active_trade['stop_loss'],
                        'take_profit': active_trade['take_profit'],
                        'risk_pips': active_trade['risk_pips'],
                        'gross_pips': gross_pips,
                        'net_pips': net_pips,
                        'bars_held': active_trade['bars_held'],
                        'exit_reason': exit_signal,
                        'profit_loss': 'WIN' if net_pips > 0 else 'LOSS'
                    }

                    self.trades.append(trade_record)

                    print(f"  EXIT  @ {m1_time.strftime('%Y-%m-%d %H:%M')} | "
                          f"{exit_signal} @ {exit_price:.5f} | "
                          f"P&L: {net_pips:+.2f}p (gross {gross_pips:+.2f}p)")

                    active_trade = None

        print("\n" + "=" * 80)
        print(f"✓ Backtest completed | Total trades: {len(self.trades)}")

        # 성과 분석
        results = self.analyze_performance()

        return results
    
    def analyze_performance(self) -> Dict:
        """성과 분석"""
        if len(self.trades) == 0:
            print("\n⚠️  No trades executed")
            return {}
        
        df_trades = pd.DataFrame(self.trades)
        
        # 기본 지표
        total_trades = len(df_trades)
        wins = (df_trades['profit_loss'] == 'WIN').sum()
        losses = (df_trades['profit_loss'] == 'LOSS').sum()
        win_rate = wins / total_trades * 100 if total_trades > 0 else 0
        
        # 수익성
        total_net_pips = df_trades['net_pips'].sum()
        cumulative_pips = df_trades['net_pips'].cumsum()
        
        # 손익비
        tp_trades = df_trades[df_trades['exit_reason'] == 'TP']
        sl_trades = df_trades[df_trades['exit_reason'] == 'SL']
        
        if len(tp_trades) > 0:
            avg_win_pips = tp_trades['net_pips'].mean()
        else:
            avg_win_pips = 0
        
        if len(sl_trades) > 0:
            avg_loss_pips = abs(sl_trades['net_pips'].mean())
        else:
            avg_loss_pips = 0
        
        rrr = avg_win_pips / avg_loss_pips if avg_loss_pips > 0 else (
            np.inf if avg_win_pips > 0 else 0
        )
        
        # Drawdown
        max_cumulative = cumulative_pips.cummax()
        drawdown = cumulative_pips - max_cumulative
        max_drawdown = drawdown.min()
        
        # 시간대별 분석
        df_trades['entry_hour'] = df_trades['entry_time'].dt.hour
        hourly_stats = df_trades.groupby('entry_hour').agg({
            'net_pips': ['count', 'sum', 'mean'],
            'profit_loss': lambda x: (x == 'WIN').sum() / len(x) * 100
        }).round(2)
        
        results = {
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate_%': round(win_rate, 2),
            'total_net_pips': round(total_net_pips, 2),
            'avg_win_pips': round(avg_win_pips, 2),
            'avg_loss_pips': round(avg_loss_pips, 2),
            'risk_reward_ratio': round(rrr, 2),
            'max_drawdown_pips': round(max_drawdown, 2),
            'tp_trades': len(tp_trades),
            'sl_trades': len(sl_trades),
            'timeout_trades': (df_trades['exit_reason'] == 'TIMEOUT').sum(),
            'avg_bars_held': round(df_trades['bars_held'].mean(), 1)
        }
        
        # 출력
        self._print_results(results, df_trades)
        
        return results
    
    def _print_results(self, results: Dict, df_trades: pd.DataFrame):
        """결과 출력"""
        print("\n" + "=" * 80)
        print("📈 PERFORMANCE SUMMARY")
        print("=" * 80)
        print(f"Total Trades:        {results['total_trades']}")
        print(f"Wins/Losses:         {results['wins']} / {results['losses']}")
        print(f"Win Rate:            {results['win_rate_%']:.2f}%")
        print(f"Total Net Pips:      {results['total_net_pips']:+.2f}p")
        print(f"Average Win:         {results['avg_win_pips']:+.2f}p")
        print(f"Average Loss:        {results['avg_loss_pips']:+.2f}p")
        print(f"Risk:Reward Ratio:   {results['risk_reward_ratio']:.2f}")
        print(f"Max Drawdown:        {results['max_drawdown_pips']:.2f}p")
        print(f"TP/SL/Timeout:       {results['tp_trades']}/{results['sl_trades']}/{results['timeout_trades']}")
        print(f"Avg Bars Held:       {results['avg_bars_held']}")
        print("=" * 80)
    
    def save_trades_csv(self, output_file: str = None):
        """거래 결과 CSV로 저장"""
        if len(self.trades) == 0:
            print("⚠️  No trades to save")
            return

        if output_file is None:
            symbol = self.strategy.symbol
            output_file = str(Path(BACKTEST_CONFIG["output_dir"]) / f"{symbol}_trades_result.csv")

        df_trades = pd.DataFrame(self.trades)
        df_trades.to_csv(output_file, index=False)
        print(f"✓ Trades saved to {output_file}")

    def plot_equity_curve(self, output_file: str = None):
        """자산 곡선 플롯 (선택)"""
        try:
            import matplotlib.pyplot as plt

            if len(self.trades) == 0:
                return

            if output_file is None:
                symbol = self.strategy.symbol
                output_file = str(Path(BACKTEST_CONFIG["output_dir"]) / f"{symbol}_equity_curve.png")

            df_trades = pd.DataFrame(self.trades)
            df_trades['cumulative_pips'] = df_trades['net_pips'].cumsum()

            plt.figure(figsize=(14, 6))
            plt.plot(df_trades['cumulative_pips'], linewidth=2)
            plt.xlabel('Trade Number')
            plt.ylabel('Cumulative Pips')
            plt.title(f'Equity Curve - {self.strategy.symbol} (Cumulative Net Pips)')
            plt.grid(True, alpha=0.3)
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"✓ Equity curve saved to {output_file}")
            plt.close()
        except ImportError:
            print("⚠️  Matplotlib not available for plotting")


def main():
    """메인 실행 함수"""

    parser = argparse.ArgumentParser(description="OB+FVG 백테스트")
    parser.add_argument(
        "--symbol",
        type=str,
        default="EURUSD",
        choices=["EURUSD", "USDJPY", "EURJPY", "XAUUSD"],
        help="거래 심볼 (기본: EURUSD)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="csv",
        choices=["csv", "db"],
        help="데이터 소스 (csv: CSV 파일, db: PostgreSQL)"
    )
    parser.add_argument("--start", type=str, default=None, help="시작일 (YYYY-MM-DD, db 모드)")
    parser.add_argument("--end", type=str, default=None, help="종료일 (YYYY-MM-DD, db 모드)")
    parser.add_argument("--compositor", action="store_true", help="SignalCompositor 사용")
    parser.add_argument("--rr", type=float, default=None, help="손익비 오버라이드")
    parser.add_argument("--sl-tf", type=str, default=None, choices=["M1", "M15"], help="SL 타임프레임 오버라이드")
    args = parser.parse_args()
    SYMBOL = args.symbol

    # 전략 파라미터 오버라이드
    strategy_kwargs = {}
    if args.rr is not None:
        strategy_kwargs['risk_reward_ratio'] = args.rr
    if args.sl_tf is not None:
        strategy_kwargs['sl_timeframe'] = args.sl_tf

    # 전략 초기화 (compositor 또는 기본 전략)
    if args.compositor:
        from signal_compositor import SignalCompositor
        strategy = SignalCompositor(symbol=SYMBOL, **strategy_kwargs)
        print(f"Mode: SignalCompositor (threshold={strategy.threshold})")
    else:
        strategy = OrderBlockFVGStrategy(symbol=SYMBOL, **strategy_kwargs)
        print(f"Mode: OB+FVG only")

    print(f"Symbol: {SYMBOL} | RR: {strategy.strategy.risk_reward_ratio if hasattr(strategy, 'strategy') else strategy.risk_reward_ratio}"
          f" | SL: {strategy.strategy.sl_timeframe if hasattr(strategy, 'strategy') else strategy.sl_timeframe}")

    # 백테스팅 엔진
    engine = BacktestEngine(strategy)

    if args.source == "db":
        # PostgreSQL에서 데이터 로드
        from db import Database
        db = Database()

        start = datetime.strptime(args.start, "%Y-%m-%d") if args.start else None
        end = datetime.strptime(args.end, "%Y-%m-%d") if args.end else None

        print(f"Loading {SYMBOL} from PostgreSQL...")
        if start:
            print(f"  Start: {args.start}")
        if end:
            print(f"  End: {args.end}")

        m1_df = db.query_ohlcv(SYMBOL, start=start, end=end)

        if len(m1_df) == 0:
            print("No data found in DB. Run init_db.py --import-csv or fetch_histdata.py first.")
            return

        # DB 데이터를 DataFrame으로 직접 전달 (CSV 중간 저장 불필요)
        results = engine.run_backtest(m1_df)

    else:
        # CSV 파일에서 데이터 로드 (기존 동작)
        data_file = str(Path(BACKTEST_CONFIG["data_dir"]) / f"{SYMBOL}_M1_data.csv")

        if not Path(data_file).exists():
            print(f"Data file not found: {data_file}")
            print(f"\nUse: python fetch_mt5_data.py --symbol {SYMBOL} --months 3")
            print(f"Or:  python backtest_ob_fvg.py --symbol {SYMBOL} --source db")
            return

        results = engine.run_backtest(data_file)

    # 결과 저장
    engine.save_trades_csv()
    engine.plot_equity_curve()

    print("\n✓ Backtest completed!")


def create_sample_data(bars: int = 1000) -> pd.DataFrame:
    """테스트용 샘플 데이터 생성"""
    times = pd.date_range('2024-01-01', periods=bars, freq='1min')
    
    np.random.seed(42)
    prices = 1.0800 + np.cumsum(np.random.randn(bars) * 0.00005)
    
    data = []
    for i, t in enumerate(times):
        open_price = prices[i]
        close_price = prices[i] + np.random.randn() * 0.00003
        high_price = max(open_price, close_price) + abs(np.random.randn()) * 0.00005
        low_price = min(open_price, close_price) - abs(np.random.randn()) * 0.00005
        volume = np.random.randint(100, 1000)
        
        data.append({
            'time': t,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'tick_volume': volume
        })
    
    return pd.DataFrame(data)


if __name__ == '__main__':
    main()
