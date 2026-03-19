# -*- coding: utf-8 -*-
"""
MT5 시장 데이터 수집 스크립트

MT5에 연결하여 지정 심볼/기간의 M1 데이터를 CSV로 저장합니다.

사용법:
    python fetch_mt5_data.py --symbol EURUSD --months 3
    python fetch_mt5_data.py --symbol USDJPY --months 6
    python fetch_mt5_data.py  # 기본값: EURUSD 3개월

주의사항:
    - MetaTrader5 앱이 실행 중이어야 합니다.
    - .env 파일에 MT5 계정 정보가 설정되어 있어야 합니다.
    - Windows 전용 (MetaTrader5 Python API는 Windows만 지원)
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Windows 콘솔 UTF-8 출력 설정
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from config import SYMBOLS, BACKTEST_CONFIG

# 디렉토리 자동 생성
Path(BACKTEST_CONFIG["data_dir"]).mkdir(parents=True, exist_ok=True)


def load_mt5_credentials():
    """
    .env 파일에서 MT5 계정 정보 로드

    Returns:
        (login, password, server) 또는 (None, None, None)
    """
    try:
        from dotenv import load_dotenv
        import os

        load_dotenv()
        login = os.getenv("MT5_LOGIN")
        password = os.getenv("MT5_PASSWORD")
        server = os.getenv("MT5_SERVER")

        if not login or not password or not server:
            print("⚠️  .env 파일에 MT5 계정 정보가 없습니다.")
            print("   .env.example을 참고하여 .env 파일을 생성하세요.")
            return None, None, None

        return int(login), password, server

    except ImportError:
        print("⚠️  python-dotenv 패키지가 필요합니다: pip install python-dotenv")
        return None, None, None


def fetch_data(symbol: str, months: int = 3) -> bool:
    """
    MT5에서 M1 데이터를 수집하여 CSV로 저장

    Args:
        symbol: 거래 심볼 (EURUSD / USDJPY / EURJPY)
        months: 수집 기간 (개월 수)

    Returns:
        성공 여부
    """
    if symbol not in SYMBOLS:
        print(f"⚠️  지원하지 않는 심볼: {symbol}")
        print(f"   지원 심볼: {list(SYMBOLS.keys())}")
        return False

    # MT5 패키지 임포트
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("⚠️  MetaTrader5 패키지가 설치되지 않았습니다.")
        print("   pip install MetaTrader5")
        return False

    # 계정 정보 로드
    login, password, server = load_mt5_credentials()
    if login is None:
        return False

    # MT5 초기화 및 로그인
    print(f"🔌 MT5 연결 중... (서버: {server})")
    if not mt5.initialize():
        print(f"❌ MT5 초기화 실패: {mt5.last_error()}")
        return False

    if not mt5.login(login, password=password, server=server):
        print(f"❌ MT5 로그인 실패: {mt5.last_error()}")
        mt5.shutdown()
        return False

    print(f"✓ MT5 연결 성공")

    try:
        # 심볼 확인
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"❌ 심볼 {symbol}을(를) 찾을 수 없습니다.")
            return False

        # 심볼을 마켓워치에 활성화 (copy_rates 전 필수)
        if not mt5.symbol_select(symbol, True):
            print(f"❌ 심볼 활성화 실패: {mt5.last_error()}")
            return False
        print(f"✓ 심볼 {symbol} 활성화됨")

        # 진단: 소량(10개) 먼저 요청해서 API 연결 확인
        test = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 1, 10)
        if test is None or len(test) == 0:
            print(f"❌ API 진단 실패 (10개 요청): {mt5.last_error()}")
            print("   MT5 Tools → Options → Charts → Max bars in chart 확인 필요")
            return False
        print(f"✓ API 진단 통과 ({len(test)}개 바 수신)")

        # 배치 방식으로 데이터 수집 (한 번에 최대 50,000개씩)
        # forex M1: 주 5일 × 24h × 60min ≈ 7,200 바/주
        total_needed = int(months * 4.3 * 7200 * 1.2)
        batch_size = 50000
        all_rates = []

        print(f"📅 수집 기간: 최근 {months}개월 (~{total_needed:,}개 바 예상)")
        print(f"📊 {symbol} M1 데이터 수집 중 (배치 크기: {batch_size:,})...")

        batch_dfs = []  # 배치마다 DataFrame으로 변환하여 저장
        start_pos = 1   # 현재 미완성 바(0) 제외
        total_collected = 0

        while start_pos < total_needed + batch_size:
            batch = mt5.copy_rates_from_pos(
                symbol,
                mt5.TIMEFRAME_M1,
                start_pos,
                batch_size
            )
            if batch is None or len(batch) == 0:
                break  # 더 이상 데이터 없음

            # numpy structured array → DataFrame (컬럼명 보존)
            df_batch = pd.DataFrame(batch)
            batch_dfs.append(df_batch)
            total_collected += len(batch)
            print(f"   수집: {total_collected:,}개 바...", end='\r')

            if len(batch) < batch_size:
                break  # 마지막 배치
            start_pos += batch_size

        print()  # 줄바꿈

        if not batch_dfs:
            print(f"❌ 데이터 수집 실패: {mt5.last_error()}")
            return False

        # 배치 합치기 및 기간 필터링
        df_raw = pd.concat(batch_dfs, ignore_index=True)
        df_raw['time'] = pd.to_datetime(df_raw['time'], unit='s')
        cutoff = datetime.utcnow() - timedelta(days=months * 30)
        df_raw = df_raw[df_raw['time'] >= cutoff]

        if len(df_raw) == 0:
            print(f"❌ 필터링 후 데이터 없음 (기준일: {cutoff.strftime('%Y-%m-%d')})")
            return False

        # DataFrame 정리
        df = df_raw[['time', 'open', 'high', 'low', 'close', 'tick_volume']].copy()
        df = df.sort_values('time').reset_index(drop=True)

        # CSV 저장
        output_path = Path(BACKTEST_CONFIG["data_dir"]) / f"{symbol}_M1_data.csv"
        df.to_csv(output_path, index=False)

        print(f"✓ 데이터 저장 완료: {output_path}")
        print(f"   총 {len(df):,}개 M1 바 ({months}개월)")
        print(f"   기간: {df['time'].iloc[0]} ~ {df['time'].iloc[-1]}")

        return True

    finally:
        mt5.shutdown()
        print("✓ MT5 연결 해제")


def main():
    parser = argparse.ArgumentParser(
        description="MT5에서 Forex M1 데이터를 수집하여 CSV로 저장합니다."
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="EURUSD",
        choices=list(SYMBOLS.keys()),
        help=f"거래 심볼 (기본: EURUSD)"
    )
    parser.add_argument(
        "--months",
        type=int,
        default=3,
        choices=[1, 3, 6, 12],
        help="수집 기간 개월 수 (기본: 3)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print(f"  MT5 데이터 수집기")
    print(f"  심볼: {args.symbol} | 기간: {args.months}개월")
    print("=" * 60)

    success = fetch_data(args.symbol, args.months)

    if success:
        print("\n✅ 데이터 수집 완료!")
        print(f"   백테스트 실행: python backtest_ob_fvg.py")
    else:
        print("\n❌ 데이터 수집 실패")
        sys.exit(1)


if __name__ == "__main__":
    main()
