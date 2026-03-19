# 📊 Order Block + FVG 거래 전략 - 사용 설명서

## 🎯 전략 개요

**Order Block + Fair Value Gap (FVG)** 기반 자동거래 시스템
- **Timeframe**: M15 (방향) + M1 (진입)
- **신호**: Order Block 감지 + FVG 확인
- **SL**: 직전 봉의 저/고가 ± 0.0001 (0.1pip)
- **TP**: SL 기반 손익비 10배 (초기값)
- **거래 비용**: 0.7 pips (스프레드 0.4 + 수수료 0.3)
- **대상**: EURUSD (확장 가능)

---

## 📁 파일 구조

```
/mnt/user-data/outputs/
├── ob_fvg_strategy.py          # 핵심 전략 로직
├── backtest_ob_fvg.py          # 백테스팅 엔진
├── live_trader_ob_fvg.py       # MT5 실시간 트레이더
├── generate_sample_data.py     # 샘플 데이터 생성
├── README_OB_FVG.md            # 이 파일
└── trades_result.csv           # 백테스트 결과 (자동 생성)
```

---

## 🚀 빠른 시작 (3단계)

### **1단계: 백테스팅 (권장)**

#### 방법 A: 샘플 데이터로 테스트 (즉시)

```bash
cd /mnt/user-data/outputs
python backtest_ob_fvg.py
```

**결과:**
- 콘솔에 거래 로그 출력
- `trades_result.csv` 생성 (거래 상세)
- `equity_curve.png` 생성 (수익 곡선)

#### 방법 B: 자신의 데이터로 테스트 (권장)

```bash
# 1. 6개월 M1 데이터 준비
# 형식: CSV (time, open, high, low, close, tick_volume)

# 2. 파일 경로 지정
# /mnt/user-data/inputs/EURUSD_M1_data.csv

# 3. 백테스팅 실행
python backtest_ob_fvg.py
```

---

### **2단계: 결과 분석**

백테스트 결과 확인:

```
📈 PERFORMANCE SUMMARY
==================================================
Total Trades:        1,234
Wins/Losses:         589 / 645
Win Rate:            47.74%
Total Net Pips:      +1,245.67p
Average Win:         +3.45p
Average Loss:        -2.10p
Risk:Reward Ratio:   1.64
Max Drawdown:        -156.23p
TP/SL/Timeout:       589/645/0
Avg Bars Held:       12.5
==================================================
```

**해석:**
- ✅ 총 수익 > 0 → 전략이 수익성 있음
- ✅ 손익비 1.6 이상 → 양호한 수익:손실
- ⚠️ 승률 < 50% → 정상 (손익비로 보정)
- ⚠️ 최대낙폭 큼 → 포지션사이징 검토 필요

---

### **3단계: MT5 실시간 거래 (선택)**

```bash
# 1. live_trader_ob_fvg.py 편집
# - LOGIN: 본인의 MT5 계정
# - PASSWORD: 본인의 비밀번호
# - SERVER: 브로커 서버명

# 2. 실시간 거래 시작
python live_trader_ob_fvg.py
```

**주의사항:**
- ⚠️ 데모 계정으로 먼저 테스트
- ⚠️ 포지션사이징 검토 필수 (현재: 0.1 lot)
- ⚠️ 하루 24시간 모니터링 필요

---

## 📊 전략 설정 변경

### Order Block 매개변수

`ob_fvg_strategy.py` 수정:

```python
# 감지 매개변수 추가 가능
MIN_OB_STRENGTH = 2  # Order Block 강도 (개발 예정)
FVG_MIN_PIPS = 5      # FVG 최소 크기 (개발 예정)
```

### 손익비 변경

```python
# backtest_ob_fvg.py 또는 live_trader_ob_fvg.py

strategy = OrderBlockFVGStrategy(
    risk_reward_ratio=15.0,  # 10 → 15로 변경
    # ... 나머지 설정
)
```

### 거래 시간 필터

```python
# 기본값: {0, 1, 8, 9, 16, 17} (런던/미국 세션)
# 변경 예:

strategy = OrderBlockFVGStrategy(
    allowed_hours={8, 9, 10, 11, 20, 21}  # 런던 세션만
)
```

---

## 📈 백테스팅 결과 분석

### CSV 결과 파일 구조

`trades_result.csv`:

```
entry_time,exit_time,direction,entry_price,exit_price,stop_loss,take_profit,risk_pips,gross_pips,net_pips,bars_held,exit_reason,profit_loss
2024-01-15 08:00,2024-01-15 08:45,BUY,1.08234,1.08456,1.08190,1.08565,26.4,22.2,20.2,45,TP,WIN
2024-01-15 09:00,2024-01-15 09:15,SELL,1.08456,1.08400,1.08500,1.08260,25.0,-5.6,-7.6,15,SL,LOSS
...
```

### Python으로 추가 분석

```python
import pandas as pd

df = pd.read_csv('trades_result.csv')

# 시간대별 승률
hourly = df.groupby(df['entry_time'].dt.hour).agg({
    'profit_loss': lambda x: (x == 'WIN').sum() / len(x) * 100,
    'net_pips': 'sum'
})

print(hourly)

# 월요일 vs 금요일
df['weekday'] = pd.to_datetime(df['entry_time']).dt.day_name()
weekday_stats = df.groupby('weekday')['net_pips'].agg(['sum', 'count'])

print(weekday_stats)
```

---

## ⚙️ 주요 클래스 및 함수

### OrderBlockFVGStrategy

```python
strategy = OrderBlockFVGStrategy(
    risk_reward_ratio=10.0,
    spread_pips=0.4,
    commission_pips=0.3,
    sl_buffer_pips=0.0001,
    allowed_hours={0, 1, 8, 9, 16, 17}
)

# Order Block 감지
signal = strategy.detect_order_block(prev_bar, curr_bar)
# Returns: 1 (BUY), -1 (SELL), or None

# FVG 감지
fvg = strategy.detect_fvg(bars_df, index)
# Returns: 1 (UP FVG), -1 (DOWN FVG), or None

# 진입 신호
entry = strategy.get_entry_signal(m15_bars, m1_bars)
# Returns: {signal, entry_time, entry_price, stop_loss, take_profit, ...}

# 손익 계산
gross, net, reason = strategy.calculate_pnl(
    entry_price, exit_price, signal, risk_pips
)
```

### BacktestEngine

```python
engine = BacktestEngine(strategy, max_bars_per_trade=100)

# 백테스팅 실행
results = engine.run_backtest('data.csv')

# 결과 저장
engine.save_trades_csv('output.csv')

# 자산 곡선 플롯
engine.plot_equity_curve('equity.png')
```

### MT5LiveTrader

```python
trader = MT5LiveTrader(
    login=1234567,
    password="password",
    server="BROKER-Server",
    symbol="EURUSD",
    lot_size=0.1
)

# MT5 연결
trader.connect()

# 실시간 거래 시작
trader.run()

# MT5 연결 해제
trader.disconnect()
```

---

## 🐛 문제 해결

### 문제: "CSV 파일을 찾을 수 없음"

**해결:**
```bash
# 1. 데이터 파일 준비
# EURUSD_M1_data.csv (M1 시간프레임)

# 2. 올바른 경로 확인
/mnt/user-data/inputs/EURUSD_M1_data.csv

# 3. CSV 헤더 확인
# time, open, high, low, close, tick_volume
```

### 문제: "신호가 너무 적음"

**확인사항:**
1. 거래 허용 시간이 맞는가?
2. Order Block 감지 로직 확인
3. FVG 감지 로직 확인
4. 데이터 품질 확인 (빠진 바가 없는가?)

**해결:**
```python
# 디버그 모드 추가 (개발 예정)
strategy.debug = True
```

### 문제: "MT5 연결 실패"

**확인사항:**
1. LOGIN, PASSWORD, SERVER 정확히 입력?
2. MT5가 실행 중인가?
3. 네트워크 연결 확인
4. 방화벽/VPN 확인

**해결:**
```bash
# MetaTrader5 라이브러리 재설치
pip install --upgrade MetaTrader5
```

---

## 📋 데이터 형식 (CSV)

예상되는 CSV 포맷:

```csv
time,open,high,low,close,tick_volume
2024-01-01 00:00,1.08000,1.08050,1.07950,1.08020,1250
2024-01-01 00:01,1.08020,1.08100,1.08010,1.08090,1350
2024-01-01 00:02,1.08090,1.08120,1.08050,1.08070,1100
...
```

**요구사항:**
- ✅ M1 (1분봉) 시간프레임
- ✅ 시간 포맷: `YYYY-MM-DD HH:MM` (ISO 8601)
- ✅ 가격 소수점: 5자리 이상
- ✅ 연속 데이터 (갭 없음)
- ✅ 최소 1,000 바 (약 16시간)

**MT5에서 데이터 추출:**

```python
import MetaTrader5 as mt5

mt5.initialize()
mt5.login(YOUR_LOGIN, password=YOUR_PASSWORD, server=YOUR_SERVER)

# 최근 1만 바 다운로드 (약 7일)
rates = mt5.copy_rates_from(
    "EURUSD", 
    mt5.TIMEFRAME_M1, 
    datetime(2024, 1, 1), 
    10000
)

df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.to_csv('EURUSD_M1_data.csv', index=False)

mt5.shutdown()
```

---

## 🎯 다음 개선 사항

### 단계 1: 신호 강화 (현재)
- ✅ Order Block 기본 감지
- ✅ FVG 기본 감지
- ⏳ 거래량 필터 추가
- ⏳ 다중 시간프레임 확인

### 단계 2: 포지션 관리 (개발 예정)
- ⏳ 동적 손익비 조정
- ⏳ Trailing Stop
- ⏳ 부분 익절

### 단계 3: 멀티 심볼 (개발 예정)
- ⏳ Cointegration 기반 전략
- ⏳ 여러 통화쌍 동시 거래

### 단계 4: AWS 배포 (개발 예정)
- ⏳ AWS EC2 자동화
- ⏳ CloudWatch 모니터링
- ⏳ 24/7 자동거래

---

## 📞 지원

### 요청사항
- 신호 조정 필요?
- 파라미터 변경?
- 버그 발견?

**피드백 제공:**
```
1. 백테스트 결과 (CSV)
2. 문제 설명
3. 원하는 개선사항
```

---

## 📄 라이센스

개인 알고리듬 거래 프로젝트

---

**행운을 빕니다! 🚀📈**

*최종 업데이트: 2025년 1월*
