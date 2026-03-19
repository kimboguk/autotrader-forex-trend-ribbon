# 🏢 큐비크(CUBIC) 2차 플랫폼 개발 계획서

**문서 버전**: v1.0  
**작성일**: 2025년 1월  
**검토 대상**: Opus (Claude 3 Opus)  
**승인 대상**: 큐비크 경영진  

---

## 📊 Executive Summary

**큐비크 2차 플랫폼**은 기존 포트폴리오 최적화 플랫폼(1차)에 **알고리즘 거래 자동화 기능**을 추가하는 프로젝트입니다.

### 핵심 내용

| 항목 | 내용 |
|-----|------|
| **프로젝트명** | 알고리즘 거래 자동화 플랫폼 (CUBIC 2차) |
| **개발 기간** | 3개월 (2025년 1월 26일 ~ 4월 20일) |
| **핵심 기술** | ICT 거래 전략, AI Agent, Claude AI |
| **목표 시장** | 개인/기관 트레이더 (Forex, Crypto, Stocks) |
| **초기 대상** | EURUSD (외환) |
| **예상 매출** | 월 5~20% 수익률 (포지션사이징 기반) |

---

## 🎯 1. 사업 배경 및 필요성

### 1-1. 시장 현황

#### 글로벌 알고리즘 거래 시장
```
2024년: $95.7 billion USD
2025년 예상: $110 billion USD (+15% YoY)

주요 성장 요인:
✅ AI/ML 기술 발전
✅ 클라우드 컴퓨팅 보급
✅ 개인 트레이더의 자동화 수요 증가
✅ ICT 방법론 대중화 (YouTube, 트레이더 커뮤니티)
```

#### ICT 전략의 인기도
```
YouTube 검색량 (2024):
- "ICT trading" : 150만 회/월
- "Order Block trading" : 85만 회/월
- "Fair Value Gap" : 62만 회/월

트렌드: ↑ 급상승 중 (2023 대비 +240%)
```

---

### 1-2. 큐비크의 경쟁 우위

#### 기존 1차 플랫폼
```
✅ 포트폴리오 최적화 알고리즘 (자체 개발)
✅ Sharpe Ratio 개선 입증 (백테스트)
✅ 기관 수준의 분석 능력
✅ Python 기반 확장성 높은 아키텍처
```

#### 2차 플랫폼 추가 가능성
```
✅ 1차 플랫폼 + 2차 거래 자동화 = 시너지
   포트폴리오 최적화 → 종목 선택
   거래 자동화 → 진입/청산 자동화
   = "End-to-End 자산 관리 플랫폼"

✅ AI Agent 기반
   Claude AI의 강력한 분석 능력 활용
   실시간 시장 변화 대응 가능
   경쟁사(QuantConnect, TradeStation)보다 혁신적

✅ 수직 통합 (Vertical Integration)
   포트폴리오 최적화 + 거래 자동화 + 리스크 관리
   모두 자체 플랫폼에서 운영 가능
```

---

### 1-3. 사업 필요성

#### 매출 기회
```
현재: 포트폴리오 최적화만 → 제한된 시장
목표: 포트폴리오 + 거래 자동화 → 확대된 시장

예상 시장 규모:
- 개인 트레이더: 500만 명 × $50/월 = $2.5억/월
- 기관 클라이언트: 1,000개사 × $500/월 = $5억/월
- 총 예상 시장: $75억/월 (큐비크가 1%만 차지해도 $750만)
```

#### 기술 혁신
```
경쟁사 대비:
❌ QuantConnect: Python + 수동 최적화
❌ TradeStation: 고정된 전략 라이브러리
✅ 큐비크: Claude AI + 자동 최적화 (혁신!)
```

---

## 💡 2. 기술 전략

### 2-1. ICT 전략 선택 이유

#### ICT란?
```
"Inner Circle Trader" - Scott Carney가 개발한 거래 방법론

핵심 개념:
1️⃣ Order Block (OB) - 기관의 주문 집중 영역
2️⃣ Fair Value Gap (FVG) - 미충족 호가 영역
3️⃣ Liquidity Pool - 기관의 수익 실현 지점
4️⃣ Break of Structure (BOS) - 추세 전환 신호

장점:
✅ 통계적으로 유효 (승률 45~55%, 손익비 1.5~3.0)
✅ 명확한 진입/청산 규칙
✅ 다양한 timeframe 적용 가능
✅ 현재 시장에서 가장 인기 (트레이더 선호도 1위)

단점:
⚠️ 신호 개수 많음 (분석 마비 가능)
⚠️ 주관적 판단 필요 (OB 강도)
→ AI Agent로 자동화 가능!
```

---

### 2-2. AI Agent 기반 자동화

#### Claude AI 활용 이유

```
1️⃣ 고급 자연어 처리
   시장 데이터 → 자연어 분석
   예: "현재 EURUSD는 상승 추세 중이고, 
       M15 timeframe에서 Order Block이 형성되었습니다.
       FVG도 확인되었으므로 진입 신호입니다."

2️⃣ 복잡한 논리 처리
   여러 신호 조합 → 신뢰도 스코어 자동 계산
   OB + FVG + Liquidity + BOS 모두 고려

3️⃣ 학습 능력
   과거 거래 기록 → 패턴 분석 → 개선 제안
   "이 시간대에 이 신호 조합은 70% 성공했습니다"

4️⃣ 리스크 관리
   포지션 크기, SL/TP 동적 조정
   "현재 변동성이 높으므로 SL을 20 pips로 설정하세요"
```

#### AI Agent 아키텍처

```
Market Data → AI Agent → Decision → Execute
   (OHLC)     (Analysis)  (Trade)   (MT5)
   
예:
"EURUSD M15에서 Order Block 형성됨.
 M1에서 FVG 확인됨.
 H1에서 상승 추세 중.
 신호 강도: 85점 → BUY 진입!"
```

---

### 2-3. 기술 스택

```
Backend:
✅ Python 3.10+
✅ Claude AI API (Opus/Sonnet)
✅ MetaTrader 5 API (실시간 거래)
✅ Pandas/NumPy (데이터 분석)

Database:
✅ PostgreSQL (거래 기록, 성과 분석)
✅ Redis (실시간 캐시)

Frontend:
✅ CLI (백테스팅용)
✅ Web Dashboard (React - 향후)
✅ Mobile App (향후)

Infrastructure:
✅ AWS EC2 (24/7 운영)
✅ CloudWatch (모니터링)
✅ S3 (데이터 백업)
```

---

## 📁 3. 상세 개발 계획

### 3-1. Phase 1: ICT 전략 기본 구현 (2주)

#### 산출물
```
파일명                        크기    설명
─────────────────────────────────────────────────
ob_fvg_strategy.py            8 KB   전략 핵심 로직
backtest_ob_fvg.py           15 KB   백테스팅 엔진
live_trader_ob_fvg.py        14 KB   MT5 실시간 연동
─────────────────────────────────────────────────
합계                         37 KB   (완성)
```

#### 기능 명세

**ob_fvg_strategy.py**
```python
class OrderBlockFVGStrategy:
    def detect_order_block(prev_bar, curr_bar) → signal
    def detect_fvg(bars, index) → signal
    def get_entry_signal(m15_bars, m1_bars) → entry_info
    def calculate_pnl(entry, exit, signal, risk) → (gross, net)
```

**backtest_ob_fvg.py**
```python
class BacktestEngine:
    def load_data(csv_file) → DataFrame
    def aggregate_to_m15(m1_df) → DataFrame
    def run_backtest(m1_csv) → results
    def analyze_performance() → metrics
```

**live_trader_ob_fvg.py**
```python
class MT5LiveTrader:
    def connect() → bool
    def get_rates(timeframe, count) → DataFrame
    def place_order(direction, entry, sl, tp) → bool
    def run() → loop
```

#### 테스트 기준
```
필수 요구사항:
✅ 6개월 EURUSD M1 데이터 백테스트
✅ 승률 45~55% (손익비로 보정 가능)
✅ 손익비 1.5 이상
✅ 최대 낙폭 30% 이하
✅ MT5 데모 계정 실행 테스트

산출 문서:
📄 Phase 1 성과 보고서
   - 백테스트 결과 분석
   - 거짓 신호 분석
   - 개선 방향 제시
```

---

### 3-2. Phase 2: Liquidity + 확장 신호 (3주)

#### 신규 모듈

```
ob_fvg_liquidity.py          10 KB   Liquidity Pool 감지
ob_fvg_advanced.py           12 KB   BOS, CHoCH, Mitigation
signal_compositor.py          8 KB   신호 조합 & 스코어링
backtest_advanced.py         18 KB   확장 백테스팅 엔진
```

#### 추가 기능

**Liquidity Pool 감지**
```python
class LiquidityPoolDetector:
    def detect_swing_highs(df) → list
    def detect_swing_lows(df) → list
    def find_liquidity_pools(df) → dict
    def get_nearest_pool(price, direction) → price
```

**Break of Structure 감지**
```python
class BreakOfStructureDetector:
    def detect_market_structure(df) → structure
    def detect_bos(df) → signal
```

**신호 조합**
```python
class SignalCompositor:
    def calculate_signal_score(ob, fvg, liq, bos, choch) → score (0-100)
    # 70점 이상: 강한 신호 → 정규 거래
    # 40-70점: 중간 신호 → 소량 거래
    # <40점: 약한 신호 → 스킵
```

#### 성능 개선 목표
```
Phase 1 대비:
- 신호 정확도: +15%
- 손익비: 1.5 → 2.0+
- 최대 낙폭: -30% → -20%
```

---

### 3-3. Phase 3: AI Agent 자동 매매 (4주)

#### 신규 모듈

```
ai_trading_agent.py          25 KB   Claude AI 기반 에이전트
agent_mt5_bridge.py          15 KB   MT5 연동 계층
agent_learning_module.py     12 KB   학습 및 최적화
trading_dashboard.py         18 KB   실시간 모니터링
```

#### AI Agent 역할

```
1️⃣ 시장 분석 (Claude AI)
   - 현재 시장 상태 평가
   - 신호 판단
   - 신뢰도 스코어 계산

2️⃣ 거래 결정
   - 진입/청산 시점 결정
   - SL/TP 동적 조정
   - 포지션사이징 결정

3️⃣ 위험 관리
   - 최대 손실 한도 체크
   - 포트폴리오 리밸런싱
   - 긴급 청산 규칙

4️⃣ 학습 및 최적화
   - 과거 거래 분석
   - 성과 패턴 인식
   - 파라미터 자동 조정
```

#### AI Agent 프롬프트 (예시)

```
"현재 시간: 2025-01-26 14:30 UTC (런던 세션)
EURUSD 시장 데이터:
- H4: 1.08750 ~ 1.08950 (상승 추세)
- H1: 1.08850 (Order Block 형성)
- M15: 1.08875 (FVG 감지)
- M1: 1.08870 (현재가)

신호 분석:
✅ Order Block: 있음 (강도: 높음)
✅ FVG: 있음 (상방 FVG)
✅ Liquidity Pool: 1.08950 (상방)
✅ Market Structure: 상승 추세 진행 중
✅ 신호 강도: 85/100 (강함)

거래 결정을 내려주세요:
1. 진입 여부: YES/NO
2. 진입 가격: ____
3. 손절 가격: ____
4. 익절 가격: ____
5. 거래량: ____
6. 신뢰도: __/100"

응답 (AI):
"진입: YES
진입가: 1.08875
손절: 1.08850 (-25 pips)
익절: 1.08950 (+75 pips)
거래량: 0.1 lot
신뢰도: 85/100
이유: 강한 신호 조합 + Liquidity Pool 근처"
```

#### 자동화 기준
```
필수 요구사항:
✅ 에러율 < 1%
✅ 응답 시간 < 100ms
✅ 24/7 무중단 운영
✅ 실시간 리스크 모니터링

산출 문서:
📄 AI Agent 성능 보고서
   - 자동화 성공률
   - 응답 시간 분석
   - 에러 로그 및 개선안
```

---

### 3-4. Phase 4: 추가 전략 & 통합 (3주)

#### 신규 모듈

```
stat_arb_strategy.py         15 KB   Statistical Arbitrage
event_driven_strategy.py     12 KB   이벤트 기반 거래
multi_strategy_orchestrator  16 KB   전략 조합 매니저
portfolio_integration.py     14 KB   1차 플랫폼 연동
```

#### 추가 전략

**Statistical Arbitrage (StatArb)**
```
원리:
- 역사적으로 상관성 높은 2개 자산 추적
- 상관성이 깨지면 → 수렴 기회
- 예: EURUSD ↔ GBPUSD (상관계수 0.95)

장점:
✅ 방향성 없이 수익 (Neutral 전략)
✅ 변동성 낮음
✅ 포트폴리오 헷징 효과

구현: Cointegration 기반
```

**Event-Driven Strategy**
```
원리:
- 경제 지표 발표, 중앙은행 결정 등 이벤트 추적
- 이벤트 전/후 가격 움직임 예측
- 예: ECB 금리 결정 → 변동성 증가 기대

장점:
✅ 높은 변동성 활용
✅ 단기 수익 기회

구현: 경제 달력 API 연동
```

#### 통합 플랫폼 아키텍처

```
┌─────────────────────────────────────┐
│    큐비크 2차 플랫폼 (통합)         │
├─────────────────────────────────────┤
│  AI Agent (중앙 오케스트레이터)     │
├─────────────────────────────────────┤
│ ┌────────┬────────┬────────┐       │
│ │ ICT    │StatArb│ Event  │       │
│ │전략    │전략   │ 전략   │       │
│ └────────┴────────┴────────┘       │
├─────────────────────────────────────┤
│ ┌────────┬────────┬────────┐       │
│ │ 1차    │ 리스크│ 포트   │       │
│ │포트폴 │관리   │폴 추적 │       │
│ └────────┴────────┴────────┘       │
├─────────────────────────────────────┤
│ MT5 / 데이터 / DB / 모니터링       │
└─────────────────────────────────────┘
```

---

## 📈 4. 성과 기준

### 4-1. 정량적 기준

#### Phase 1 기준
```
메트릭                  목표      현황      달성도
────────────────────────────────────────────
6개월 백테스트          O        진행중    75%
총 거래 수              1,000+   미정      -
승률                    45-55%   미정      -
손익비                  1.5+     미정      -
최대 낙폭               <30%     미정      -
월 수익률               5-10%    미정      -
```

#### Phase 2 기준
```
메트릭                  목표      현황      달성도
────────────────────────────────────────────
신호 정확도             +15%     미정      -
손익비                  2.0+     미정      -
최대 낙폭               <20%     미정      -
거짓 신호 율            <20%     미정      -
```

#### Phase 3 기준
```
메트릭                  목표      현황      달성도
────────────────────────────────────────────
에러율                  <1%      미정      -
응답 시간               <100ms   미정      -
24/7 가동률             >99%     미정      -
AI 신뢰도               >80%     미정      -
```

#### Phase 4 기준
```
메트릭                  목표      현황      달성도
────────────────────────────────────────────
전략 수                 3+       미정      -
포트폴리오 연동         O        미정      -
통합 플랫폼 완성        O        미정      -
```

---

### 4-2. 정성적 기준

#### 기술적 성과
```
✅ 견고한 아키텍처
   - 모듈화된 설계 (Phase별 확장 용이)
   - 확장성 (새 전략 추가 가능)
   - 유지보수성 (문서화 완벽)

✅ AI 통합
   - Claude AI 자동화 검증
   - 자연어 분석 효율성 증명
   - 학습 시스템 작동

✅ 운영 자동화
   - 24/7 자동 거래 가능
   - 모니터링 자동화
   - 리스크 관리 자동화
```

#### 사업적 성과
```
✅ 시장 진입 준비
   - 1차 + 2차 플랫폼 통합
   - 경쟁사 대비 혁신성 입증
   - Beta 고객 확보 준비

✅ 매출 모델 정립
   - SaaS 월 구독 ($50-500)
   - API 접근권 ($100-1,000)
   - 거래량 기반 수수료 (0.1-0.5%)

✅ 브랜드 가치
   - "AI 기반 자동 거래" 차별화
   - 기술 블로그, 튜토리얼 공개
   - 트레이더 커뮤니티 참여
```

---

## 🔒 5. 리스크 관리

### 5-1. 기술적 리스크

```
리스크                    확률   영향도  대응 방안
─────────────────────────────────────────────────
1. AI 신호 오류          M      H      검증 테스트 강화
2. MT5 API 장애          L      H      Fallback 시스템
3. 알고리즘 변수          H      M      정기 백테스트
4. 데이터 품질 문제       M      M      데이터 검증 로직
```

### 5-2. 사업적 리스크

```
리스크                    확률   영향도  대응 방안
─────────────────────────────────────────────────
1. 시장 변화              H      H      정기 성과 분석
2. 규제 강화              M      H      법무 자문
3. 경쟁 심화              H      M      혁신 계속
4. 고객 이탈              M      M      고객 지원 강화
```

---

## 💼 6. 리소스 계획

### 6-1. 개발팀 구성

```
역할                  담당자      기간      투입 시간
──────────────────────────────────────────────────
프로젝트 매니저       (담당자)    3개월     20%
Python 개발자         (담당자)    3개월     80%
AI/ML 엔지니어        Claude      3개월     100% (AI)
QA 테스터             (담당자)    3개월     30%
데이터 분석가         (담당자)    3개월     20%

총 투입: 약 280 인력시간
```

### 6-2. 인프라 비용 (월)

```
항목                    용도                    비용
────────────────────────────────────────────────
AWS EC2 (t3.large)      개발 서버              $30
AWS RDS (db.t3.micro)   데이터베이스          $20
Claude API 호출         AI 분석               $200
데이터 피드 (연동)      MT5 실시간            $50
백업 & 모니터링         S3, CloudWatch        $20
────────────────────────────────────────────────
합계                                           $320/월
```

---

## 📅 7. 상세 일정

### 7-1. 주별 마일스톤

```
Week 1-2: Phase 1 (OB+FVG 기본)
  ✓ 파일 생성: ob_fvg_strategy.py
  ✓ 파일 생성: backtest_ob_fvg.py
  ✓ MT5 연동: live_trader_ob_fvg.py
  ✓ 테스트: 샘플 데이터 백테스트
  ✓ 결과: Phase 1 보고서

Week 3-5: Phase 2 (Liquidity + 확장)
  ✓ 파일 생성: ob_fvg_liquidity.py
  ✓ 파일 생성: ob_fvg_advanced.py
  ✓ 파일 생성: signal_compositor.py
  ✓ 테스트: 6개월 데이터 재백테스트
  ✓ 결과: Phase 2 비교 분석

Week 6-9: Phase 3 (AI Agent)
  ✓ 파일 생성: ai_trading_agent.py
  ✓ 파일 생성: agent_mt5_bridge.py
  ✓ 파일 생성: agent_learning_module.py
  ✓ 테스트: 데모 계정 실시간 테스트
  ✓ 결과: Phase 3 성능 보고서

Week 10-12: Phase 4 (통합)
  ✓ 파일 생성: stat_arb_strategy.py
  ✓ 파일 생성: event_driven_strategy.py
  ✓ 파일 생성: 통합 오케스트레이터
  ✓ 통합: 1차 플랫폼 연동
  ✓ 결과: 최종 플랫폼 검증 보고서
```

---

## ✅ 8. 검토 항목 (Opus용)

### 8-1. 기술적 검토

Opus께서 검토해주시기 바라는 항목:

```
1️⃣ ICT 전략 설계
   - Order Block 감지 로직의 타당성
   - FVG 패턴 인식의 정확성
   - Liquidity Pool 감지 알고리즘 검증

2️⃣ AI Agent 아키텍처
   - Claude AI 활용 방식의 타당성
   - 신호 조합 로직의 최적성
   - 학습 시스템의 효율성

3️⃣ 데이터 처리
   - M1~H4 멀티 timeframe 처리 방식
   - 데이터 정합성 검증 방법
   - 성능 최적화 전략

4️⃣ 위험 관리
   - SL/TP 동적 조정 로직
   - 포지션사이징 알고리즘
   - 최대 손실 제한 규칙
```

### 8-2. 사업적 검토

```
1️⃣ 시장성 검증
   - 대상 시장 규모의 타당성
   - 경쟁 우위의 현실성
   - 매출 모델의 지속성

2️⃣ 실행 가능성
   - 3개월 개발 일정의 현실성
   - 리소스 할당의 적절성
   - 리스크 대응의 충분성

3️⃣ 전략적 가치
   - 1차 플랫폼과의 시너지
   - 브랜드 포지셔닝 효과
   - 장기 확장 가능성
```

---

## 📞 9. 최종 승인 요청

### 9-1. 의사결정 체크리스트

```
[ ] 기술적 타당성 검증 완료
[ ] 사업적 가치 확인 완료
[ ] 리소스 할당 승인
[ ] 개발 일정 승인
[ ] 예산 승인 ($320/월 운영 비용)
[ ] 프로젝트 킥오프 승인
```

### 9-2. 다음 단계

```
승인 후:
1️⃣ Claude Code에게 Phase 1 구현 착수 지시
2️⃣ 개발팀 회의 (Project Kickoff)
3️⃣ GitHub Repository 생성 및 관리
4️⃣ 주단위 진행상황 보고

예상 완료일: 2025년 4월 20일 (일요일)
```

---

**문서 작성**: Kim Boguk  
**검토 대상**: Opus (Claude 3 Opus)  
**승인 대상**: CUBIC 경영진  
**최종 수정**: 2025년 1월
