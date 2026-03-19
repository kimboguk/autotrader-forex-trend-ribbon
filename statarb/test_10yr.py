# 10년 cointegration 검정: 전체 페어 조합, D1 데이터
import sys, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from data_loader import DataLoader
from cointegration_analyzer import CointegrationAnalyzer

loader = DataLoader()
analyzer = CointegrationAnalyzer()

symbols = ['EURUSD','GBPUSD','USDCHF','AUDUSD','NZDUSD','USDCAD','EURGBP','EURCHF',
           'USDJPY','EURJPY']

tf = 'D1'
print(f"{'='*70}", flush=True)
print(f"  10-Year Cointegration Test ({tf}) : 2015-01-01 ~ 2025-12-31", flush=True)
print(f"{'='*70}", flush=True)
print(f"{'Pair':<20} {'p-value':>8} {'beta':>8} {'HL(d)':>8} {'corr':>8} {'Result':>8}", flush=True)
print("-" * 70, flush=True)

coint_count = 0
total = len(symbols)*(len(symbols)-1)//2
done = 0

for i in range(len(symbols)):
    for j in range(i+1, len(symbols)):
        sy, sx = symbols[i], symbols[j]
        done += 1
        try:
            data = loader.load_pair(sy, sx, tf, '2015-01-01', '2025-12-31')
            y, x = data[sy], data[sx]
            eg = analyzer.test_engle_granger(y, x)
            hl = analyzer.estimate_half_life(eg['residuals'])
            tag = "COINT" if eg["is_cointegrated"] else "no"
            if eg["is_cointegrated"]:
                coint_count += 1
            hl_str = f"{hl:.0f}" if np.isfinite(hl) else "inf"
            marker = "***" if eg["is_cointegrated"] else "   "
            print(f"{sy}/{sx:<12} {eg['p_value']:>8.4f} {eg['hedge_ratio']:>8.4f} "
                  f"{hl_str:>7}d {y.corr(x):>8.3f}  {marker} {tag}  ({done}/{total})", flush=True)
        except Exception as e:
            print(f"{sy}/{sx:<12} ERROR: {e}  ({done}/{total})", flush=True)

print(f"\nCointegrated: {coint_count} / {total}", flush=True)
