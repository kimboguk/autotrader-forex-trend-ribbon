# -*- coding: utf-8 -*-
"""
Trend Grid - TP/SL Optimization

Tests various TP/SL combinations and finds optimal ratio.
Supports per-timeframe parameter ranges (D1 ~5x H1 volatility).
"""
import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(1, os.path.join(_here, '..', 'statarb'))

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from backtest import run_backtest

# TP/SL combinations per timeframe
# Format: (tp_pips, sl_pips, label)
COMBOS_H1 = [
    (None, None, "No TP/SL"),
    # 2:1 ratio
    (60,  30,  "60/30"),
    (80,  40,  "80/40"),
    (100, 50,  "100/50"),
    (120, 60,  "120/60"),
    (150, 75,  "150/75"),
    (200, 100, "200/100"),
    # 3:1 ratio
    (90,  30,  "90/30"),
    (120, 40,  "120/40"),
    (150, 50,  "150/50"),
    (200, 67,  "200/67"),
    # 1.5:1 ratio
    (75,  50,  "75/50"),
    (100, 67,  "100/67"),
    (150, 100, "150/100"),
    # SL only
    (None, 30,  "-/30"),
    (None, 50,  "-/50"),
    (None, 75,  "-/75"),
    (None, 100, "-/100"),
    # TP only
    (50,  None, "50/-"),
    (100, None, "100/-"),
    (150, None, "150/-"),
    (200, None, "200/-"),
]

# D1: ~5x H1 volatility
COMBOS_D1 = [
    (None, None, "No TP/SL"),
    # 2:1 ratio
    (300, 150, "300/150"),
    (400, 200, "400/200"),
    (500, 250, "500/250"),
    (600, 300, "600/300"),
    (800, 400, "800/400"),
    (1000,500, "1000/500"),
    # 3:1 ratio
    (450, 150, "450/150"),
    (600, 200, "600/200"),
    (750, 250, "750/250"),
    (1000,333, "1000/333"),
    # 1.5:1 ratio
    (375, 250, "375/250"),
    (500, 333, "500/333"),
    (750, 500, "750/500"),
    # SL only
    (None, 150, "-/150"),
    (None, 200, "-/200"),
    (None, 250, "-/250"),
    (None, 300, "-/300"),
    (None, 400, "-/400"),
    (None, 500, "-/500"),
    # TP only
    (300, None, "300/-"),
    (500, None, "500/-"),
    (750, None, "750/-"),
    (1000,None, "1000/-"),
]

# H4: ~2x H1 volatility
COMBOS_H4 = [
    (None, None, "No TP/SL"),
    # 2:1 ratio
    (120, 60,  "120/60"),
    (160, 80,  "160/80"),
    (200, 100, "200/100"),
    (250, 125, "250/125"),
    (300, 150, "300/150"),
    (400, 200, "400/200"),
    # 3:1 ratio
    (180, 60,  "180/60"),
    (240, 80,  "240/80"),
    (300, 100, "300/100"),
    (400, 133, "400/133"),
    # 1.5:1 ratio
    (150, 100, "150/100"),
    (200, 133, "200/133"),
    (300, 200, "300/200"),
    # SL only
    (None, 50,  "-/50"),
    (None, 75,  "-/75"),
    (None, 100, "-/100"),
    (None, 150, "-/150"),
    (None, 200, "-/200"),
    # TP only
    (100, None, "100/-"),
    (200, None, "200/-"),
    (300, None, "300/-"),
    (400, None, "400/-"),
]

COMBOS_MAP = {
    "D1": COMBOS_D1,
    "H4": COMBOS_H4,
    "H1": COMBOS_H1,
}

def main():
    timeframe = sys.argv[1] if len(sys.argv) > 1 else "H1"
    COMBOS = COMBOS_MAP.get(timeframe, COMBOS_H1)

    print(f"\n{'='*100}")
    print(f"  TP/SL Optimization: EURUSD {timeframe} EMA")
    print(f"{'='*100}")

    # Load data once
    results = []
    for tp, sl, label in COMBOS:
        r = run_backtest('EURUSD', timeframe, ma_type='ema',
                         tp_pips=tp, sl_pips=sl, verbose=False)
        s = r['stats']
        trades_df = r['trades']

        # Exit reason breakdown
        if len(trades_df) > 0 and 'exit_reason' in trades_df.columns:
            reason_counts = trades_df['exit_reason'].value_counts().to_dict()
        else:
            reason_counts = {}

        results.append({
            'label': label,
            'tp': tp or '-',
            'sl': sl or '-',
            'trades': s['total_trades'],
            'win_pct': s['win_rate'],
            'pf': s['profit_factor'],
            'net_pips': s['total_pnl_pips'],
            'net_usd': s['total_pnl_usd'],
            'exp_pips': s['expectancy_pips'],
            'max_dd': s['max_drawdown_pct'],
            'ann_ret': s['annual_return_pct'],
            'tp_exits': reason_counts.get('tp', 0),
            'sl_exits': reason_counts.get('sl', 0),
            'sig_exits': reason_counts.get('signal', 0),
        })

    df = pd.DataFrame(results)
    df = df.sort_values('net_pips', ascending=False).reset_index(drop=True)

    pd.set_option('display.width', 140)
    pd.set_option('display.max_columns', 20)

    print(f"\n{'─'*100}")
    print(f"  Sorted by Net P&L (pips)")
    print(f"{'─'*100}")
    print(df.to_string(index=False))
    print(f"{'─'*100}")

    # Top 5
    print(f"\n  Top 5:")
    for i, row in df.head(5).iterrows():
        print(f"    {i+1}. {row['label']:>10s}  "
              f"Net: {row['net_pips']:+8.1f} pips  "
              f"PF: {row['pf']:.2f}  "
              f"Win: {row['win_pct']:.1f}%  "
              f"MaxDD: {row['max_dd']:.1f}%  "
              f"Trades: {row['trades']}  "
              f"(TP:{row['tp_exits']} SL:{row['sl_exits']} Sig:{row['sig_exits']})")


if __name__ == "__main__":
    main()
