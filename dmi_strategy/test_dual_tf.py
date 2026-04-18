import sys, numpy as np, pandas as pd
sys.path.insert(0, '.')
sys.path.insert(0, '../common')
sys.path.insert(0, '../statarb')
from data_loader import DataLoader
from dmi_strategy import compute_dmi, detect_fractal_swings, get_last_confirmed_swing

loader = DataLoader()
m1 = loader.load_m1('EURUSD', '2000-01-01', '2026-12-31')

PIP_SIZE = 0.0001
COST_PIPS = 0.7
FRACTAL_K = 2
SWING_LB = 15
WARMUP = 100
DMI_PERIOD = 14

def resample(m1, rule):
    return m1.resample(rule).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna(subset=['open'])

def calc_mdd(pnl_list):
    equity = 10000; peak = equity; max_dd = 0
    for pnl in pnl_list:
        equity += pnl
        if equity > peak: peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd: max_dd = dd
    return max_dd

def run_dmi_dual_tf(entry_df, filter_df):
    dmi_e = compute_dmi(entry_df, DMI_PERIOD)
    swings = detect_fractal_swings(entry_df, FRACTAL_K)
    pdi_e = dmi_e['plus_di'].values
    ndi_e = dmi_e['minus_di'].values
    adx_e = dmi_e['adx'].values
    opens = entry_df['open'].values
    highs = entry_df['high'].values
    lows = entry_df['low'].values
    closes = entry_df['close'].values
    times = entry_df.index
    n = len(entry_df)

    # Filter TF: DI direction from completed bars
    dmi_f = compute_dmi(filter_df, DMI_PERIOD)
    pdi_f = dmi_f['plus_di']
    ndi_f = dmi_f['minus_di']
    filter_dir = (pdi_f > ndi_f).astype(int) * 2 - 1
    filter_dir_shifted = filter_dir.shift(1)
    filter_dir_aligned = filter_dir_shifted.reindex(entry_df.index, method='ffill').values

    trades = []
    entry_price = 0.0; entry_dir = 0; entry_time = None; stop_price = 0.0
    prev_filter_dir = 0

    for i in range(WARMUP, n):
        if np.isnan(pdi_e[i]) or np.isnan(adx_e[i]) or np.isnan(filter_dir_aligned[i]):
            continue

        cur_filter_dir = int(filter_dir_aligned[i])

        # 1. Stop hit
        if entry_dir == 1 and lows[i] <= stop_price:
            pnl = (stop_price - entry_price) / PIP_SIZE - COST_PIPS
            trades.append({'entry_time': entry_time, 'exit_time': times[i], 'pnl': pnl, 'reason': 'stop'})
            entry_dir = 0; continue
        elif entry_dir == -1 and highs[i] >= stop_price:
            pnl = (entry_price - stop_price) / PIP_SIZE - COST_PIPS
            trades.append({'entry_time': entry_time, 'exit_time': times[i], 'pnl': pnl, 'reason': 'stop'})
            entry_dir = 0; continue

        # 2. Exit: filter TF direction change OR entry TF DI reverse
        if entry_dir != 0 and i >= 1 and i + 1 < n:
            filter_exit = False
            if entry_dir == 1 and cur_filter_dir == -1:
                filter_exit = True
            elif entry_dir == -1 and cur_filter_dir == 1:
                filter_exit = True

            entry_reverse = False
            if entry_dir == 1:
                entry_reverse = (ndi_e[i] > pdi_e[i]) and (ndi_e[i-1] <= pdi_e[i-1])
            else:
                entry_reverse = (pdi_e[i] > ndi_e[i]) and (pdi_e[i-1] <= ndi_e[i-1])

            if filter_exit or entry_reverse:
                ep = opens[i + 1]
                pnl = entry_dir * (ep - entry_price) / PIP_SIZE - COST_PIPS
                reason = 'filter_reverse' if filter_exit else 'entry_reverse'
                trades.append({'entry_time': entry_time, 'exit_time': times[i+1], 'pnl': pnl, 'reason': reason})
                entry_dir = 0; continue

        # 3. Entry
        if entry_dir == 0 and i >= 1 and i + 1 < n:
            cross_long = (pdi_e[i] > ndi_e[i]) and (pdi_e[i-1] <= ndi_e[i-1])
            cross_short = (ndi_e[i] > pdi_e[i]) and (ndi_e[i-1] <= pdi_e[i-1])
            adx_ok = (adx_e[i] > 15) and (adx_e[i] > adx_e[i-1])

            long_ok = cross_long and cur_filter_dir == 1
            short_ok = cross_short and cur_filter_dir == -1

            if (long_ok or short_ok) and adx_ok:
                direction = 1 if long_ok else -1
                ep = opens[i + 1]
                if direction == 1:
                    sw = get_last_confirmed_swing(swings['is_swing_low'], entry_df['low'], i, SWING_LB, FRACTAL_K)
                else:
                    sw = get_last_confirmed_swing(swings['is_swing_high'], entry_df['high'], i, SWING_LB, FRACTAL_K)
                if sw is None: continue
                entry_dir = direction; entry_price = ep; entry_time = times[i+1]; stop_price = sw

    if entry_dir != 0:
        pnl = entry_dir * (closes[-1] - entry_price) / PIP_SIZE - COST_PIPS
        trades.append({'entry_time': entry_time, 'exit_time': times[-1], 'pnl': pnl, 'reason': 'eot'})

    return trades

def summary(trades, name):
    df = pd.DataFrame(trades)
    n = len(df)
    if n == 0:
        print(f"{name:<25} No trades"); return
    wr = (df['pnl'] > 0).mean() * 100
    gw = df[df['pnl'] > 0]['pnl'].sum()
    gl = abs(df[df['pnl'] <= 0]['pnl'].sum())
    pf = gw / gl if gl > 0 else 99
    mdd = calc_mdd(list(df['pnl']))
    reasons = df['reason'].value_counts().to_dict()
    print(f"{name:<25} {n:>6} {wr:>5.1f}% {pf:>5.2f} {df['pnl'].sum():>+9.0f} {mdd:>6.2f}%  {reasons}")

combos = [
    ('M30+H4', '30min', '4h'),
    ('M15+H4', '15min', '4h'),
    ('M15+H1', '15min', '1h'),
    ('H1+H4', '1h', '4h'),
    ('H1+D1', '1h', '1D'),
    ('H4+D1', '4h', '1D'),
]

print(f"{'Combo':<25} {'Trades':>6} {'WR%':>6} {'PF':>5} {'Pips':>9} {'MDD%':>7}  Exit Reasons")
print('-' * 90)

all_results = {}
for name, entry_rule, filter_rule in combos:
    entry_df = resample(m1, entry_rule)
    filter_df = resample(m1, filter_rule)
    trades = run_dmi_dual_tf(entry_df, filter_df)
    summary(trades, name)
    all_results[name] = trades

# Yearly breakdown for top combos
for combo_name in ['M30+H4', 'H4+D1']:
    trades = all_results.get(combo_name, [])
    if not trades:
        continue
    df = pd.DataFrame(trades)
    df['year'] = pd.to_datetime(df['entry_time']).dt.year
    print(f"\n=== {combo_name} — Yearly Breakdown ===")
    print(f"{'Year':>6} {'Trades':>7} {'WR%':>7} {'PF':>7} {'Pips':>10}")
    print('-' * 42)
    for yr in sorted(df['year'].unique()):
        sub = df[df['year'] == yr]
        n2 = len(sub)
        wr2 = (sub['pnl'] > 0).mean() * 100
        gw2 = sub[sub['pnl'] > 0]['pnl'].sum()
        gl2 = abs(sub[sub['pnl'] <= 0]['pnl'].sum())
        pf2 = gw2 / gl2 if gl2 > 0 else 99
        print(f"{yr:>6} {n2:>7} {wr2:>6.1f}% {pf2:>6.2f} {sub['pnl'].sum():>+10.0f}")
    n = len(df)
    wr = (df['pnl'] > 0).mean() * 100
    gw = df[df['pnl'] > 0]['pnl'].sum()
    gl = abs(df[df['pnl'] <= 0]['pnl'].sum())
    pf = gw / gl if gl > 0 else 99
    print(f"{'TOTAL':>6} {n:>7} {wr:>6.1f}% {pf:>6.2f} {df['pnl'].sum():>+10.0f}")
