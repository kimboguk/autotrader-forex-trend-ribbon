# -*- coding: utf-8 -*-
"""
DMI Strategy Backtest Engine

Wilder 1978 DMI rules + Bill Williams fractal swing stops.
All parameters are literature-based fixed values — no tuning.

Entry: DI cross + ADX > 25
Stop: Most recent fractal swing (fixed at entry, no trailing)
Exit: Stop hit (intra-bar) > ADX peak rule > DI reverse cross
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
sys.path.insert(0, '../common')
sys.path.insert(0, '../statarb')

from dmi_strategy import compute_dmi, detect_fractal_swings, get_last_confirmed_swing


# ── Parameters (ALL FIXED, no tuning) ─────────────────────────

DMI_PERIOD = 14       # Wilder 1978
ADX_THRESH = 25       # Wilder 1978 (strong trend)
PEAK_LB = 2           # ADX 2-bar consecutive decline
SWING_LB = 15         # Swing lookback (15 bars = 7.5h = ~1 session)
FRACTAL_K = 2         # Bill Williams (5-bar pattern)
WARMUP = 100          # DMI/ADX double smoothing warmup

PIP_SIZE = 0.0001     # EURUSD
SPREAD_PIPS = 0.4
COMMISSION_PIPS = 0.3
COST_PIPS = SPREAD_PIPS + COMMISSION_PIPS  # 0.7 round-trip


def run_backtest(
    df: pd.DataFrame,
    symbol: str = 'EURUSD',
    pip_size: float = PIP_SIZE,
    cost_pips: float = COST_PIPS,
    initial_capital: float = 10000,
    allowed_entry_hours: set = None,
    verbose: bool = False,
) -> dict:
    """
    Run DMI strategy backtest.

    Args:
        df: M30 OHLC DataFrame
        symbol: currency pair
        pip_size: pip size
        cost_pips: round-trip cost in pips
        initial_capital: starting capital
        verbose: print trade details

    Returns:
        dict with 'trades' (list), 'stats' (dict), 'equity' (array)
    """
    # ── Compute indicators ──
    dmi = compute_dmi(df, DMI_PERIOD)
    swings = detect_fractal_swings(df, FRACTAL_K)

    plus_di = dmi['plus_di'].values
    minus_di = dmi['minus_di'].values
    adx = dmi['adx'].values

    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    times = df.index

    n = len(df)
    equity = initial_capital
    equity_arr = np.full(n, initial_capital)

    # ── State ──
    entry_dir = 0       # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_time = None
    stop_price = 0.0
    tp_price = 0.0
    adx_at_entry = 0.0

    trades = []

    def record_trade(exit_time, exit_price, reason):
        nonlocal equity, entry_dir, entry_price, entry_time, stop_price
        direction = "long" if entry_dir == 1 else "short"
        if entry_dir == 1:
            pnl_pips = (exit_price - entry_price) / pip_size - cost_pips
        else:
            pnl_pips = (entry_price - exit_price) / pip_size - cost_pips
        pnl_usd = pnl_pips  # 0.1 lot, 1 pip = $1
        equity += pnl_usd
        trades.append({
            'entry_time': entry_time,
            'exit_time': exit_time,
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'stop_price': stop_price,
            'exit_reason': reason,
            'pnl_pips': pnl_pips,
            'pnl_usd': pnl_usd,
            'adx_at_entry': adx_at_entry,
        })
        if verbose:
            print(f"  {direction.upper()} {entry_time} → {exit_time} "
                  f"@ {entry_price:.5f} → {exit_price:.5f} "
                  f"PnL={pnl_pips:+.1f}p [{reason}]")
        entry_dir = 0
        entry_price = 0.0
        entry_time = None
        stop_price = 0.0

    # ── Bar-by-bar simulation ──
    for i in range(WARMUP, n):
        # Skip if indicators not ready
        if np.isnan(plus_di[i]) or np.isnan(adx[i]):
            equity_arr[i] = equity
            continue

        # ── 1. Check TP/SL hit (intra-bar, highest priority) ──
        if entry_dir == 1 and stop_price > 0:
            hit_sl = lows[i] <= stop_price
            hit_tp = tp_price > 0 and highs[i] >= tp_price
            if hit_sl and hit_tp:
                # Both hit — assume SL first (conservative)
                hit_tp = False
            if hit_tp:
                record_trade(times[i], tp_price, 'tp')
                entry_dir = 0
                equity_arr[i] = equity
                continue
            if hit_sl:
                record_trade(times[i], stop_price, 'stop')
                entry_dir = 0
                equity_arr[i] = equity
                continue
        elif entry_dir == -1 and stop_price > 0:
            hit_sl = highs[i] >= stop_price
            hit_tp = tp_price > 0 and lows[i] <= tp_price
            if hit_sl and hit_tp:
                hit_tp = False
            if hit_tp:
                record_trade(times[i], tp_price, 'tp')
                entry_dir = 0
                equity_arr[i] = equity
                continue
            if hit_sl:
                record_trade(times[i], stop_price, 'stop')
                entry_dir = 0
                equity_arr[i] = equity
                continue

        # ── 2. Check signal-based exits (execute at next bar open) ──
        if entry_dir != 0 and i + 1 < n:
            # DI reverse cross only (no ADX peak rule)
            di_reverse = False
            if i >= 1:
                if entry_dir == 1:
                    # Long: -DI crosses above +DI
                    di_reverse = (minus_di[i] > plus_di[i]) and (minus_di[i-1] <= plus_di[i-1])
                else:
                    # Short: +DI crosses above -DI
                    di_reverse = (plus_di[i] > minus_di[i]) and (plus_di[i-1] <= minus_di[i-1])

            if di_reverse:
                exec_price = opens[i + 1]
                record_trade(times[i + 1], exec_price, 'reverse_cross')
                entry_dir = 0
                equity_arr[i] = equity
                continue

        # ── 3. Check entry signals (only if flat) ──
        if entry_dir == 0 and i >= 1 and i + 1 < n:
            # DI cross detection
            cross_long = (plus_di[i] > minus_di[i]) and (plus_di[i-1] <= minus_di[i-1])
            cross_short = (minus_di[i] > plus_di[i]) and (minus_di[i-1] <= plus_di[i-1])

            # ADX filter: ADX > 15 AND ADX rising
            adx_ok = (adx[i] > 15) and (adx[i] > adx[i-1])

            # Time filter (entries only)
            if allowed_entry_hours is not None:
                exec_hour = times[i + 1].hour if hasattr(times[i + 1], 'hour') else pd.Timestamp(times[i + 1]).hour
                if exec_hour not in allowed_entry_hours:
                    equity_arr[i] = equity
                    continue

            if (cross_long or cross_short) and adx_ok:
                direction = 1 if cross_long else -1
                exec_price = opens[i + 1]

                # Find swing stop
                if direction == 1:
                    sw = get_last_confirmed_swing(
                        swings['is_swing_low'], df['low'],
                        current_idx=i, lookback=SWING_LB, K=FRACTAL_K)
                else:
                    sw = get_last_confirmed_swing(
                        swings['is_swing_high'], df['high'],
                        current_idx=i, lookback=SWING_LB, K=FRACTAL_K)

                if sw is None:
                    # No swing found — skip entry
                    equity_arr[i] = equity
                    continue

                entry_dir = direction
                entry_price = exec_price
                entry_time = times[i + 1]
                stop_price = sw
                adx_at_entry = adx[i]

                # TP = 2x SL distance (risk:reward = 1:2)
                sl_dist = abs(exec_price - sw)
                if direction == 1:
                    tp_price = exec_price + 2 * sl_dist
                else:
                    tp_price = exec_price - 2 * sl_dist

                if verbose:
                    dir_str = "LONG" if direction == 1 else "SHORT"
                    stop_dist = abs(exec_price - stop_price) / pip_size
                    print(f"  ENTER {dir_str} @ {exec_price:.5f} "
                          f"stop={stop_price:.5f} ({stop_dist:.0f}p) "
                          f"ADX={adx[i]:.1f} time={times[i+1]}")

        equity_arr[i] = equity

    # ── Close last open trade at last close ──
    if entry_dir != 0:
        record_trade(times[-1], closes[-1], 'eot')
        entry_dir = 0

    # ── Compute stats ──
    stats = compute_stats(trades, equity_arr, initial_capital)

    return {
        'trades': trades,
        'stats': stats,
        'equity': equity_arr,
    }


def compute_stats(trades: list, equity_arr: np.ndarray, initial_capital: float) -> dict:
    """Compute summary statistics from trade list."""
    if not trades:
        return {'total_trades': 0}

    df = pd.DataFrame(trades)
    n = len(df)
    wins = df[df['pnl_pips'] > 0]
    losses = df[df['pnl_pips'] <= 0]
    wr = len(wins) / n * 100
    gross_win = wins['pnl_pips'].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses['pnl_pips'].sum()) if len(losses) > 0 else 0
    pf = gross_win / gross_loss if gross_loss > 0 else 99.99
    total_pnl = df['pnl_pips'].sum()

    # MDD
    peak = initial_capital
    max_dd = 0
    equity = initial_capital
    for pnl in df['pnl_pips']:
        equity += pnl
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # Exit reason distribution
    reason_dist = df['exit_reason'].value_counts().to_dict()

    return {
        'total_trades': n,
        'win_rate': wr,
        'profit_factor': pf,
        'total_pnl_pips': total_pnl,
        'avg_pnl': df['pnl_pips'].mean(),
        'avg_win': wins['pnl_pips'].mean() if len(wins) > 0 else 0,
        'avg_loss': losses['pnl_pips'].mean() if len(losses) > 0 else 0,
        'max_drawdown_pct': max_dd,
        'exit_reasons': reason_dist,
        'avg_adx_entry': df['adx_at_entry'].mean(),
    }


# ── Main runner ───────────────────────────────────────────────

def main():
    from data_loader import DataLoader

    loader = DataLoader()
    m1 = loader.load_m1('EURUSD', '2000-01-01', '2026-12-31')

    # EURUSD allowed hours (Python data UTC-7): KST 17,18,20,21,22 → data hours 1,2,4,5,6
    DATA_TO_KST = 16
    KST_HOURS = {17, 18, 20, 21, 22}
    allowed_hours = {(h - DATA_TO_KST) % 24 for h in KST_HOURS}

    timeframes = {
        'M30': '30min',
        'H1': '1h',
        'H4': '4h',
    }

    print(f"{'TF':<6} {'Filter':<8} {'Trades':>7} {'WR%':>6} {'PF':>5} {'Pips':>9} {'MDD%':>7} {'AvgW':>7} {'AvgL':>7}")
    print('-' * 75)

    for tf_name, rule in timeframes.items():
        df = m1.resample(rule).agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'tick_volume': 'sum',
        }).dropna(subset=['open'])

        # Without time filter
        r1 = run_backtest(df, verbose=False)
        s1 = r1['stats']
        if s1['total_trades'] > 0:
            print(f"{tf_name:<6} {'none':<8} {s1['total_trades']:>7} {s1['win_rate']:>5.1f}% {s1['profit_factor']:>5.2f} {s1['total_pnl_pips']:>+9.0f} {s1['max_drawdown_pct']:>6.2f}% {s1['avg_win']:>+7.1f} {s1['avg_loss']:>+7.1f}")

        # With time filter
        r2 = run_backtest(df, allowed_entry_hours=allowed_hours, verbose=False)
        s2 = r2['stats']
        if s2['total_trades'] > 0:
            print(f"{tf_name:<6} {'KST':<8} {s2['total_trades']:>7} {s2['win_rate']:>5.1f}% {s2['profit_factor']:>5.2f} {s2['total_pnl_pips']:>+9.0f} {s2['max_drawdown_pct']:>6.2f}% {s2['avg_win']:>+7.1f} {s2['avg_loss']:>+7.1f}")

    # H4 yearly breakdown (best combo)
    print(f"\n=== H4 (no time filter) — Yearly Breakdown ===")
    df_h4 = m1.resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'tick_volume': 'sum',
    }).dropna(subset=['open'])
    r = run_backtest(df_h4, verbose=False)
    df_trades = pd.DataFrame(r['trades'])
    df_trades['year'] = pd.to_datetime(df_trades['entry_time']).dt.year
    print(f"{'Year':>6} {'Trades':>7} {'WR%':>7} {'PF':>7} {'Pips':>10}")
    print('-' * 42)
    for yr in sorted(df_trades['year'].unique()):
        sub = df_trades[df_trades['year'] == yr]
        n = len(sub)
        wr = (sub['pnl_pips'] > 0).mean() * 100
        gw = sub[sub['pnl_pips'] > 0]['pnl_pips'].sum()
        gl = abs(sub[sub['pnl_pips'] <= 0]['pnl_pips'].sum())
        pf = gw / gl if gl > 0 else 99
        print(f"{yr:>6} {n:>7} {wr:>6.1f}% {pf:>6.2f} {sub['pnl_pips'].sum():>+10.0f}")


if __name__ == '__main__':
    main()
