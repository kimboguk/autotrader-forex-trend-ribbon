# -*- coding: utf-8 -*-
"""
StatArb - H1 적응형 전략

Rolling cointegration 추적 기반 적응형 mean-reversion.
공적분 유효 구간에서만 진입, 붕괴 시 이중 조건(tight TP + short timeout) 적용.
"""

import sys
import numpy as np
import pandas as pd

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from cointegration_analyzer import CointegrationAnalyzer


class AdaptiveStatArbStrategy:
    """
    H1 적응형 StatArb 전략

    vs StatArbStrategy:
    1. Rolling cointegration recheck → 동적 hedge ratio
    2. 공적분 유효할 때만 진입
    3. 공적분 붕괴 시 이중 조건 (degraded mode)
    """

    def __init__(
        self,
        coint_window: int = 240,
        coint_recheck: int = 24,
        coint_pvalue: float = 0.05,
        z_entry: float = 2.0,
        z_exit: float = 0.5,
        z_stop: float = 4.0,
        max_holding_bars: int = 240,
        lookback: int = 100,
        degraded_z_exit: float = 0.25,
        degraded_timeout: int = 72,
    ):
        self.coint_window = coint_window
        self.coint_recheck = coint_recheck
        self.coint_pvalue = coint_pvalue
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.z_stop = z_stop
        self.max_holding_bars = max_holding_bars
        self.lookback = lookback
        self.degraded_z_exit = degraded_z_exit
        self.degraded_timeout = degraded_timeout

    def generate_signals(
        self,
        y: pd.Series,
        x: pd.Series,
        verbose: bool = False,
    ) -> pd.DataFrame:
        n = len(y)
        analyzer = CointegrationAnalyzer()

        y_vals = y.values
        x_vals = x.values

        # Output arrays
        spreads = np.full(n, np.nan)
        z_scores = np.full(n, np.nan)
        hedge_ratios = np.full(n, np.nan)
        coint_pvalues = np.full(n, np.nan)
        is_coint_arr = np.zeros(n, dtype=bool)
        signals = np.zeros(n)
        positions = np.zeros(n)
        bars_held_arr = np.zeros(n, dtype=int)
        exit_reasons = np.empty(n, dtype=object)
        exit_reasons[:] = ""

        # State
        current_hedge = np.nan
        current_pvalue = 1.0
        is_coint = False
        hedge_ok = False

        pos = 0
        held = 0
        entry_hedge = 0.0
        degraded_held = 0
        in_degraded = False

        start = self.coint_window
        n_checks = 0

        for t in range(start, n):
            # ── 1. Cointegration recheck ──
            if (t - start) % self.coint_recheck == 0:
                n_checks += 1
                if verbose and n_checks % 500 == 0:
                    pct = (t - start) / (n - start) * 100
                    print(f"  [{pct:.0f}%] {n_checks} EG checks...", flush=True)

                wy = y.iloc[t - self.coint_window:t]
                wx = x.iloc[t - self.coint_window:t]
                try:
                    eg = analyzer.test_engle_granger(wy, wx)
                    current_hedge = eg["hedge_ratio"]
                    current_pvalue = eg["p_value"]
                    is_coint = eg["is_cointegrated"]
                    hedge_ok = True
                except Exception:
                    current_pvalue = 1.0
                    is_coint = False

            if not hedge_ok:
                continue

            hedge_ratios[t] = current_hedge
            coint_pvalues[t] = current_pvalue
            is_coint_arr[t] = is_coint

            # ── 2. Spread & z-score (current hedge ratio) ──
            spreads[t] = y_vals[t] - current_hedge * x_vals[t]

            lb_start = max(start, t - self.lookback + 1)
            lb_spread = y_vals[lb_start:t + 1] - current_hedge * x_vals[lb_start:t + 1]

            if len(lb_spread) >= 20:
                mean_s = lb_spread.mean()
                std_s = lb_spread.std()
                z = (spreads[t] - mean_s) / std_s if std_s > 0 else 0.0
            else:
                z = 0.0
            z_scores[t] = z

            # ── 3. Exit check ──
            if pos != 0:
                held += 1
                close_signal = False
                reason = ""

                if is_coint:
                    in_degraded = False
                    degraded_held = 0

                    # lowest → highest priority
                    if held >= self.max_holding_bars:
                        close_signal = True
                        reason = "timeout"
                    if (pos == 1 and z >= -self.z_exit) or \
                       (pos == -1 and z <= self.z_exit):
                        close_signal = True
                        reason = "tp"
                    if abs(z) >= self.z_stop:
                        close_signal = True
                        reason = "sl"
                else:
                    # Degraded mode
                    if not in_degraded:
                        in_degraded = True
                        degraded_held = 0
                    degraded_held += 1

                    if degraded_held >= self.degraded_timeout:
                        close_signal = True
                        reason = "deg_timeout"
                    if (pos == 1 and z >= -self.degraded_z_exit) or \
                       (pos == -1 and z <= self.degraded_z_exit):
                        close_signal = True
                        reason = "deg_tp"
                    if abs(z) >= self.z_stop:
                        close_signal = True
                        reason = "deg_sl"

                if close_signal:
                    signals[t] = -pos
                    exit_reasons[t] = reason
                    pos = 0
                    held = 0
                    entry_hedge = 0.0
                    in_degraded = False
                    degraded_held = 0
                    positions[t] = 0
                    bars_held_arr[t] = 0
                    continue  # no same-bar re-entry

                positions[t] = pos
                bars_held_arr[t] = held
                continue

            # ── 4. Entry check ──
            if pos == 0 and is_coint:
                if z <= -self.z_entry:
                    signals[t] = 1
                    pos = 1
                    held = 0
                    entry_hedge = current_hedge
                    in_degraded = False
                    degraded_held = 0
                elif z >= self.z_entry:
                    signals[t] = -1
                    pos = -1
                    held = 0
                    entry_hedge = current_hedge
                    in_degraded = False
                    degraded_held = 0

            positions[t] = pos
            bars_held_arr[t] = held

        if verbose:
            print(f"  Done: {n_checks} EG checks total", flush=True)

        # Build DataFrame
        df = pd.DataFrame(index=y.index)
        df["y"] = y
        df["x"] = x
        df["spread"] = spreads
        df["z_score"] = z_scores
        df["hedge_ratio"] = hedge_ratios
        df["coint_pvalue"] = coint_pvalues
        df["is_cointegrated"] = is_coint_arr
        df["signal"] = signals
        df["position"] = positions
        df["bars_held"] = bars_held_arr
        df["exit_reason"] = exit_reasons
        df["mr_prob"] = np.where(is_coint_arr, 1.0, 0.0)

        return df


class AdaptiveBacktester:
    """적응형 전략 백테스트 엔진 (동적 hedge ratio)"""

    def __init__(
        self,
        spread_cost_y: float = 0.0,
        spread_cost_x: float = 0.0,
        slippage_y: float = 0.0,
        slippage_x: float = 0.0,
    ):
        self.cost_y = spread_cost_y + slippage_y
        self.cost_x = spread_cost_x + slippage_x

    def run(self, signals_df: pd.DataFrame, initial_capital: float = 10000.0) -> dict:
        df = signals_df.copy()
        sig = df["signal"].values
        y_vals = df["y"].values
        x_vals = df["x"].values
        hr_vals = df["hedge_ratio"].values

        trades = []
        equity = [initial_capital]
        cum_pnl = 0.0

        entry_y = entry_x = 0.0
        entry_t = 0
        direction = 0
        beta_entry = 0.0
        entry_z = 0.0
        entry_coint_p = 0.0

        for t in range(len(df)):
            s = sig[t]

            if s != 0 and direction == 0:
                # Entry
                direction = int(s)
                entry_y = y_vals[t]
                entry_x = x_vals[t]
                entry_t = t
                entry_z = df["z_score"].values[t]
                beta_entry = hr_vals[t]
                entry_coint_p = df["coint_pvalue"].values[t]

            elif s != 0 and direction != 0:
                # Exit
                exit_y = y_vals[t]
                exit_x = x_vals[t]

                y_pnl = exit_y - entry_y
                x_pnl = exit_x - entry_x

                if direction == 1:
                    trade_pnl = y_pnl - beta_entry * x_pnl
                else:
                    trade_pnl = -y_pnl + beta_entry * x_pnl

                cost = 2 * (self.cost_y + abs(beta_entry) * self.cost_x)
                trade_pnl -= cost

                cum_pnl += trade_pnl
                bars = t - entry_t

                trades.append({
                    "entry_time": df.index[entry_t],
                    "exit_time": df.index[t],
                    "direction": "LONG" if direction == 1 else "SHORT",
                    "entry_z": entry_z,
                    "exit_z": df["z_score"].values[t],
                    "entry_y": entry_y,
                    "exit_y": exit_y,
                    "entry_x": entry_x,
                    "exit_x": exit_x,
                    "hedge_ratio": beta_entry,
                    "y_pnl": y_pnl if direction == 1 else -y_pnl,
                    "x_pnl": -beta_entry * x_pnl if direction == 1 else beta_entry * x_pnl,
                    "cost": cost,
                    "pnl": trade_pnl,
                    "cum_pnl": cum_pnl,
                    "bars_held": bars,
                    "exit_reason": df["exit_reason"].values[t],
                    "entry_coint_p": entry_coint_p,
                    "exit_coint_p": df["coint_pvalue"].values[t],
                    "coint_at_exit": bool(df["is_cointegrated"].values[t]),
                })

                direction = 0

            equity.append(initial_capital + cum_pnl)

        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        equity_arr = equity[:len(df)]
        equity_series = pd.Series(equity_arr, index=df.index[:len(equity_arr)])

        metrics = self._compute_metrics(trades_df, equity_series, initial_capital)

        return {
            "trades": trades_df,
            "equity": equity_series,
            "metrics": metrics,
        }

    @staticmethod
    def _compute_metrics(trades, equity, initial_capital):
        if len(trades) == 0:
            return {"total_trades": 0}

        pnls = trades["pnl"]
        wins = pnls > 0
        losses = pnls < 0

        peak = equity.cummax()
        dd = equity - peak
        max_dd = dd.min()
        max_dd_pct = max_dd / initial_capital * 100

        # Daily returns for Sharpe (comparable to D1 backtester)
        eq_daily = equity.resample('1D').last().dropna()
        daily_rets = eq_daily.pct_change().dropna()

        sharpe = 0.0
        if len(daily_rets) > 0 and daily_rets.std() > 0:
            sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252)

        sortino = 0.0
        neg_rets = daily_rets[daily_rets < 0]
        if len(neg_rets) > 0 and neg_rets.std() > 0:
            sortino = daily_rets.mean() / neg_rets.std() * np.sqrt(252)

        gross_profit = pnls[wins].sum() if wins.any() else 0
        gross_loss = abs(pnls[losses].sum()) if losses.any() else 1e-10
        profit_factor = gross_profit / gross_loss

        # Exit reason breakdown
        reason_counts = {}
        if "exit_reason" in trades.columns:
            reason_counts = trades["exit_reason"].value_counts().to_dict()

        return {
            "total_trades": len(trades),
            "win_rate": wins.mean() * 100,
            "avg_pnl": pnls.mean(),
            "total_pnl": pnls.sum(),
            "avg_win": pnls[wins].mean() if wins.any() else 0,
            "avg_loss": pnls[losses].mean() if losses.any() else 0,
            "profit_factor": profit_factor,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "max_drawdown_pct": max_dd_pct,
            "avg_bars_held": trades["bars_held"].mean(),
            "max_bars_held": trades["bars_held"].max(),
            "long_trades": (trades["direction"] == "LONG").sum(),
            "short_trades": (trades["direction"] == "SHORT").sum(),
            "exit_reasons": reason_counts,
        }
