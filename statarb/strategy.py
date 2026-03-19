# -*- coding: utf-8 -*-
"""
StatArb - 전략 코어

OLS spread + rolling z-score 기반 mean-reversion 전략.
HMM 레짐 필터로 trending 구간 거래 차단.

시그널 파이프라인:
  1. OLS hedge ratio → spread = y - β*x
  2. Rolling z-score → 진입/청산 시그널
  3. HMM regime filter → MR 구간에서만 거래
  4. GARCH vol ratio → 변동성 극단시 거래 차단 (선택)

진입 조건 (ALL 충족):
  1. |z_score| >= z_entry (2.0)
  2. HMM mr_probability >= mr_threshold (0.7)
  3. Rolling cointegration 유효 (선택)

청산 조건 (ANY):
  1. |z_score| <= z_exit (0.5) → 정상 청산
  2. |z_score| >= z_stop (4.0) → 손절
  3. holding_bars > max_holding → 타임아웃

포지션:
  z > +entry → SHORT spread (sell y, buy x*β)
  z < -entry → LONG spread (buy y, sell x*β)
"""

import sys

import numpy as np
import pandas as pd

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from cointegration_analyzer import CointegrationAnalyzer
from hmm_regime import HMMRegimeDetector


class StatArbStrategy:
    """StatArb Mean-Reversion 전략"""

    def __init__(
        self,
        z_entry: float = 2.0,
        z_exit: float = 0.5,
        z_stop: float = 4.0,
        max_holding_bars: int = 500,
        lookback: int = 100,
        use_hmm: bool = True,
        mr_threshold: float = 0.7,
        hmm_lookback: int = 20,
    ):
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.z_stop = z_stop
        self.max_holding_bars = max_holding_bars
        self.lookback = lookback
        self.use_hmm = use_hmm
        self.mr_threshold = mr_threshold
        self.hmm_lookback = hmm_lookback

    def generate_signals(
        self,
        y: pd.Series,
        x: pd.Series,
    ) -> pd.DataFrame:
        """
        전체 시리즈에 대한 시그널 생성.

        Returns:
            DataFrame with: spread, z_score, hedge_ratio, signal, position,
                           mr_prob, entry_price_y, entry_price_x, bars_held, pnl
        """
        # ── 1. OLS spread + z-score ──
        analyzer = CointegrationAnalyzer()
        eg = analyzer.test_engle_granger(y, x)
        hedge_ratio = eg["hedge_ratio"]
        intercept = eg["intercept"]
        spread = eg["residuals"]

        rm = spread.rolling(self.lookback, min_periods=20).mean()
        rs = spread.rolling(self.lookback, min_periods=20).std()
        z_score = (spread - rm) / rs.replace(0, np.nan)

        df = pd.DataFrame(index=y.index)
        df["y"] = y
        df["x"] = x
        df["spread"] = spread
        df["z_score"] = z_score
        df["hedge_ratio"] = hedge_ratio

        # ── 2. HMM 레짐 필터 ──
        if self.use_hmm:
            hmm = HMMRegimeDetector(
                n_states=2,
                feature_lookback=self.hmm_lookback,
            )
            regimes = hmm.fit_predict(spread)
            # regimes index가 spread와 다를 수 있으므로 reindex
            df["mr_prob"] = regimes["mr_probability"].reindex(df.index)
            df["mr_prob"] = df["mr_prob"].ffill().fillna(0.0)
        else:
            df["mr_prob"] = 1.0  # HMM 없으면 항상 MR

        # ── 3. 시그널 + 포지션 시뮬레이션 ──
        n = len(df)
        signals = np.zeros(n)       # 1=long spread, -1=short spread, 0=flat
        positions = np.zeros(n)     # 현재 포지션
        bars_held = np.zeros(n, dtype=int)
        entry_z = np.zeros(n)

        pos = 0       # current position
        held = 0      # bars held
        e_z = 0.0     # entry z-score

        z_vals = df["z_score"].values
        mr_vals = df["mr_prob"].values

        for t in range(1, n):
            z = z_vals[t]
            mr = mr_vals[t]

            if np.isnan(z):
                positions[t] = pos
                bars_held[t] = held
                entry_z[t] = e_z
                continue

            # ── 청산 조건 ──
            if pos != 0:
                held += 1
                close_signal = False

                # 정상 청산: z가 exit 이내로 복귀
                if pos == 1 and z >= -self.z_exit:
                    close_signal = True
                elif pos == -1 and z <= self.z_exit:
                    close_signal = True

                # 손절: z가 stop 이상으로 확대
                if abs(z) >= self.z_stop:
                    close_signal = True

                # 타임아웃
                if held >= self.max_holding_bars:
                    close_signal = True

                # 레짐 변화 청산: MR 확률이 0.5 이하
                if self.use_hmm and mr < 0.5:
                    close_signal = True

                if close_signal:
                    signals[t] = -pos  # 반대 방향 = 청산
                    pos = 0
                    held = 0
                    e_z = 0.0
                else:
                    positions[t] = pos
                    bars_held[t] = held
                    entry_z[t] = e_z
                    continue

            # ── 진입 조건 ──
            if pos == 0:
                # HMM 레짐 필터
                if self.use_hmm and mr < self.mr_threshold:
                    positions[t] = 0
                    bars_held[t] = 0
                    entry_z[t] = 0.0
                    continue

                if z <= -self.z_entry:
                    # z가 충분히 낮음 → spread가 저평가 → LONG spread
                    signals[t] = 1
                    pos = 1
                    held = 0
                    e_z = z
                elif z >= self.z_entry:
                    # z가 충분히 높음 → spread가 고평가 → SHORT spread
                    signals[t] = -1
                    pos = -1
                    held = 0
                    e_z = z

            positions[t] = pos
            bars_held[t] = held
            entry_z[t] = e_z

        df["signal"] = signals
        df["position"] = positions
        df["bars_held"] = bars_held
        df["entry_z"] = entry_z

        return df


class StatArbBacktester:
    """StatArb 백테스트 엔진"""

    def __init__(
        self,
        spread_cost_y: float = 0.0,
        spread_cost_x: float = 0.0,
        slippage_y: float = 0.0,
        slippage_x: float = 0.0,
    ):
        """
        Args:
            spread_cost_y: y심볼 편도 스프레드+수수료 (가격 단위)
            spread_cost_x: x심볼 편도 스프레드+수수료 (가격 단위)
            slippage_y/x: 슬리피지 (가격 단위)
        """
        self.cost_y = spread_cost_y + slippage_y
        self.cost_x = spread_cost_x + slippage_x

    def run(
        self,
        signals_df: pd.DataFrame,
        initial_capital: float = 10000.0,
    ) -> dict:
        """
        시그널 DataFrame을 기반으로 백테스트 실행.

        P&L 계산:
          LONG spread:  pnl = (y_exit - y_entry) - β*(x_exit - x_entry)
          SHORT spread: pnl = -(y_exit - y_entry) + β*(x_exit - x_entry)
          양쪽 leg 비용 차감: 2 * (cost_y + cost_x)

        Returns:
            dict with trades, equity_curve, metrics
        """
        df = signals_df.copy()
        sig = df["signal"].values
        y_vals = df["y"].values
        x_vals = df["x"].values
        beta = df["hedge_ratio"].iloc[0]  # OLS 고정 β

        trades = []
        equity = [initial_capital]
        cum_pnl = 0.0

        # 진입/청산 추적
        entry_y = entry_x = entry_t = 0
        direction = 0
        entry_z = 0.0

        for t in range(len(df)):
            s = sig[t]

            if s != 0 and direction == 0:
                # 진입
                direction = int(s)
                entry_y = y_vals[t]
                entry_x = x_vals[t]
                entry_t = t
                entry_z = df["z_score"].values[t]

            elif s != 0 and direction != 0:
                # 청산 (signal = -direction)
                exit_y = y_vals[t]
                exit_x = x_vals[t]

                # P&L 계산
                y_pnl = exit_y - entry_y
                x_pnl = exit_x - entry_x

                if direction == 1:
                    # LONG spread: buy y, sell x*β
                    trade_pnl = y_pnl - beta * x_pnl
                else:
                    # SHORT spread: sell y, buy x*β
                    trade_pnl = -y_pnl + beta * x_pnl

                # 거래 비용 (진입 + 청산 = 2회)
                cost = 2 * (self.cost_y + abs(beta) * self.cost_x)
                trade_pnl -= cost

                cum_pnl += trade_pnl
                bars = t - entry_t

                exit_z = df["z_score"].values[t]

                trades.append({
                    "entry_time": df.index[entry_t],
                    "exit_time": df.index[t],
                    "direction": "LONG" if direction == 1 else "SHORT",
                    "entry_z": entry_z,
                    "exit_z": exit_z,
                    "entry_y": entry_y,
                    "exit_y": exit_y,
                    "entry_x": entry_x,
                    "exit_x": exit_x,
                    "y_pnl": y_pnl if direction == 1 else -y_pnl,
                    "x_pnl": -beta * x_pnl if direction == 1 else beta * x_pnl,
                    "cost": cost,
                    "pnl": trade_pnl,
                    "cum_pnl": cum_pnl,
                    "bars_held": bars,
                    "mr_prob_entry": df["mr_prob"].values[entry_t],
                })

                direction = 0

            equity.append(initial_capital + cum_pnl)

        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        equity_series = pd.Series(equity[:len(df)], index=df.index[:len(equity[:len(df)])])

        metrics = self._compute_metrics(trades_df, equity_series, initial_capital)

        return {
            "trades": trades_df,
            "equity": equity_series,
            "metrics": metrics,
        }

    @staticmethod
    def _compute_metrics(trades: pd.DataFrame, equity: pd.Series,
                         initial_capital: float) -> dict:
        """성과 메트릭 계산"""
        if len(trades) == 0:
            return {"total_trades": 0}

        pnls = trades["pnl"]
        wins = pnls > 0
        losses = pnls < 0

        # Drawdown
        peak = equity.cummax()
        dd = equity - peak
        max_dd = dd.min()
        max_dd_pct = max_dd / initial_capital * 100

        # Sharpe (annualized, 252 trading days)
        daily_returns = equity.pct_change().dropna()
        sharpe = 0.0
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)

        # Sortino
        neg_returns = daily_returns[daily_returns < 0]
        sortino = 0.0
        if len(neg_returns) > 0 and neg_returns.std() > 0:
            sortino = daily_returns.mean() / neg_returns.std() * np.sqrt(252)

        # Profit factor
        gross_profit = pnls[wins].sum() if wins.any() else 0
        gross_loss = abs(pnls[losses].sum()) if losses.any() else 1e-10
        profit_factor = gross_profit / gross_loss

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
        }


def print_backtest_report(result: dict, pair_name: str = ""):
    """백테스트 결과 출력"""
    m = result["metrics"]
    trades = result["trades"]

    print(f"\n{'='*65}")
    print(f"  StatArb Backtest Report: {pair_name}")
    print(f"{'='*65}")

    if m["total_trades"] == 0:
        print("  No trades generated.")
        return

    print(f"\n  Trades:        {m['total_trades']}  "
          f"(Long: {m['long_trades']}, Short: {m['short_trades']})")
    print(f"  Win Rate:      {m['win_rate']:.1f}%")
    print(f"  Profit Factor: {m['profit_factor']:.2f}")
    print(f"  Avg P&L:       {m['avg_pnl']:.6f}")
    print(f"  Total P&L:     {m['total_pnl']:.4f}")
    print(f"  Avg Win:       {m['avg_win']:.6f}")
    print(f"  Avg Loss:      {m['avg_loss']:.6f}")
    print(f"  Sharpe:        {m['sharpe']:.2f}")
    print(f"  Sortino:       {m['sortino']:.2f}")
    print(f"  Max DD:        {m['max_drawdown']:.4f} ({m['max_drawdown_pct']:.1f}%)")
    print(f"  Avg Hold:      {m['avg_bars_held']:.0f} bars")
    print(f"  Max Hold:      {m['max_bars_held']} bars")

    # 연도별 분해
    if len(trades) > 0 and "entry_time" in trades.columns:
        trades["year"] = pd.to_datetime(trades["entry_time"]).dt.year
        yearly = trades.groupby("year").agg(
            n=("pnl", "count"),
            pnl=("pnl", "sum"),
            wr=("pnl", lambda x: (x > 0).mean() * 100),
        )
        print(f"\n  Year   Trades    P&L       WR")
        print(f"  {'-'*38}")
        for yr, row in yearly.iterrows():
            print(f"  {yr}   {row['n']:>5}   {row['pnl']:>+.4f}   {row['wr']:.0f}%")
