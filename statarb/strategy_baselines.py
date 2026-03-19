# -*- coding: utf-8 -*-
"""
StatArb - H1 기본 전략 3종

1. RatioBollingerStrategy: log(y/x) 볼린저 밴드 (가장 단순)
2. KalmanStrategy: 칼만 필터 동적 hedge ratio
3. OUOptimalStrategy: OU process 기반 동적 임계값

모든 전략은 AdaptiveBacktester 호환 DataFrame을 출력.
"""

import sys
import numpy as np
import pandas as pd

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from kalman_filter import KalmanFilterHedgeRatio
from cointegration_analyzer import CointegrationAnalyzer


# ── 공통 시그널 루프 ──────────────────────────────────────────

def _run_signal_loop(
    z_scores, hedge_ratios,
    z_entry_arr, z_exit_arr, z_stop_arr, max_hold_arr,
    start_idx: int,
):
    """
    z-score 기반 진입/청산 시그널 생성 (공통 루프).

    Args:
        z_scores: z-score 배열 (n,)
        hedge_ratios: hedge ratio 배열 (n,)
        z_entry_arr: 진입 임계값 배열 (n,) — OU는 동적, 나머지는 고정
        z_exit_arr: 청산 임계값 배열 (n,)
        z_stop_arr: 손절 임계값 배열 (n,)
        max_hold_arr: 최대 보유 배열 (n,) — OU는 동적
        start_idx: 시그널 생성 시작 인덱스

    Returns:
        signals, positions, bars_held_arr, exit_reasons
    """
    n = len(z_scores)
    signals = np.zeros(n)
    positions = np.zeros(n)
    bars_held_arr = np.zeros(n, dtype=int)
    exit_reasons = np.empty(n, dtype=object)
    exit_reasons[:] = ""

    pos = 0
    held = 0

    for t in range(start_idx, n):
        z = z_scores[t]
        ze = z_entry_arr[t]
        zx = z_exit_arr[t]
        zs = z_stop_arr[t]
        mh = max_hold_arr[t]

        if np.isnan(z) or np.isnan(ze):
            positions[t] = pos
            bars_held_arr[t] = held
            continue

        # ── 청산 ──
        if pos != 0:
            held += 1
            close_signal = False
            reason = ""

            if held >= mh:
                close_signal = True
                reason = "timeout"
            if (pos == 1 and z >= -zx) or (pos == -1 and z <= zx):
                close_signal = True
                reason = "tp"
            if abs(z) >= zs:
                close_signal = True
                reason = "sl"

            if close_signal:
                signals[t] = -pos
                exit_reasons[t] = reason
                pos = 0
                held = 0
                positions[t] = 0
                bars_held_arr[t] = 0
                continue

            positions[t] = pos
            bars_held_arr[t] = held
            continue

        # ── 진입 ──
        if pos == 0 and ze > 0:
            if z <= -ze:
                signals[t] = 1
                pos = 1
                held = 0
            elif z >= ze:
                signals[t] = -1
                pos = -1
                held = 0

        positions[t] = pos
        bars_held_arr[t] = held

    return signals, positions, bars_held_arr, exit_reasons


def _build_df(y, x, spreads, z_scores, hedge_ratios,
              signals, positions, bars_held_arr, exit_reasons,
              coint_pvalues=None, is_coint_arr=None):
    """공통 DataFrame 빌더."""
    n = len(y)
    df = pd.DataFrame(index=y.index)
    df["y"] = y
    df["x"] = x
    df["spread"] = spreads
    df["z_score"] = z_scores
    df["hedge_ratio"] = hedge_ratios
    df["signal"] = signals
    df["position"] = positions
    df["bars_held"] = bars_held_arr
    df["exit_reason"] = exit_reasons
    df["coint_pvalue"] = coint_pvalues if coint_pvalues is not None else np.zeros(n)
    df["is_cointegrated"] = is_coint_arr if is_coint_arr is not None else np.ones(n, dtype=bool)
    df["mr_prob"] = np.ones(n)
    return df


# ═══════════════════════════════════════════════════════════════
# 1. Ratio Bollinger
# ═══════════════════════════════════════════════════════════════

class RatioBollingerStrategy:
    """
    log(y/x) 가격비 기반 볼린저 밴드 전략.
    공적분 검정 없이 가격 비율의 평균회귀만 활용.
    """

    def __init__(
        self,
        lookback: int = 100,
        z_entry: float = 2.0,
        z_exit: float = 0.5,
        z_stop: float = 4.0,
        max_holding_bars: int = 240,
    ):
        self.lookback = lookback
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.z_stop = z_stop
        self.max_holding_bars = max_holding_bars

    def generate_signals(self, y: pd.Series, x: pd.Series,
                         verbose: bool = False) -> pd.DataFrame:
        n = len(y)
        y_vals = y.values
        x_vals = x.values

        # Spread = log(y/x)
        log_ratio = np.log(y_vals / x_vals)
        spreads = log_ratio.copy()

        # Rolling z-score
        z_scores = np.full(n, np.nan)
        for t in range(self.lookback, n):
            window = log_ratio[t - self.lookback + 1:t + 1]
            m = window.mean()
            s = window.std()
            z_scores[t] = (log_ratio[t] - m) / s if s > 0 else 0.0

        # Hedge ratio = y/x (가격비, PnL 계산용)
        hedge_ratios = y_vals / x_vals

        # 고정 임계값
        ze = np.full(n, self.z_entry)
        zx = np.full(n, self.z_exit)
        zs = np.full(n, self.z_stop)
        mh = np.full(n, self.max_holding_bars)

        if verbose:
            print("  RatioBollinger: generating signals...", flush=True)

        signals, positions, bars_held, exits = _run_signal_loop(
            z_scores, hedge_ratios, ze, zx, zs, mh, self.lookback)

        if verbose:
            print("  Done.", flush=True)

        return _build_df(y, x, spreads, z_scores, hedge_ratios,
                         signals, positions, bars_held, exits)


# ═══════════════════════════════════════════════════════════════
# 2. Kalman Filter
# ═══════════════════════════════════════════════════════════════

class KalmanStrategy:
    """
    칼만 필터 기반 동적 hedge ratio + rolling z-score.
    매 바마다 β를 업데이트하여 연속 적응.
    """

    def __init__(
        self,
        delta: float = 1e-4,
        Ve: float = 1e-3,
        lookback: int = 100,
        z_entry: float = 2.0,
        z_exit: float = 0.5,
        z_stop: float = 4.0,
        max_holding_bars: int = 240,
    ):
        self.delta = delta
        self.Ve = Ve
        self.lookback = lookback
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.z_stop = z_stop
        self.max_holding_bars = max_holding_bars

    def generate_signals(self, y: pd.Series, x: pd.Series,
                         verbose: bool = False) -> pd.DataFrame:
        n = len(y)

        if verbose:
            print("  Kalman: running filter...", flush=True)

        # Kalman filter
        kf = KalmanFilterHedgeRatio(delta=self.delta, Ve=self.Ve)
        kf_result = kf.fit(y, x)

        betas = kf_result["beta"].values
        kf_spread = kf_result["spread"].values
        converged = kf_result["converged"].values

        # Rolling z-score on Kalman spread
        z_scores = np.full(n, np.nan)
        warmup = max(50, self.lookback)

        for t in range(warmup, n):
            if not converged[t]:
                continue
            lb = max(0, t - self.lookback + 1)
            window = kf_spread[lb:t + 1]
            # Only use converged bars
            mask = converged[lb:t + 1]
            valid = window[mask]
            if len(valid) < 20:
                continue
            m = valid.mean()
            s = valid.std()
            z_scores[t] = (kf_spread[t] - m) / s if s > 0 else 0.0

        if verbose:
            print("  Kalman: generating signals...", flush=True)

        # 고정 임계값
        ze = np.full(n, self.z_entry)
        zx = np.full(n, self.z_exit)
        zs = np.full(n, self.z_stop)
        mh = np.full(n, self.max_holding_bars)

        signals, positions, bars_held, exits = _run_signal_loop(
            z_scores, betas, ze, zx, zs, mh, warmup)

        if verbose:
            print("  Done.", flush=True)

        return _build_df(y, x, kf_spread, z_scores, betas,
                         signals, positions, bars_held, exits)


# ═══════════════════════════════════════════════════════════════
# 3. OU-Optimal
# ═══════════════════════════════════════════════════════════════

class OUOptimalStrategy:
    """
    Ornstein-Uhlenbeck 프로세스 파라미터 기반 적응형 전략.
    OU 반감기에서 동적 진입 임계값, 최대 보유 기간 도출.
    z_exit=0 (평균 완전 회귀 시 청산).
    """

    def __init__(
        self,
        lookback: int = 240,
        recheck: int = 24,
        min_half_life: float = 5,
        max_half_life: float = 120,
        z_exit: float = 0.0,
        z_stop: float = 4.0,
        holding_mult: float = 2.0,
    ):
        self.lookback = lookback
        self.recheck = recheck
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.z_exit = z_exit
        self.z_stop = z_stop
        self.holding_mult = holding_mult

    def generate_signals(self, y: pd.Series, x: pd.Series,
                         verbose: bool = False) -> pd.DataFrame:
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

        # Dynamic thresholds
        z_entry_arr = np.full(n, np.nan)
        z_exit_arr = np.full(n, self.z_exit)
        z_stop_arr = np.full(n, self.z_stop)
        max_hold_arr = np.full(n, 240.0)  # default

        # State
        current_hedge = np.nan
        current_pvalue = 1.0
        current_hl = np.nan
        tradeable = False
        hedge_ok = False

        start = self.lookback
        n_checks = 0

        for t in range(start, n):
            # ── 1. Recheck OU parameters ──
            if (t - start) % self.recheck == 0:
                n_checks += 1
                if verbose and n_checks % 500 == 0:
                    pct = (t - start) / (n - start) * 100
                    print(f"  [{pct:.0f}%] {n_checks} OU checks...", flush=True)

                wy = y.iloc[t - self.lookback:t]
                wx = x.iloc[t - self.lookback:t]

                try:
                    eg = analyzer.test_engle_granger(wy, wx)
                    current_hedge = eg["hedge_ratio"]
                    current_pvalue = eg["p_value"]
                    hl = analyzer.estimate_half_life(eg["residuals"])
                    hedge_ok = True

                    if np.isfinite(hl) and \
                       self.min_half_life <= hl <= self.max_half_life:
                        current_hl = hl
                        tradeable = True
                    else:
                        tradeable = False
                except Exception:
                    current_pvalue = 1.0
                    tradeable = False

            if not hedge_ok:
                continue

            hedge_ratios[t] = current_hedge
            coint_pvalues[t] = current_pvalue
            is_coint_arr[t] = tradeable

            # Dynamic thresholds from OU half-life
            if tradeable:
                # z_entry: shorter HL → lower entry
                z_entry_arr[t] = max(1.25, min(2.5, np.sqrt(current_hl / 10)))
                max_hold_arr[t] = max(24, self.holding_mult * current_hl)
            else:
                z_entry_arr[t] = np.nan  # no trading

            # ── 2. Spread & z-score ──
            spreads[t] = y_vals[t] - current_hedge * x_vals[t]

            lb_start = max(start, t - 100 + 1)  # z-score lookback = 100
            lb_spread = y_vals[lb_start:t + 1] - current_hedge * x_vals[lb_start:t + 1]

            if len(lb_spread) >= 20:
                mean_s = lb_spread.mean()
                std_s = lb_spread.std()
                z_scores[t] = (spreads[t] - mean_s) / std_s if std_s > 0 else 0.0

        if verbose:
            print(f"  Done: {n_checks} OU checks total", flush=True)

        # ── 3. Signal loop ──
        signals, positions, bars_held, exits = _run_signal_loop(
            z_scores, hedge_ratios,
            z_entry_arr, z_exit_arr, z_stop_arr, max_hold_arr,
            start)

        return _build_df(y, x, spreads, z_scores, hedge_ratios,
                         signals, positions, bars_held, exits,
                         coint_pvalues, is_coint_arr)
