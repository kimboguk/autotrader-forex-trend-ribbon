# -*- coding: utf-8 -*-
"""
StatArb - Kalman Filter 동적 헤지 비율

OLS 고정 헤지 비율 대신 실시간 적응하는 동적 헤지 비율 제공.
Kalman innovation(예측 오차)이 곧 spread signal.

State:       [alpha_t, beta_t]  (intercept, hedge ratio)
Observation: y_t = alpha_t + beta_t * x_t + ε_t
Transition:  [alpha_{t+1}, beta_{t+1}] = [alpha_t, beta_t] + w_t  (random walk)

핵심 파라미터:
  delta (1e-6 ~ 1e-2): 상태 전이 노이즈 = 헤지 비율 변화 속도
    - 작을수록 안정적 (smooth), 클수록 빠른 적응 (reactive)
  Ve (관측 노이즈): innovation 분산 초기값, 온라인으로 적응

사용법:
    kf = KalmanFilterHedgeRatio(delta=1e-4, Ve=1e-3)
    result = kf.fit(y_series, x_series)
    # result: DataFrame with alpha, beta, spread, spread_var, z_score
"""

import sys
import argparse
import warnings

import numpy as np
import pandas as pd

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


class KalmanFilterHedgeRatio:
    """
    2-state Kalman Filter for dynamic hedge ratio estimation.

    State vector: θ = [alpha, beta]
    Observation:  y = H @ θ + ε,  where H = [1, x]
    Transition:   θ_{t+1} = θ_t + w,  w ~ N(0, Q)
    """

    def __init__(self, delta: float = 1e-4, Ve: float = 1e-3):
        self.delta = delta
        self.Ve = Ve
        # State
        self.theta = np.zeros(2)          # [alpha, beta]
        self.P = np.eye(2) * 1.0          # State covariance
        self.Q = np.eye(2) * delta        # Process noise (scaled by delta)
        self.R = Ve                        # Observation noise
        self._initialized = False

    def reset(self):
        """필터 상태 초기화"""
        self.theta = np.zeros(2)
        self.P = np.eye(2) * 1.0
        self.Q = np.eye(2) * self.delta
        self.R = self.Ve
        self._initialized = False

    def update(self, y: float, x: float) -> dict:
        """
        단일 관측값에 대한 Predict → Update 스텝.

        Returns:
            dict with alpha, beta, spread (innovation), spread_var, z_score
        """
        H = np.array([1.0, x])  # Observation matrix row

        # ── Predict ──
        # θ_{t|t-1} = θ_{t-1|t-1}  (random walk → no change)
        # P_{t|t-1} = P_{t-1|t-1} + Q
        P_pred = self.P + self.Q

        # ── Innovation ──
        y_hat = H @ self.theta
        innovation = y - y_hat                     # = spread signal
        S = H @ P_pred @ H + self.R                # Innovation variance

        # ── Update ──
        K = P_pred @ H / S                         # Kalman gain
        self.theta = self.theta + K * innovation   # State update
        self.P = P_pred - np.outer(K, K) * S       # Covariance update (Joseph form simplified)

        # z-score: innovation / sqrt(innovation variance)
        z_score = innovation / np.sqrt(S) if S > 0 else 0.0

        return {
            "alpha": self.theta[0],
            "beta": self.theta[1],
            "spread": innovation,
            "spread_var": S,
            "z_score": z_score,
            "kalman_gain_beta": K[1],
        }

    def fit(self, y: pd.Series, x: pd.Series) -> pd.DataFrame:
        """
        전체 시리즈에 대해 Kalman Filter 실행.

        Args:
            y: dependent variable (e.g., EURCHF close prices)
            x: independent variable (e.g., USDJPY close prices)

        Returns:
            DataFrame with columns: alpha, beta, spread, spread_var, z_score
        """
        self.reset()

        n = len(y)
        results = {
            "alpha": np.zeros(n),
            "beta": np.zeros(n),
            "spread": np.zeros(n),
            "spread_var": np.zeros(n),
            "z_score": np.zeros(n),
        }

        y_vals = y.values
        x_vals = x.values

        for t in range(n):
            r = self.update(y_vals[t], x_vals[t])
            results["alpha"][t] = r["alpha"]
            results["beta"][t] = r["beta"]
            results["spread"][t] = r["spread"]
            results["spread_var"][t] = r["spread_var"]
            results["z_score"][t] = r["z_score"]

        df = pd.DataFrame(results, index=y.index)

        # 초기 수렴 기간 (처음 ~50바) 마킹
        df["converged"] = True
        df.iloc[:50, df.columns.get_loc("converged")] = False

        return df

    def log_likelihood(self, y: pd.Series, x: pd.Series) -> float:
        """
        Innovation log-likelihood 계산 (파라미터 최적화 목적함수).

        L = -0.5 * Σ [log(2π*S_t) + e_t² / S_t]
        """
        self.reset()
        n = len(y)
        y_vals = y.values
        x_vals = x.values

        ll = 0.0
        for t in range(n):
            r = self.update(y_vals[t], x_vals[t])
            S = r["spread_var"]
            e = r["spread"]
            if S > 0:
                ll -= 0.5 * (np.log(2 * np.pi * S) + e ** 2 / S)

        return ll


def optimize_kalman_params(
    y: pd.Series,
    x: pd.Series,
    delta_range: tuple = (1e-6, 1e-2, 15),
    Ve_range: tuple = (1e-4, 1e-1, 15),
) -> dict:
    """
    Grid search로 Kalman 파라미터 최적화.
    Innovation log-likelihood 최대화.

    Args:
        delta_range: (start, end, n_points) for logspace
        Ve_range: (start, end, n_points) for logspace

    Returns:
        dict with best_delta, best_Ve, best_ll, grid_results
    """
    deltas = np.logspace(
        np.log10(delta_range[0]), np.log10(delta_range[1]), int(delta_range[2])
    )
    Ves = np.logspace(
        np.log10(Ve_range[0]), np.log10(Ve_range[1]), int(Ve_range[2])
    )

    best_ll = -np.inf
    best_delta = deltas[0]
    best_Ve = Ves[0]
    grid = []

    for d in deltas:
        for v in Ves:
            kf = KalmanFilterHedgeRatio(delta=d, Ve=v)
            ll = kf.log_likelihood(y, x)
            grid.append({"delta": d, "Ve": v, "log_likelihood": ll})
            if ll > best_ll:
                best_ll = ll
                best_delta = d
                best_Ve = v

    return {
        "best_delta": best_delta,
        "best_Ve": best_Ve,
        "best_ll": best_ll,
        "grid": pd.DataFrame(grid),
    }


def main():
    """CLI: Kalman Filter 테스트"""
    parser = argparse.ArgumentParser(description="Kalman Filter Hedge Ratio")
    parser.add_argument("--pair", type=str, default="EURCHF,USDJPY",
                        help="Pair (comma-separated, e.g., EURCHF,USDJPY)")
    parser.add_argument("--tf", type=str, default="D1")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    parser.add_argument("--optimize", action="store_true",
                        help="Grid search로 최적 파라미터 탐색")
    args = parser.parse_args()

    from data_loader import DataLoader
    from config import KALMAN_CONFIG

    sym_y, sym_x = args.pair.split(",")

    print(f"Loading {sym_y}/{sym_x} {args.tf} ({args.start} ~ {args.end})...")
    loader = DataLoader()
    data = loader.load_pair(sym_y, sym_x, args.tf, args.start, args.end)
    y, x = data[sym_y], data[sym_x]
    print(f"Loaded: {len(y)} bars")

    if args.optimize:
        print(f"\nOptimizing Kalman parameters (grid search)...")
        opt = optimize_kalman_params(
            y, x,
            delta_range=KALMAN_CONFIG["delta_range"],
            Ve_range=KALMAN_CONFIG["Ve_range"],
        )
        print(f"Best delta={opt['best_delta']:.2e}, Ve={opt['best_Ve']:.2e}, "
              f"LL={opt['best_ll']:.2f}")
        delta, Ve = opt["best_delta"], opt["best_Ve"]
    else:
        delta = KALMAN_CONFIG["delta"]
        Ve = KALMAN_CONFIG["Ve"]

    print(f"\nRunning Kalman Filter (delta={delta:.2e}, Ve={Ve:.2e})...")
    kf = KalmanFilterHedgeRatio(delta=delta, Ve=Ve)
    result = kf.fit(y, x)

    # 수렴 후 통계만 표시
    conv = result[result["converged"]]
    print(f"\n{'='*60}")
    print(f"  Kalman Filter Results: {sym_y}/{sym_x} ({args.tf})")
    print(f"{'='*60}")
    print(f"\nBeta (hedge ratio):")
    print(f"  mean={conv['beta'].mean():.4f}  std={conv['beta'].std():.4f}")
    print(f"  min={conv['beta'].min():.4f}   max={conv['beta'].max():.4f}")
    print(f"\nSpread z-score:")
    print(f"  mean={conv['z_score'].mean():.4f}  std={conv['z_score'].std():.4f}")
    print(f"  min={conv['z_score'].min():.4f}   max={conv['z_score'].max():.4f}")

    # z-score 분포
    zs = conv["z_score"]
    print(f"\nZ-score distribution:")
    for thresh in [1.0, 1.5, 2.0, 2.5, 3.0]:
        pct = (zs.abs() >= thresh).mean()
        print(f"  |z| >= {thresh}: {pct:.1%}")

    # OLS 비교
    from cointegration_analyzer import CointegrationAnalyzer
    analyzer = CointegrationAnalyzer()
    eg = analyzer.test_engle_granger(y, x)
    print(f"\nOLS comparison:")
    print(f"  OLS beta (fixed): {eg['hedge_ratio']:.4f}")
    print(f"  Kalman beta (final): {conv['beta'].iloc[-1]:.4f}")
    print(f"  OLS residual std: {eg['residuals'].std():.6f}")
    print(f"  Kalman spread std: {conv['spread'].std():.6f}")


if __name__ == "__main__":
    main()
