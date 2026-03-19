# -*- coding: utf-8 -*-
"""
StatArb - 스프레드 계산 + 정규화

Kalman Filter 또는 OLS 기반 스프레드 계산, z-score / CDF-score 정규화.

스프레드 정의:
  - Kalman: innovation (예측 오차) = y_t - (α_t + β_t * x_t)
  - OLS:    residual = y_t - (α + β * x_t)  (고정 회귀 계수)

정규화 방법:
  - Rolling z-score: (spread - rolling_mean) / rolling_std
  - KDE CDF-score:   비모수 분위수 기반 (fat tail에 강건)

사용법:
    calc = SpreadCalculator(method="kalman", delta=1e-4, Ve=1e-3)
    result = calc.fit(y_series, x_series)
    # result: DataFrame with spread, z_score, cdf_score, hedge_ratio
"""

import sys
import argparse

import numpy as np
import pandas as pd
from scipy import stats

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from kalman_filter import KalmanFilterHedgeRatio, optimize_kalman_params


class SpreadCalculator:
    """Kalman 또는 OLS 기반 스프레드 계산 + 정규화"""

    def __init__(
        self,
        method: str = "kalman",
        lookback: int = 100,
        delta: float = 1e-4,
        Ve: float = 1e-3,
    ):
        """
        Args:
            method: "kalman" or "ols"
            lookback: rolling z-score 계산 윈도우
            delta: Kalman 상태 전이 노이즈 (method="kalman"일 때)
            Ve: Kalman 관측 노이즈
        """
        self.method = method
        self.lookback = lookback
        self.delta = delta
        self.Ve = Ve

    def fit(self, y: pd.Series, x: pd.Series) -> pd.DataFrame:
        """
        전체 시리즈에 대해 스프레드 계산 + 정규화.

        Returns:
            DataFrame with: spread, hedge_ratio, z_score, rolling_z_score, cdf_score
        """
        if self.method == "kalman":
            return self._fit_kalman(y, x)
        elif self.method == "ols":
            return self._fit_ols(y, x)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _fit_kalman(self, y: pd.Series, x: pd.Series) -> pd.DataFrame:
        """Kalman Filter 기반 스프레드"""
        kf = KalmanFilterHedgeRatio(delta=self.delta, Ve=self.Ve)
        kf_result = kf.fit(y, x)

        df = pd.DataFrame(index=y.index)
        df["hedge_ratio"] = kf_result["beta"]
        df["intercept"] = kf_result["alpha"]
        df["converged"] = kf_result["converged"]

        # 스프레드: y - β_t * x (동적 헤지 비율 사용)
        # Kalman의 핵심 가치 = 시간에 따라 적응하는 β
        df["spread"] = y.values - kf_result["beta"].values * x.values

        # Rolling z-score on the dynamic spread (주 시그널)
        spread = df["spread"]
        rm = spread.rolling(self.lookback, min_periods=20).mean()
        rs = spread.rolling(self.lookback, min_periods=20).std()
        df["z_score"] = (spread - rm) / rs.replace(0, np.nan)

        # Kalman innovation z-score (보조 지표)
        df["innovation_z"] = kf_result["z_score"]
        df["spread_var"] = kf_result["spread_var"]

        # CDF score (비모수 분위수)
        df["cdf_score"] = self._compute_cdf_score(spread, self.lookback)

        return df

    def _fit_ols(self, y: pd.Series, x: pd.Series) -> pd.DataFrame:
        """OLS 고정 헤지 비율 기반 스프레드"""
        from cointegration_analyzer import CointegrationAnalyzer
        analyzer = CointegrationAnalyzer()

        eg = analyzer.test_engle_granger(y, x)
        spread = eg["residuals"]

        df = pd.DataFrame(index=y.index)
        df["spread"] = spread
        df["hedge_ratio"] = eg["hedge_ratio"]
        df["intercept"] = eg["intercept"]

        # Rolling z-score
        rm = spread.rolling(self.lookback, min_periods=20).mean()
        rs = spread.rolling(self.lookback, min_periods=20).std()
        df["z_score"] = (spread - rm) / rs.replace(0, np.nan)
        df["rolling_z_score"] = df["z_score"]  # OLS에서는 동일

        # CDF score
        df["cdf_score"] = self._compute_cdf_score(spread, self.lookback)

        df["spread_var"] = rs ** 2
        df["converged"] = True

        return df

    @staticmethod
    def _compute_cdf_score(spread: pd.Series, lookback: int) -> pd.Series:
        """
        Rolling KDE 기반 CDF score 계산.

        각 시점에서 lookback 윈도우 내 분포 대비 현재 값의 백분위수 [0, 1].
        0.05 이하 = 극단적 저평가, 0.95 이상 = 극단적 고평가.
        """
        n = len(spread)
        cdf_scores = np.full(n, np.nan)

        vals = spread.values
        min_samples = max(30, lookback // 5)

        for t in range(min_samples, n):
            start = max(0, t - lookback)
            window = vals[start:t]
            valid = window[~np.isnan(window)]
            if len(valid) < min_samples:
                continue

            # Empirical CDF (KDE보다 빠르고 대부분 충분)
            cdf_scores[t] = np.mean(valid <= vals[t])

        return pd.Series(cdf_scores, index=spread.index)


def main():
    """CLI: Spread Calculator 비교 테스트"""
    parser = argparse.ArgumentParser(description="Spread Calculator")
    parser.add_argument("--pair", type=str, default="EURCHF,USDJPY")
    parser.add_argument("--tf", type=str, default="D1")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    parser.add_argument("--optimize", action="store_true",
                        help="Kalman 파라미터 최적화")
    args = parser.parse_args()

    from data_loader import DataLoader
    from cointegration_analyzer import CointegrationAnalyzer
    from config import KALMAN_CONFIG

    sym_y, sym_x = args.pair.split(",")

    print(f"Loading {sym_y}/{sym_x} {args.tf} ({args.start} ~ {args.end})...")
    loader = DataLoader()
    data = loader.load_pair(sym_y, sym_x, args.tf, args.start, args.end)
    y, x = data[sym_y], data[sym_x]
    print(f"Loaded: {len(y)} bars\n")

    # 파라미터 최적화
    delta = KALMAN_CONFIG["delta"]
    Ve = KALMAN_CONFIG["Ve"]

    if args.optimize:
        print("Optimizing Kalman parameters...")
        opt = optimize_kalman_params(
            y, x,
            delta_range=KALMAN_CONFIG["delta_range"],
            Ve_range=KALMAN_CONFIG["Ve_range"],
        )
        delta, Ve = opt["best_delta"], opt["best_Ve"]
        print(f"Optimal: delta={delta:.2e}, Ve={Ve:.2e}, LL={opt['best_ll']:.2f}\n")

    # ── Kalman vs OLS 비교 ──
    print(f"{'='*65}")
    print(f"  Spread Comparison: {sym_y}/{sym_x} ({args.tf})")
    print(f"{'='*65}")

    # Kalman
    calc_kf = SpreadCalculator(method="kalman", delta=delta, Ve=Ve)
    r_kf = calc_kf.fit(y, x)
    kf_conv = r_kf[r_kf["converged"]] if "converged" in r_kf.columns else r_kf

    # OLS
    calc_ols = SpreadCalculator(method="ols")
    r_ols = calc_ols.fit(y, x)

    print(f"\n{'Metric':<25} {'Kalman':>12} {'OLS':>12}")
    print("-" * 50)
    print(f"{'Hedge ratio (final)':<25} {kf_conv['hedge_ratio'].iloc[-1]:>12.4f} {r_ols['hedge_ratio'].iloc[0]:>12.4f}")
    print(f"{'Hedge ratio std':<25} {kf_conv['hedge_ratio'].std():>12.4f} {'(fixed)':>12}")
    print(f"{'Spread std':<25} {kf_conv['spread'].std():>12.6f} {r_ols['spread'].std():>12.6f}")
    print(f"{'Z-score std':<25} {kf_conv['z_score'].std():>12.4f} {r_ols['z_score'].dropna().std():>12.4f}")

    # Stationarity 비교 (ADF test)
    analyzer = CointegrationAnalyzer()
    adf_kf = analyzer.test_stationarity(kf_conv["spread"].dropna())
    adf_ols = analyzer.test_stationarity(r_ols["spread"].dropna())
    print(f"{'ADF p-value':<25} {adf_kf['p_value']:>12.4f} {adf_ols['p_value']:>12.4f}")
    print(f"{'ADF stationary?':<25} {'Yes' if adf_kf['is_stationary'] else 'No':>12} {'Yes' if adf_ols['is_stationary'] else 'No':>12}")

    # Z-score 분포
    print(f"\nZ-score extremes (|z| >= threshold):")
    for thresh in [1.0, 1.5, 2.0, 2.5, 3.0]:
        pct_kf = (kf_conv["z_score"].abs() >= thresh).mean()
        pct_ols = (r_ols["z_score"].dropna().abs() >= thresh).mean()
        print(f"  |z| >= {thresh}: Kalman={pct_kf:.1%}  OLS={pct_ols:.1%}")

    # CDF score 분포
    print(f"\nCDF score extremes:")
    for q in [0.05, 0.10, 0.90, 0.95]:
        pct_kf = (kf_conv["cdf_score"].dropna() <= q).mean() if q < 0.5 else (kf_conv["cdf_score"].dropna() >= q).mean()
        pct_ols = (r_ols["cdf_score"].dropna() <= q).mean() if q < 0.5 else (r_ols["cdf_score"].dropna() >= q).mean()
        label = f"CDF <= {q}" if q < 0.5 else f"CDF >= {q}"
        print(f"  {label}: Kalman={pct_kf:.1%}  OLS={pct_ols:.1%}")


if __name__ == "__main__":
    main()
