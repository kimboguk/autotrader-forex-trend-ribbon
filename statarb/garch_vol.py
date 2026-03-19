# -*- coding: utf-8 -*-
"""
StatArb - GARCH 조건부 변동성 모델링

Rolling std 대신 GARCH(1,1) 조건부 변동성으로 z-score 정규화.
변동성 클러스터링을 반영하여 vol spike 시 threshold 자동 확대 → 거짓 시그널 감소.

모델: σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}
분포: Student-t (fat tail 대응)

사용법:
    garch = GARCHSpreadVol()
    garch.fit(spread_series)
    result = garch.get_garch_zscore(spread_series, lookback=100)
    # result: DataFrame with garch_vol, garch_z_score, rolling_z_score

CLI:
    python garch_vol.py --pair EURCHF,USDJPY --tf D1 --start 2015-01-01
"""

import sys
import argparse
import warnings

import numpy as np
import pandas as pd
from arch import arch_model

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


class GARCHSpreadVol:
    """
    GARCH(1,1) 기반 스프레드 조건부 변동성 모델.

    Rolling std z-score 대비 장점:
    - 변동성 클러스터링 반영 (vol spike 후 자동 확대)
    - 앞선 변동성 정보 활용 (예측적)
    - Fat tail 분포 대응 (Student-t)
    """

    def __init__(self, p: int = 1, q: int = 1, dist: str = "t",
                 vol_lookback: int = 100):
        self.p = p
        self.q = q
        self.dist = dist
        self.vol_lookback = vol_lookback
        self.model_result = None

    def fit(self, spread: pd.Series) -> "GARCHSpreadVol":
        """
        스프레드 수익률에 GARCH 모델 학습.

        GARCH는 수익률(변화량)에 적용 — 수준(level)이 아님.
        """
        returns = spread.diff().dropna()

        # 스케일링 (GARCH는 큰 값에서 수렴 문제)
        self._scale = returns.std()
        if self._scale == 0 or np.isnan(self._scale):
            self._scale = 1.0
        scaled_returns = returns / self._scale * 100  # % 스케일

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = arch_model(
                scaled_returns,
                vol="Garch",
                p=self.p,
                q=self.q,
                dist=self.dist,
                rescale=False,
            )
            self.model_result = model.fit(disp="off")

        return self

    def get_conditional_vol(self, spread: pd.Series) -> pd.Series:
        """
        학습된 GARCH 모델로 조건부 변동성 추출.

        Returns:
            conditional volatility (원래 스케일)
        """
        if self.model_result is None:
            raise RuntimeError("Call fit() first")

        # 조건부 분산 → 변동성 (원래 스케일로 복원)
        cond_var = self.model_result.conditional_volatility
        cond_vol = cond_var / 100 * self._scale  # 원래 스케일

        return cond_vol

    def get_garch_zscore(self, spread: pd.Series) -> pd.DataFrame:
        """
        GARCH 조건부 변동성 기반 z-score + rolling z-score 비교.

        garch_z_score = (spread - rolling_mean) / garch_conditional_vol
        rolling_z_score = (spread - rolling_mean) / rolling_std
        """
        if self.model_result is None:
            self.fit(spread)

        lb = self.vol_lookback

        df = pd.DataFrame(index=spread.index)
        df["spread"] = spread

        # Rolling mean (공통)
        rm = spread.rolling(lb, min_periods=20).mean()

        # Rolling std z-score (기준선)
        rs = spread.rolling(lb, min_periods=20).std()
        df["rolling_z_score"] = (spread - rm) / rs.replace(0, np.nan)

        # GARCH conditional volatility
        cond_vol = self.get_conditional_vol(spread)

        # spread index와 cond_vol index 정렬 (GARCH는 diff()로 1개 적음)
        df["garch_vol"] = np.nan
        common_idx = df.index.intersection(cond_vol.index)
        df.loc[common_idx, "garch_vol"] = cond_vol.loc[common_idx].values

        # GARCH z-score
        df["garch_z_score"] = (spread - rm) / df["garch_vol"].replace(0, np.nan)

        # Vol ratio (GARCH vol / rolling std) — vol spike 감지
        df["vol_ratio"] = df["garch_vol"] / rs.replace(0, np.nan)

        return df

    def forecast_vol(self, horizon: int = 1) -> float:
        """향후 변동성 예측 (원래 스케일)"""
        if self.model_result is None:
            raise RuntimeError("Call fit() first")

        forecast = self.model_result.forecast(horizon=horizon)
        var_forecast = forecast.variance.iloc[-1, 0]
        return np.sqrt(var_forecast) / 100 * self._scale


def main():
    """CLI: GARCH 변동성 테스트"""
    parser = argparse.ArgumentParser(description="GARCH Spread Volatility")
    parser.add_argument("--pair", type=str, default="EURCHF,USDJPY")
    parser.add_argument("--tf", type=str, default="D1")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    args = parser.parse_args()

    from data_loader import DataLoader
    from cointegration_analyzer import CointegrationAnalyzer
    from config import GARCH_CONFIG

    sym_y, sym_x = args.pair.split(",")

    print(f"Loading {sym_y}/{sym_x} {args.tf} ({args.start} ~ {args.end})...")
    loader = DataLoader()
    data = loader.load_pair(sym_y, sym_x, args.tf, args.start, args.end)
    y, x = data[sym_y], data[sym_x]
    print(f"Loaded: {len(y)} bars")

    # OLS spread
    analyzer = CointegrationAnalyzer()
    eg = analyzer.test_engle_granger(y, x)
    spread = eg["residuals"]
    print(f"OLS hedge ratio: {eg['hedge_ratio']:.4f}")

    # GARCH fit
    print(f"\nFitting GARCH({GARCH_CONFIG['p']},{GARCH_CONFIG['q']})...")
    garch = GARCHSpreadVol(
        p=GARCH_CONFIG["p"],
        q=GARCH_CONFIG["q"],
        dist=GARCH_CONFIG["dist"],
        vol_lookback=GARCH_CONFIG["vol_lookback"],
    )
    result = garch.get_garch_zscore(spread)

    # GARCH 모델 요약
    print(f"\n{'='*60}")
    print(f"  GARCH Results: {sym_y}/{sym_x} ({args.tf})")
    print(f"{'='*60}")

    mr = garch.model_result
    print(f"\nModel: GARCH({GARCH_CONFIG['p']},{GARCH_CONFIG['q']}), dist={GARCH_CONFIG['dist']}")
    print(f"  omega = {mr.params.get('omega', 0):.6f}")
    print(f"  alpha = {mr.params.get('alpha[1]', 0):.4f}")
    print(f"  beta  = {mr.params.get('beta[1]', 0):.4f}")
    persistence = mr.params.get('alpha[1]', 0) + mr.params.get('beta[1]', 0)
    print(f"  persistence (α+β) = {persistence:.4f}")

    # Z-score 비교
    r_clean = result.dropna()
    print(f"\n{'Metric':<25} {'GARCH z':>12} {'Rolling z':>12}")
    print("-" * 50)
    print(f"{'std':<25} {r_clean['garch_z_score'].std():>12.4f} {r_clean['rolling_z_score'].std():>12.4f}")
    print(f"{'min':<25} {r_clean['garch_z_score'].min():>12.4f} {r_clean['rolling_z_score'].min():>12.4f}")
    print(f"{'max':<25} {r_clean['garch_z_score'].max():>12.4f} {r_clean['rolling_z_score'].max():>12.4f}")

    print(f"\nZ-score extremes:")
    for thresh in [1.0, 1.5, 2.0, 2.5, 3.0]:
        pct_g = (r_clean["garch_z_score"].abs() >= thresh).mean()
        pct_r = (r_clean["rolling_z_score"].abs() >= thresh).mean()
        print(f"  |z| >= {thresh}: GARCH={pct_g:.1%}  Rolling={pct_r:.1%}")

    # Vol ratio 통계
    print(f"\nVol ratio (GARCH/Rolling):")
    vr = r_clean["vol_ratio"].dropna()
    print(f"  mean={vr.mean():.3f}  std={vr.std():.3f}")
    print(f"  min={vr.min():.3f}   max={vr.max():.3f}")
    print(f"  vol_ratio > 1.5 (spike): {(vr > 1.5).mean():.1%}")

    # Forecast
    vol_1d = garch.forecast_vol(horizon=1)
    print(f"\n1-day ahead vol forecast: {vol_1d:.6f}")


if __name__ == "__main__":
    main()
