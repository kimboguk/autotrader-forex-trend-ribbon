# -*- coding: utf-8 -*-
"""
StatArb - HMM 레짐 감지

Gaussian HMM으로 스프레드의 레짐을 분류:
  State 0: Mean-Reverting (낮은 변동성, 스프레드 수렴) → 거래 허용
  State 1: Trending (높은 변동성, 스프레드 발산) → 거래 금지

Feature 벡터: [spread_return, spread_volatility, spread_zscore]

cointegration이 항상 유효하지 않으므로, mean-reverting 구간에서만 거래하면
Drawdown 대폭 감소 가능.

사용법:
    hmm = HMMRegimeDetector(n_states=2)
    hmm.fit(spread_series)
    regimes = hmm.predict(spread_series)
    # regimes: DataFrame with regime, is_mean_reverting, mr_probability

CLI:
    python hmm_regime.py --pair EURCHF,USDJPY --tf D1 --start 2015-01-01
"""

import sys
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


class HMMRegimeDetector:
    """
    Gaussian HMM 기반 스프레드 레짐 감지.

    학습 데이터로 HMM 파라미터를 추정한 후,
    새로운 데이터에 대해 레짐 확률을 예측.
    """

    def __init__(self, n_states: int = 2, n_iter: int = 100,
                 feature_lookback: int = 20, random_state: int = 42):
        self.n_states = n_states
        self.n_iter = n_iter
        self.feature_lookback = feature_lookback
        self.random_state = random_state

        self.model = None
        self.scaler = StandardScaler()
        self.mr_state_idx = None  # mean-reverting state index

    def prepare_features(self, spread: pd.Series) -> pd.DataFrame:
        """
        스프레드에서 HMM 입력 feature 추출.

        Features:
          1. spread_return: 스프레드 변화량
          2. spread_vol: rolling 변동성
          3. spread_z: rolling z-score (mean-reversion 정도)
        """
        lb = self.feature_lookback

        df = pd.DataFrame(index=spread.index)
        df["spread_return"] = spread.diff()
        df["spread_vol"] = spread.rolling(lb, min_periods=5).std()

        rm = spread.rolling(lb, min_periods=5).mean()
        rs = spread.rolling(lb, min_periods=5).std()
        df["spread_z"] = (spread - rm) / rs.replace(0, np.nan)

        return df.dropna()

    def fit(self, spread: pd.Series) -> "HMMRegimeDetector":
        """
        스프레드 데이터로 HMM 학습.

        학습 후 각 state의 변동성을 비교하여
        최소 변동성 state = mean_reverting으로 라벨링.
        """
        features_df = self.prepare_features(spread)
        features = features_df.values

        # StandardScaler 적용
        X = self.scaler.fit_transform(features)

        # GaussianHMM 학습
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
            self.model.fit(X)

        # State 라벨링: 변동성(feature 1 = spread_vol) 평균이 낮은 state = MR
        state_vol_means = []
        for s in range(self.n_states):
            # Scaled feature space에서 spread_vol (index=1)의 평균
            state_vol_means.append(self.model.means_[s, 1])

        self.mr_state_idx = int(np.argmin(state_vol_means))

        return self

    def predict(self, spread: pd.Series) -> pd.DataFrame:
        """
        레짐 예측 + 확률 계산.

        Returns:
            DataFrame with:
              regime: 0 or 1 (state index)
              is_mean_reverting: True/False
              mr_probability: mean-reverting state 확률 [0, 1]
        """
        if self.model is None:
            raise RuntimeError("Call fit() first")

        features_df = self.prepare_features(spread)
        X = self.scaler.transform(features_df.values)

        # Viterbi decoding (most likely state sequence)
        states = self.model.predict(X)

        # Forward-backward (state posterior probabilities)
        posteriors = self.model.predict_proba(X)
        mr_prob = posteriors[:, self.mr_state_idx]

        result = pd.DataFrame(index=features_df.index)
        result["regime"] = states
        result["is_mean_reverting"] = (states == self.mr_state_idx)
        result["mr_probability"] = mr_prob

        return result

    def fit_predict(self, spread: pd.Series) -> pd.DataFrame:
        """학습 + 예측을 한번에 수행 (in-sample)"""
        self.fit(spread)
        return self.predict(spread)

    def get_state_stats(self) -> pd.DataFrame:
        """각 state의 통계 요약"""
        if self.model is None:
            raise RuntimeError("Call fit() first")

        rows = []
        feature_names = ["spread_return", "spread_vol", "spread_z"]

        for s in range(self.n_states):
            row = {"state": s}
            for i, name in enumerate(feature_names):
                row[f"{name}_mean"] = self.model.means_[s, i]
                row[f"{name}_var"] = self.model.covars_[s, i, i]
            row["is_mr"] = (s == self.mr_state_idx)
            row["stationary_prob"] = self.model.startprob_[s]
            rows.append(row)

        return pd.DataFrame(rows)


def main():
    """CLI: HMM 레짐 감지 테스트"""
    parser = argparse.ArgumentParser(description="HMM Regime Detector")
    parser.add_argument("--pair", type=str, default="EURCHF,USDJPY")
    parser.add_argument("--tf", type=str, default="D1")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    parser.add_argument("--states", type=int, default=2)
    args = parser.parse_args()

    from data_loader import DataLoader
    from cointegration_analyzer import CointegrationAnalyzer
    from config import HMM_CONFIG

    sym_y, sym_x = args.pair.split(",")

    print(f"Loading {sym_y}/{sym_x} {args.tf} ({args.start} ~ {args.end})...")
    loader = DataLoader()
    data = loader.load_pair(sym_y, sym_x, args.tf, args.start, args.end)
    y, x = data[sym_y], data[sym_x]
    print(f"Loaded: {len(y)} bars")

    # OLS spread 계산
    analyzer = CointegrationAnalyzer()
    eg = analyzer.test_engle_granger(y, x)
    spread = eg["residuals"]
    print(f"OLS hedge ratio: {eg['hedge_ratio']:.4f}, p-value: {eg['p_value']:.4f}")

    # HMM 학습 + 예측
    print(f"\nFitting HMM (n_states={args.states})...")
    hmm = HMMRegimeDetector(
        n_states=args.states,
        feature_lookback=HMM_CONFIG["feature_lookback"],
    )
    regimes = hmm.fit_predict(spread)

    # 결과 출력
    print(f"\n{'='*60}")
    print(f"  HMM Regime Detection: {sym_y}/{sym_x} ({args.tf})")
    print(f"{'='*60}")

    # State 통계
    stats = hmm.get_state_stats()
    print(f"\nState Statistics (scaled features):")
    print(stats.to_string(index=False))

    # 레짐 분포
    mr_pct = regimes["is_mean_reverting"].mean()
    print(f"\nRegime Distribution:")
    print(f"  Mean-Reverting: {mr_pct:.1%} of time ({regimes['is_mean_reverting'].sum()} bars)")
    print(f"  Trending:       {1-mr_pct:.1%} of time ({(~regimes['is_mean_reverting']).sum()} bars)")

    # MR probability 분포
    mr_prob = regimes["mr_probability"]
    print(f"\nMR Probability Distribution:")
    print(f"  mean={mr_prob.mean():.3f}  std={mr_prob.std():.3f}")
    print(f"  min={mr_prob.min():.3f}   max={mr_prob.max():.3f}")

    for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
        pct = (mr_prob >= thresh).mean()
        print(f"  P(MR) >= {thresh}: {pct:.1%}")

    # 레짐별 스프레드 특성
    print(f"\nSpread characteristics by regime:")
    for is_mr in [True, False]:
        label = "MR" if is_mr else "Trend"
        mask = regimes["is_mean_reverting"] == is_mr
        idx = regimes.index[mask]
        s = spread.loc[idx]
        ret = s.diff().dropna()
        print(f"  [{label}] spread_std={s.std():.6f}  "
              f"return_std={ret.std():.6f}  "
              f"mean_reversion={-ret.autocorr(1):.3f}")

    # 연도별 레짐 비율
    print(f"\nYearly MR ratio:")
    regimes["year"] = regimes.index.year
    yearly = regimes.groupby("year")["is_mean_reverting"].mean()
    for yr, ratio in yearly.items():
        bar = "#" * int(ratio * 40)
        print(f"  {yr}: {ratio:.0%} {bar}")


if __name__ == "__main__":
    main()
