# -*- coding: utf-8 -*-
"""
StatArb - Cointegration 분석 모듈

Engle-Granger, Johansen 검정 + Rolling Window 안정성 평가 + OU 반감기 추정.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple

from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.vector_ar.vecm import coint_johansen


class CointegrationAnalyzer:
    """Cointegration 검정 및 분석 도구"""

    # ── Engle-Granger 2단계 검정 ──

    @staticmethod
    def test_engle_granger(
        y: pd.Series,
        x: pd.Series,
        significance: float = 0.05,
    ) -> Dict:
        """
        Engle-Granger 2단계 cointegration 검정.

        1. OLS: y = α + β*x + ε
        2. ADF test on residuals (ε)

        Args:
            y: 종속 변수 가격 시리즈
            x: 독립 변수 가격 시리즈
            significance: 유의수준 (기본 0.05)

        Returns:
            {
                'p_value': float,        # cointegration p-value
                'test_stat': float,      # 검정 통계량
                'crit_values': dict,     # 임계값 (1%, 5%, 10%)
                'hedge_ratio': float,    # β (헤지 비율)
                'intercept': float,      # α (절편)
                'is_cointegrated': bool, # p < significance
                'residuals': Series,     # 잔차 (스프레드)
            }
        """
        # OLS regression: y = α + β*x
        x_const = add_constant(x.values)
        model = OLS(y.values, x_const).fit()
        intercept = model.params[0]
        hedge_ratio = model.params[1]
        residuals = pd.Series(model.resid, index=y.index, name="spread")

        # statsmodels coint test (내부적으로 ADF 수행)
        test_stat, p_value, crit_values = coint(y, x)

        return {
            "p_value": p_value,
            "test_stat": test_stat,
            "crit_values": {"1%": crit_values[0], "5%": crit_values[1], "10%": crit_values[2]},
            "hedge_ratio": hedge_ratio,
            "intercept": intercept,
            "is_cointegrated": p_value < significance,
            "residuals": residuals,
        }

    # ── Johansen 검정 (다변량) ──

    @staticmethod
    def test_johansen(
        data: pd.DataFrame,
        det_order: int = 0,
        k_ar_diff: int = 1,
    ) -> Dict:
        """
        Johansen cointegration 검정 (3+ 변수 가능).

        Args:
            data: 가격 DataFrame (각 열 = 심볼)
            det_order: 결정론적 항 (0=no const, 1=const)
            k_ar_diff: VAR 차분 차수

        Returns:
            {
                'trace_stat': array,     # trace 통계량
                'crit_values_95': array, # 95% 임계값
                'eigen_vectors': array,  # 공적분 벡터
                'n_cointegrated': int,   # 공적분 관계 수
                'max_eigen_stat': array, # max eigenvalue 통계량
            }
        """
        result = coint_johansen(data.values, det_order=det_order, k_ar_diff=k_ar_diff)

        # trace test: 몇 개의 공적분 관계가 있는지
        trace_stat = result.lr1       # trace 통계량
        crit_95 = result.cvt[:, 1]    # 95% 임계값 (열 0=90%, 1=95%, 2=99%)
        n_coint = int(np.sum(trace_stat > crit_95))

        return {
            "trace_stat": trace_stat,
            "crit_values_95": crit_95,
            "eigen_vectors": result.evec,
            "n_cointegrated": n_coint,
            "max_eigen_stat": result.lr2,
            "max_eigen_crit_95": result.cvm[:, 1],
        }

    # ── Rolling Window Cointegration ──

    def rolling_cointegration(
        self,
        y: pd.Series,
        x: pd.Series,
        window: int = 500,
        step: int = 50,
        significance: float = 0.05,
    ) -> pd.DataFrame:
        """
        Rolling window cointegration 검정.
        시간에 따른 cointegration 안정성 평가.

        Args:
            y, x: 가격 시리즈
            window: 롤링 윈도우 크기 (바)
            step: 롤링 스텝 크기 (바)
            significance: 유의수준

        Returns:
            DataFrame(time, p_value, hedge_ratio, is_cointegrated, half_life)
        """
        results = []
        total = len(y)

        for i in range(window, total, step):
            y_win = y.iloc[i - window:i]
            x_win = x.iloc[i - window:i]

            try:
                eg = self.test_engle_granger(y_win, x_win, significance)
                hl = self.estimate_half_life(eg["residuals"])

                results.append({
                    "time": y.index[i - 1],
                    "p_value": eg["p_value"],
                    "hedge_ratio": eg["hedge_ratio"],
                    "is_cointegrated": eg["is_cointegrated"],
                    "half_life": hl,
                })
            except Exception:
                results.append({
                    "time": y.index[i - 1],
                    "p_value": 1.0,
                    "hedge_ratio": np.nan,
                    "is_cointegrated": False,
                    "half_life": np.nan,
                })

        return pd.DataFrame(results)

    # ── OU Process 반감기 추정 ──

    @staticmethod
    def estimate_half_life(spread: pd.Series) -> float:
        """
        Ornstein-Uhlenbeck 프로세스 반감기 추정.

        ΔS(t) = θ(μ - S(t-1)) + ε
        half_life = -ln(2) / ln(1 + θ) ≈ ln(2) / θ  (θ가 작을 때)

        AR(1) regression: ΔS = a + b*S(-1) → θ = -b

        Returns:
            반감기 (바 단위). 음수이면 mean-reverting 하지 않음.
        """
        spread_clean = spread.dropna()
        if len(spread_clean) < 10:
            return np.nan

        lag = spread_clean.shift(1).dropna()
        delta = spread_clean.diff().dropna()

        # 길이 맞추기
        lag = lag.iloc[1:]
        delta = delta.iloc[1:] if len(delta) > len(lag) else delta

        min_len = min(len(lag), len(delta))
        lag = lag.iloc[:min_len]
        delta = delta.iloc[:min_len]

        if len(lag) < 5:
            return np.nan

        # AR(1): ΔS = a + b*S(-1)
        x = add_constant(lag.values)
        model = OLS(delta.values, x).fit()
        b = model.params[1]

        if b >= 0:
            return np.inf  # non-mean-reverting

        half_life = -np.log(2) / b
        return half_life

    # ── ADF 정상성 검정 ──

    @staticmethod
    def test_stationarity(series: pd.Series) -> Dict:
        """
        Augmented Dickey-Fuller 정상성 검정.

        Returns:
            {
                'adf_stat': float,
                'p_value': float,
                'is_stationary': bool,
                'n_lags': int,
            }
        """
        result = adfuller(series.dropna(), autolag="AIC")
        return {
            "adf_stat": result[0],
            "p_value": result[1],
            "is_stationary": result[1] < 0.05,
            "n_lags": result[2],
            "n_obs": result[3],
            "crit_values": result[4],
        }

    # ── 상관계수 ──

    @staticmethod
    def rolling_correlation(
        y: pd.Series,
        x: pd.Series,
        window: int = 100,
    ) -> pd.Series:
        """Rolling Pearson 상관계수"""
        return y.rolling(window).corr(x)

    # ── 페어 스캐닝 ──

    def scan_pairs(
        self,
        price_data: pd.DataFrame,
        significance: float = 0.05,
    ) -> pd.DataFrame:
        """
        모든 2-심볼 조합에 대해 cointegration 검정.

        Args:
            price_data: 심볼별 close 가격 (각 열 = 심볼)
            significance: 유의수준

        Returns:
            DataFrame 정렬: p_value 오름차순
        """
        from itertools import combinations

        symbols = price_data.columns.tolist()
        results = []

        for sym_y, sym_x in combinations(symbols, 2):
            y = price_data[sym_y].dropna()
            x = price_data[sym_x].dropna()

            # 겹치는 인덱스만
            common = y.index.intersection(x.index)
            y = y.loc[common]
            x = x.loc[common]

            if len(y) < 100:
                continue

            try:
                eg = self.test_engle_granger(y, x, significance)
                hl = self.estimate_half_life(eg["residuals"])
                corr = y.corr(x)

                results.append({
                    "pair": f"{sym_y}/{sym_x}",
                    "symbol_y": sym_y,
                    "symbol_x": sym_x,
                    "p_value": eg["p_value"],
                    "test_stat": eg["test_stat"],
                    "hedge_ratio": eg["hedge_ratio"],
                    "half_life": hl,
                    "correlation": corr,
                    "is_cointegrated": eg["is_cointegrated"],
                    "n_obs": len(y),
                })
            except Exception as e:
                results.append({
                    "pair": f"{sym_y}/{sym_x}",
                    "symbol_y": sym_y,
                    "symbol_x": sym_x,
                    "p_value": 1.0,
                    "test_stat": np.nan,
                    "hedge_ratio": np.nan,
                    "half_life": np.nan,
                    "correlation": np.nan,
                    "is_cointegrated": False,
                    "n_obs": 0,
                    "error": str(e),
                })

        df = pd.DataFrame(results)
        if len(df) > 0:
            df = df.sort_values("p_value").reset_index(drop=True)
        return df
