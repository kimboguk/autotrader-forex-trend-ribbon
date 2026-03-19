# -*- coding: utf-8 -*-
"""
StatArb - Walk-Forward + CPCV 검증 프레임워크

과적합 여부를 검증하기 위한 두 가지 방법론:

1. Walk-Forward Optimization (WFO):
   - In-Sample (IS): 파라미터 최적화
   - Out-of-Sample (OOS): 최적 파라미터로 검증
   - 결과: OOS Sharpe 일관성, IS vs OOS 성과 비교

2. Combinatorial Purged Cross-Validation (CPCV):
   - Lopez de Prado 방법론
   - N개 순차 블록 → C(N,k) 조합 테스트
   - 결과: PBO (과적합 확률), Deflated Sharpe

사용법:
    # Walk-Forward
    python walk_forward.py --pair EURCHF,USDJPY --mode wfo

    # CPCV
    python walk_forward.py --pair EURCHF,USDJPY --mode cpcv

    # 전체 (WFO + CPCV)
    python walk_forward.py --pair EURCHF,USDJPY --mode all
"""

import sys
import argparse
import itertools
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from data_loader import DataLoader
from config import SYMBOLS, TRADING_CONFIG, RISK_CONFIG, WFO_CONFIG
from strategy import StatArbStrategy, StatArbBacktester


# ── Walk-Forward Optimization ──────────────────────────────────


class WalkForwardOptimizer:
    """
    Rolling Window Walk-Forward Optimization.

    전체 기간을 IS/OOS 윈도우로 분할하여:
    1. IS에서 파라미터 최적화 (Sharpe 최대화)
    2. 최적 파라미터로 OOS 성과 측정
    3. 모든 OOS 결과를 누적하여 최종 검증
    """

    def __init__(
        self,
        is_days: int = 500,
        oos_days: int = 126,
        roll_days: int = 63,
        param_grid: Optional[dict] = None,
    ):
        """
        Args:
            is_days: In-Sample 기간 (D1 바 수)
            oos_days: Out-of-Sample 기간
            roll_days: 롤링 스텝 (각 윈도우 이동량)
            param_grid: 최적화할 파라미터 그리드
        """
        self.is_days = is_days
        self.oos_days = oos_days
        self.roll_days = roll_days

        self.param_grid = param_grid or {
            "z_entry": [1.5, 2.0, 2.5],
            "z_exit": [0.0, 0.25, 0.5],
            "lookback": [50, 100, 200],
        }

    def _generate_windows(
        self, n_bars: int
    ) -> List[Tuple[int, int, int, int]]:
        """
        IS/OOS 윈도우 인덱스 생성.

        Returns:
            List of (is_start, is_end, oos_start, oos_end)
        """
        windows = []
        start = 0

        while start + self.is_days + self.oos_days <= n_bars:
            is_start = start
            is_end = start + self.is_days
            oos_start = is_end
            oos_end = min(is_end + self.oos_days, n_bars)

            windows.append((is_start, is_end, oos_start, oos_end))
            start += self.roll_days

        return windows

    def _run_backtest(
        self,
        y: pd.Series, x: pd.Series,
        sym_y: str, sym_x: str,
        z_entry: float, z_exit: float, lookback: int,
        use_hmm: bool = True,
    ) -> dict:
        """단일 파라미터 조합으로 백테스트 실행"""
        strategy = StatArbStrategy(
            z_entry=z_entry,
            z_exit=z_exit,
            z_stop=TRADING_CONFIG["z_score_stop"],
            max_holding_bars=TRADING_CONFIG["max_holding_bars"],
            lookback=lookback,
            use_hmm=use_hmm,
            mr_threshold=TRADING_CONFIG["mr_prob_threshold"],
        )

        signals = strategy.generate_signals(y, x)

        # 거래 비용
        sym_y_cfg = SYMBOLS[sym_y]
        sym_x_cfg = SYMBOLS[sym_x]
        cost_y = sym_y_cfg["pip_size"] * (sym_y_cfg["spread_pips"] + sym_y_cfg["commission_pips"])
        cost_x = sym_x_cfg["pip_size"] * (sym_x_cfg["spread_pips"] + sym_x_cfg["commission_pips"])
        slip_y = sym_y_cfg["pip_size"] * RISK_CONFIG["slippage_pips"]
        slip_x = sym_x_cfg["pip_size"] * RISK_CONFIG["slippage_pips"]

        backtester = StatArbBacktester(
            spread_cost_y=cost_y,
            spread_cost_x=cost_x,
            slippage_y=slip_y,
            slippage_x=slip_x,
        )

        return backtester.run(signals)

    def _optimize_is(
        self,
        y: pd.Series, x: pd.Series,
        sym_y: str, sym_x: str,
        use_hmm: bool = True,
    ) -> Tuple[dict, float]:
        """
        IS 기간에서 Sharpe 최대화 파라미터 탐색.

        Returns:
            (best_params, best_sharpe)
        """
        keys = list(self.param_grid.keys())
        combos = list(itertools.product(*self.param_grid.values()))

        best_sharpe = -np.inf
        best_params = dict(zip(keys, combos[0]))

        for combo in combos:
            params = dict(zip(keys, combo))
            try:
                result = self._run_backtest(
                    y, x, sym_y, sym_x,
                    z_entry=params["z_entry"],
                    z_exit=params["z_exit"],
                    lookback=params["lookback"],
                    use_hmm=use_hmm,
                )
                m = result["metrics"]
                if m["total_trades"] >= 3:
                    sharpe = m.get("sharpe", 0.0)
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_params = params
            except Exception:
                continue

        return best_params, best_sharpe

    def run(
        self,
        sym_y: str, sym_x: str,
        tf: str = "D1",
        start: str = "2015-01-01",
        end: str = "2025-12-31",
        use_hmm: bool = True,
    ) -> dict:
        """
        Walk-Forward Optimization 실행.

        Returns:
            dict with:
              windows: List[dict] - 각 윈도우의 IS/OOS 결과
              oos_sharpes: List[float]
              oos_pnls: List[float]
              oos_equity: pd.Series - 누적 OOS 에퀴티
              summary: dict - 전체 WFO 요약 메트릭
        """
        loader = DataLoader()
        data = loader.load_pair(sym_y, sym_x, tf, start, end)
        y_full, x_full = data[sym_y], data[sym_x]

        n = len(y_full)
        windows = self._generate_windows(n)

        if len(windows) == 0:
            raise ValueError(
                f"Not enough data for WFO. Need {self.is_days + self.oos_days} bars, "
                f"have {n}."
            )

        print(f"\nWalk-Forward Optimization: {sym_y}/{sym_x} ({tf})")
        print(f"Total bars: {n}, Windows: {len(windows)}")
        print(f"IS: {self.is_days}d, OOS: {self.oos_days}d, Roll: {self.roll_days}d")
        print(f"{'='*75}")

        results = []
        oos_equities = []

        for i, (is_s, is_e, oos_s, oos_e) in enumerate(windows):
            is_dates = f"{y_full.index[is_s].strftime('%Y-%m-%d')} ~ {y_full.index[is_e-1].strftime('%Y-%m-%d')}"
            oos_dates = f"{y_full.index[oos_s].strftime('%Y-%m-%d')} ~ {y_full.index[oos_e-1].strftime('%Y-%m-%d')}"

            # ── IS 최적화 ──
            y_is = y_full.iloc[is_s:is_e]
            x_is = x_full.iloc[is_s:is_e]

            best_params, is_sharpe = self._optimize_is(
                y_is, x_is, sym_y, sym_x, use_hmm
            )

            # ── OOS 검증 ──
            y_oos = y_full.iloc[oos_s:oos_e]
            x_oos = x_full.iloc[oos_s:oos_e]

            try:
                oos_result = self._run_backtest(
                    y_oos, x_oos, sym_y, sym_x,
                    z_entry=best_params["z_entry"],
                    z_exit=best_params["z_exit"],
                    lookback=best_params["lookback"],
                    use_hmm=use_hmm,
                )
                oos_m = oos_result["metrics"]
                oos_sharpe = oos_m.get("sharpe", 0.0)
                oos_pnl = oos_m.get("total_pnl", 0.0)
                oos_trades = oos_m.get("total_trades", 0)
                oos_wr = oos_m.get("win_rate", 0.0)

                if "equity" in oos_result and len(oos_result["equity"]) > 0:
                    oos_equities.append(oos_result["equity"])
            except Exception as e:
                oos_sharpe = 0.0
                oos_pnl = 0.0
                oos_trades = 0
                oos_wr = 0.0

            window_result = {
                "window": i + 1,
                "is_period": is_dates,
                "oos_period": oos_dates,
                "is_sharpe": is_sharpe,
                "oos_sharpe": oos_sharpe,
                "oos_pnl": oos_pnl,
                "oos_trades": oos_trades,
                "oos_wr": oos_wr,
                "best_params": best_params,
            }
            results.append(window_result)

            # 진행 출력
            p = best_params
            print(
                f"  [{i+1:>2}/{len(windows)}] "
                f"IS={is_sharpe:+.2f} OOS={oos_sharpe:+.2f} "
                f"pnl={oos_pnl:+.4f} tr={oos_trades} "
                f"z_e={p['z_entry']} z_x={p['z_exit']} lb={p['lookback']}",
                flush=True,
            )

        # ── 요약 ──
        oos_sharpes = [r["oos_sharpe"] for r in results]
        oos_pnls = [r["oos_pnl"] for r in results]
        is_sharpes = [r["is_sharpe"] for r in results]

        # OOS 에퀴티 누적
        oos_equity = None
        if oos_equities:
            oos_equity = pd.concat(oos_equities)
            oos_equity = oos_equity[~oos_equity.index.duplicated(keep='first')]
            oos_equity = oos_equity.sort_index()

        summary = {
            "n_windows": len(windows),
            "avg_is_sharpe": np.mean(is_sharpes),
            "avg_oos_sharpe": np.mean(oos_sharpes),
            "median_oos_sharpe": np.median(oos_sharpes),
            "std_oos_sharpe": np.std(oos_sharpes),
            "pct_positive_oos": np.mean([s > 0 for s in oos_sharpes]) * 100,
            "total_oos_pnl": np.sum(oos_pnls),
            "avg_oos_pnl": np.mean(oos_pnls),
            "is_oos_correlation": np.corrcoef(is_sharpes, oos_sharpes)[0, 1]
                if len(is_sharpes) > 1 else 0.0,
            "efficiency_ratio": (
                np.mean(oos_sharpes) / np.mean(is_sharpes)
                if np.mean(is_sharpes) != 0 else 0.0
            ),
        }

        return {
            "windows": results,
            "oos_sharpes": oos_sharpes,
            "oos_pnls": oos_pnls,
            "oos_equity": oos_equity,
            "summary": summary,
        }


# ── Combinatorial Purged Cross-Validation ──────────────────────


class CombinatorialPurgedCV:
    """
    CPCV (Lopez de Prado, 2018)

    데이터를 N개 순차 블록으로 나누고,
    C(N, k)개의 train/test 조합을 생성.
    각 조합에서 test 블록 앞뒤로 purge 적용.

    PBO (Probability of Backtest Overfitting):
      IS 최적 파라미터가 OOS에서 중앙값 이하 성과를 보일 확률.
      PBO > 0.5 → 과적합 가능성 높음.
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test: int = 2,
        purge_length: int = 10,
        param_grid: Optional[dict] = None,
    ):
        self.n_groups = n_groups
        self.n_test = n_test
        self.purge_length = purge_length

        self.param_grid = param_grid or {
            "z_entry": [1.5, 2.0, 2.5],
            "z_exit": [0.0, 0.25, 0.5],
            "lookback": [50, 100, 200],
        }

    def _split_groups(self, n_bars: int) -> List[Tuple[int, int]]:
        """데이터를 N개 순차 블록으로 분할"""
        group_size = n_bars // self.n_groups
        groups = []
        for i in range(self.n_groups):
            start = i * group_size
            end = (i + 1) * group_size if i < self.n_groups - 1 else n_bars
            groups.append((start, end))
        return groups

    def _get_train_test_indices(
        self, groups: List[Tuple[int, int]], test_groups: Tuple[int, ...]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        train/test 인덱스 생성 (purge 적용).

        test 블록 앞뒤 purge_length만큼의 데이터를 train에서 제거.
        """
        n_groups = len(groups)
        test_set = set(test_groups)

        test_indices = []
        train_indices = []

        for g_idx in range(n_groups):
            s, e = groups[g_idx]
            indices = np.arange(s, e)

            if g_idx in test_set:
                test_indices.extend(indices)
            else:
                # Purge: test 블록과 인접한 train 데이터 제거
                purge_mask = np.ones(len(indices), dtype=bool)

                for t_idx in test_set:
                    t_s, t_e = groups[t_idx]
                    # test 블록 바로 앞 그룹이면 끝부분 purge
                    if g_idx == t_idx - 1:
                        purge_end = min(self.purge_length, len(indices))
                        purge_mask[-purge_end:] = False
                    # test 블록 바로 뒤 그룹이면 시작부분 purge
                    if g_idx == t_idx + 1:
                        purge_start = min(self.purge_length, len(indices))
                        purge_mask[:purge_start] = False

                train_indices.extend(indices[purge_mask])

        return np.array(train_indices), np.array(test_indices)

    def _run_backtest_on_indices(
        self,
        y_full: pd.Series, x_full: pd.Series,
        indices: np.ndarray,
        sym_y: str, sym_x: str,
        z_entry: float, z_exit: float, lookback: int,
        use_hmm: bool,
    ) -> dict:
        """인덱스로 선택된 데이터에서 백테스트 실행"""
        y = y_full.iloc[indices]
        x = x_full.iloc[indices]

        strategy = StatArbStrategy(
            z_entry=z_entry,
            z_exit=z_exit,
            z_stop=TRADING_CONFIG["z_score_stop"],
            max_holding_bars=TRADING_CONFIG["max_holding_bars"],
            lookback=lookback,
            use_hmm=use_hmm,
            mr_threshold=TRADING_CONFIG["mr_prob_threshold"],
        )

        signals = strategy.generate_signals(y, x)

        sym_y_cfg = SYMBOLS[sym_y]
        sym_x_cfg = SYMBOLS[sym_x]
        cost_y = sym_y_cfg["pip_size"] * (sym_y_cfg["spread_pips"] + sym_y_cfg["commission_pips"])
        cost_x = sym_x_cfg["pip_size"] * (sym_x_cfg["spread_pips"] + sym_x_cfg["commission_pips"])
        slip_y = sym_y_cfg["pip_size"] * RISK_CONFIG["slippage_pips"]
        slip_x = sym_x_cfg["pip_size"] * RISK_CONFIG["slippage_pips"]

        backtester = StatArbBacktester(
            spread_cost_y=cost_y,
            spread_cost_x=cost_x,
            slippage_y=slip_y,
            slippage_x=slip_x,
        )

        return backtester.run(signals)

    def run(
        self,
        sym_y: str, sym_x: str,
        tf: str = "D1",
        start: str = "2015-01-01",
        end: str = "2025-12-31",
        use_hmm: bool = True,
    ) -> dict:
        """
        CPCV 실행.

        Returns:
            dict with:
              pbo: float - 과적합 확률 [0, 1]
              deflated_sharpe: float - 조정 Sharpe
              is_sharpes: dict - 파라미터별 IS Sharpe
              oos_sharpes: dict - 파라미터별 OOS Sharpe
              combo_results: List[dict] - 각 조합별 결과
        """
        loader = DataLoader()
        data = loader.load_pair(sym_y, sym_x, tf, start, end)
        y_full, x_full = data[sym_y], data[sym_x]

        n = len(y_full)
        groups = self._split_groups(n)

        # C(N, k) 조합 생성
        test_combos = list(itertools.combinations(range(self.n_groups), self.n_test))
        total_combos = len(test_combos)

        print(f"\nCPCV: {sym_y}/{sym_x} ({tf})")
        print(f"Total bars: {n}, Groups: {self.n_groups}, Test groups: {self.n_test}")
        print(f"Combinations: C({self.n_groups},{self.n_test}) = {total_combos}")
        print(f"Purge length: {self.purge_length} bars")
        print(f"{'='*75}")

        # 파라미터 조합
        keys = list(self.param_grid.keys())
        param_combos = list(itertools.product(*self.param_grid.values()))

        # 각 CV 조합에서 모든 파라미터의 IS/OOS Sharpe 수집
        # is_perf[combo_idx][param_idx] = sharpe
        # oos_perf[combo_idx][param_idx] = sharpe
        is_perf = []
        oos_perf = []

        for c_idx, test_groups in enumerate(test_combos):
            train_idx, test_idx = self._get_train_test_indices(groups, test_groups)

            if len(train_idx) < 100 or len(test_idx) < 50:
                continue

            is_sharpes = []
            oos_sharpes = []

            for p_idx, combo in enumerate(param_combos):
                params = dict(zip(keys, combo))

                # IS (train)
                try:
                    is_result = self._run_backtest_on_indices(
                        y_full, x_full, train_idx, sym_y, sym_x,
                        params["z_entry"], params["z_exit"], params["lookback"],
                        use_hmm,
                    )
                    is_m = is_result["metrics"]
                    is_s = is_m.get("sharpe", 0.0) if is_m.get("total_trades", 0) >= 3 else 0.0
                except Exception:
                    is_s = 0.0

                # OOS (test)
                try:
                    oos_result = self._run_backtest_on_indices(
                        y_full, x_full, test_idx, sym_y, sym_x,
                        params["z_entry"], params["z_exit"], params["lookback"],
                        use_hmm,
                    )
                    oos_m = oos_result["metrics"]
                    oos_s = oos_m.get("sharpe", 0.0) if oos_m.get("total_trades", 0) >= 2 else 0.0
                except Exception:
                    oos_s = 0.0

                is_sharpes.append(is_s)
                oos_sharpes.append(oos_s)

            is_perf.append(is_sharpes)
            oos_perf.append(oos_sharpes)

            # 진행 출력
            best_is_idx = np.argmax(is_sharpes)
            best_is_p = dict(zip(keys, param_combos[best_is_idx]))
            print(
                f"  [{c_idx+1:>2}/{total_combos}] "
                f"test={test_groups} "
                f"best_IS={is_sharpes[best_is_idx]:+.2f} "
                f"→ OOS={oos_sharpes[best_is_idx]:+.2f} "
                f"z_e={best_is_p['z_entry']} z_x={best_is_p['z_exit']} lb={best_is_p['lookback']}",
                flush=True,
            )

        # ── PBO 계산 ──
        is_perf = np.array(is_perf)
        oos_perf = np.array(oos_perf)

        n_overfit = 0
        n_valid = 0
        logit_values = []

        for c_idx in range(len(is_perf)):
            # IS 최적 파라미터
            best_is_idx = np.argmax(is_perf[c_idx])
            best_oos_sharpe = oos_perf[c_idx, best_is_idx]

            # OOS에서 IS 최적 파라미터의 순위
            oos_rank = np.sum(oos_perf[c_idx] <= best_oos_sharpe) / len(oos_perf[c_idx])

            if oos_rank <= 0.5:
                n_overfit += 1

            # logit for PBO
            if 0 < oos_rank < 1:
                logit_values.append(np.log(oos_rank / (1 - oos_rank)))

            n_valid += 1

        pbo = n_overfit / n_valid if n_valid > 0 else 1.0

        # ── Deflated Sharpe Ratio ──
        # Bailey & Lopez de Prado (2014)
        n_trials = len(param_combos)
        all_oos_sharpes = oos_perf.flatten()
        all_oos_sharpes = all_oos_sharpes[~np.isnan(all_oos_sharpes)]

        best_sharpe = np.max(all_oos_sharpes) if len(all_oos_sharpes) > 0 else 0.0
        sharpe_std = np.std(all_oos_sharpes) if len(all_oos_sharpes) > 0 else 1.0

        # Expected max Sharpe under null hypothesis
        euler_mascheroni = 0.5772
        expected_max_sharpe = sharpe_std * (
            (1 - euler_mascheroni) * sp_stats.norm.ppf(1 - 1 / n_trials)
            + euler_mascheroni * sp_stats.norm.ppf(1 - 1 / (n_trials * np.e))
        ) if n_trials > 1 else 0.0

        # Deflated Sharpe
        if sharpe_std > 0 and n > 0:
            deflated_sharpe = (best_sharpe - expected_max_sharpe) / (
                sharpe_std / np.sqrt(n_valid)
            ) if n_valid > 0 else 0.0
        else:
            deflated_sharpe = 0.0

        summary = {
            "pbo": pbo,
            "deflated_sharpe": deflated_sharpe,
            "expected_max_sharpe": expected_max_sharpe,
            "best_oos_sharpe": best_sharpe,
            "n_combos_tested": n_valid,
            "n_param_combos": n_trials,
            "avg_oos_sharpe": np.mean(all_oos_sharpes) if len(all_oos_sharpes) > 0 else 0.0,
        }

        return {
            "pbo": pbo,
            "deflated_sharpe": deflated_sharpe,
            "is_perf": is_perf,
            "oos_perf": oos_perf,
            "summary": summary,
            "logit_values": logit_values,
        }


# ── 리포트 출력 ───────────────────────────────────────────────


def print_wfo_report(result: dict, pair_name: str = ""):
    """Walk-Forward 결과 출력"""
    s = result["summary"]

    print(f"\n{'='*65}")
    print(f"  Walk-Forward Report: {pair_name}")
    print(f"{'='*65}")

    print(f"\n  Windows:           {s['n_windows']}")
    print(f"  Avg IS Sharpe:     {s['avg_is_sharpe']:+.3f}")
    print(f"  Avg OOS Sharpe:    {s['avg_oos_sharpe']:+.3f}")
    print(f"  Median OOS Sharpe: {s['median_oos_sharpe']:+.3f}")
    print(f"  OOS Sharpe Std:    {s['std_oos_sharpe']:.3f}")
    print(f"  % Positive OOS:    {s['pct_positive_oos']:.0f}%")
    print(f"  Total OOS P&L:     {s['total_oos_pnl']:+.4f}")
    print(f"  IS/OOS Correlation: {s['is_oos_correlation']:+.3f}")
    print(f"  Efficiency Ratio:  {s['efficiency_ratio']:.2f}")

    # 해석
    print(f"\n  Interpretation:")
    if s["avg_oos_sharpe"] > 0.5:
        print(f"    OOS Sharpe > 0.5 → Strong out-of-sample performance")
    elif s["avg_oos_sharpe"] > 0:
        print(f"    OOS Sharpe > 0 → Positive but moderate")
    else:
        print(f"    OOS Sharpe <= 0 → Poor out-of-sample performance")

    if s["efficiency_ratio"] > 0.5:
        print(f"    Efficiency > 0.5 → Low overfitting risk")
    elif s["efficiency_ratio"] > 0.2:
        print(f"    Efficiency 0.2~0.5 → Moderate overfitting risk")
    else:
        print(f"    Efficiency < 0.2 → High overfitting risk")

    if s["pct_positive_oos"] >= 60:
        print(f"    {s['pct_positive_oos']:.0f}% positive OOS → Consistent edge")
    else:
        print(f"    {s['pct_positive_oos']:.0f}% positive OOS → Inconsistent")

    # 윈도우별 상세
    print(f"\n  Window Details:")
    print(f"  {'#':>3} {'IS Sharpe':>10} {'OOS Sharpe':>11} {'OOS P&L':>10} {'Trades':>7} {'Params'}")
    print(f"  {'-'*70}")

    for w in result["windows"]:
        p = w["best_params"]
        print(
            f"  {w['window']:>3} "
            f"{w['is_sharpe']:>+10.3f} "
            f"{w['oos_sharpe']:>+11.3f} "
            f"{w['oos_pnl']:>+10.4f} "
            f"{w['oos_trades']:>7} "
            f"z_e={p['z_entry']} z_x={p['z_exit']} lb={p['lookback']}"
        )


def print_cpcv_report(result: dict, pair_name: str = ""):
    """CPCV 결과 출력"""
    s = result["summary"]

    print(f"\n{'='*65}")
    print(f"  CPCV Report: {pair_name}")
    print(f"{'='*65}")

    print(f"\n  PBO (Prob. of Backtest Overfitting): {s['pbo']:.2f}")
    print(f"  Deflated Sharpe Ratio:               {s['deflated_sharpe']:+.3f}")
    print(f"  Expected Max Sharpe (null):           {s['expected_max_sharpe']:+.3f}")
    print(f"  Best OOS Sharpe:                      {s['best_oos_sharpe']:+.3f}")
    print(f"  Avg OOS Sharpe:                       {s['avg_oos_sharpe']:+.3f}")
    print(f"  CV Combinations Tested:               {s['n_combos_tested']}")
    print(f"  Parameter Combinations:               {s['n_param_combos']}")

    # 해석
    print(f"\n  Interpretation:")
    if s["pbo"] < 0.3:
        print(f"    PBO = {s['pbo']:.2f} → LOW overfitting risk (good)")
    elif s["pbo"] < 0.5:
        print(f"    PBO = {s['pbo']:.2f} → MODERATE overfitting risk")
    else:
        print(f"    PBO = {s['pbo']:.2f} → HIGH overfitting risk (bad)")

    if s["deflated_sharpe"] > 0:
        print(f"    Deflated Sharpe > 0 → Strategy has genuine edge after adjustment")
    else:
        print(f"    Deflated Sharpe <= 0 → Edge may be due to multiple testing")


# ── CLI ───────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="StatArb Walk-Forward + CPCV Validation")
    parser.add_argument("--pair", type=str, default="EURCHF,USDJPY",
                        help="Pair (y,x)")
    parser.add_argument("--tf", type=str, default="D1")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["wfo", "cpcv", "all"],
                        help="Validation mode")
    parser.add_argument("--no-hmm", action="store_true")
    parser.add_argument("--is-days", type=int, default=500,
                        help="IS window (D1 bars)")
    parser.add_argument("--oos-days", type=int, default=126,
                        help="OOS window (D1 bars)")
    parser.add_argument("--roll-days", type=int, default=63,
                        help="Roll step (D1 bars)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to CSV")
    args = parser.parse_args()

    sym_y, sym_x = args.pair.split(",")
    pair_name = f"{sym_y}/{sym_x} ({args.tf})"
    use_hmm = not args.no_hmm

    if args.mode in ("wfo", "all"):
        print(f"\n{'#'*65}")
        print(f"  Walk-Forward Optimization")
        print(f"{'#'*65}")

        wfo = WalkForwardOptimizer(
            is_days=args.is_days,
            oos_days=args.oos_days,
            roll_days=args.roll_days,
        )
        wfo_result = wfo.run(
            sym_y, sym_x, args.tf, args.start, args.end, use_hmm
        )
        print_wfo_report(wfo_result, pair_name)

        if args.output:
            df = pd.DataFrame(wfo_result["windows"])
            df.to_csv(args.output.replace(".csv", "_wfo.csv"), index=False)
            print(f"\nSaved WFO results to {args.output.replace('.csv', '_wfo.csv')}")

    if args.mode in ("cpcv", "all"):
        print(f"\n{'#'*65}")
        print(f"  Combinatorial Purged Cross-Validation")
        print(f"{'#'*65}")

        cpcv = CombinatorialPurgedCV(
            n_groups=WFO_CONFIG.get("cpcv_n_groups", 6),
            n_test=WFO_CONFIG.get("cpcv_n_test", 2),
            purge_length=WFO_CONFIG.get("purge_length", 10),
        )
        cpcv_result = cpcv.run(
            sym_y, sym_x, args.tf, args.start, args.end, use_hmm
        )
        print_cpcv_report(cpcv_result, pair_name)


if __name__ == "__main__":
    main()
