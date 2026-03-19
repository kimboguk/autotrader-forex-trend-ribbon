# -*- coding: utf-8 -*-
"""
Phase 1: H1 Rolling Cointegration Scan

H1 타임프레임에서 짧은 rolling window로 공적분이 얼마나 자주 나타나는지 탐색.
구현 전 검증 단계 — 공적분 발생 빈도, 지속 기간, 연간 기회 횟수 확인.
"""

import sys
import os
import numpy as np
import pandas as pd
import warnings
from itertools import groupby

warnings.filterwarnings("ignore")

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from data_loader import DataLoader
from cointegration_analyzer import CointegrationAnalyzer


PAIRS = [
    ("AUDUSD", "NZDUSD"),
    ("EURCHF", "USDJPY"),
    ("USDCHF", "EURJPY"),
]

WINDOWS = [240, 500, 720]  # 2주, 1개월, 1.5개월 (H1 바)
STEP = 24                   # 1일 (H1 24바)
P_THRESHOLD = 0.05


def load_h1_pair_chunked(loader, sym_y, sym_x, start_year=2006, end_year=2026):
    """M1 데이터를 연도별로 분할 로드 후 H1 리샘플링, 병합"""
    chunks = []
    for yr in range(start_year, end_year + 1):
        s = f"{yr}-01-01"
        e = f"{yr}-12-31"
        try:
            data = loader.load_pair(sym_y, sym_x, "H1", s, e)
            if len(data) > 0:
                chunks.append(data)
        except Exception:
            continue
    if not chunks:
        raise ValueError(f"No H1 data for {sym_y}/{sym_x}")
    merged = pd.concat(chunks)
    merged = merged[~merged.index.duplicated(keep='first')].sort_index()
    return merged


def scan_pair(sym_y: str, sym_x: str, window: int) -> dict:
    """단일 페어 + 단일 윈도우 크기로 rolling cointegration 스캔"""
    loader = DataLoader()
    analyzer = CointegrationAnalyzer()

    data = load_h1_pair_chunked(loader, sym_y, sym_x)
    y, x = data[sym_y], data[sym_x]
    n = len(data)

    results = []
    total_checks = 0

    for t in range(window, n, STEP):
        wy = y.iloc[t - window:t]
        wx = x.iloc[t - window:t]

        try:
            eg = analyzer.test_engle_granger(wy, wx)
            hl = analyzer.estimate_half_life(eg["residuals"])
        except Exception:
            continue

        total_checks += 1
        results.append({
            "time": data.index[t],
            "p_value": eg["p_value"],
            "is_coint": eg["is_cointegrated"],
            "hedge_ratio": eg["hedge_ratio"],
            "half_life": hl,
        })

    df = pd.DataFrame(results)
    if len(df) == 0:
        return None

    # ── 통계 계산 ──
    coint_mask = df["is_coint"].values
    n_coint = coint_mask.sum()
    coint_pct = n_coint / len(df) * 100

    # 연속 공적분 구간 분석
    runs = []
    for val, group in groupby(coint_mask):
        length = sum(1 for _ in group)
        if val:
            runs.append(length * STEP)  # H1 바 단위

    avg_run = np.mean(runs) if runs else 0
    max_run = max(runs) if runs else 0
    n_episodes = len(runs)

    # 연간 통계
    df["year"] = df["time"].dt.year
    years = df["year"].unique()
    n_years = len(years)

    yearly_stats = []
    for yr in sorted(years):
        yr_df = df[df["year"] == yr]
        yr_coint = yr_df["is_coint"].sum()
        yr_pct = yr_coint / len(yr_df) * 100
        # 연간 에피소드 수
        yr_mask = yr_df["is_coint"].values
        yr_episodes = sum(1 for val, _ in groupby(yr_mask) if val)
        yearly_stats.append({
            "year": yr, "checks": len(yr_df),
            "coint": yr_coint, "pct": yr_pct,
            "episodes": yr_episodes,
        })

    return {
        "pair": f"{sym_y}/{sym_x}",
        "window": window,
        "total_bars": n,
        "period": f"{data.index[0].date()} ~ {data.index[-1].date()}",
        "total_checks": len(df),
        "coint_checks": n_coint,
        "coint_pct": coint_pct,
        "n_episodes": n_episodes,
        "avg_run_h1": avg_run,
        "max_run_h1": max_run,
        "avg_episodes_per_year": n_episodes / n_years if n_years > 0 else 0,
        "yearly": yearly_stats,
        "df": df,
    }


def main():
    os.makedirs("outputs", exist_ok=True)

    for sym_y, sym_x in PAIRS:
        print(f"\n{'='*80}")
        print(f"  {sym_y}/{sym_x} — H1 Rolling Cointegration Scan")
        print(f"{'='*80}")

        for window in WINDOWS:
            window_label = f"{window}h ({window/24:.0f}d)"
            print(f"\n  Window: {window_label}, Step: {STEP}h (1d)")
            print(f"  {'─'*60}")

            result = scan_pair(sym_y, sym_x, window)
            if result is None:
                print("  No results")
                continue

            print(f"  Period: {result['period']}")
            print(f"  Total checks: {result['total_checks']}")
            print(f"  Cointegrated: {result['coint_checks']} ({result['coint_pct']:.1f}%)")
            print(f"  Episodes: {result['n_episodes']} "
                  f"(avg {result['avg_episodes_per_year']:.1f}/year)")
            print(f"  Avg duration: {result['avg_run_h1']:.0f}h "
                  f"({result['avg_run_h1']/24:.1f}d)")
            print(f"  Max duration: {result['max_run_h1']:.0f}h "
                  f"({result['max_run_h1']/24:.1f}d)")

            # 연도별 통계
            print(f"\n  {'Year':>6} {'Checks':>7} {'Coint':>6} {'%':>6} {'Episodes':>9}")
            print(f"  {'-'*40}")
            for ys in result["yearly"]:
                print(f"  {ys['year']:>6} {ys['checks']:>7} {ys['coint']:>6} "
                      f"{ys['pct']:>5.1f}% {ys['episodes']:>9}")

            # CSV 저장
            safe = f"{sym_y}_{sym_x}"
            csv_path = f"outputs/h1_coint_scan_{safe}_w{window}.csv"
            result["df"].to_csv(csv_path, index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
