# -*- coding: utf-8 -*-
"""
StatArb - 자동 Cointegration 페어 스캐닝

DB에 있는 심볼들 간의 cointegration 관계를 자동 검정하고,
복수 타임프레임에서 안정성을 평가하여 최적 페어를 추천.

사용법:
    # DB에 있는 심볼로 EURUSD 기준 스캔
    python pair_scanner.py --base EURUSD --timeframe H1 --start 2020-01-01 --end 2025-12-31

    # 전체 조합 스캔
    python pair_scanner.py --all --timeframe H1 --start 2020-01-01 --end 2025-12-31

    # 복수 타임프레임 스캔
    python pair_scanner.py --base EURUSD --timeframe H1,H4,D1 --start 2020-01-01
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from data_loader import DataLoader
from cointegration_analyzer import CointegrationAnalyzer
from config import CANDIDATE_PAIRS, COINT_CONFIG


def scan_base_pairs(
    loader: DataLoader,
    analyzer: CointegrationAnalyzer,
    base_symbol: str,
    candidate_symbols: list,
    timeframe: str = "H1",
    start: str = None,
    end: str = None,
) -> pd.DataFrame:
    """
    기준 심볼과 후보 심볼들 간의 cointegration 검정.

    Returns:
        검정 결과 DataFrame (p_value 오름차순 정렬)
    """
    results = []

    for sym_x in candidate_symbols:
        if sym_x == base_symbol:
            continue

        print(f"  Testing {base_symbol}/{sym_x} ({timeframe})...", end=" ")

        try:
            pair_data = loader.load_pair(base_symbol, sym_x, timeframe, start, end)

            y = pair_data[base_symbol]
            x = pair_data[sym_x]

            # Engle-Granger 검정
            eg = analyzer.test_engle_granger(y, x, COINT_CONFIG["significance_level"])

            # 반감기
            half_life = analyzer.estimate_half_life(eg["residuals"])

            # 상관계수
            corr = y.corr(x)

            # 스프레드 정상성 직접 검정
            adf = analyzer.test_stationarity(eg["residuals"])

            # Rolling cointegration 안정성 (간략 버전)
            rolling = analyzer.rolling_cointegration(
                y, x,
                window=COINT_CONFIG["rolling_window"],
                step=max(1, len(y) // 20),  # ~20개 구간
            )
            stability = rolling["is_cointegrated"].mean() if len(rolling) > 0 else 0.0

            result = {
                "pair": f"{base_symbol}/{sym_x}",
                "timeframe": timeframe,
                "p_value": eg["p_value"],
                "test_stat": eg["test_stat"],
                "hedge_ratio": eg["hedge_ratio"],
                "half_life": half_life,
                "correlation": corr,
                "adf_spread_p": adf["p_value"],
                "stability": stability,
                "is_cointegrated": eg["is_cointegrated"],
                "n_obs": len(y),
            }

            status = "COINT" if eg["is_cointegrated"] else "no"
            hl_str = f"{half_life:.0f}" if np.isfinite(half_life) else "inf"
            print(f"p={eg['p_value']:.4f} β={eg['hedge_ratio']:.4f} "
                  f"HL={hl_str} stab={stability:.0%} [{status}]")

        except Exception as e:
            result = {
                "pair": f"{base_symbol}/{sym_x}",
                "timeframe": timeframe,
                "p_value": 1.0,
                "test_stat": np.nan,
                "hedge_ratio": np.nan,
                "half_life": np.nan,
                "correlation": np.nan,
                "adf_spread_p": np.nan,
                "stability": 0.0,
                "is_cointegrated": False,
                "n_obs": 0,
            }
            print(f"ERROR: {e}")

        results.append(result)

    df = pd.DataFrame(results).sort_values("p_value").reset_index(drop=True)
    return df


def print_results(results: pd.DataFrame, title: str = ""):
    """검정 결과를 깔끔하게 출력"""
    if title:
        print(f"\n{'=' * 80}")
        print(f"  {title}")
        print(f"{'=' * 80}")

    if len(results) == 0:
        print("  No results.")
        return

    # 주요 컬럼만 출력
    display_cols = ["pair", "timeframe", "p_value", "hedge_ratio", "half_life",
                    "correlation", "stability", "is_cointegrated", "n_obs"]
    available = [c for c in display_cols if c in results.columns]

    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(results[available].to_string(index=False))

    # Cointegrated 페어 요약
    coint_pairs = results[results["is_cointegrated"]]
    print(f"\nCointegrated pairs: {len(coint_pairs)} / {len(results)}")

    if len(coint_pairs) > 0:
        # 반감기 유효 범위 내 페어
        min_hl = COINT_CONFIG["min_half_life"]
        max_hl = COINT_CONFIG["max_half_life"]
        valid = coint_pairs[
            (coint_pairs["half_life"] >= min_hl) &
            (coint_pairs["half_life"] <= max_hl)
        ]
        print(f"Valid half-life ({min_hl}-{max_hl} bars): {len(valid)} pairs")

        if len(valid) > 0:
            best = valid.iloc[0]
            print(f"\nBest pair: {best['pair']} "
                  f"(p={best['p_value']:.4f}, HL={best['half_life']:.0f}, "
                  f"stab={best['stability']:.0%})")


def main():
    parser = argparse.ArgumentParser(description="StatArb Cointegration Pair Scanner")
    parser.add_argument("--base", type=str, default="EURUSD",
                        help="Base symbol for pair scanning")
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated candidate symbols (default: all DB symbols)")
    parser.add_argument("--timeframe", type=str, default="H1",
                        help="Timeframe(s), comma-separated (e.g., H1,H4,D1)")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--all", action="store_true",
                        help="Scan all pairwise combinations (ignores --base)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to CSV")
    args = parser.parse_args()

    loader = DataLoader()
    analyzer = CointegrationAnalyzer()

    # DB 상태 출력
    print("DB Status:")
    loader.print_status()

    # 후보 심볼 결정
    if args.symbols:
        candidates = args.symbols.split(",")
    else:
        candidates = loader.list_symbols()
        # CANDIDATE_PAIRS에서 추출
        if not args.all:
            pair_symbols = set()
            for sy, sx in CANDIDATE_PAIRS:
                if sy == args.base:
                    pair_symbols.add(sx)
            # DB에 있는 것만 필터
            db_symbols = set(candidates)
            candidates = sorted(pair_symbols & db_symbols)
            if not candidates:
                candidates = [s for s in db_symbols if s != args.base]

    print(f"\nBase: {args.base}")
    print(f"Candidates: {candidates}")

    # 타임프레임 파싱
    timeframes = args.timeframe.split(",")

    # 스캔 실행
    all_results = []
    for tf in timeframes:
        tf = tf.strip()
        print(f"\n--- Scanning {tf} ---")

        if args.all:
            # 전체 조합
            try:
                all_symbols = list(dict.fromkeys(candidates + [args.base]))  # 중복 제거
                data = loader.load_multi(all_symbols, tf, args.start, args.end)
                results = analyzer.scan_pairs(data, COINT_CONFIG["significance_level"])
                results["timeframe"] = tf
            except Exception as e:
                print(f"Error loading multi data: {e}")
                continue
        else:
            results = scan_base_pairs(
                loader, analyzer, args.base, candidates, tf, args.start, args.end,
            )

        all_results.append(results)
        print_results(results, f"{args.base} Cointegration Scan - {tf}")

    # 복수 타임프레임 결과 병합
    if len(all_results) > 1:
        combined = pd.concat(all_results, ignore_index=True)
        print_results(combined, "Combined Multi-Timeframe Results")

        if args.output:
            combined.to_csv(args.output, index=False)
            print(f"\nResults saved to {args.output}")
    elif len(all_results) == 1 and args.output:
        all_results[0].to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
