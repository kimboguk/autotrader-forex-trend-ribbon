# -*- coding: utf-8 -*-
"""
StatArb - Full Backtest Report (전체 거래 내역 + 연간 통계)

상위 3개 페어에 대해 DB 전체 데이터로 백테스트 실행.
모든 진입/청산 내역을 CSV로 저장하고, 연간 통계를 산출.
"""

import sys
import os
import numpy as np
import pandas as pd

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from data_loader import DataLoader
from config import SYMBOLS, TRADING_CONFIG, RISK_CONFIG
from strategy import StatArbStrategy, StatArbBacktester


# ── 상위 3개 페어 + 최적 파라미터 (그리드 서치 결과) ──

PAIRS = [
    {
        "sym_y": "AUDUSD", "sym_x": "NZDUSD",
        "z_entry": 1.5, "z_exit": 0.25, "use_hmm": False, "lookback": 200,
        "start": "2006-01-01",  # NZDUSD 데이터 시작: 2005-08
    },
    {
        "sym_y": "EURCHF", "sym_x": "USDJPY",
        "z_entry": 1.5, "z_exit": 0.0, "use_hmm": True, "lookback": 100,
        "start": "2003-01-01",  # EURCHF 데이터 시작: 2002-01
    },
    {
        "sym_y": "USDCHF", "sym_x": "EURJPY",
        "z_entry": 2.0, "z_exit": 0.25, "use_hmm": False, "lookback": 200,
        "start": "2003-01-01",  # EURJPY 데이터 시작: 2002-03
    },
]

END_DATE = "2026-02-28"
TIMEFRAME = "D1"


def run_pair_backtest(pair_cfg: dict) -> dict:
    """단일 페어 백테스트 실행, 전체 거래 내역 반환"""
    sym_y = pair_cfg["sym_y"]
    sym_x = pair_cfg["sym_x"]
    start = pair_cfg["start"]

    print(f"\n{'='*70}")
    print(f"  {sym_y}/{sym_x} ({TIMEFRAME})  |  {start} ~ {END_DATE}")
    print(f"  Params: z_e={pair_cfg['z_entry']} z_x={pair_cfg['z_exit']} "
          f"hmm={pair_cfg['use_hmm']} lb={pair_cfg['lookback']}")
    print(f"{'='*70}")

    # 데이터 로드
    loader = DataLoader()
    data = loader.load_pair(sym_y, sym_x, TIMEFRAME, start, END_DATE)
    y, x = data[sym_y], data[sym_x]
    print(f"  Data: {len(data)} bars ({data.index[0].date()} ~ {data.index[-1].date()})")

    # 전략 실행
    strategy = StatArbStrategy(
        z_entry=pair_cfg["z_entry"],
        z_exit=pair_cfg["z_exit"],
        z_stop=TRADING_CONFIG["z_score_stop"],
        max_holding_bars=TRADING_CONFIG["max_holding_bars"],
        lookback=pair_cfg["lookback"],
        use_hmm=pair_cfg["use_hmm"],
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
        spread_cost_y=cost_y, spread_cost_x=cost_x,
        slippage_y=slip_y, slippage_x=slip_x,
    )
    result = backtester.run(signals)

    trades = result["trades"]
    if len(trades) == 0:
        print("  *** No trades ***")
        return {"pair": f"{sym_y}/{sym_x}", "trades": pd.DataFrame(), "equity": result["equity"]}

    # ── 거래별 상세 컬럼 추가 ──
    trades["pair"] = f"{sym_y}/{sym_x}"
    trades["trade_no"] = range(1, len(trades) + 1)

    # Y leg pnl (pips)
    trades["y_pnl_pips"] = trades["y_pnl"] / sym_y_cfg["pip_size"]
    trades["x_pnl_pips"] = trades["x_pnl"] / sym_x_cfg["pip_size"]
    trades["cost_pips"] = trades["cost"] / sym_y_cfg["pip_size"]

    # Spread PnL in pips (y 기준)
    trades["pnl_pips"] = trades["pnl"] / sym_y_cfg["pip_size"]

    # 승/패 구분
    trades["result"] = np.where(trades["pnl"] > 0, "WIN", np.where(trades["pnl"] < 0, "LOSS", "BE"))

    # RRR 계산: z-score 기반
    # 진입 z에서 TP(z_exit)까지 거리 vs 진입 z에서 SL(z_stop)까지 거리
    z_entry_thresh = pair_cfg["z_entry"]
    z_exit_thresh = pair_cfg["z_exit"]
    z_stop = TRADING_CONFIG["z_score_stop"]

    # Planned RRR = (|entry_z| - z_exit) / (z_stop - |entry_z|)
    trades["planned_rrr"] = (abs(trades["entry_z"]) - z_exit_thresh) / (z_stop - abs(trades["entry_z"])).replace(0, np.nan)
    trades["planned_rrr"] = trades["planned_rrr"].clip(lower=0)

    # Actual RRR = |exit_z - entry_z| / |entry_z - 0| (z-score 이동 비율)
    # 더 직관적: actual_reward / planned_risk
    # 승리 거래: pnl > 0 → RRR = pnl / avg_loss_size
    # 패배 거래: pnl < 0 → RRR = -pnl / avg_loss_size (negative)
    avg_loss = abs(trades.loc[trades["pnl"] < 0, "pnl"].mean()) if (trades["pnl"] < 0).any() else 1e-10
    trades["actual_R"] = trades["pnl"] / avg_loss  # R-multiple

    # 보유 기간 (일)
    trades["hold_days"] = trades["bars_held"]  # D1이므로 bars = days

    # 정렬된 출력 컬럼
    output_cols = [
        "pair", "trade_no", "direction",
        "entry_time", "exit_time", "hold_days",
        "entry_y", "exit_y", "entry_x", "exit_x",
        "entry_z", "exit_z",
        "y_pnl_pips", "x_pnl_pips", "cost_pips", "pnl_pips",
        "pnl", "cum_pnl", "result",
        "planned_rrr", "actual_R",
        "mr_prob_entry",
    ]
    available_cols = [c for c in output_cols if c in trades.columns]
    trades_out = trades[available_cols].copy()

    print(f"  Trades: {len(trades)}  |  Win Rate: {(trades['pnl']>0).mean()*100:.1f}%")
    print(f"  Total PnL: {trades['pnl'].sum():.4f} ({trades['pnl_pips'].sum():.0f} pips)")

    return {
        "pair": f"{sym_y}/{sym_x}",
        "trades": trades_out,
        "equity": result["equity"],
        "metrics": result["metrics"],
        "sym_y_pip": sym_y_cfg["pip_size"],
    }


def compute_annual_stats(trades: pd.DataFrame, equity: pd.Series, sym_y_pip: float) -> pd.DataFrame:
    """연간 통계 계산"""
    if len(trades) == 0:
        return pd.DataFrame()

    trades = trades.copy()
    trades["year"] = pd.to_datetime(trades["entry_time"]).dt.year

    years = sorted(trades["year"].unique())
    rows = []

    for yr in years:
        yr_trades = trades[trades["year"] == yr]
        n = len(yr_trades)
        wins = (yr_trades["pnl"] > 0).sum()
        losses = (yr_trades["pnl"] < 0).sum()
        wr = wins / n * 100 if n > 0 else 0

        total_pnl = yr_trades["pnl"].sum()
        pnl_pips = yr_trades["pnl_pips"].sum() if "pnl_pips" in yr_trades.columns else total_pnl / sym_y_pip

        avg_pnl = yr_trades["pnl"].mean()
        avg_win = yr_trades.loc[yr_trades["pnl"] > 0, "pnl"].mean() if wins > 0 else 0
        avg_loss = yr_trades.loc[yr_trades["pnl"] < 0, "pnl"].mean() if losses > 0 else 0

        # Profit factor
        gross_win = yr_trades.loc[yr_trades["pnl"] > 0, "pnl"].sum() if wins > 0 else 0
        gross_loss = abs(yr_trades.loc[yr_trades["pnl"] < 0, "pnl"].sum()) if losses > 0 else 1e-10
        pf = gross_win / gross_loss

        # 평균 보유 기간
        avg_hold = yr_trades["hold_days"].mean() if "hold_days" in yr_trades.columns else yr_trades["bars_held"].mean()

        # Avg R-multiple
        avg_R = yr_trades["actual_R"].mean() if "actual_R" in yr_trades.columns else 0

        # Max Drawdown (해당 연도 equity curve에서)
        yr_start = f"{yr}-01-01"
        yr_end = f"{yr}-12-31"
        yr_eq = equity[(equity.index >= yr_start) & (equity.index <= yr_end)]
        if len(yr_eq) > 1:
            peak = yr_eq.cummax()
            dd = yr_eq - peak
            max_dd = dd.min()
            # MDD를 가격 단위 → pips 변환
            max_dd_pips = max_dd / sym_y_pip if sym_y_pip > 0 else max_dd
        else:
            max_dd = 0
            max_dd_pips = 0

        # 연간 Sharpe (daily returns 기반)
        yr_returns = yr_eq.pct_change().dropna()
        if len(yr_returns) > 10 and yr_returns.std() > 0:
            sharpe = yr_returns.mean() / yr_returns.std() * np.sqrt(252)
        else:
            sharpe = 0.0

        rows.append({
            "Year": yr,
            "Trades": n,
            "Wins": wins,
            "Losses": losses,
            "WinRate%": round(wr, 1),
            "PnL(price)": round(total_pnl, 6),
            "PnL(pips)": round(pnl_pips, 1),
            "AvgPnL": round(avg_pnl, 6),
            "AvgWin": round(avg_win, 6) if avg_win else 0,
            "AvgLoss": round(avg_loss, 6) if avg_loss else 0,
            "PF": round(pf, 2),
            "AvgR": round(avg_R, 2),
            "AvgHold(d)": round(avg_hold, 0),
            "MDD(pips)": round(max_dd_pips, 1),
            "Sharpe": round(sharpe, 2),
        })

    df = pd.DataFrame(rows)

    # 전체 합계 행
    all_n = len(trades)
    all_wins = (trades["pnl"] > 0).sum()
    all_losses = (trades["pnl"] < 0).sum()
    all_wr = all_wins / all_n * 100 if all_n > 0 else 0
    all_pnl = trades["pnl"].sum()
    all_pips = trades["pnl_pips"].sum() if "pnl_pips" in trades.columns else all_pnl / sym_y_pip
    all_avg = trades["pnl"].mean()
    all_avgwin = trades.loc[trades["pnl"] > 0, "pnl"].mean() if all_wins > 0 else 0
    all_avgloss = trades.loc[trades["pnl"] < 0, "pnl"].mean() if all_losses > 0 else 0
    gw = trades.loc[trades["pnl"] > 0, "pnl"].sum() if all_wins > 0 else 0
    gl = abs(trades.loc[trades["pnl"] < 0, "pnl"].sum()) if all_losses > 0 else 1e-10
    all_pf = gw / gl
    all_avghold = trades["hold_days"].mean() if "hold_days" in trades.columns else trades["bars_held"].mean()
    all_avgR = trades["actual_R"].mean() if "actual_R" in trades.columns else 0

    # 전체 MDD
    if len(equity) > 1:
        peak = equity.cummax()
        dd = equity - peak
        all_mdd = dd.min() / sym_y_pip
    else:
        all_mdd = 0

    # 전체 Sharpe
    all_ret = equity.pct_change().dropna()
    if len(all_ret) > 10 and all_ret.std() > 0:
        all_sharpe = all_ret.mean() / all_ret.std() * np.sqrt(252)
    else:
        all_sharpe = 0

    total_row = pd.DataFrame([{
        "Year": "TOTAL",
        "Trades": all_n,
        "Wins": all_wins,
        "Losses": all_losses,
        "WinRate%": round(all_wr, 1),
        "PnL(price)": round(all_pnl, 6),
        "PnL(pips)": round(all_pips, 1),
        "AvgPnL": round(all_avg, 6),
        "AvgWin": round(all_avgwin, 6),
        "AvgLoss": round(all_avgloss, 6),
        "PF": round(all_pf, 2),
        "AvgR": round(all_avgR, 2),
        "AvgHold(d)": round(all_avghold, 0),
        "MDD(pips)": round(all_mdd, 1),
        "Sharpe": round(all_sharpe, 2),
    }])

    df = pd.concat([df, total_row], ignore_index=True)
    return df


def main():
    os.makedirs("outputs", exist_ok=True)

    all_trades = []
    all_annual = {}

    for pair_cfg in PAIRS:
        result = run_pair_backtest(pair_cfg)
        pair_name = result["pair"]

        if len(result["trades"]) == 0:
            continue

        trades = result["trades"]
        all_trades.append(trades)

        # CSV 저장: 전체 거래 내역
        safe_name = pair_name.replace("/", "_")
        csv_path = f"outputs/trades_{safe_name}.csv"
        trades.to_csv(csv_path, index=False, float_format="%.6f")
        print(f"  → Saved: {csv_path} ({len(trades)} trades)")

        # 연간 통계
        annual = compute_annual_stats(trades, result["equity"], result["sym_y_pip"])
        all_annual[pair_name] = annual

        # 연간 통계 출력
        print(f"\n  ── {pair_name} Annual Stats ──")
        pd.set_option("display.max_columns", 20)
        pd.set_option("display.width", 200)
        pd.set_option("display.float_format", lambda x: f"{x:.2f}" if abs(x) >= 0.01 else f"{x:.6f}")
        print(annual.to_string(index=False))

        # 연간 통계 CSV 저장
        annual_path = f"outputs/annual_{safe_name}.csv"
        annual.to_csv(annual_path, index=False)
        print(f"  → Saved: {annual_path}")

    # ── 전체 합산 거래 내역 CSV ──
    if all_trades:
        combined = pd.concat(all_trades, ignore_index=True)
        combined.to_csv("outputs/trades_ALL_3pairs.csv", index=False, float_format="%.6f")
        print(f"\n  → Combined trades: outputs/trades_ALL_3pairs.csv ({len(combined)} trades)")

    # ── 3개 페어 비교 요약 ──
    if all_annual:
        print(f"\n\n{'='*80}")
        print(f"  SUMMARY: Top 3 Pairs Comparison (TOTAL row)")
        print(f"{'='*80}")
        rows = []
        for pair_name, annual in all_annual.items():
            total = annual[annual["Year"] == "TOTAL"].iloc[0]
            rows.append({
                "Pair": pair_name,
                "Period": f"{annual.iloc[0]['Year']}~{annual.iloc[-2]['Year']}",
                "Trades": int(total["Trades"]),
                "WinRate%": total["WinRate%"],
                "PnL(pips)": total["PnL(pips)"],
                "PF": total["PF"],
                "AvgR": total["AvgR"],
                "MDD(pips)": total["MDD(pips)"],
                "Sharpe": total["Sharpe"],
                "AvgHold(d)": total["AvgHold(d)"],
            })
        summary = pd.DataFrame(rows)
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
