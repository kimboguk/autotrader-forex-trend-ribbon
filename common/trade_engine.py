# -*- coding: utf-8 -*-
"""
Common Trade Engine - strategy-agnostic trade simulation and statistics.

Extracted from trend_grid/backtest.py. Any strategy that produces a DataFrame
with a 'signal' column (0, 1, -1, 2, -2) can reuse these functions.
"""

import numpy as np
import pandas as pd

from config import SYMBOLS, BACKTEST_CONFIG, RESAMPLE_RULES


# ── Data Loading ────────────────────────────────────────────

_m1_cache: dict[tuple, pd.DataFrame] = {}


def clear_m1_cache():
    """Clear M1 cache (call after each backtest run to free memory)."""
    _m1_cache.clear()


def _load_m1_cached(symbol: str, start: str = None, end: str = None) -> pd.DataFrame:
    """Load M1 data with caching. Same (symbol, start, end) returns cached copy."""
    key = (symbol, start, end)
    if key not in _m1_cache:
        from data_loader import DataLoader
        loader = DataLoader()
        _m1_cache[key] = loader.load_m1(symbol, start, end)
    return _m1_cache[key]


def load_ohlcv(symbol: str, timeframe: str,
               start: str = None, end: str = None) -> pd.DataFrame:
    """Load M1 data from DB and resample to target timeframe."""
    m1 = _load_m1_cached(symbol, start, end)

    if len(m1) == 0:
        raise ValueError(f"No M1 data for {symbol}")

    if timeframe == "M1":
        return m1.copy()

    rule = RESAMPLE_RULES.get(timeframe)
    if rule is None:
        raise ValueError(f"Unknown timeframe: {timeframe}")

    resampled = m1.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "tick_volume": "sum",
    }).dropna(subset=["open"])

    return resampled


# ── Trade Cost ──────────────────────────────────────────────

def calc_trade_cost(symbol: str) -> float:
    """
    One-way trade cost in price units.
    For forex/index: (spread_pips + commission_pips) / 2 * pip_size
    For crypto: 0 (fee calculated as percentage in _record_trade)
    """
    sym_cfg = SYMBOLS[symbol]
    category = sym_cfg.get("category", "forex")
    if category == "crypto" and sym_cfg.get("fee_rate"):
        return 0.0  # crypto uses percentage-based fees
    pip_size = sym_cfg["pip_size"]
    one_way = (sym_cfg["spread_pips"] + sym_cfg["commission_pips"]) / 2 * pip_size
    return one_way


# ── Trade Simulation ────────────────────────────────────────

def simulate_trades(
    grid: pd.DataFrame,
    symbol: str,
    tp_pips: float = None,
    sl_pips: float = None,
    filter_positions: dict = None,
    alignment_cols: list = None,
    progress_callback=None,
    compound: bool = False,
    leverage: int = 1,
    kelly_fraction: float = 0.0,
    next_bar_open: bool = True,
    allowed_entry_hours: set = None,
    htf_exit: bool = False,
) -> tuple[list[dict], np.ndarray]:
    """
    Bar-by-bar trade simulation on a DataFrame with 'signal' column.

    Args:
        grid: DataFrame with OHLC + 'signal' column. Index = DatetimeIndex.
        symbol: Symbol name (for cost/pip config lookup).
        tp_pips, sl_pips: Optional TP/SL in pips.
        filter_positions: Dict of {tf_name: np.array} for higher TF position filter.
        alignment_cols: List of MA column names for alignment filter.
        progress_callback: Called with year int on year change.

    Returns:
        (trades_list, equity_array)
    """
    sym_cfg = SYMBOLS[symbol]
    cost_per_side = calc_trade_cost(symbol)
    pip_size = sym_cfg["pip_size"]
    quote_ccy = sym_cfg.get("quote_ccy", "USD")
    category = sym_cfg.get("category", "forex")

    # Position sizing depends on asset category
    if category == "index":
        point_value = sym_cfg.get("point_value", 1)
        contracts = sym_cfg.get("lot_size", 1)
    elif category == "crypto":
        contracts = sym_cfg.get("lot_size", 1)
        fee_rate = sym_cfg.get("fee_rate", 0.001)
    else:  # forex
        lot_size = sym_cfg.get("lot_size", BACKTEST_CONFIG["lot_size"])
        pos_lots = BACKTEST_CONFIG["position_size_lots"]
        units = pos_lots * lot_size

    initial_capital = BACKTEST_CONFIG["initial_capital"]
    equity = initial_capital

    # Kelly position sizer (kelly_fraction > 0 enables Kelly compound mode)
    kelly_sizer = None
    if kelly_fraction > 0:
        from kelly import KellyPositionSizer
        kelly_sizer = KellyPositionSizer(kelly_fraction=kelly_fraction)

    tp_price = tp_pips * pip_size if tp_pips else None
    sl_price = sl_pips * pip_size if sl_pips else None

    filter_positions = filter_positions or {}
    filter_arrs = {k: v.values if hasattr(v, 'values') else v for k, v in filter_positions.items()}

    alignment_arrs = [grid[col].values for col in (alignment_cols or [])]

    trades = []
    n_bars = len(grid)
    equity_arr = np.full(n_bars, initial_capital, dtype=np.float64)

    entry_price = 0.0
    entry_time = None
    entry_dir = 0  # 1=long, -1=short

    def _record_trade(exit_time, exit_price, direction, exit_reason="signal"):
        nonlocal equity
        if direction == "long":
            pnl_price = exit_price - entry_price
        else:
            pnl_price = entry_price - exit_price

        # Position scale
        if compound and kelly_sizer is not None:
            scale = kelly_sizer.get_scale(equity, initial_capital) * leverage
        elif compound and equity > 0:
            scale = (equity / initial_capital) * leverage
        else:
            scale = 1.0 * leverage

        if category == "index":
            pnl = pnl_price * point_value * contracts * scale
            total_cost = cost_per_side * point_value * contracts * 2 * scale
            pnl -= total_cost
        elif category == "crypto":
            pnl = pnl_price * contracts * scale
            total_cost = (entry_price + exit_price) * contracts * fee_rate * scale
            pnl -= total_cost
        else:  # forex
            pnl = (pnl_price * units - cost_per_side * units * 2) * scale
            total_cost = cost_per_side * units * 2 * scale
            if quote_ccy == "JPY":
                pnl = pnl / exit_price
                total_cost = total_cost / exit_price

        equity += pnl

        # Record trade result for Kelly position sizer
        if kelly_sizer is not None:
            kelly_sizer.record_trade(pnl)

        trades.append({
            "entry_time": entry_time,
            "exit_time": exit_time,
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_pips": pnl_price / pip_size,
            "cost_pips": (cost_per_side * 2) / pip_size if category != "crypto" else total_cost / pip_size,
            "net_pnl_pips": pnl_price / pip_size - ((cost_per_side * 2) / pip_size if category != "crypto" else total_cost / pip_size),
            "pnl_usd": pnl,
            "equity_after": equity,
            "exit_reason": exit_reason,
        })

    close_arr = grid["close"].values
    high_arr = grid["high"].values
    low_arr = grid["low"].values
    open_arr = grid["open"].values
    time_idx = grid.index

    # EA-style: compute signals from actual position using grid columns
    # Legacy: use pre-computed 'signal' column (no filter interaction issues)
    ea_style = all(col in grid.columns for col in
                   ["grid_top", "grid_bottom", "body_mid", "is_bullish"])

    if ea_style:
        bm_arr = grid["body_mid"].values
        gt_arr = grid["grid_top"].values
        gb_arr = grid["grid_bottom"].values
        bull_arr = grid["is_bullish"].values
        from config import VWMA_PERIODS
        warmup = max(VWMA_PERIODS) + 1
        # Check for relaxed_entry config
        try:
            from config import MA_TYPE as _ma_type
        except ImportError:
            pass
    else:
        sig_arr = grid["signal"].values
        warmup = 0

    current_year = None
    for i in range(n_bars):
        close = close_arr[i]
        high = high_arr[i]
        low = low_arr[i]
        t = time_idx[i]

        bar_year = t.year
        if bar_year != current_year:
            current_year = bar_year
            if progress_callback:
                progress_callback(current_year)

        # ── TP/SL check ──
        if entry_dir != 0 and (tp_price or sl_price):
            hit_tp = False
            hit_sl = False
            direction = "long" if entry_dir == 1 else "short"

            if entry_dir == 1:
                if sl_price and low <= entry_price - sl_price:
                    hit_sl = True
                if tp_price and high >= entry_price + tp_price:
                    hit_tp = True
            else:
                if sl_price and high >= entry_price + sl_price:
                    hit_sl = True
                if tp_price and low <= entry_price - tp_price:
                    hit_tp = True

            if hit_sl and hit_tp:
                open_p = open_arr[i]
                if entry_dir == 1:
                    dist_to_sl = open_p - (entry_price - sl_price)
                    dist_to_tp = (entry_price + tp_price) - open_p
                    if dist_to_tp < dist_to_sl:
                        hit_sl = False
                    else:
                        hit_tp = False
                else:
                    dist_to_sl = (entry_price + sl_price) - open_p
                    dist_to_tp = open_p - (entry_price - tp_price)
                    if dist_to_tp < dist_to_sl:
                        hit_sl = False
                    else:
                        hit_tp = False

            if hit_sl:
                if entry_dir == 1:
                    exit_p = entry_price - sl_price
                else:
                    exit_p = entry_price + sl_price
                _record_trade(t, exit_p, direction, "sl")
                entry_dir = 0
                equity_arr[i] = equity
                continue

            if hit_tp:
                if entry_dir == 1:
                    exit_p = entry_price + tp_price
                else:
                    exit_p = entry_price - tp_price
                _record_trade(t, exit_p, direction, "tp")
                entry_dir = 0
                equity_arr[i] = equity
                continue

        # ── Compute signal based on actual position (EA-style) ──
        if ea_style:
            if i < warmup or np.isnan(gt_arr[i]) or np.isnan(gt_arr[i - 1]):
                equity_arr[i] = equity
                continue

            # Entry conditions using actual entry_dir
            prev_below_top = bm_arr[i - 1] <= gt_arr[i - 1]
            prev_above_bot = bm_arr[i - 1] >= gb_arr[i - 1]

            # Relaxed entry
            prev_bull = bull_arr[i - 1]
            prev_open = open_arr[i - 1]
            prev_close = close_arr[i - 1]
            if not prev_bull and prev_close < gt_arr[i - 1] < prev_open:
                prev_below_top = True
            if prev_bull and prev_open < gb_arr[i - 1] < prev_close:
                prev_above_bot = True

            long_entry = bull_arr[i] and (bm_arr[i] > gt_arr[i]) and prev_below_top
            short_entry = (not bull_arr[i]) and (bm_arr[i] < gb_arr[i]) and prev_above_bot
            long_exit = (not bull_arr[i]) and (bm_arr[i] < gt_arr[i])
            short_exit = bull_arr[i] and (bm_arr[i] > gb_arr[i])

            # htf_exit: replace current-TF exit trigger with HTF position deviation
            if htf_exit and filter_arrs:
                eff_long_exit = not all(fp[i] == 1 for fp in filter_arrs.values())
                eff_short_exit = not all(fp[i] == -1 for fp in filter_arrs.values())
            else:
                eff_long_exit = long_exit
                eff_short_exit = short_exit

            # Determine action from actual position
            action = None  # "enter_long", "enter_short", "exit", "reverse_long", "reverse_short"
            if entry_dir == 0:
                if long_entry:
                    action = "enter_long"
                elif short_entry:
                    action = "enter_short"
            elif entry_dir == 1:
                if eff_long_exit:
                    action = "reverse_short" if short_entry else "exit"
            elif entry_dir == -1:
                if eff_short_exit:
                    action = "reverse_long" if long_entry else "exit"

            if action is None:
                equity_arr[i] = equity
                continue

            # Execution price
            if next_bar_open:
                if i + 1 >= n_bars:
                    equity_arr[i] = equity
                    continue
                exec_price = open_arr[i + 1]
                exec_time = time_idx[i + 1]
            else:
                exec_price = close
                exec_time = t

            def _filter_allows_ea(direction):
                if not filter_arrs:
                    return True
                return all(fp[i] == direction for fp in filter_arrs.values())

            def _hour_allows_entry():
                if not allowed_entry_hours:  # None or empty set
                    return True
                return exec_time.hour in allowed_entry_hours

            # Apply H4 filter + time filter + execute (exit always allowed)
            if action == "enter_long":
                if _filter_allows_ea(1) and _hour_allows_entry():
                    entry_price = exec_price
                    entry_time = exec_time
                    entry_dir = 1
            elif action == "enter_short":
                if _filter_allows_ea(-1) and _hour_allows_entry():
                    entry_price = exec_price
                    entry_time = exec_time
                    entry_dir = -1
            elif action == "exit":
                direction = "long" if entry_dir == 1 else "short"
                _record_trade(exec_time, exec_price, direction)
                entry_dir = 0
            elif action == "reverse_long":
                _record_trade(exec_time, exec_price, "short")
                if _filter_allows_ea(1) and _hour_allows_entry():
                    entry_price = exec_price
                    entry_time = exec_time
                    entry_dir = 1
                else:
                    entry_dir = 0
            elif action == "reverse_short":
                _record_trade(exec_time, exec_price, "long")
                if _filter_allows_ea(-1) and _hour_allows_entry():
                    entry_price = exec_price
                    entry_time = exec_time
                    entry_dir = -1
                else:
                    entry_dir = 0

        else:
            # ── Legacy: pre-computed signal column ──
            sig = sig_arr[i]

            # htf_exit: HTF deviation forces exit (ignoring current-TF signal)
            if htf_exit and filter_arrs and entry_dir != 0:
                htf_deviated = not all(fp[i] == entry_dir for fp in filter_arrs.values())
                if htf_deviated:
                    if next_bar_open and i + 1 < n_bars:
                        htf_exec_price = open_arr[i + 1]
                        htf_exec_time = time_idx[i + 1]
                    else:
                        htf_exec_price = close
                        htf_exec_time = t
                    direction = "long" if entry_dir == 1 else "short"
                    _record_trade(htf_exec_time, htf_exec_price, direction, "htf_exit")
                    entry_dir = 0

            if sig == 0:
                equity_arr[i] = equity
                continue

            if next_bar_open:
                if i + 1 >= n_bars:
                    equity_arr[i] = equity
                    continue
                exec_price = open_arr[i + 1]
                exec_time = time_idx[i + 1]
            else:
                exec_price = close
                exec_time = t

            def _filter_allows(direction):
                if not filter_arrs:
                    return True
                return all(fp[i] == direction for fp in filter_arrs.values())

            def _alignment_allows(direction):
                if not alignment_arrs:
                    return True
                vals = [arr[i] for arr in alignment_arrs]
                if any(np.isnan(v) for v in vals):
                    return True
                if direction == 1:
                    return all(vals[j] > vals[j+1] for j in range(len(vals)-1))
                else:
                    return all(vals[j] < vals[j+1] for j in range(len(vals)-1))

            if sig == 1:
                if entry_dir == -1 and not htf_exit:
                    _record_trade(exec_time, exec_price, "short", "signal")
                    entry_dir = 0
                if entry_dir == 0:
                    if _filter_allows(1) and _alignment_allows(1):
                        entry_price = exec_price
                        entry_time = exec_time
                        entry_dir = 1
            elif sig == -1:
                if entry_dir == 1 and not htf_exit:
                    _record_trade(exec_time, exec_price, "long", "signal")
                    entry_dir = 0
                if entry_dir == 0:
                    if _filter_allows(-1) and _alignment_allows(-1):
                        entry_price = exec_price
                        entry_time = exec_time
                        entry_dir = -1
            elif sig == 2:
                if entry_dir == -1 and not htf_exit:
                    _record_trade(exec_time, exec_price, "short", "signal")
                    entry_dir = 0
                if entry_dir == 0:
                    if _filter_allows(1) and _alignment_allows(1):
                        entry_price = exec_price
                        entry_time = exec_time
                        entry_dir = 1
            elif sig == -2:
                if entry_dir == 1 and not htf_exit:
                    _record_trade(exec_time, exec_price, "long", "signal")
                    entry_dir = 0
                if entry_dir == 0:
                    if _filter_allows(-1) and _alignment_allows(-1):
                        entry_price = exec_price
                        entry_time = exec_time
                        entry_dir = -1

        equity_arr[i] = equity

    # Force-close open position at last bar
    if entry_dir != 0:
        close = close_arr[-1]
        t = time_idx[-1]
        direction = "long" if entry_dir == 1 else "short"
        _record_trade(t, close, direction, "forced")
        equity_arr[-1] = equity

    return trades, equity_arr


# ── Statistics ──────────────────────────────────────────────

def compute_stats(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    initial_capital: float,
    symbol: str,
    timeframe: str,
    ma_type: str,
    compound: bool = False,
) -> dict:
    """Compute trade performance statistics."""

    if len(trades_df) == 0:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "ma_type": ma_type or "ema",
            "total_trades": 0,
            "note": "No trades generated",
        }

    n = len(trades_df)
    winners = trades_df[trades_df["pnl_usd"] > 0]
    losers = trades_df[trades_df["pnl_usd"] <= 0]

    win_rate = len(winners) / n if n > 0 else 0
    avg_win = winners["pnl_usd"].mean() if len(winners) > 0 else 0
    avg_loss = losers["pnl_usd"].mean() if len(losers) > 0 else 0

    gross_profit = winners["pnl_usd"].sum() if len(winners) > 0 else 0
    gross_loss = abs(losers["pnl_usd"].sum()) if len(losers) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    total_pnl = trades_df["pnl_usd"].sum()
    total_pnl_pips = trades_df["net_pnl_pips"].sum()
    total_cost_pips = trades_df["cost_pips"].sum()

    if len(equity_df) > 0:
        eq = equity_df["equity"]
        peak = eq.cummax()
        dd = (eq - peak) / peak * 100
        max_dd_pct = dd.min()
    else:
        max_dd_pct = 0.0

    if "entry_time" in trades_df.columns and "exit_time" in trades_df.columns:
        holding = pd.to_datetime(trades_df["exit_time"]) - pd.to_datetime(trades_df["entry_time"])
        avg_holding = holding.mean()
    else:
        avg_holding = pd.Timedelta(0)

    final_equity = trades_df["equity_after"].iloc[-1]
    if len(equity_df) > 1:
        total_days = (equity_df["time"].iloc[-1] - equity_df["time"].iloc[0]).days
        if total_days > 0:
            years = total_days / 365.25
            if compound:
                # CAGR for compound mode
                annual_return = ((final_equity / initial_capital) ** (1 / years) - 1) * 100
            else:
                # Simple annualised return for fixed-lot mode
                annual_return = (final_equity / initial_capital - 1) / years * 100
        else:
            annual_return = 0.0
    else:
        annual_return = 0.0
        total_days = 0

    expectancy_pips = trades_df["net_pnl_pips"].mean()

    # Annualised Sharpe Ratio & Volatility (daily returns)
    sharpe_ratio = 0.0
    annual_volatility = 0.0
    if len(equity_df) > 1:
        eq_daily = equity_df.set_index("time")["equity"].resample("1D").last().dropna()
        if len(eq_daily) > 1:
            daily_returns = eq_daily.pct_change().dropna()
            if len(daily_returns) > 0 and daily_returns.std() > 0:
                sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
                annual_volatility = daily_returns.std() * np.sqrt(252) * 100

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "ma_type": ma_type or "ema",
        "data_period_days": total_days,
        "total_trades": n,
        "long_trades": len(trades_df[trades_df["direction"] == "long"]),
        "short_trades": len(trades_df[trades_df["direction"] == "short"]),
        "win_rate": round(win_rate * 100, 1),
        "profit_factor": round(profit_factor, 2),
        "total_pnl_pips": round(total_pnl_pips, 1),
        "total_cost_pips": round(total_cost_pips, 1),
        "total_pnl_usd": round(total_pnl, 2),
        "avg_win_usd": round(avg_win, 2),
        "avg_loss_usd": round(avg_loss, 2),
        "expectancy_pips": round(expectancy_pips, 1),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "annual_return_pct": round(annual_return, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "annual_volatility_pct": round(annual_volatility, 2),
        "avg_holding": str(avg_holding),
        "initial_capital": initial_capital,
        "final_equity": round(final_equity, 2),
    }
