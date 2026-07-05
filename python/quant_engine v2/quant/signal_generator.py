"""
quant/signal_generator.py
Python replica of SR-Probability v9fix Pine logic for backtesting.

This module bridges the optimizer with historical OHLCV data. It replicates
the Pine Script zone detection and signal conditions in vectorized Python so
that Optuna can replay thousands of parameter combinations efficiently.

Note: Pine's pivot detection uses lookahead (pivotLen bars right). In
backtesting this is valid on historical data; in live use only Pine is used.
"""

import numpy as np
import pandas as pd
import ta


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=period
    ).average_true_range()


def _pivot_highs(series: pd.Series, left: int = 5, right: int = 5) -> pd.Series:
    """Return True at bars that are pivot highs (look-ahead used, valid for backtest only)."""
    n = len(series)
    result = pd.Series(False, index=series.index)
    vals = series.values
    for i in range(left, n - right):
        window = vals[i - left: i + right + 1]
        if vals[i] == window.max():
            result.iloc[i] = True
    return result


def _pivot_lows(series: pd.Series, left: int = 5, right: int = 5) -> pd.Series:
    n = len(series)
    result = pd.Series(False, index=series.index)
    vals = series.values
    for i in range(left, n - right):
        window = vals[i - left: i + right + 1]
        if vals[i] == window.min():
            result.iloc[i] = True
    return result


def _in_session(ts: pd.DatetimeTZDtype, params: dict) -> bool:
    """Check if a bar falls within the defined session windows (UTC HHMM)."""
    hhmm = ts.hour * 100 + (ts.minute // 30) * 30
    a_start = params.get("ses_a_start", 1600)
    a_end   = params.get("ses_a_end",   100)
    b_start = params.get("ses_b_start", 1100)
    b_end   = params.get("ses_b_end",   1559)

    in_a = (hhmm >= a_start) or (hhmm <= a_end)
    in_b = b_start <= hhmm <= b_end
    return in_a or in_b


def generate_signals(ohlcv: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Replicate SR-Probability v9fix signal detection in Python.

    Parameters
    ----------
    ohlcv  : DataFrame with DatetimeTZDtype index (UTC) and
             columns: open, high, low, close, volume
    params : dict matching Optuna parameter space keys

    Returns
    -------
    DataFrame of signals with columns:
      ts, direction, trigger_type, score, entry_price, sl_price, tp_price,
      ftr_confirmed, session, zone_top, zone_bot
    """
    df = ohlcv.copy()
    pivot_len       = int(params.get("pivot_len", 5))
    atr_buf_mult    = float(params.get("exit_buffer_mult", 0.50))
    score_threshold = float(params.get("score_threshold", 65.0))
    ftr_lookback    = int(params.get("ftr_lookback", 10))
    vol_ma_len      = int(params.get("vol_ma_len", 20))
    signal_cooldown = int(params.get("signal_cooldown", 10))
    rr_ratio        = float(params.get("rr_ratio", 2.0))

    # ── Features ────────────────────────────────────────────────────────────────
    df["atr"]     = _compute_atr(df, 14)
    df["vol_ma"]  = df["volume"].rolling(vol_ma_len).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma"].replace(0, np.nan)

    ph_mask = _pivot_highs(df["high"], pivot_len, pivot_len)
    pl_mask = _pivot_lows(df["low"], pivot_len, pivot_len)

    # ── Zone construction ───────────────────────────────────────────────────────
    res_zones = []   # list of [top, bottom, touches, created_bar, last_signal_bar, broken]
    sup_zones = []

    for i, (ts_val, row) in enumerate(df.iterrows()):
        buf = row["atr"] * 0.15 if not np.isnan(row["atr"]) else 0

        if ph_mask.iloc[i]:
            top = row["high"] + buf
            bot = max(row["open"], row["close"])
            res_zones.append({"top": top, "bot": bot, "touches": 0, "bar": i,
                              "last_sig": -99, "broken": False})

        if pl_mask.iloc[i]:
            top = min(row["open"], row["close"])
            bot = row["low"] - buf
            sup_zones.append({"top": top, "bot": bot, "touches": 0, "bar": i,
                              "last_sig": -99, "broken": False})

    # ── Signal detection pass ────────────────────────────────────────────────────
    signals = []
    last_any_signal = -99

    for i, (ts_val, row) in enumerate(df.iterrows()):
        if np.isnan(row["atr"]) or np.isnan(row["vol_ratio"]):
            continue

        in_sess    = _in_session(ts_val, params) if hasattr(ts_val, "hour") else True
        if not in_sess:
            continue

        vol_ratio  = row["vol_ratio"]
        # Session volume inversion: low volume is good in session
        vol_score  = 20.0 * (1.5 - min(vol_ratio, 1.5)) / 1.5  # higher when vol is low

        # FTR checks
        lookback_start = max(0, i - ftr_lookback)
        recent_low  = df["low"].iloc[lookback_start:i].min() if i > 0 else row["low"]
        recent_high = df["high"].iloc[lookback_start:i].max() if i > 0 else row["high"]
        ftr_buy  = row["low"] > recent_low   # Higher Low
        ftr_sell = row["high"] < recent_high  # Lower High

        global_ok = (i - last_any_signal) >= int(params.get("global_cooldown", 5))

        # ── Resistance zones ────────────────────────────────────────────────────
        for z in res_zones:
            if z["broken"]:
                continue
            if row["close"] > z["top"]:
                z["broken"] = True
                continue
            if z["top"] >= row["high"] >= z["bot"] or z["top"] >= row["low"] >= z["bot"]:
                # Touch event
                z_ok = (i - z["last_sig"]) >= signal_cooldown
                if not (z_ok and global_ok and ftr_sell):
                    continue

                z["touches"] += 1
                freshness = max(0, 30 * (1 - z["touches"] * 0.3))
                pattern   = 15.0 if (row["open"] - row["close"]) > 0 else 5.0  # bearish bar
                trigger   = 25.0  # rejection
                sc        = freshness + vol_score + pattern + trigger
                if sc >= score_threshold:
                    sl_dist = abs(row["close"] - z["top"]) + row["atr"] * 0.5
                    signals.append({
                        "ts":            ts_val,
                        "direction":     "SELL",
                        "trigger_type":  "Rejection +FTR" if ftr_sell else "Rejection",
                        "score":         round(sc, 1),
                        "ftr_confirmed": int(ftr_sell),
                        "session":       "Active",
                        "entry_price":   row["close"],
                        "sl_price":      row["close"] + sl_dist,
                        "tp_price":      row["close"] - sl_dist * rr_ratio,
                        "zone_top":      z["top"],
                        "zone_bot":      z["bot"],
                    })
                    z["last_sig"]  = i
                    last_any_signal = i

        # ── Support zones ───────────────────────────────────────────────────────
        for z in sup_zones:
            if z["broken"]:
                continue
            if row["close"] < z["bot"]:
                z["broken"] = True
                continue
            if z["top"] >= row["high"] >= z["bot"] or z["top"] >= row["low"] >= z["bot"]:
                z_ok = (i - z["last_sig"]) >= signal_cooldown
                if not (z_ok and global_ok and ftr_buy):
                    continue

                z["touches"] += 1
                freshness = max(0, 30 * (1 - z["touches"] * 0.3))
                pattern   = 15.0 if (row["close"] - row["open"]) > 0 else 5.0  # bullish bar
                trigger   = 25.0
                sc        = freshness + vol_score + pattern + trigger
                if sc >= score_threshold:
                    sl_dist = abs(row["close"] - z["bot"]) + row["atr"] * 0.5
                    signals.append({
                        "ts":            ts_val,
                        "direction":     "BUY",
                        "trigger_type":  "Rejection +FTR" if ftr_buy else "Rejection",
                        "score":         round(sc, 1),
                        "ftr_confirmed": int(ftr_buy),
                        "session":       "Active",
                        "entry_price":   row["close"],
                        "sl_price":      row["close"] - sl_dist,
                        "tp_price":      row["close"] + sl_dist * rr_ratio,
                        "zone_top":      z["top"],
                        "zone_bot":      z["bot"],
                    })
                    z["last_sig"]  = i
                    last_any_signal = i

    return pd.DataFrame(signals) if signals else pd.DataFrame()
