"""
quant/session_analysis.py
Session window win-rate heatmap and rolling statistical analysis.

Answers: "What is the empirically optimal pre-market entry window
for Gold 5m, given the actual closed signal history?"

Outputs:
  - 30-min bucket win-rate matrix (hour x minute) for heatmap
  - Statistical significance test per bucket (chi-square vs. overall win rate)
  - Recommended optimal windows based on min sample threshold
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu
from typing import Tuple


def _to_utc_hhmm(signals: pd.DataFrame) -> pd.DataFrame:
    """
    Parse ts column (ISO8601) to UTC datetime and extract hour + minute.
    Adds columns: utc_dt, utc_hour, utc_minute, utc_hhmm (int e.g. 1630)
    """
    df = signals.copy()
    df["utc_dt"] = pd.to_datetime(df["ts"], utc=True)
    df["utc_hour"]   = df["utc_dt"].dt.hour
    df["utc_minute"] = df["utc_dt"].dt.minute
    df["utc_hhmm"]   = df["utc_hour"] * 100 + (df["utc_minute"] // 30) * 30
    return df


def session_winrate_heatmap(
    signals: pd.DataFrame,
    min_bucket_size: int = 5,
    bucket_minutes: int = 30,
) -> pd.DataFrame:
    """
    Compute win rate per time bucket across the 24-hour trading day.

    Parameters
    ----------
    signals          : closed signals DataFrame (must have ts, outcome, pnl_r columns)
    min_bucket_size  : suppress buckets with fewer than this many signals
    bucket_minutes   : bucket size in minutes (30 default = 48 buckets/day)

    Returns
    -------
    DataFrame with columns: utc_hhmm, ph_hhmm, n_signals, win_rate,
                             avg_r, p_value, significant (bool at 0.05)
    """
    closed = signals[signals["outcome"].isin(["WIN", "LOSS"])].copy()
    if closed.empty:
        return pd.DataFrame()

    df = _to_utc_hhmm(closed)
    overall_wr = (df["outcome"] == "WIN").mean()
    total_n = len(df)
    overall_wins = (df["outcome"] == "WIN").sum()

    results = []
    for hhmm, grp in df.groupby("utc_hhmm"):
        n = len(grp)
        if n < min_bucket_size:
            continue

        wins = (grp["outcome"] == "WIN").sum()
        wr   = wins / n
        avg_r = grp["pnl_r"].mean()

        # Chi-square: is this bucket's win rate different from overall?
        # 2x2 contingency: [wins_in, losses_in] vs [wins_out, losses_out]
        wins_out   = overall_wins - wins
        losses_in  = n - wins
        losses_out = (total_n - n) - wins_out
        contingency = np.array([[wins, losses_in], [wins_out, losses_out]])
        try:
            _, p_val, _, _ = chi2_contingency(contingency, correction=True)
        except ValueError:
            p_val = 1.0

        # Convert UTC HHMM to PH time (UTC+8)
        h, m = int(str(hhmm).zfill(4)[:2]), int(str(hhmm).zfill(4)[2:])
        ph_h = (h + 8) % 24
        ph_hhmm = ph_h * 100 + m

        results.append({
            "utc_hhmm":    hhmm,
            "ph_hhmm":     ph_hhmm,
            "ph_time":     f"{ph_h:02d}:{m:02d}",
            "n_signals":   n,
            "win_rate":    round(wr, 4),
            "avg_r":       round(avg_r, 3),
            "p_value":     round(p_val, 4),
            "significant": p_val < 0.05 and wr > overall_wr,
        })

    return pd.DataFrame(results).sort_values("ph_hhmm").reset_index(drop=True)


def heatmap_matrix(heatmap_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the heatmap data into a matrix format for Plotly heatmap rendering.
    Returns: (pivot_matrix, x_labels, y_labels) suitable for go.Heatmap.
    """
    if heatmap_df.empty:
        return pd.DataFrame()
    df = heatmap_df.copy()
    df["ph_hour"]   = df["ph_hhmm"] // 100
    df["ph_minute"] = df["ph_hhmm"] % 100
    pivot = df.pivot_table(index="ph_hour", columns="ph_minute", values="win_rate")
    return pivot


def best_session_windows(
    heatmap_df: pd.DataFrame,
    top_n: int = 5,
    min_n: int = 10,
) -> pd.DataFrame:
    """
    Return the top_n time buckets by win rate, filtered by significance
    and minimum sample size. Used to recommend session window parameters.
    """
    if heatmap_df.empty:
        return pd.DataFrame()
    filtered = heatmap_df[
        (heatmap_df["significant"] == True) &
        (heatmap_df["n_signals"] >= min_n)
    ].sort_values("win_rate", ascending=False)
    return filtered.head(top_n)[
        ["ph_time", "n_signals", "win_rate", "avg_r", "p_value", "utc_hhmm"]
    ].reset_index(drop=True)


def compare_session_vs_nonSession(
    signals: pd.DataFrame,
    ses_a_start_utc: int = 1600,
    ses_a_end_utc:   int = 100,
    ses_b_start_utc: int = 1100,
    ses_b_end_utc:   int = 1559,
) -> dict:
    """
    Mann-Whitney U test: are R-multiples in-session > out-of-session?
    Returns summary statistics and test result.
    """
    closed = signals[signals["outcome"].isin(["WIN", "LOSS"])].copy()
    if closed.empty:
        return {}

    df = _to_utc_hhmm(closed)

    def in_window_a(hhmm):
        return (hhmm >= ses_a_start_utc) or (hhmm <= ses_a_end_utc)

    def in_window_b(hhmm):
        return ses_b_start_utc <= hhmm <= ses_b_end_utc

    df["in_session"] = df["utc_hhmm"].apply(lambda x: in_window_a(x) or in_window_b(x))

    in_sess  = df[df["in_session"]]["pnl_r"].dropna().values
    out_sess = df[~df["in_session"]]["pnl_r"].dropna().values

    if len(in_sess) < 5 or len(out_sess) < 5:
        return {"error": "Insufficient data for Mann-Whitney test"}

    stat, p = mannwhitneyu(in_sess, out_sess, alternative="greater")

    return {
        "n_in_session":     len(in_sess),
        "n_out_session":    len(out_sess),
        "avg_r_in":         round(float(in_sess.mean()), 4),
        "avg_r_out":        round(float(out_sess.mean()), 4),
        "win_rate_in":      round(float((df[df["in_session"]]["outcome"] == "WIN").mean()), 4),
        "win_rate_out":     round(float((df[~df["in_session"]]["outcome"] == "WIN").mean()), 4),
        "mann_whitney_u":   round(float(stat), 2),
        "p_value":          round(float(p), 4),
        "session_superior": p < 0.05,
    }
