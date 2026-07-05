"""
quant/kelly.py
Fractional Kelly Criterion position sizing, segmented by signal type.

Kelly formula: f* = (b*p - q) / b
  b = average win / average |loss| ratio (in R-multiples)
  p = win probability (calibrated posterior)
  q = 1 - p

Half-Kelly (f*/2) is the default fraction for practical use.
Quarter-Kelly (f*/4) is recommended for any regime_id == 2 (high-vol).
"""

import numpy as np
import pandas as pd
from typing import Optional


KELLY_FRACTIONS = {
    "full":    1.0,
    "half":    0.5,
    "quarter": 0.25,
}


def compute_kelly(
    signals: pd.DataFrame,
    group_by: str = "trigger_type",
    fraction: str = "half",
    min_samples: int = 20,
) -> pd.DataFrame:
    """
    Compute fractional Kelly for each unique value of group_by column.

    Parameters
    ----------
    signals     : DataFrame with columns [trigger_type, outcome, pnl_r, score]
    group_by    : column to segment by ('trigger_type', 'session', or 'direction')
    fraction    : 'full' | 'half' | 'quarter'
    min_samples : minimum closed trades required to compute Kelly for a segment

    Returns
    -------
    DataFrame with columns:
      segment, n_trades, win_rate, avg_win_r, avg_loss_r, b_ratio,
      kelly_full, kelly_applied, recommended_risk_pct
    """
    closed = signals[signals["outcome"].isin(["WIN", "LOSS"])].copy()
    k_frac = KELLY_FRACTIONS.get(fraction, 0.5)
    rows = []

    for seg, grp in closed.groupby(group_by):
        n = len(grp)
        if n < min_samples:
            rows.append({group_by: seg, "n_trades": n, "note": f"Insufficient data (<{min_samples})"})
            continue

        wins  = grp[grp["outcome"] == "WIN"]["pnl_r"]
        losses= grp[grp["outcome"] == "LOSS"]["pnl_r"].abs()

        p = len(wins) / n
        q = 1 - p
        avg_win  = wins.mean()  if len(wins)  > 0 else 0.0
        avg_loss = losses.mean() if len(losses) > 0 else 1.0  # default 1R loss

        b = avg_win / avg_loss if avg_loss > 0 else 0.0
        kelly_full = (b * p - q) / b if b > 0 else 0.0
        kelly_full = max(kelly_full, 0.0)  # no negative sizing
        kelly_applied = kelly_full * k_frac

        # Recommended risk as % of account equity
        # Cap at 2% per trade regardless of Kelly output (risk management floor)
        recommended_risk_pct = min(kelly_applied * 100, 2.0)

        rows.append({
            group_by:             seg,
            "n_trades":           n,
            "win_rate":           round(p, 4),
            "avg_win_r":          round(avg_win, 3),
            "avg_loss_r":         round(avg_loss, 3),
            "b_ratio":            round(b, 3),
            "kelly_full":         round(kelly_full, 4),
            f"kelly_{fraction}":  round(kelly_applied, 4),
            "recommended_risk_%": round(recommended_risk_pct, 2),
            "note":               "OK",
        })

    return pd.DataFrame(rows)


def kelly_confidence_bounds(
    p: float, b: float, n: int, fraction: float = 0.5, ci: float = 0.95
) -> dict:
    """
    Compute confidence interval around Kelly fraction using Wilson score
    interval for win rate uncertainty.

    Parameters
    ----------
    p        : observed win rate
    b        : win/loss ratio
    n        : number of trades
    fraction : Kelly fraction applied (0.5 = half-Kelly)
    ci       : confidence level (0.95 default)

    Returns dict with kelly_lower, kelly_central, kelly_upper
    """
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - ci) / 2)

    # Wilson score CI for proportion p
    center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / (1 + z**2 / n)
    p_lo = max(center - margin, 0.0)
    p_hi = min(center + margin, 1.0)

    def kelly_f(prob: float) -> float:
        q = 1 - prob
        f = (b * prob - q) / b if b > 0 else 0.0
        return max(f, 0.0) * fraction

    return {
        "kelly_lower":   round(kelly_f(p_lo), 4),
        "kelly_central": round(kelly_f(p), 4),
        "kelly_upper":   round(kelly_f(p_hi), 4),
        "p_lower":       round(p_lo, 4),
        "p_upper":       round(p_hi, 4),
    }


def expected_growth_rate(p: float, b: float, f: float) -> float:
    """
    Log-optimal growth rate G(f) = p*log(1 + b*f) + q*log(1 - f)
    Used to verify that the applied fraction is near the growth-optimal point.
    """
    q = 1 - p
    if f >= 1.0 or (1 + b * f) <= 0:
        return float("-inf")
    return p * np.log(1 + b * f) + q * np.log(1 - f)
