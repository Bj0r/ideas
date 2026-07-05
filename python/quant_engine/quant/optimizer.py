"""
quant/optimizer.py
Optuna-based Bayesian parameter optimization for SR-Probability strategy.

Optimizes over:
  - Session window A start/end (pre-market PH, expressed as UTC HHMM)
  - Session window B start/end (late session PH, UTC HHMM)
  - exit_buffer_mult  (0.25 to 1.5 ATR)
  - signal_cooldown   (5 to 20 bars)
  - global_cooldown   (2 to 10 bars)
  - ftr_lookback      (3 to 30 bars)
  - score_threshold   (60 to 80)

Objective: maximize Calmar ratio on out-of-sample walk-forward fold.
Penalizes low trade count to prevent degenerate over-filtered solutions.
"""

import optuna
import numpy as np
import pandas as pd
from typing import Callable, Optional
import warnings

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


# ── Walk-forward split ─────────────────────────────────────────────────────────

def walk_forward_splits(df: pd.DataFrame, n_splits: int = 5, test_frac: float = 0.2):
    """
    Yield (train_df, test_df) tuples using anchored walk-forward expansion.
    The test window is always the rightmost test_frac of each fold.
    """
    n = len(df)
    fold_size = n // n_splits
    for i in range(1, n_splits + 1):
        train_end = int(fold_size * i * (1 - test_frac))
        test_end  = fold_size * i
        if test_end > n:
            break
        yield df.iloc[:train_end], df.iloc[train_end:test_end]


# ── Signal replay engine ───────────────────────────────────────────────────────

def replay_signals(
    ohlcv: pd.DataFrame,
    params: dict,
    signal_generator: Callable,
) -> pd.DataFrame:
    """
    Apply a signal_generator function (wraps Pine logic in Python)
    with given params over an OHLCV DataFrame.

    signal_generator(ohlcv, params) -> pd.DataFrame with columns:
      [ts, direction, score, entry_price, sl_price, tp_price]

    Returns a trades DataFrame with outcome and pnl_r computed.
    """
    signals_df = signal_generator(ohlcv, params)
    if signals_df.empty:
        return pd.DataFrame()

    trades = []
    for _, sig in signals_df.iterrows():
        entry = sig["entry_price"]
        sl    = sig["sl_price"]
        tp    = sig["tp_price"]
        sl_dist = abs(entry - sl)
        if sl_dist == 0:
            continue

        direction = sig["direction"]
        future = ohlcv[ohlcv.index > sig["ts"]].head(24)  # max 24 bars to resolve

        outcome, pnl_r = "OPEN", 0.0
        for _, bar in future.iterrows():
            if direction == "BUY":
                if bar["low"] <= sl:
                    outcome, pnl_r = "LOSS", -1.0
                    break
                if bar["high"] >= tp:
                    outcome, pnl_r = "WIN", (tp - entry) / sl_dist
                    break
            else:  # SELL
                if bar["high"] >= sl:
                    outcome, pnl_r = "LOSS", -1.0
                    break
                if bar["low"] <= tp:
                    outcome, pnl_r = "WIN", (entry - tp) / sl_dist
                    break

        trades.append({**sig.to_dict(), "outcome": outcome, "pnl_r": pnl_r})

    return pd.DataFrame(trades)


# ── Objective metrics ──────────────────────────────────────────────────────────

def calmar_ratio(trades: pd.DataFrame, min_trades: int = 15) -> float:
    """
    Calmar = annualized return / max drawdown.
    Returns 0.0 for insufficient trades or zero drawdown.
    Penalized negatively if trade count is below min_trades.
    """
    if len(trades) < min_trades:
        return -10.0 * (min_trades - len(trades)) / min_trades  # penalty

    pnl = trades["pnl_r"].values
    cumulative = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    max_dd = drawdown.max()

    if max_dd == 0:
        return 0.0

    total_r  = cumulative[-1]
    n_bars   = len(trades)
    annual_r = total_r * (252 * 78 / n_bars)  # 78 5m bars per trading day
    return annual_r / max_dd


def sharpe_ratio(trades: pd.DataFrame, min_trades: int = 15) -> float:
    if len(trades) < min_trades:
        return -5.0
    pnl = trades["pnl_r"].values
    if pnl.std() == 0:
        return 0.0
    return (pnl.mean() / pnl.std()) * np.sqrt(252 * 78)


# ── Optuna study ───────────────────────────────────────────────────────────────

def create_study(
    ohlcv: pd.DataFrame,
    signal_generator: Callable,
    n_trials: int = 100,
    objective_fn: str = "calmar",
    n_splits: int = 5,
    storage_url: Optional[str] = None,
) -> optuna.Study:
    """
    Run Bayesian optimization over the parameter space.

    Parameters
    ----------
    ohlcv             : full OHLCV history DataFrame
    signal_generator  : callable wrapping the Pine strategy logic in Python
    n_trials          : number of Optuna trials (100 is practical on i3/12GB)
    objective_fn      : 'calmar' | 'sharpe'
    n_splits          : walk-forward folds
    storage_url       : optional SQLite URL for persistent study e.g.
                        'sqlite:///optuna_study.db'
    """
    obj_map = {"calmar": calmar_ratio, "sharpe": sharpe_ratio}
    metric_fn = obj_map[objective_fn]

    def objective(trial: optuna.Trial) -> float:
        params = {
            # Session window A (pre-market PH, UTC HHMM)
            "ses_a_start": trial.suggest_int("ses_a_start", 1400, 1800, step=30),
            "ses_a_end":   trial.suggest_int("ses_a_end",   0,    200,  step=30),
            # Session window B (late PH, UTC HHMM)
            "ses_b_start": trial.suggest_int("ses_b_start", 900,  1300, step=30),
            "ses_b_end":   trial.suggest_int("ses_b_end",   1300, 1600, step=30),
            # Pine strategy params
            "exit_buffer_mult": trial.suggest_float("exit_buffer_mult", 0.25, 1.5, step=0.05),
            "signal_cooldown":  trial.suggest_int("signal_cooldown", 5, 20),
            "global_cooldown":  trial.suggest_int("global_cooldown", 2, 10),
            "ftr_lookback":     trial.suggest_int("ftr_lookback", 3, 30),
            "score_threshold":  trial.suggest_float("score_threshold", 60.0, 80.0, step=1.0),
        }

        scores = []
        for train_df, test_df in walk_forward_splits(ohlcv, n_splits=n_splits):
            trades = replay_signals(test_df, params, signal_generator)
            if trades.empty:
                scores.append(-10.0)
                continue
            scores.append(metric_fn(trades))

        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=20),
        storage=storage_url,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)
    return study


def best_params_report(study: optuna.Study) -> pd.DataFrame:
    """Return top-20 trials as a formatted DataFrame for dashboard display."""
    trials = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials = trials[trials["state"] == "COMPLETE"].sort_values("value", ascending=False)
    return trials.head(20).reset_index(drop=True)
