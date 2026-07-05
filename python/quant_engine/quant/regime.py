"""
quant/regime.py
Hidden Markov Model regime detection for Gold 5m.

States
------
0 — Low-volatility / range-bound (Asian pre-market sweep conditions)
1 — Trending / directional
2 — High-volatility / event-driven (NFP, FOMC — avoid session-sweep entries)

Features used: 14-period ATR (normalized by price) + volume ratio vs 20-bar MA.
Both features are z-scored before fitting so the HMM is scale-invariant.
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "assets" / "hmm_model.pkl"
SCALER_PATH = Path(__file__).parent.parent / "assets" / "hmm_scaler.pkl"

# Human-readable labels mapped from HMM state integers
# Remapped after fitting based on mean ATR ordering (low → high)
REGIME_LABELS = {0: "Low-Vol / Range", 1: "Trending", 2: "High-Vol / Event"}
REGIME_COLORS = {0: "#00b894", 1: "#fdcb6e", 2: "#d63031"}


def _build_features(df: pd.DataFrame) -> np.ndarray:
    """
    Compute ATR14 (normalized) and volume ratio, return as (N, 2) array.
    df must have columns: high, low, close, volume
    """
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    vol = df["volume"].values

    # ATR via Wilder's smoothing
    tr = np.maximum(high[1:] - low[1:],
         np.maximum(np.abs(high[1:] - close[:-1]),
                    np.abs(low[1:] - close[:-1])))
    atr = np.empty(len(df))
    atr[:14] = np.nan
    atr[14] = tr[:13].mean()
    for i in range(15, len(df)):
        atr[i] = (atr[i - 1] * 13 + tr[i - 1]) / 14

    atr_norm = atr / close  # normalize by price level

    # Volume ratio vs 20-bar MA
    vol_ma = pd.Series(vol).rolling(20).mean().values
    vol_ratio = np.where(vol_ma > 0, vol / vol_ma, 1.0)

    # Stack and drop NaN rows (first 20 bars)
    feat = np.column_stack([atr_norm, vol_ratio])
    return feat


def fit_regime_model(df: pd.DataFrame, n_states: int = 3, random_state: int = 42) -> GaussianHMM:
    """
    Fit a Gaussian HMM on OHLCV data.
    Saves model and scaler to assets/ for later inference.
    Returns the fitted model with state order remapped by ascending ATR mean.
    """
    feat = _build_features(df)
    valid_mask = ~np.isnan(feat).any(axis=1)
    feat_clean = feat[valid_mask]

    scaler = StandardScaler()
    feat_scaled = scaler.fit_transform(feat_clean)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        random_state=random_state,
        tol=1e-4,
    )
    model.fit(feat_scaled)

    # Remap states so state 0 = lowest ATR, state 2 = highest ATR
    mean_atrs = model.means_[:, 0]  # first feature is ATR
    order = np.argsort(mean_atrs)
    remap = {old: new for new, old in enumerate(order)}
    model._remap = remap

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    return model


def predict_regimes(df: pd.DataFrame) -> pd.Series:
    """
    Load saved model, predict regime for each bar in df.
    Returns pd.Series of regime integers aligned to df index.
    Bars with insufficient data return NaN.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError("HMM model not found. Run fit_regime_model() first.")

    model: GaussianHMM = joblib.load(MODEL_PATH)
    scaler: StandardScaler = joblib.load(SCALER_PATH)

    feat = _build_features(df)
    valid_mask = ~np.isnan(feat).any(axis=1)

    feat_clean = feat[valid_mask]
    feat_scaled = scaler.transform(feat_clean)
    raw_states = model.predict(feat_scaled)

    remap = getattr(model, "_remap", {i: i for i in range(model.n_components)})
    remapped = np.array([remap.get(s, s) for s in raw_states])

    regimes = np.full(len(df), np.nan)
    regimes[valid_mask] = remapped
    return pd.Series(regimes, index=df.index, name="regime")


def get_transition_matrix(model: GaussianHMM) -> pd.DataFrame:
    """Return the transition probability matrix as a labeled DataFrame."""
    labels = [REGIME_LABELS[i] for i in range(model.n_components)]
    return pd.DataFrame(model.transmat_, index=labels, columns=labels)


def current_regime_summary(df: pd.DataFrame) -> dict:
    """
    Convenience function returning the current bar's regime + context.
    Returns dict with: regime_id, label, color, atr_norm, vol_ratio, win_rate_modifier
    """
    regimes = predict_regimes(df)
    current = int(regimes.dropna().iloc[-1])

    # Win-rate modifier: reduces score threshold recommendation by regime
    # Low-vol sweep conditions are favorable for the strategy
    modifiers = {0: +5, 1: 0, 2: -10}

    feat = _build_features(df)
    last_valid = feat[~np.isnan(feat).any(axis=1)][-1]

    return {
        "regime_id": current,
        "label": REGIME_LABELS[current],
        "color": REGIME_COLORS[current],
        "atr_norm": round(float(last_valid[0]) * 100, 4),  # as % of price
        "vol_ratio": round(float(last_valid[1]), 3),
        "score_threshold_modifier": modifiers[current],
    }
