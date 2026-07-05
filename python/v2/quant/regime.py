"""
quant/regime.py
Market regime detection using sklearn GaussianMixture.

GaussianMixture is functionally equivalent to a Gaussian HMM for regime
classification when temporal ordering is less critical than cluster identity.
It ships with scikit-learn, requires no C compilation, and deploys cleanly
on Streamlit Cloud.

States (remapped by ascending ATR mean after fitting):
  0 — Low-volatility / range-bound  (optimal for session-sweep entries)
  1 — Trending / directional
  2 — High-volatility / event-driven (avoid session-sweep entries)
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

MODEL_PATH  = Path(__file__).parent.parent / "assets" / "gmm_model.pkl"
SCALER_PATH = Path(__file__).parent.parent / "assets" / "gmm_scaler.pkl"

REGIME_LABELS = {0: "Low-Vol / Range", 1: "Trending", 2: "High-Vol / Event"}
REGIME_COLORS = {0: "#00d2aa", 1: "#fdcb6e", 2: "#d63031"}


def _build_features(df: pd.DataFrame) -> np.ndarray:
    """
    Compute ATR-14 (price-normalized) and 20-bar volume ratio.
    Returns (N, 2) float array. Rows with NaN are preserved for index alignment.
    """
    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values
    vol   = df["volume"].values

    # True Range
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:]  - close[:-1])
        )
    )

    # ATR via Wilder smoothing
    atr = np.full(len(df), np.nan)
    if len(tr) >= 14:
        atr[14] = tr[:14].mean()
        for i in range(15, len(df)):
            atr[i] = (atr[i - 1] * 13 + tr[i - 1]) / 14

    atr_norm  = atr / close

    # Volume ratio vs 20-bar MA
    vol_ma    = pd.Series(vol).rolling(20).mean().values
    vol_ratio = np.where(vol_ma > 0, vol / vol_ma, 1.0)

    return np.column_stack([atr_norm, vol_ratio])


def fit_regime_model(df: pd.DataFrame, n_states: int = 3, random_state: int = 42):
    """
    Fit a GaussianMixture on OHLCV features.
    Saves model + scaler to assets/. States remapped by ascending ATR mean.
    """
    feat        = _build_features(df)
    valid_mask  = ~np.isnan(feat).any(axis=1)
    feat_clean  = feat[valid_mask]

    scaler      = StandardScaler()
    feat_scaled = scaler.fit_transform(feat_clean)

    gmm = GaussianMixture(
        n_components=n_states,
        covariance_type="full",
        max_iter=300,
        random_state=random_state,
        n_init=5,
    )
    gmm.fit(feat_scaled)

    # Remap component labels so 0 = lowest ATR, 2 = highest ATR
    mean_atrs = gmm.means_[:, 0]
    order     = np.argsort(mean_atrs)
    remap     = {int(old): int(new) for new, old in enumerate(order)}
    gmm._remap = remap

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(gmm,    MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    return gmm


def predict_regimes(df: pd.DataFrame) -> pd.Series:
    """
    Load saved GMM, predict regime for each bar in df.
    Returns pd.Series of int regime labels aligned to df.index.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Regime model not found. Run fit_regime_model() first.")

    gmm: GaussianMixture = joblib.load(MODEL_PATH)
    scaler: StandardScaler = joblib.load(SCALER_PATH)

    feat       = _build_features(df)
    valid_mask = ~np.isnan(feat).any(axis=1)

    feat_scaled = scaler.transform(feat[valid_mask])
    raw_states  = gmm.predict(feat_scaled)
    remap       = getattr(gmm, "_remap", {i: i for i in range(gmm.n_components)})
    remapped    = np.array([remap.get(int(s), int(s)) for s in raw_states])

    regimes            = np.full(len(df), np.nan)
    regimes[valid_mask] = remapped
    return pd.Series(regimes, index=df.index, name="regime")


def get_component_stats(gmm: GaussianMixture, scaler: StandardScaler) -> pd.DataFrame:
    """
    Return readable component statistics (means in original scale).
    Replaces HMM transition matrix with weight + mean table.
    """
    remap   = getattr(gmm, "_remap", {i: i for i in range(gmm.n_components)})
    means_s = scaler.inverse_transform(gmm.means_)
    rows    = []
    for raw_id in range(gmm.n_components):
        mapped = remap.get(raw_id, raw_id)
        rows.append({
            "regime":    mapped,
            "label":     REGIME_LABELS.get(mapped, str(mapped)),
            "weight":    round(float(gmm.weights_[raw_id]), 4),
            "atr_norm":  round(float(means_s[raw_id, 0]), 6),
            "vol_ratio": round(float(means_s[raw_id, 1]), 4),
        })
    return pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)


def current_regime_summary(df: pd.DataFrame) -> dict:
    """
    Returns current bar regime context including recommended score threshold modifier.
    """
    regimes    = predict_regimes(df)
    current    = int(regimes.dropna().iloc[-1])
    modifiers  = {0: +5, 1: 0, 2: -10}

    feat       = _build_features(df)
    last_valid = feat[~np.isnan(feat).any(axis=1)][-1]

    return {
        "regime_id":                current,
        "label":                    REGIME_LABELS.get(current, "Unknown"),
        "color":                    REGIME_COLORS.get(current, "#7faaaf"),
        "atr_norm":                 round(float(last_valid[0]) * 100, 4),
        "vol_ratio":                round(float(last_valid[1]), 3),
        "score_threshold_modifier": modifiers.get(current, 0),
    }
