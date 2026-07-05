"""
quant/calibration.py
Maps the Pine heuristic probability score (60-100 range) to a
true posterior win probability via Platt scaling and isotonic regression.

Input: DataFrame of closed signals with columns [score, outcome]
       outcome must be binary-encoded: WIN=1, LOSS=0, SCRATCH excluded.

Output: fitted calibrators + calibration curve for display.
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss, log_loss
import joblib
from pathlib import Path
from typing import Tuple

PLATT_PATH  = Path(__file__).parent.parent / "assets" / "platt_calibrator.pkl"
ISO_PATH    = Path(__file__).parent.parent / "assets" / "iso_calibrator.pkl"


def _prepare(signals: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Filter to closed WIN/LOSS signals, return (scores, labels)."""
    closed = signals[signals["outcome"].isin(["WIN", "LOSS"])].copy()
    if len(closed) < 30:
        raise ValueError(f"Insufficient closed signals for calibration: {len(closed)} (need ≥30)")
    X = closed["score"].values.reshape(-1, 1)
    y = (closed["outcome"] == "WIN").astype(int).values
    return X, y


def fit_platt(signals: pd.DataFrame) -> LogisticRegression:
    """
    Platt scaling: fits a logistic regression on the Pine score.
    P(WIN | score) = sigmoid(a * score + b)
    """
    X, y = _prepare(signals)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
    lr.fit(Xs, y)

    PLATT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": lr, "scaler": scaler}, PLATT_PATH)
    return lr


def fit_isotonic(signals: pd.DataFrame) -> IsotonicRegression:
    """
    Isotonic regression calibration: non-parametric, monotone mapping.
    Better than Platt when there are enough samples (≥100).
    """
    X, y = _prepare(signals)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(X.ravel(), y)
    ISO_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(iso, ISO_PATH)
    return iso


def predict_calibrated_probability(score: float, method: str = "isotonic") -> float:
    """
    Predict calibrated win probability for a given Pine score.
    method: 'isotonic' | 'platt'
    """
    if method == "isotonic":
        iso: IsotonicRegression = joblib.load(ISO_PATH)
        return float(np.clip(iso.predict([score])[0], 0.0, 1.0))
    elif method == "platt":
        bundle = joblib.load(PLATT_PATH)
        lr, scaler = bundle["model"], bundle["scaler"]
        Xs = scaler.transform([[score]])
        return float(lr.predict_proba(Xs)[0, 1])
    raise ValueError(f"Unknown calibration method: {method}")


def calibration_report(signals: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      mean_score, fraction_win, count, brier_score, method
    for both Platt and Isotonic calibrators.
    Suitable for plotting reliability diagrams.
    """
    X, y = _prepare(signals)
    scores_raw = X.ravel()

    rows = []

    # Raw (uncalibrated)
    prob_true, prob_pred = calibration_curve(y, scores_raw / 100, n_bins=n_bins, strategy="uniform")
    for pt, pp in zip(prob_true, prob_pred):
        rows.append({"method": "Raw Pine Score", "mean_predicted": pp, "fraction_win": pt})

    # Platt
    if PLATT_PATH.exists():
        bundle = joblib.load(PLATT_PATH)
        lr, scaler = bundle["model"], bundle["scaler"]
        platt_probs = lr.predict_proba(scaler.transform(X))[:, 1]
        prob_true_p, prob_pred_p = calibration_curve(y, platt_probs, n_bins=n_bins, strategy="uniform")
        for pt, pp in zip(prob_true_p, prob_pred_p):
            rows.append({"method": "Platt Scaling", "mean_predicted": pp, "fraction_win": pt})

    # Isotonic
    if ISO_PATH.exists():
        iso = joblib.load(ISO_PATH)
        iso_probs = np.clip(iso.predict(scores_raw), 0, 1)
        prob_true_i, prob_pred_i = calibration_curve(y, iso_probs, n_bins=n_bins, strategy="uniform")
        for pt, pp in zip(prob_true_i, prob_pred_i):
            rows.append({"method": "Isotonic Regression", "mean_predicted": pp, "fraction_win": pt})

    return pd.DataFrame(rows)


def cross_validated_brier(signals: pd.DataFrame, cv: int = 5) -> pd.DataFrame:
    """
    Stratified k-fold cross-validation of Brier scores for all three methods.
    Returns DataFrame with [method, fold, brier_score].
    """
    X, y = _prepare(signals)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Raw
        raw_probs = np.clip(X_te.ravel() / 100, 0, 1)
        results.append({"method": "Raw", "fold": fold, "brier": brier_score_loss(y_te, raw_probs)})

        # Platt
        sc = StandardScaler()
        lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
        lr.fit(sc.fit_transform(X_tr), y_tr)
        platt_p = lr.predict_proba(sc.transform(X_te))[:, 1]
        results.append({"method": "Platt", "fold": fold, "brier": brier_score_loss(y_te, platt_p)})

        # Isotonic
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(X_tr.ravel(), y_tr)
        iso_p = np.clip(iso.predict(X_te.ravel()), 0, 1)
        results.append({"method": "Isotonic", "fold": fold, "brier": brier_score_loss(y_te, iso_p)})

    return pd.DataFrame(results)
