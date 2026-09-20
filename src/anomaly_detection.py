"""
Anomaly Detection
=================
Isolation Forest for detecting unusual battery operating conditions.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
from src.preprocessing import ANOMALY_FEATURES


def train_anomaly_detector(
    df: pd.DataFrame, contamination: float = 0.08, seed: int = 42
) -> IsolationForest:
    """Train an Isolation Forest on the multivariate operating conditions."""
    X = df[ANOMALY_FEATURES].copy()
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X)
    return model


def detect_anomaly(model: IsolationForest, features: dict) -> dict:
    """
    Detect whether a single observation is anomalous.

    Returns:
        {
            "label": "NORMAL" or "ANOMALY",
            "score": float (lower = more anomalous),
            "message": str
        }
    """
    row = {}
    for feat in ANOMALY_FEATURES:
        row[feat] = features.get(feat, 0.0)
    df = pd.DataFrame([row])

    prediction = model.predict(df)[0]  # 1 = normal, -1 = anomaly
    score = model.decision_function(df)[0]

    if prediction == -1:
        label = "ANOMALY"
        message = "Unusual thermal/operating behaviour detected"
    else:
        label = "NORMAL"
        message = "Operating within expected parameters"

    return {
        "label": label,
        "score": round(float(score), 4),
        "message": message,
    }


def save_anomaly_model(model: IsolationForest, model_dir: str) -> Path:
    """Save the trained anomaly detector."""
    path = Path(model_dir) / "isolation_forest_model.joblib"
    joblib.dump(model, path)
    return path


def load_anomaly_model(model_dir: str) -> IsolationForest:
    """Load a trained anomaly detector."""
    path = Path(model_dir) / "isolation_forest_model.joblib"
    return joblib.load(path)
