"""
Prediction Engine
=================
Load trained XGBoost and PyTorch MLP models and generate failure probability predictions.
"""

import torch
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from src.preprocessing import FEATURE_COLUMNS
from src.models.mlp_predictor import FailureClassifier

class MLPPredictor:
    def __init__(self, model_path: str, scaler_x_path: str, imputer_path: str):
        self.model = FailureClassifier(input_dim=len(FEATURE_COLUMNS))
        self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        self.model.eval()
        self.scaler_x = joblib.load(scaler_x_path)
        self.imputer = joblib.load(imputer_path)
        
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Returns probability of failure (class 1)."""
        X_imputed = self.imputer.transform(df[FEATURE_COLUMNS].values)
        X_scaled = self.scaler_x.transform(X_imputed)
        with torch.no_grad():
            logits = self.model(torch.FloatTensor(X_scaled)).numpy().squeeze()
            
        # Handle scalar vs array return safely
        if logits.ndim == 0:
            logits = np.array([logits])
            
        probs = 1.0 / (1.0 + np.exp(-logits)) # sigmoid
        return probs

def load_mlp_model(model_dir: str = "models/failure_predictor"):
    """Load the trained PyTorch Failure Predictor."""
    try:
        return MLPPredictor(
            model_path=f"{model_dir}/mlp.pt",
            scaler_x_path=f"{model_dir}/scaler_x.pkl",
            imputer_path=f"{model_dir}/imputer.pkl"
        )
    except Exception as e:
        print(f"Warning: Could not load MLP model: {e}")
        return None

def load_model(model_path: str = "models/xgboost_failure_model.joblib"):
    """Load the trained XGBoost model from disk."""
    try:
        return joblib.load(model_path)
    except Exception as e:
        print(f"Warning: Could not load XGBoost model: {e}")
        return None

def predict_failure_probability(model, features: dict) -> float:
    """Predict failure probability from a feature dictionary using either model."""
    df = pd.DataFrame([features])
    df = df[FEATURE_COLUMNS]
    if isinstance(model, MLPPredictor):
        prediction = model.predict_proba(df)[0]
    else:
        prediction = model.predict_proba(df)[0][1] # XGBoost predict_proba returns [P(0), P(1)]
    return float(prediction)

def predict_batch(model, df: pd.DataFrame) -> np.ndarray:
    """Predict failure probabilities for a batch of inputs."""
    if isinstance(model, MLPPredictor):
        return model.predict_proba(df[FEATURE_COLUMNS])
    return model.predict_proba(df[FEATURE_COLUMNS])[:, 1]

def get_default_features() -> dict:
    """Return a set of moderate/normal operating conditions for the new dataset."""
    return {
        "cycle_count": 500,
        "state_of_charge": 80.0,
        "depth_of_discharge": 20.0,
        "cell_voltage_avg": 3.8,
        "cell_voltage_std": 0.01,
        "cell_temperature_avg": 30.0,
        "cell_temperature_max": 32.0,
        "internal_resistance": 1.5,
        "charge_efficiency": 95.0,
        "fast_charge_ratio": 0.1,
        "average_charge_power_kw": 50.0,
        "charging_interruptions": 1,
        "cooling_system_health": 100.0,
        "average_ambient_temperature": 25.0,
    }
