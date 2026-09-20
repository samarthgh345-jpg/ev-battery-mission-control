"""
Prediction Engine
=================
Load trained XGBoost and PyTorch MLP models and generate predictions.
"""

import torch
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from src.preprocessing import FEATURE_COLUMNS
from src.models.mlp_predictor import ThermalMLP

class MLPPredictor:
    def __init__(self, model_path: str, scaler_x_path: str, scaler_y_path: str):
        self.model = ThermalMLP(input_dim=len(FEATURE_COLUMNS))
        self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        self.model.eval()
        self.scaler_x = joblib.load(scaler_x_path)
        self.scaler_y = joblib.load(scaler_y_path)
        
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler_x.transform(df[FEATURE_COLUMNS].values)
        with torch.no_grad():
            preds_scaled = self.model(torch.FloatTensor(X_scaled)).numpy()
        preds = self.scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        return preds

def load_mlp_model(model_dir: str = "models/thermal_predictor"):
    """Load the trained PyTorch MLP Predictor."""
    try:
        return MLPPredictor(
            model_path=f"{model_dir}/mlp.pt",
            scaler_x_path=f"{model_dir}/scaler_x.pkl",
            scaler_y_path=f"{model_dir}/scaler_y.pkl"
        )
    except Exception as e:
        print(f"Warning: Could not load MLP model: {e}")
        return None

def load_model(model_path: str = "models/xgboost_temperature_model.joblib"):
    """Load the trained XGBoost model from disk."""
    try:
        return joblib.load(model_path)
    except Exception as e:
        print(f"Warning: Could not load XGBoost model: {e}")
        return None

def predict_temperature(model, features: dict) -> float:
    """Predict max battery temperature from a feature dictionary using either model."""
    df = pd.DataFrame([features])
    df = df[FEATURE_COLUMNS]
    if isinstance(model, MLPPredictor):
        prediction = model.predict(df)[0]
    else:
        prediction = model.predict(df)[0]
    return float(prediction)

def predict_batch(model, df: pd.DataFrame) -> np.ndarray:
    """Predict for a batch of inputs."""
    if isinstance(model, MLPPredictor):
        return model.predict(df[FEATURE_COLUMNS])
    return model.predict(df[FEATURE_COLUMNS])

def get_default_features() -> dict:
    """Return a set of moderate/normal operating conditions."""
    return {
        "battery_current_A": 5.0,
        "battery_voltage_V": 3.7,
        "state_of_charge_percent": 60.0,
        "ambient_temperature_C": 28.0,
        "battery_temperature_C": 35.0,
        "coolant_inlet_temperature_C": 23.0,
        "coolant_flow_rate_kg_s": 0.020,
        "coolant_pressure_Pa": 120000.0,
        "nanoparticle_concentration_percent": 2.0,
        "reynolds_number": 1500.0,
        "heat_transfer_coefficient_W_m2K": 4500.0,
        "microchannel_width_mm": 0.6,
        "microchannel_height_mm": 0.4,
        "discharge_rate_C": 1.5,
    }
