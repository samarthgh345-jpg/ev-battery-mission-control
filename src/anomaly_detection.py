"""
Anomaly Detection
=================
Autoencoder for detecting unusual battery operating conditions.
"""

import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
from src.preprocessing import ANOMALY_FEATURES
from src.models.autoencoder import ThermalAutoencoder

def load_autoencoder(model_dir: str):
    """Load the trained Autoencoder, scaler, imputer, and threshold."""
    path = Path(model_dir)
    ae_path = path / "ae.pt"
    scaler_path = path / "scaler.pkl"
    imputer_path = path / "imputer.pkl"
    threshold_path = path / "threshold.txt"
    
    model = ThermalAutoencoder(input_dim=len(ANOMALY_FEATURES))
    model.load_state_dict(torch.load(ae_path))
    model.eval()
    
    scaler = joblib.load(scaler_path)
    imputer = joblib.load(imputer_path)
    
    with open(threshold_path, "r") as f:
        threshold = float(f.read().strip())
        
    return model, scaler, imputer, threshold

def detect_anomaly(model, scaler, imputer, threshold, features: dict) -> dict:
    """
    Detect whether a single observation is anomalous using the Autoencoder.

    Returns:
        {
            "label": "NORMAL" or "ANOMALY",
            "score": float (reconstruction error),
            "threshold": float,
            "message": str
        }
    """
    row = []
    for feat in ANOMALY_FEATURES:
        row.append(features.get(feat, 0.0))
        
    X = np.array([row])
    
    # Preprocess
    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)
    X_tensor = torch.FloatTensor(X_scaled)
    
    with torch.no_grad():
        reconstruction = model(X_tensor)
        mse = torch.mean((X_tensor - reconstruction)**2).item()
        
    if mse > threshold:
        label = "ANOMALY"
        message = "High reconstruction error: Unusual operating behaviour detected"
    else:
        label = "NORMAL"
        message = "Operating within expected parameters"

    return {
        "label": label,
        "score": round(float(mse), 4),
        "threshold": round(float(threshold), 4),
        "message": message,
    }
