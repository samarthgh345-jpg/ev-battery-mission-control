"""
Temperature Forecasting
========================
Forecast battery temperature at +1, +3, +5 minutes.

Methodology:
    1. Take the current sensor state as the baseline.
    2. Project sensor trajectories forward assuming current trends continue
       (e.g., current increasing, coolant flow decreasing).
    3. Run the trained XGBoost model at each future timestep to predict
       the max battery temperature.

This is a simulation-based forecast using the trained ML model, not a
time-series model. It provides an estimate of "what temperature would
the model predict if conditions continue on their current trajectory."
"""

import copy
import numpy as np
from src.prediction_engine import predict_temperature


def forecast_temperatures(
    model,
    current_features: dict,
    trend: dict = None,
    horizons_min: list = None,
) -> list:
    """
    Forecast future temperatures at specified horizons.

    Args:
        model: Trained XGBoost model
        current_features: Current sensor readings as a feature dict
        trend: Per-minute rate of change for each feature (optional).
               If None, uses small default trends.
        horizons_min: List of forecast horizons in minutes [1, 3, 5]

    Returns:
        List of dicts: [{"horizon_min": 1, "predicted_C": 42.6}, ...]
    """
    if horizons_min is None:
        horizons_min = [1, 3, 5]

    # Default trends: slight increase in thermal load
    if trend is None:
        trend = {
            "battery_current_A": 0.15,
            "ambient_temperature_C": 0.05,
            "battery_temperature_C": 0.3,
            "coolant_inlet_temperature_C": 0.02,
            "coolant_flow_rate_kg_s": -0.0002,
            "discharge_rate_C": 0.05,
        }

    forecasts = []
    for horizon in horizons_min:
        future_features = copy.deepcopy(current_features)
        for feat, rate in trend.items():
            if feat in future_features:
                future_features[feat] = future_features[feat] + rate * horizon

        # Clip to physically plausible ranges
        future_features["battery_current_A"] = max(0.5, min(15.0, future_features.get("battery_current_A", 5.0)))
        future_features["coolant_flow_rate_kg_s"] = max(0.003, min(0.04, future_features.get("coolant_flow_rate_kg_s", 0.02)))
        future_features["ambient_temperature_C"] = max(15.0, min(50.0, future_features.get("ambient_temperature_C", 28.0)))
        future_features["battery_temperature_C"] = max(20.0, min(60.0, future_features.get("battery_temperature_C", 35.0)))
        future_features["discharge_rate_C"] = max(0.5, min(5.0, future_features.get("discharge_rate_C", 1.5)))

        predicted_temp = predict_temperature(model, future_features)
        forecasts.append({
            "horizon_min": horizon,
            "predicted_C": round(predicted_temp, 1),
        })

    return forecasts
