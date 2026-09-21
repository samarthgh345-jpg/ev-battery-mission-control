"""
Failure Probability Forecasting
================================
Forecast battery failure probability at +1, +3, +5 minutes.
"""

import copy
from src.prediction_engine import predict_failure_probability

def forecast_failure_probability(
    model,
    current_features: dict,
    trend: dict = None,
    horizons_min: list = None,
) -> list:
    """
    Forecast future failure probability at specified horizons.
    """
    if horizons_min is None:
        horizons_min = [1, 3, 5]

    # Default trends: slight increase in thermal load / aging
    if trend is None:
        trend = {
            "cell_temperature_avg": 0.5,
            "cell_temperature_max": 0.8,
            "internal_resistance": 0.05,
            "state_of_charge": -0.5,
        }

    forecasts = []
    for horizon in horizons_min:
        future_features = copy.deepcopy(current_features)
        for feat, rate in trend.items():
            if feat in future_features:
                future_features[feat] = future_features[feat] + rate * horizon

        # Clip to physically plausible ranges
        future_features["cell_temperature_avg"] = max(15.0, min(80.0, future_features.get("cell_temperature_avg", 30.0)))
        future_features["cell_temperature_max"] = max(15.0, min(85.0, future_features.get("cell_temperature_max", 32.0)))
        future_features["internal_resistance"] = max(0.5, min(10.0, future_features.get("internal_resistance", 1.5)))
        future_features["state_of_charge"] = max(0.0, min(100.0, future_features.get("state_of_charge", 80.0)))

        predicted_prob = predict_failure_probability(model, future_features)
        forecasts.append({
            "horizon_min": horizon,
            "predicted_prob": round(predicted_prob, 4),
        })

    return forecasts
