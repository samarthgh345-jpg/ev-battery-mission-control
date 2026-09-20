"""
Thermal Risk Engine
===================
Transparent, configurable risk assessment layer.

Loads thresholds from config/risk_thresholds.json.
These are prototype thresholds for demonstration — NOT experimentally validated safety limits.
"""

import json
from pathlib import Path
from typing import Optional


def load_risk_config(config_path: str = None) -> dict:
    """Load risk thresholds from JSON config."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config" / "risk_thresholds.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def assess_thermal_risk(
    current_temp: float,
    predicted_temp_5min: float = None,
    temp_rate_per_min: float = 0.0,
    anomaly_score: float = 0.0,
    hotspot_risk_pct: float = 0.0,
    config: dict = None,
) -> dict:
    """
    Assess overall thermal risk.

    Returns:
        {
            "risk_level": "NORMAL" | "CAUTION" | "HIGH" | "CRITICAL",
            "risk_score": float (0-100),
            "color": str,
            "icon": str,
            "description": str,
            "factors": list[str]
        }
    """
    if config is None:
        config = load_risk_config()

    t_thresh = config["temperature_thresholds_C"]
    r_thresh = config["temperature_rate_thresholds_C_per_min"]
    h_thresh = config["hotspot_risk_thresholds_percent"]
    labels = config["risk_labels"]

    factors = []

    # Temperature-based risk
    if current_temp > t_thresh["critical_above"]:
        temp_risk = 3
        factors.append(f"Temperature {current_temp:.1f}°C exceeds critical threshold")
    elif current_temp > t_thresh["high_max"]:
        temp_risk = 3
        factors.append(f"Temperature {current_temp:.1f}°C in critical range")
    elif current_temp > t_thresh["caution_max"]:
        temp_risk = 2
        factors.append(f"Temperature {current_temp:.1f}°C in high range")
    elif current_temp > t_thresh["normal_max"]:
        temp_risk = 1
        factors.append(f"Temperature {current_temp:.1f}°C above normal range")
    else:
        temp_risk = 0

    # Predicted temperature risk
    pred_risk = 0
    if predicted_temp_5min is not None:
        if predicted_temp_5min > t_thresh["critical_above"]:
            pred_risk = 3
            factors.append(f"Predicted temperature {predicted_temp_5min:.1f}°C (critical)")
        elif predicted_temp_5min > t_thresh["high_max"]:
            pred_risk = 2
            factors.append(f"Predicted temperature {predicted_temp_5min:.1f}°C (high)")
        elif predicted_temp_5min > t_thresh["caution_max"]:
            pred_risk = 1
            factors.append(f"Predicted temperature {predicted_temp_5min:.1f}°C (caution)")

    # Rate of change risk
    rate_risk = 0
    if abs(temp_rate_per_min) > r_thresh["critical_above"]:
        rate_risk = 3
        factors.append(f"Rapid temperature increase: {temp_rate_per_min:.2f}°C/min")
    elif abs(temp_rate_per_min) > r_thresh["high_max"]:
        rate_risk = 2
        factors.append(f"High temperature rate: {temp_rate_per_min:.2f}°C/min")
    elif abs(temp_rate_per_min) > r_thresh["caution_max"]:
        rate_risk = 1
        factors.append(f"Elevated temperature rate: {temp_rate_per_min:.2f}°C/min")

    # Hotspot risk
    hot_risk = 0
    if hotspot_risk_pct > h_thresh["critical_above"]:
        hot_risk = 3
        factors.append(f"Hotspot risk {hotspot_risk_pct:.0f}% (critical)")
    elif hotspot_risk_pct > h_thresh["high_max"]:
        hot_risk = 2
        factors.append(f"Hotspot risk {hotspot_risk_pct:.0f}% (high)")
    elif hotspot_risk_pct > h_thresh["caution_max"]:
        hot_risk = 1
        factors.append(f"Hotspot risk {hotspot_risk_pct:.0f}% (elevated)")

    # Anomaly boost
    anomaly_risk = 0
    if anomaly_score < -0.1:
        anomaly_risk = 1
        factors.append("Anomalous operating conditions detected")

    # Overall risk = max of individual risks
    overall = max(temp_risk, pred_risk, rate_risk, hot_risk, anomaly_risk)
    level_map = {0: "NORMAL", 1: "CAUTION", 2: "HIGH", 3: "CRITICAL"}
    risk_level = level_map[overall]

    # Numeric risk score (0-100)
    risk_score = min(100, overall * 25 + (current_temp - 30) * 2 + hotspot_risk_pct * 0.2)
    risk_score = max(0, risk_score)

    label_info = labels[risk_level]

    return {
        "risk_level": risk_level,
        "risk_score": round(risk_score, 1),
        "color": label_info["color"],
        "icon": label_info["icon"],
        "description": label_info["description"],
        "factors": factors if factors else ["All parameters within normal range"],
    }
