"""
Thermal AI Agent
=================
Agentic AI decision layer for battery thermal management.

The agent observes operating conditions, assesses risk,
makes predictions, and recommends cooling actions.

Decision workflow:
    OBSERVATION → RISK ASSESSMENT → PREDICTION → ACTION SELECTION → COOLING RESPONSE
"""

import json
import copy
from pathlib import Path
from src.risk_engine import load_risk_config, assess_thermal_risk
from src.prediction_engine import predict_temperature


def create_agent_decision(
    xgb_model,
    current_features: dict,
    forecasts: list,
    temp_trend: float,
    current_risk_level: str,
    hotspot_risk_pct: float,
    anomaly_label: str = "NORMAL",
) -> dict:
    """
    Run the AI agent decision workflow using actual ML predictions.
    """
    # ── 1. OBSERVATION ─────────────────────────────────
    battery_current = current_features.get("battery_current_A", 5.0)
    ambient_temp = current_features.get("ambient_temperature_C", 25.0)
    coolant_flow = current_features.get("coolant_flow_rate_kg_s", 0.02)
    current_temp = predict_temperature(xgb_model, current_features)

    observations = []
    if temp_trend > 1.0:
        observations.append("Temperature rising rapidly")
    elif temp_trend > 0.3:
        observations.append("Temperature gradually increasing")
    elif temp_trend < -0.3:
        observations.append("Temperature decreasing")
    else:
        observations.append("Temperature stable")

    if battery_current > 10:
        observations.append("High battery discharge current")
    if ambient_temp > 35:
        observations.append("Elevated ambient temperature")
    if coolant_flow < 0.012:
        observations.append("Low coolant flow rate")
    if anomaly_label == "ANOMALY":
        observations.append("Anomalous operating conditions detected")

    observation_text = ". ".join(observations) if observations else "Normal conditions"

    # ── 2. PREDICTION ──────────────────────────────────
    max_predicted = current_temp
    if forecasts:
        max_predicted = max(p.get("predicted_C", current_temp) for p in forecasts)
    prediction_text = f"5-minute temperature: {max_predicted:.1f}C"

    # ── 3. EVALUATE & CONSIDER ACTIONS ─────────────────
    candidates = [
        {"action": "Maintain cooling", "flow": coolant_flow, "target_pct": int(coolant_flow * 1000)},
        {"action": "Cooling 20%", "flow": 0.020, "target_pct": 20},
        {"action": "Cooling 35%", "flow": 0.035, "target_pct": 35},
        {"action": "Emergency cooling", "flow": 0.040, "target_pct": 40},
    ]

    options_evaluated = []
    for cand in candidates:
        sim_features = copy.deepcopy(current_features)
        sim_features["coolant_flow_rate_kg_s"] = cand["flow"]
        pred_temp = predict_temperature(xgb_model, sim_features)
        risk = assess_thermal_risk(pred_temp, hotspot_risk_pct=hotspot_risk_pct)
        options_evaluated.append({
            "action": cand["action"],
            "target_pct": cand["target_pct"],
            "flow": cand["flow"],
            "predicted_temp": pred_temp,
            "risk_level": risk["risk_level"],
            "risk_score": risk["risk_score"]
        })

    # ── 4. SELECT ACTION ───────────────────────────────
    # Rank by: (1) risk level severity, (2) risk score, (3) coolant flow efficiency.
    _level_rank = {"NORMAL": 0, "CAUTION": 1, "HIGH": 2, "CRITICAL": 3}
    best_option = min(
        options_evaluated,
        key=lambda x: (
            _level_rank.get(x["risk_level"], 9),  # lower is better
            x["flow"] if x["risk_level"] == "NORMAL" else x["risk_score"],
            x["flow"],                             # lower flow preferred when tied
        ),
    )

    decision_text = (
        f"Increase cooling to {best_option['target_pct']}%."
        if best_option["flow"] > coolant_flow
        else f"Decrease cooling to {best_option['target_pct']}%." if best_option["flow"] < coolant_flow
        else "Maintain current cooling."
    )
    expected_result_text = f"Thermal risk projected: {best_option['risk_level']}."
    if current_risk_level != best_option["risk_level"]:
        expected_result_text = f"Thermal risk reduced from {current_risk_level} to {best_option['risk_level']}."

    return {
        "observation": observation_text,
        "prediction": prediction_text,
        "options_evaluated": options_evaluated,
        "decision": decision_text,
        "expected_result": expected_result_text,
        "target_coolant_percent": best_option["target_pct"],
        "target_coolant_flow": best_option["flow"],
    }


def simulate_cooling_response(
    xgb_model,
    start_features: dict,
    target_coolant_flow: float,
    steps: int = 5,
) -> dict:
    """
    Simulate the temperature trajectory using the XGBoost model.
    """
    current_flow = start_features.get("coolant_flow_rate_kg_s", 0.01)
    flow_step = (target_coolant_flow - current_flow) / max(steps - 1, 1)

    coolant_trajectory = []
    temp_trajectory = []
    
    current_features = copy.deepcopy(start_features)
    peak_before = predict_temperature(xgb_model, current_features)

    for i in range(steps):
        flow = current_flow + flow_step * i
        current_features["coolant_flow_rate_kg_s"] = flow
        
        # Actual XGBoost prediction
        temp = predict_temperature(xgb_model, current_features)
        
        coolant_trajectory.append(round(flow * 1000, 1)) # approx percent
        temp_trajectory.append(round(temp, 1))

    return {
        "coolant_trajectory": coolant_trajectory,
        "temp_trajectory": temp_trajectory,
        "peak_before": round(peak_before, 1),
        "peak_after": temp_trajectory[-1],
        "reduction_C": round(peak_before - temp_trajectory[-1], 1),
    }


def run_thermal_agent_loop(
    xgb_model,
    iso_model=None,
    n_steps: int = 6,
    seed: int = 42,
) -> list:
    """
    Run a simulated thermal stress event through the agentic decision loop.

    Generates n_steps simulation states representing:
      - Steps 0-2: Progressive thermal stress (temperature rising)
      - Steps 3-5: Agent intervenes with cooling → temperature stabilises/falls

    Each step returns a dict:
        {
            "time_sec": int,
            "features": dict,
            "agent_decision": dict or None,  # populated at step 2
        }
    """
    from src.simulation_engine import get_simulation_state
    from src.forecasting import forecast_temperatures
    from src.anomaly_detection import detect_anomaly
    from src.risk_engine import assess_thermal_risk
    from src.hotspot_risk import estimate_hotspot_risk

    trajectory = []
    intervention_flow = None  # set after agent decides

    for i in range(n_steps):
        # Phase 1 (0-2): escalating thermal stress
        if i < 3:
            features = get_simulation_state("THERMAL_STRESS", step=i + 4, seed=seed)
        else:
            # Phase 2 (3-5): agent applies selected cooling
            features = get_simulation_state("THERMAL_STRESS", step=i + 4, seed=seed)
            if intervention_flow is not None:
                features["coolant_flow_rate_kg_s"] = min(
                    0.04,
                    features["coolant_flow_rate_kg_s"] + intervention_flow * (i - 2),
                )

        agent_decision = None

        # At step 2 (third observation) the agent runs its decision loop
        if i == 2:
            forecasts = forecast_temperatures(xgb_model, features)
            hotspot = estimate_hotspot_risk(
                predict_temperature(xgb_model, features),
                features["battery_current_A"],
                features["discharge_rate_C"],
                features["coolant_flow_rate_kg_s"],
                features["ambient_temperature_C"],
            )
            pred_5min = forecasts[-1]["predicted_C"] if forecasts else predict_temperature(xgb_model, features)
            risk = assess_thermal_risk(
                predict_temperature(xgb_model, features),
                pred_5min,
                temp_rate_per_min=1.5,
                hotspot_risk_pct=hotspot["hotspot_risk_percent"],
            )

            agent_decision = create_agent_decision(
                xgb_model=xgb_model,
                current_features=features,
                forecasts=forecasts,
                temp_trend=1.5,
                current_risk_level=risk["risk_level"],
                hotspot_risk_pct=hotspot["hotspot_risk_percent"],
            )
            # Determine the incremental flow adjustment per post-decision step
            target_flow = agent_decision["target_coolant_flow"]
            current_flow = features["coolant_flow_rate_kg_s"]
            intervention_flow = max(0, (target_flow - current_flow) / 3)

        trajectory.append({
            "time_sec": i * 25,
            "features": copy.deepcopy(features),
            "agent_decision": agent_decision,
        })

    return trajectory
