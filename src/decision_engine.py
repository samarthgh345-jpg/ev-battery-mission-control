"""
Deterministic Thermal Decision Engine
=======================================
Transparent, rule-based thermal safety and cooling decision component
for EV battery mission control. Evaluates current simulated telemetry,
identifies triggered safety boundaries, and suggests actionable simulated responses.
"""

from typing import Dict, Any, List


def evaluate_thermal_recommendation(
    hotspot_temp: float,
    avg_temp: float,
    cooling_health: float,
    failure_prob: float,
    anomaly_score: float,
    anomaly_threshold: float = 0.1929,
    coolant_flow: float = 0.02,
    current: float = 10.0,
) -> Dict[str, Any]:
    """
    Evaluate current battery operating state against explicit deterministic rules.

    Returns:
        Dict containing:
            - observation: str
            - triggers: List[str]
            - predicted_outcome: str
            - suggested_action: str
            - reason: str
            - severity: "NORMAL" | "CAUTION" | "HIGH" | "CRITICAL"
            - recommended_flow_boost_pct: float
    """
    triggers: List[str] = []
    hotspot_temp = float(hotspot_temp)
    avg_temp = float(avg_temp)
    cooling_health = float(cooling_health)
    failure_prob = float(failure_prob)
    anomaly_score = float(anomaly_score)

    # 1. Rule 1: Critical Runaway Onset (Hotspot >= 48.0°C or Failure Prob >= 0.80)
    if hotspot_temp >= 48.0 or failure_prob >= 0.80:
        if hotspot_temp >= 48.0:
            triggers.append(f"Hotspot temperature ({hotspot_temp:.1f}°C) exceeded exothermic runaway boundary (48.0°C)")
        if failure_prob >= 0.80:
            triggers.append(f"MLP failure probability ({failure_prob * 100:.1f}%) crossed critical safety threshold (80%)")

        act = "APPLY EMERGENCY MAXIMUM COOLING (+35%) & POWER SHED"
        return {
            "observation": f"Critical thermal stress detected: Hotspot at {hotspot_temp:.1f}°C with {failure_prob * 100:.1f}% failure probability.",
            "triggers": triggers,
            "predicted_outcome": "Uncontrolled cell self-heating and thermal runaway propagation without immediate intervention.",
            "suggested_action": act,
            "recommended_action": act,
            "reason": "Exothermic cell degradation accelerates above 48°C. Maximum liquid coolant heat extraction is required.",
            "severity": "CRITICAL",
            "status": "CRITICAL",
            "recommended_flow_boost_pct": 35.0,
        }

    # 2. Rule 2: High Thermal Caution (Hotspot >= 42.0°C or Failure Prob >= 0.40 or Cooling Health <= 50%)
    if hotspot_temp >= 42.0 or failure_prob >= 0.40 or cooling_health <= 50.0:
        if hotspot_temp >= 42.0:
            triggers.append(f"Hotspot temperature ({hotspot_temp:.1f}°C) crossed caution threshold (42.0°C)")
        if failure_prob >= 0.40:
            triggers.append(f"Predicted failure probability ({failure_prob * 100:.1f}%) is elevated (>= 40%)")
        if cooling_health <= 50.0:
            triggers.append(f"Cooling system health ({cooling_health:.1f}%) is degraded (<= 50%)")

        act = "INCREASE COOLANT FLOW (+18%) & MODERATE FAST-CHARGE RATE"
        return {
            "observation": f"Elevated thermal risk: Hotspot at {hotspot_temp:.1f}°C (delta: +{hotspot_temp - avg_temp:.1f}°C above mean).",
            "triggers": triggers,
            "predicted_outcome": "Continued load current will elevate cell temperature toward the 48°C runaway threshold.",
            "suggested_action": act,
            "recommended_action": act,
            "reason": "Enhanced convective heat removal stabilizes the localized thermal gradient before exothermic onset.",
            "severity": "HIGH",
            "status": "HIGH",
            "recommended_flow_boost_pct": 18.0,
        }

    # 3. Rule 3: Multivariate Anomaly (Autoencoder MSE > threshold)
    if anomaly_score > anomaly_threshold:
        triggers.append(f"Autoencoder reconstruction error ({anomaly_score:.4f}) exceeded threshold ({anomaly_threshold:.4f})")
        act = "MAINTAIN COOLING & RUN WHAT-IF SENSITIVITY CHECK"
        return {
            "observation": "Atypical operating state detected by unsupervised Autoencoder.",
            "triggers": triggers,
            "predicted_outcome": "Possible sub-component impedance anomaly or localized voltage variance.",
            "suggested_action": act,
            "recommended_action": act,
            "reason": "Multi-variable feature relationship deviates from normal operational manifold, warranting monitoring.",
            "severity": "CAUTION",
            "status": "CAUTION",
            "recommended_flow_boost_pct": 10.0,
        }

    # 4. Rule 4: Moderate Temperature Elevation (Hotspot >= 36.0°C or variance >= 10.0°C)
    if hotspot_temp >= 36.0 or (hotspot_temp - avg_temp) >= 10.0:
        if hotspot_temp >= 36.0:
            triggers.append(f"Hotspot temperature ({hotspot_temp:.1f}°C) is mildly elevated")
        if (hotspot_temp - avg_temp) >= 10.0:
            triggers.append(f"Temperature variance across pack ({hotspot_temp - avg_temp:.1f}°C) exceeds normal 10°C envelope")

        act = "PREEMPTIVE COOLANT MODULATION (+10%)"
        return {
            "observation": f"Mild thermal gradient: Hotspot at {hotspot_temp:.1f}°C vs pack average of {avg_temp:.1f}°C.",
            "triggers": triggers,
            "predicted_outcome": "Gradual heat buildup under sustained high current.",
            "suggested_action": act,
            "recommended_action": act,
            "reason": "Early flow increase dissipates heat efficiently with minimal auxiliary pump power consumption.",
            "severity": "CAUTION",
            "status": "CAUTION",
            "recommended_flow_boost_pct": 10.0,
        }

    # 5. Rule 5: Nominal Safe Envelopes
    triggers.append("All physical parameters, model probabilities, and anomaly scores within nominal limits")
    act = "MAINTAIN NOMINAL OPERATION"
    return {
        "observation": f"Nominal thermal state: Pack average {avg_temp:.1f}°C, Hotspot {hotspot_temp:.1f}°C, Failure prob {failure_prob * 100:.2f}%.",
        "triggers": triggers,
        "predicted_outcome": "Stable steady-state operation within calibrated automotive battery safety margins.",
        "suggested_action": act,
        "recommended_action": act,
        "reason": "Heat generation matches convective cooling dissipation capacity.",
        "severity": "NORMAL",
        "status": "NORMAL",
        "recommended_flow_boost_pct": 0.0,
    }
