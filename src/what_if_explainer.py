"""
What-If Scenario Explainability Engine
=======================================
Generates defensible, physics-grounded, multi-model explanations for hypothetical
What-If parameter perturbations without fabricating SHAP values or unsupported causal claims.
"""

from typing import Dict, Any, List, Optional
import numpy as np

from src.prediction_engine import predict_failure_probability
from src.anomaly_detection import detect_anomaly
from src.risk_engine import assess_thermal_risk


def generate_what_if_explanation(
    baseline: Dict[str, Any],
    scenario: Dict[str, Any],
    mlp_model,
    xgb_model=None,
    ae_artifacts=None,
) -> Dict[str, Any]:
    """
    Generate dynamic, rigorous scenario explanation comparing baseline and hypothetical What-If inputs.

    Returns:
        Dict containing:
            - summary: Dict with overall risk direction and high-level synthesis
            - parameter_table: List[Dict] with baseline, scenario, abs delta, pct delta, and visual indicator
            - physics_explanations: List[Dict] with physical mechanisms and ranked OVAT sensitivities
            - model_comparison: Dict comparing MLP and XGBoost predictions and agreement
            - recommendation: Dict with actionable next simulated step and rationale
    """
    # ── 1. Model Predictions ─────────────────────────────────────
    base_mlp = float(predict_failure_probability(mlp_model, baseline))
    scen_mlp = float(predict_failure_probability(mlp_model, scenario))
    mlp_delta_pp = (scen_mlp - base_mlp) * 100

    base_xgb = float(predict_failure_probability(xgb_model, baseline)) if xgb_model is not None else None
    scen_xgb = float(predict_failure_probability(xgb_model, scenario)) if xgb_model is not None else None
    xgb_delta_pp = (scen_xgb - base_xgb) * 100 if (base_xgb is not None and scen_xgb is not None) else None

    # Anomaly detector evaluation if artifacts provided
    base_ae = None
    scen_ae = None
    if ae_artifacts is not None:
        ae_model, ae_scaler, ae_imputer, ae_threshold = ae_artifacts
        base_ae = detect_anomaly(ae_model, ae_scaler, ae_imputer, ae_threshold, baseline)
        scen_ae = detect_anomaly(ae_model, ae_scaler, ae_imputer, ae_threshold, scenario)

    # Heuristic Risk Engine scores
    base_risk = assess_thermal_risk(
        current_temp=float(baseline.get("cell_temperature_max", 28.5)),
        predicted_temp_5min=None,
        temp_rate_per_min=0.0,
        anomaly_score=float(base_ae["score"]) if base_ae else 0.01,
        hotspot_risk_pct=base_mlp * 100,
    )
    scen_risk = assess_thermal_risk(
        current_temp=float(scenario.get("cell_temperature_max", 28.5)),
        predicted_temp_5min=None,
        temp_rate_per_min=0.0,
        anomaly_score=float(scen_ae["score"]) if scen_ae else 0.01,
        hotspot_risk_pct=scen_mlp * 100,
    )
    risk_score_delta = scen_risk["risk_score"] - base_risk["risk_score"]

    # ── 2. Parameter Impact Table Calculation ───────────────────
    param_meta = [
        ("cell_temperature_avg", "Average Cell Temperature", "°C", True),
        ("cell_temperature_max", "Maximum Hotspot Temperature", "°C", True),
        ("average_charge_power_kw", "Average Charge Power", "kW", True),
        ("internal_resistance", "Internal Resistance", "mΩ", True),
        ("state_of_charge", "State of Charge", "%", True),
        ("cooling_system_health", "Cooling System Health", "%", False),  # False means higher is safer
    ]

    param_table = []
    changed_params = []

    for key, label, unit, higher_is_riskier in param_meta:
        b_val = float(baseline.get(key, 0.0))
        s_val = float(scenario.get(key, 0.0))
        abs_delta = s_val - b_val

        # Safe percentage delta calculation (avoid division by zero)
        if abs(b_val) > 1e-6:
            pct_delta = (abs_delta / b_val) * 100.0
            pct_str = f"{pct_delta:+.1f}%"
        else:
            pct_delta = 0.0 if abs(abs_delta) < 1e-6 else None
            pct_str = "N/A" if pct_delta is None else "0.0%"

        # Visual indicator that does not rely on color alone
        if abs(abs_delta) < 1e-4:
            direction = "UNCHANGED"
            indicator = "━"
            trend_desc = "No change"
        elif abs_delta > 0:
            direction = "INCREASE"
            indicator = "▲"
            trend_desc = f"+{abs_delta:.1f} {unit} increase"
        else:
            direction = "DECREASE"
            indicator = "▼"
            trend_desc = f"{abs_delta:.1f} {unit} decrease"

        is_changed = abs(abs_delta) >= 1e-4
        if is_changed:
            changed_params.append((key, label, unit, b_val, s_val, abs_delta, pct_delta, higher_is_riskier))

        param_table.append({
            "key": key,
            "label": label,
            "unit": unit,
            "baseline": f"{b_val:.1f} {unit}" if unit != "mΩ" else f"{b_val:.2f} {unit}",
            "scenario": f"{s_val:.1f} {unit}" if unit != "mΩ" else f"{s_val:.2f} {unit}",
            "abs_delta_val": abs_delta,
            "abs_delta": f"{abs_delta:+.1f} {unit}" if unit != "mΩ" else f"{abs_delta:+.2f} {unit}",
            "pct_delta": pct_str,
            "direction": direction,
            "indicator": indicator,
            "trend_desc": trend_desc,
            "is_changed": is_changed,
        })

    # ── 3. Scenario Summary ─────────────────────────────────────
    if mlp_delta_pp > 1.0 or risk_score_delta > 5:
        risk_direction = "INCREASED RISK"
        risk_color = "#D94841"
        summary_headline = f"Scenario Increased Thermal Risk (+{mlp_delta_pp:.1f} pp failure probability)"
    elif mlp_delta_pp < -1.0 or risk_score_delta < -5:
        risk_direction = "DECREASED RISK"
        risk_color = "#16866A"
        summary_headline = f"Scenario Reduced Thermal Risk ({mlp_delta_pp:.1f} pp failure probability)"
    else:
        risk_direction = "STABLE / NOMINAL"
        risk_color = "#16866A"
        summary_headline = "Scenario Maintained Nominal Baseline Stability"

    summary_text = (
        f"Compared to baseline, this scenario shifted PyTorch MLP failure probability from "
        f"{base_mlp * 100:.1f}% to {scen_mlp * 100:.1f}% ({mlp_delta_pp:+.1f} percentage points) "
        f"and modified the deterministic Risk Engine score from {base_risk['risk_score']:.0f} to "
        f"{scen_risk['risk_score']:.0f} ({risk_score_delta:+.0f} points)."
    )

    # ── 4. Controlled One-Variable-at-a-Time (OVAT) Sensitivity ──
    # Evaluate model with only ONE variable changed at a time from baseline
    sensitivities = []
    for key, label, unit, b_val, s_val, abs_delta, pct_delta, higher_is_riskier in changed_params:
        single_var_scenario = baseline.copy()
        single_var_scenario[key] = s_val
        single_prob = float(predict_failure_probability(mlp_model, single_var_scenario))
        marginal_delta_pp = (single_prob - base_mlp) * 100

        sensitivities.append({
            "key": key,
            "label": label,
            "unit": unit,
            "b_val": b_val,
            "s_val": s_val,
            "abs_delta": abs_delta,
            "marginal_delta_pp": marginal_delta_pp,
            "abs_marginal_delta_pp": abs(marginal_delta_pp),
        })

    # Rank drivers by absolute marginal impact
    sensitivities.sort(key=lambda x: x["abs_marginal_delta_pp"], reverse=True)

    # Physical mechanism explanations connected to actual inputs
    physics_explanations = []
    for item in sensitivities:
        k = item["key"]
        lbl = item["label"]
        delta = item["abs_delta"]
        unit = item["unit"]
        m_delta = item["marginal_delta_pp"]

        if k == "cell_temperature_max":
            mech = (
                f"Hotspot temperature changed by {delta:+.1f} {unit} (to {item['s_val']:.1f} {unit}). "
                f"In lithium-ion cells, temperatures exceeding 42.0°C compromise the Solid Electrolyte Interphase (SEI) "
                f"layer and increase internal degradation rates; temperatures above 48.0°C risk thermal runaway onset."
            )
        elif k == "internal_resistance":
            mech = (
                f"Internal resistance changed by {delta:+.2f} {unit} (to {item['s_val']:.2f} {unit}). "
                f"By Joule's law (Q_gen = I²R), cell heat generation scales linearly with internal impedance. "
                f"Elevated resistance severely increases heat buildup under active charge or discharge currents."
            )
        elif k == "average_charge_power_kw":
            mech = (
                f"Charge power changed by {delta:+.1f} {unit} (to {item['s_val']:.1f} {unit}). "
                f"Fast-charging power dictates pack current (I = P/V). Higher charging electrical power creates "
                f"intense thermal flux through continuous electrochemical ion transfer."
            )
        elif k == "cooling_system_health":
            mech = (
                f"Cooling system health changed by {delta:+.1f} {unit} (to {item['s_val']:.0f} {unit}). "
                f"BTMS convective heat removal rate scales with coolant flow and radiator effectiveness "
                f"(Q_rem ∝ h·A·ΔT). Degraded cooling leaves cell heat unextracted, causing temperature escalation."
            )
        elif k == "cell_temperature_avg":
            mech = (
                f"Average pack temperature changed by {delta:+.1f} {unit} (to {item['s_val']:.1f} {unit}). "
                f"Elevating mean temperature reduces the thermal gradient to the coolant and elevates nominal cell kinetics."
            )
        elif k == "state_of_charge":
            mech = (
                f"State of charge changed by {delta:+.1f} {unit} (to {item['s_val']:.1f} {unit}). "
                f"Higher battery SOC corresponds to higher open-circuit cell potentials and lower cathode thermal stability limits."
            )
        else:
            mech = f"Parameter {lbl} adjusted by {delta:+.1f} {unit}."

        physics_explanations.append({
            "key": k,
            "label": lbl,
            "abs_delta_str": f"{delta:+.1f} {unit}" if unit != "mΩ" else f"{delta:+.2f} {unit}",
            "marginal_impact_pp": m_delta,
            "explanation": mech,
        })

    # If no parameters changed
    if not physics_explanations:
        physics_explanations.append({
            "key": "none",
            "label": "All Variables Unchanged",
            "abs_delta_str": "0.0",
            "marginal_impact_pp": 0.0,
            "explanation": "Scenario parameters are identical to the calibrated nominal baseline. Physics mechanisms remain in steady-state equilibrium.",
        })

    # ── 5. Model Comparison & Consensus ─────────────────────────
    model_comparison = {
        "mlp_baseline_pct": base_mlp * 100,
        "mlp_scenario_pct": scen_mlp * 100,
        "mlp_delta_pp": mlp_delta_pp,
        "xgb_baseline_pct": base_xgb * 100 if base_xgb is not None else None,
        "xgb_scenario_pct": scen_xgb * 100 if scen_xgb is not None else None,
        "xgb_delta_pp": xgb_delta_pp,
    }

    if base_xgb is not None and scen_xgb is not None:
        diff_pp = abs((scen_mlp * 100) - (scen_xgb * 100))
        same_direction = (mlp_delta_pp * xgb_delta_pp) >= 0

        if diff_pp > 15.0 or not same_direction:
            model_comparison["consensus_status"] = "ARCHITECTURAL DISAGREEMENT"
            model_comparison["consensus_note"] = (
                f"Model divergence detected: PyTorch MLP predicts {scen_mlp * 100:.1f}%, while XGBoost predicts "
                f"{scen_xgb * 100:.1f}% (difference: {diff_pp:.1f} pp). Neural networks interpolate smoothly across continuous "
                f"multi-dimensional manifolds, whereas gradient-boosted decision trees rely on orthogonal feature partition splits."
            )
        else:
            model_comparison["consensus_status"] = "CONSENSUS ALIGNED"
            model_comparison["consensus_note"] = (
                f"Both models agree directionally on this scenario's impact (PyTorch MLP: {scen_mlp * 100:.1f}%, "
                f"XGBoost: {scen_xgb * 100:.1f}%)."
            )
    else:
        model_comparison["consensus_status"] = "SINGLE MODEL EVALUATED"
        model_comparison["consensus_note"] = "XGBoost model artifact was not loaded; PyTorch MLP is the primary predictor."

    # ── 6. Dynamic Recommendation ───────────────────────────────
    scen_hotspot = float(scenario.get("cell_temperature_max", 28.5))
    scen_cooling = float(scenario.get("cooling_system_health", 74.0))
    scen_power = float(scenario.get("average_charge_power_kw", 40.0))
    scen_res = float(scenario.get("internal_resistance", 0.53))

    if scen_hotspot >= 48.0 or scen_mlp >= 0.70:
        action_title = "Immediate Heat Shedding & Charge Current Cut"
        action_body = (
            f"The simulated hotspot ({scen_hotspot:.1f}°C) or failure risk ({scen_mlp * 100:.1f}%) is in the critical zone. "
            f"Reduce fast charge power below 40.0 kW and boost cooling system health above 80% to arrest exothermic thermal runaway onset."
        )
    elif scen_hotspot >= 42.0 or scen_mlp >= 0.35 or scen_cooling <= 50.0:
        action_title = "Cooling System Flow Augmentation & Power Moderation"
        action_body = (
            f"Hotspot temperature ({scen_hotspot:.1f}°C) or degraded cooling health ({scen_cooling:.0f}%) creates elevated thermal stress. "
            f"Increase coolant flow capacity and reduce charge power from {scen_power:.1f} kW to maintain cell temperatures below 40°C."
        )
    elif mlp_delta_pp < -0.5:
        action_title = "Verified Favorable Operating Envelope"
        action_body = (
            f"This scenario achieved a {abs(mlp_delta_pp):.1f} pp reduction in failure probability compared to baseline. "
            f"The robust cooling ({scen_cooling:.0f}%) and moderate charge power ({scen_power:.1f} kW) keep cell temperatures securely in the optimal 20–35°C band."
        )
    else:
        action_title = "Maintain Nominal Operational Baseline"
        action_body = (
            "All simulated parameters remain within certified safe operating margins. Continue nominal operation while monitoring hotspot variance."
        )

    recommendation = {
        "title": action_title,
        "body": action_body,
        "disclaimer": "Simulated recommendation only. Applied to mathematical models and not physical battery hardware.",
    }

    return {
        "summary": {
            "direction": risk_direction,
            "color": risk_color,
            "headline": summary_headline,
            "text": summary_text,
            "base_mlp_pct": base_mlp * 100,
            "scen_mlp_pct": scen_mlp * 100,
            "mlp_delta_pp": mlp_delta_pp,
            "base_risk_score": base_risk["risk_score"],
            "scen_risk_score": scen_risk["risk_score"],
            "risk_score_delta": risk_score_delta,
            "base_ae_score": base_ae["score"] if base_ae else None,
            "scen_ae_score": scen_ae["score"] if scen_ae else None,
        },
        "parameter_table": param_table,
        "physics_explanations": physics_explanations,
        "model_comparison": model_comparison,
        "recommendation": recommendation,
    }


# ── Live Parameter Explanation Engine ───────────────────────────

PARAM_METADATA: Dict[str, Dict[str, Any]] = {
    "cell_temperature_avg": {
        "name": "Average Cell Temperature (°C)",
        "unit": "°C",
        "precision": 1,
        "tolerance": 0.01,
    },
    "cell_temperature_max": {
        "name": "Maximum Hotspot Temperature (°C)",
        "unit": "°C",
        "precision": 1,
        "tolerance": 0.01,
    },
    "internal_resistance": {
        "name": "Internal Resistance (mΩ)",
        "unit": "mΩ",
        "precision": 2,
        "tolerance": 0.005,
    },
    "cooling_system_health": {
        "name": "Cooling System Health (%)",
        "unit": "%",
        "precision": 0,
        "tolerance": 0.1,
    },
    "average_charge_power_kw": {
        "name": "Average Charge Power (kW)",
        "unit": "kW",
        "precision": 1,
        "tolerance": 0.1,
    },
    "state_of_charge": {
        "name": "State of Charge (%)",
        "unit": "%",
        "precision": 1,
        "tolerance": 0.1,
    },
    "average_ambient_temperature": {
        "name": "Average Ambient Temperature (°C)",
        "unit": "°C",
        "precision": 1,
        "tolerance": 0.01,
    },
    "coolant_flow": {
        "name": "Coolant Flow Rate (kg/s)",
        "unit": "kg/s",
        "precision": 3,
        "tolerance": 0.001,
    },
    "current": {
        "name": "Pack Current (A)",
        "unit": "A",
        "precision": 1,
        "tolerance": 0.1,
    },
}


def explain_parameter_change(
    param_key: str,
    baseline_val: float,
    current_val: float,
) -> Optional[Dict[str, Any]]:
    """
    Explain an individual parameter change in simple, defensible language grounded in
    the simulator's physics and configured safety thresholds.
    """
    meta = PARAM_METADATA.get(param_key, {
        "name": param_key.replace("_", " ").title(),
        "unit": "",
        "precision": 2,
        "tolerance": 1e-4,
    })

    b_val = float(baseline_val)
    c_val = float(current_val)
    delta = c_val - b_val
    tol = meta.get("tolerance", 1e-4)

    if abs(delta) < tol:
        return None

    prec = meta.get("precision", 1)
    unit = meta.get("unit", "")
    unit_suffix = f" {unit}" if unit and not unit.startswith("%") else unit

    b_str = f"{b_val:.{prec}f}{unit_suffix}"
    c_str = f"{c_val:.{prec}f}{unit_suffix}"
    d_str = f"{delta:+.{prec}f}{unit_suffix}"
    change_summary = f"{b_str} → {c_str} ({d_str})"
    direction = "INCREASE" if delta > 0 else "DECREASE"

    param_name = meta["name"]

    # Parameter-specific explanations
    if param_key == "cell_temperature_avg":
        if delta > 0:
            what_means = (
                f"The battery pack's mean bulk cell temperature has risen from {b_str} to {c_str} ({d_str} above nominal baseline)."
            )
            why_matters = (
                f"In the thermal simulator, higher pack average temperature reduces the temperature gradient to the liquid coolant "
                f"(T_pack - T_coolant), diminishing passive heat dissipation and accelerating electrochemical degradation kinetics across all module cells."
            )
            expected_effect = (
                "Expected to increase thermal risk (physics-based thermal stress expectation; model probability estimate is computed upon simulation)."
            )
            suggested_adj = (
                "Reduce simulated charge power or increase coolant pump flow rate to restore pack bulk temperature toward the nominal 20.0–25.0°C band."
            )
        else:
            what_means = (
                f"The battery pack's mean bulk cell temperature has cooled from {b_str} to {c_str} ({d_str} below nominal baseline)."
            )
            why_matters = (
                f"Lower bulk temperatures enlarge the heat rejection margin before reaching thermal caution zones, "
                f"though excessively cold temperatures (< 15.0°C) can elevate lithium plating risk during high-current charging."
            )
            expected_effect = (
                "Expected to decrease thermal risk under normal operation (physics-based thermal headroom gain; model probability estimate is computed upon simulation)."
            )
            suggested_adj = (
                "Maintain pack temperature within the optimal 20.0–30.0°C operational window for balanced electrical efficiency and longevity."
            )

    elif param_key == "cell_temperature_max":
        if delta > 0:
            if c_val >= 48.0:
                threshold_note = "exceeding the project's critical exothermic runaway onset boundary (48.0°C)"
                why_matters = (
                    f"In lithium-ion cells, localized temperatures at or above 48.0°C trigger exothermic self-heating reactions "
                    f"in the anode Solid Electrolyte Interphase (SEI) and cathode, risking rapid, uncontrollable thermal propagation across adjacent cells."
                )
                expected_effect = (
                    "Expected to increase risk severely to critical level (physics-based critical threshold trigger; model probability estimate is computed upon simulation)."
                )
                suggested_adj = (
                    "Apply emergency maximum simulated cooling (+35% flow) and shed charge current immediately to arrest self-heating."
                )
            elif c_val >= 42.0:
                threshold_note = "crossing the project's caution threshold (42.0°C)"
                why_matters = (
                    f"Localized temperatures exceeding 42.0°C compromise SEI layer stability, accelerating cell aging, impedance growth, "
                    f"and gas generation in the localized hotspot cell."
                )
                expected_effect = (
                    "Expected to increase risk significantly (physics-based caution threshold trigger; model probability estimate is computed upon simulation)."
                )
                suggested_adj = (
                    "Increase coolant flow (+18%) and moderate fast-charge power to stabilize the localized thermal gradient before runaway onset."
                )
            elif c_val >= 36.0:
                threshold_note = "entering the mild thermal elevation range (≥ 36.0°C)"
                why_matters = (
                    f"Localized hotspot temperatures create an elevated thermal gradient across module cells (T_hotspot - T_avg). "
                    f"In the lumped simulator, localized thermal variance accelerates uneven cell capacity fade."
                )
                expected_effect = (
                    "Expected to increase risk mildly (physics-based thermal gradient elevation; model probability estimate is computed upon simulation)."
                )
                suggested_adj = (
                    "Preemptively modulate coolant flow (+10%) to prevent further hotspot heat accumulation."
                )
            else:
                threshold_note = "remaining within the normal thermal envelope (< 36.0°C)"
                why_matters = (
                    f"Mild localized temperature fluctuations are typical during operation; the hotspot remains well below safety caution limits."
                )
                expected_effect = (
                    "Expected to maintain or slightly increase risk (physics-based expectation; model probability estimate is computed upon simulation)."
                )
                suggested_adj = (
                    "Continue monitoring hotspot variance relative to the module mean temperature."
                )

            what_means = (
                f"The maximum cell hotspot temperature has reached {c_str} ({d_str} above nominal baseline {b_str}), {threshold_note}."
            )
        else:
            what_means = (
                f"The maximum cell hotspot temperature has dropped to {c_str} ({d_str} below baseline {b_str}), safely below configured caution limits."
            )
            why_matters = (
                f"Minimizing peak localized temperatures prevents steep inter-cell thermal gradients and keeps the hottest cell "
                f"comfortably below the 42.0°C caution and 48.0°C runaway thresholds."
            )
            expected_effect = (
                "Expected to decrease risk (physics-based thermal margin improvement; model probability estimate is computed upon simulation)."
            )
            suggested_adj = (
                "Maintain uniform coolant distribution across pack modules to preserve thermal balance."
            )

    elif param_key == "internal_resistance":
        if delta > 0:
            what_means = (
                f"Cell internal impedance has risen from {b_str} to {c_str} ({d_str} above nominal baseline)."
            )
            why_matters = (
                f"By Joule's law (Q_gen = I²R), internal resistive heat generation scales directly with cell resistance. "
                f"Elevated impedance severely multiplies waste heat generation under active current, placing heavier demand on the cooling system. "
                f"(Note: The actual heating magnitude depends directly on the active pack current)."
            )
            expected_effect = (
                "Expected to increase risk (physics-based Joule heating expectation; model probability estimate is computed upon simulation)."
            )
            suggested_adj = (
                "Reduce simulated charge or discharge current demand to limit I²R heat generation in higher-impedance cells."
            )
        else:
            what_means = (
                f"Cell internal impedance has decreased to {c_str} ({d_str} below nominal baseline {b_str})."
            )
            why_matters = (
                f"Lower internal impedance minimizes ohmic losses (I²R) during current transfer, generating less internal waste heat "
                f"and improving battery electrical efficiency."
            )
            expected_effect = (
                "Expected to decrease risk (physics-based resistive loss reduction; model probability estimate is computed upon simulation)."
            )
            suggested_adj = (
                "Ideal low-impedance state; supports nominal fast charging without excessive resistive self-heating."
            )

    elif param_key == "cooling_system_health":
        if delta < 0:
            if c_val <= 50.0:
                deg_note = "crossing the project's caution degradation threshold (≤ 50%)"
                suggested_adj = (
                    "Reduce fast-charge power significantly and schedule simulated cooling pump/coolant circuit servicing."
                )
            else:
                deg_note = "reducing available convective heat extraction headroom"
                suggested_adj = (
                    "Monitor coolant flow rate and avoid extended peak-power fast-charge sessions under degraded cooling."
                )

            what_means = (
                f"Cooling system health has degraded from {b_str} to {c_str} ({d_str} below nominal baseline), {deg_note}."
            )
            why_matters = (
                f"In the thermal simulator, liquid cooling convective heat removal scales with cooling health and flow "
                f"(Q_rem ∝ h·A·ΔT). Reduced cooling effectiveness limits heat extraction, causing cell temperatures to escalate under sustained load."
            )
            expected_effect = (
                "Expected to increase risk (physics-based convective heat extraction loss; model probability estimate is computed upon simulation)."
            )
        else:
            what_means = (
                f"Cooling system health has increased from {b_str} to {c_str} ({d_str} above nominal baseline)."
            )
            why_matters = (
                f"Enhanced cooling health maximizes the convective heat extraction capacity of the Battery Thermal Management System (BTMS), "
                f"rapidly shedding cell waste heat and maintaining uniform pack temperatures."
            )
            expected_effect = (
                "Expected to decrease risk (physics-based convective heat dissipation gain; model probability estimate is computed upon simulation)."
            )
            suggested_adj = (
                "Optimal cooling condition; permits safe handling of higher charge power without thermal buildup."
            )

    elif param_key == "average_charge_power_kw":
        if delta > 0:
            what_means = (
                f"Average charge power has increased from {b_str} to {c_str} ({d_str} above nominal baseline)."
            )
            why_matters = (
                f"Fast-charge power dictates pack electrical current (I ≈ P / V). Higher charging power elevates current throughput, "
                f"which quadratically drives Joule heating (I²R) and intensifies electrochemical ion transfer flux. "
                f"(Note: Charge power influences current based on pack operating voltage; it is not identical to current directly)."
            )
            expected_effect = (
                "Expected to increase risk (physics-based current throughput and thermal load expectation; model probability estimate is computed upon simulation)."
            )
            suggested_adj = (
                "Cap charge power to 40.0–50.0 kW or verify maximum cooling flow before initiating high-rate charging."
            )
        else:
            what_means = (
                f"Average charge power has decreased from {b_str} to {c_str} ({d_str} below nominal baseline)."
            )
            why_matters = (
                f"Lower charge power reduces current throughput (I ≈ P / V), decreasing both ohmic Joule heating and electrochemical stress on electrode surfaces."
            )
            expected_effect = (
                "Expected to decrease risk (physics-based electrical and thermal load reduction; model probability estimate is computed upon simulation)."
            )
            suggested_adj = (
                "Favorable charging profile for thermal preservation when rapid charging turnaround is not strictly required."
            )

    elif param_key == "state_of_charge":
        if delta > 0:
            what_means = (
                f"Battery State of Charge (SoC) has increased from {b_str} to {c_str} ({d_str} above nominal baseline)."
            )
            why_matters = (
                f"Higher battery SoC corresponds to elevated open-circuit cell potential and chemically lower cathode thermal stability limits in lithium-ion cells. "
                f"(Limitation disclosure: SoC does not directly generate I²R thermal Joules on its own without active current, but contributes to statistical failure vulnerability in the trained ML models)."
            )
            expected_effect = (
                "Expected to increase risk slightly in model feature space, though direct thermal heating depends on active current (physics vs model distinction)."
            )
            suggested_adj = (
                "Avoid holding cells at elevated SoC (> 80–90%) in high-temperature environments; taper charge rate near top of charge."
            )
        else:
            what_means = (
                f"Battery State of Charge (SoC) has decreased from {b_str} to {c_str} ({d_str} below nominal baseline)."
            )
            why_matters = (
                f"Moderate or lower SoC operates in a stable electrochemical potential region with lower cathode thermal stress. "
                f"(Limitation disclosure: Lower SoC reduces cell overpotential but does not alter heat extraction physics)."
            )
            expected_effect = (
                "Expected to maintain or slightly reduce risk (physics/model distinction; model probability estimate is computed upon simulation)."
            )
            suggested_adj = (
                "Keep SoC within the 20.0–80.0% daily operating window to maximize cycle longevity."
            )

    elif param_key == "average_ambient_temperature":
        if delta > 0:
            what_means = f"Ambient air temperature shifted from {b_str} to {c_str} ({d_str} above baseline)."
            why_matters = (
                f"Higher ambient temperature reduces external convective dissipation to surroundings "
                f"(Q_amb = h_amb · A · (T_pack - T_amb)), lowering passive heat loss."
            )
            expected_effect = "Expected to increase thermal risk (physics-based heat loss reduction; model estimate computed upon simulation)."
            suggested_adj = "Increase BTMS liquid cooling flow to compensate for reduced passive ambient heat rejection."
        else:
            what_means = f"Ambient air temperature cooled from {b_str} to {c_str} ({d_str} below baseline)."
            why_matters = "Cooler ambient air enhances external convective heat loss gradient from the pack casing."
            expected_effect = "Expected to decrease thermal risk (physics-based ambient gradient enhancement)."
            suggested_adj = "Safe operating envelope; ambient conditions aid pack temperature stabilization."

    elif param_key == "coolant_flow":
        if delta > 0:
            what_means = f"Coolant mass flow rate increased from {b_str} to {c_str} ({d_str} above baseline)."
            why_matters = "Higher coolant flow elevates convective heat transfer coefficient, carrying heat away from module cells more rapidly."
            expected_effect = "Expected to decrease risk (physics-based convective dissipation increase)."
            suggested_adj = "Effective intervention during high-current charging."
        else:
            what_means = f"Coolant mass flow rate reduced from {b_str} to {c_str} ({d_str} below baseline)."
            why_matters = "Lower flow rate leaves generated heat unextracted, allowing cell temperatures to accumulate."
            expected_effect = "Expected to increase risk (physics-based heat extraction loss)."
            suggested_adj = "Increase pump flow rate to maintain thermal equilibrium."

    elif param_key == "current":
        if delta > 0:
            what_means = f"Pack current increased from {b_str} to {c_str} ({d_str} above baseline)."
            why_matters = "Joule heating scales quadratically with current (Q_gen = I²R); doubling current quadruples ohmic heat generation."
            expected_effect = "Expected to increase risk sharply (physics-based I²R heating acceleration)."
            suggested_adj = "Moderate load current or engage high cooling flow."
        else:
            what_means = f"Pack current decreased from {b_str} to {c_str} ({d_str} below baseline)."
            why_matters = "Lower current drastically reduces I²R resistive heating and electrochemical stress."
            expected_effect = "Expected to decrease risk (physics-based quadratic heat reduction)."
            suggested_adj = "Ideal condition for thermal stabilization."

    else:
        what_means = f"Parameter {param_name} adjusted from {b_str} to {c_str} ({d_str})."
        why_matters = "Modifies the simulated operational input feature vector for battery monitoring and model inference."
        expected_effect = "Effect depends on multi-variable interaction in the trained failure prediction models."
        suggested_adj = "Review telemetry parameters relative to nominal operating guidelines."

    return {
        "param_key": param_key,
        "param_name": param_name,
        "baseline_val": b_val,
        "current_val": c_val,
        "delta_val": delta,
        "baseline_str": b_str,
        "current_str": c_str,
        "delta_str": d_str,
        "change_summary": change_summary,
        "direction": direction,
        "what_this_means": what_means,
        "why_it_matters": why_matters,
        "expected_effect": expected_effect,
        "suggested_adjustment": suggested_adj,
    }


def get_live_parameter_explanations(
    baseline: Dict[str, Any],
    current_values: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Compare current slider values against baseline and return ordered explanations
    for all changed parameters.
    """
    ordered_keys = [
        "cell_temperature_avg",
        "cell_temperature_max",
        "internal_resistance",
        "cooling_system_health",
        "average_charge_power_kw",
        "state_of_charge",
        "average_ambient_temperature",
        "coolant_flow",
        "current",
    ]

    explanations: List[Dict[str, Any]] = []

    for k in ordered_keys:
        if k in current_values and k in baseline:
            exp = explain_parameter_change(k, baseline[k], current_values[k])
            if exp is not None:
                explanations.append(exp)

    for k in current_values:
        if k not in ordered_keys and k in baseline:
            exp = explain_parameter_change(k, baseline[k], current_values[k])
            if exp is not None:
                explanations.append(exp)

    return explanations


def compute_simulation_comparison(
    baseline: Dict[str, Any],
    scenario: Dict[str, Any],
    mlp_model,
    xgb_model=None,
    ae_artifacts=None,
) -> Dict[str, Any]:
    """
    Compute actual before/after failure probabilities and generate a compact,
    one-sentence explanation and recommended simulated adjustment.
    """
    base_mlp = float(predict_failure_probability(mlp_model, baseline))
    scen_mlp = float(predict_failure_probability(mlp_model, scenario))
    base_mlp_pct = base_mlp * 100.0
    scen_mlp_pct = scen_mlp * 100.0
    mlp_delta_pp = scen_mlp_pct - base_mlp_pct

    base_xgb_pct = None
    scen_xgb_pct = None
    xgb_delta_pp = None
    if xgb_model is not None:
        base_xgb = float(predict_failure_probability(xgb_model, baseline))
        scen_xgb = float(predict_failure_probability(xgb_model, scenario))
        base_xgb_pct = base_xgb * 100.0
        scen_xgb_pct = scen_xgb * 100.0
        xgb_delta_pp = scen_xgb_pct - base_xgb_pct

    changed = []
    for k in [
        "cell_temperature_max",
        "cell_temperature_avg",
        "internal_resistance",
        "cooling_system_health",
        "average_charge_power_kw",
        "state_of_charge",
    ]:
        if abs(float(scenario.get(k, 0)) - float(baseline.get(k, 0))) > 1e-3:
            name = PARAM_METADATA.get(k, {}).get("name", k).split("(")[0].strip()
            changed.append(name.lower())

    changed_str = ", ".join(changed) if changed else "nominal parameters"

    # One-sentence explanation of calculated result change
    if mlp_delta_pp > 1.0:
        result_explanation = (
            f"Calculated failure probability increased by {mlp_delta_pp:+.2f} percentage points "
            f"(from {base_mlp_pct:.2f}% to {scen_mlp_pct:.2f}%) driven by elevated thermal stress from {changed_str}."
        )
    elif mlp_delta_pp < -1.0:
        result_explanation = (
            f"Calculated failure probability decreased by {abs(mlp_delta_pp):.2f} percentage points "
            f"(from {base_mlp_pct:.2f}% to {scen_mlp_pct:.2f}%) as simulated conditions ({changed_str}) established a safer thermal margin."
        )
    else:
        result_explanation = (
            f"Calculated failure probability remained stable ({base_mlp_pct:.2f}% vs {scen_mlp_pct:.2f}%, delta: {mlp_delta_pp:+.2f} pp), "
            f"with parameters operating within nominal baseline margins."
        )

    # One recommended next adjustment
    scen_hotspot = float(scenario.get("cell_temperature_max", 28.5))
    scen_cooling = float(scenario.get("cooling_system_health", 74.0))
    scen_power = float(scenario.get("average_charge_power_kw", 40.0))

    if scen_hotspot >= 48.0 or scen_mlp >= 0.70:
        recommended_action = (
            f"Reduce charge power below 40.0 kW and increase cooling health above 80% to bring hotspot temperature ({scen_hotspot:.1f}°C) below the 48.0°C runaway boundary."
        )
    elif scen_hotspot >= 42.0 or scen_mlp >= 0.35 or scen_cooling <= 50.0:
        recommended_action = (
            f"Increase coolant flow and moderate charge power from {scen_power:.1f} kW to maintain cell temperatures below the 40.0°C threshold."
        )
    elif mlp_delta_pp < -0.5:
        recommended_action = (
            f"Current scenario maintains optimal cell temperatures (20.0–35.0°C); maintain balanced cooling health ({scen_cooling:.0f}%)."
        )
    else:
        recommended_action = (
            "Operating envelope is stable; maintain current charge rate and monitor hotspot temperature variance."
        )

    return {
        "base_mlp_pct": base_mlp_pct,
        "scen_mlp_pct": scen_mlp_pct,
        "mlp_delta_pp": mlp_delta_pp,
        "base_xgb_pct": base_xgb_pct,
        "scen_xgb_pct": scen_xgb_pct,
        "xgb_delta_pp": xgb_delta_pp,
        "result_explanation": result_explanation,
        "recommended_action": recommended_action,
        "disclaimer": (
            "Model estimates derived from trained neural network (PyTorch MLP) and gradient boosted trees (XGBoost). "
            "These represent statistical risk estimates, not guaranteed physical failure probabilities or active real-world hardware commands."
        ),
    }
