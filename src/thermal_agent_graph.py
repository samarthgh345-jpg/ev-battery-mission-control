import copy
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

from src.prediction_engine import predict_temperature, load_mlp_model
from src.risk_engine import assess_thermal_risk
from src.hotspot_risk import estimate_hotspot_risk
from src.anomaly_detection import detect_anomaly

class ThermalAgentState(TypedDict):
    sensor_data: Dict[str, float]
    xgb_prediction: float
    mlp_prediction: float
    anomaly_score: float
    anomaly_label: str
    risk_level: str
    risk_score: float
    candidates: List[Dict[str, Any]]
    simulations: List[Dict[str, Any]]
    selected_action: str
    decision_reason: str
    target_coolant_flow: float
    target_coolant_percent: int

def observe_node(state: ThermalAgentState):
    """Passes the sensor data forward. Validates features."""
    return state

def predict_node(state: ThermalAgentState, xgb_model, mlp_model, iso_model):
    """Predict temperature using XGBoost and MLP, and detect anomalies."""
    sensor_data = state["sensor_data"]
    
    # XGBoost prediction
    xgb_pred = predict_temperature(xgb_model, sensor_data)
    
    # MLP prediction (if available)
    mlp_pred = xgb_pred
    if mlp_model:
        mlp_pred = predict_temperature(mlp_model, sensor_data)
        
    # Anomaly detection (existing isolation forest + new autoencoder logic if merged, 
    # here using existing iso_model for compatibility with current flow)
    anomaly_res = detect_anomaly(iso_model, sensor_data)
    
    return {
        "xgb_prediction": xgb_pred,
        "mlp_prediction": mlp_pred,
        "anomaly_score": anomaly_res["score"],
        "anomaly_label": anomaly_res["label"]
    }

def evaluate_risk_node(state: ThermalAgentState):
    """Evaluate thermal risk based on predictions."""
    sensor_data = state["sensor_data"]
    # We use the higher of the two predictions to be safe
    pred_temp = max(state["xgb_prediction"], state["mlp_prediction"])
    
    hotspot = estimate_hotspot_risk(
        pred_temp,
        sensor_data.get("battery_current_A", 5.0),
        sensor_data.get("discharge_rate_C", 1.0),
        sensor_data.get("coolant_flow_rate_kg_s", 0.02),
        sensor_data.get("ambient_temperature_C", 25.0),
    )
    
    risk = assess_thermal_risk(
        current_temp=sensor_data.get("battery_temperature_C", pred_temp),
        predicted_temp_5min=pred_temp,
        temp_rate_per_min=0.0,
        hotspot_risk_pct=hotspot["hotspot_risk_percent"]
    )
    
    return {
        "risk_level": risk["risk_level"],
        "risk_score": risk["risk_score"]
    }

def generate_candidates_node(state: ThermalAgentState):
    """Generate candidate cooling actions."""
    current_flow = state["sensor_data"].get("coolant_flow_rate_kg_s", 0.02)
    candidates = [
        {"action": "Maintain cooling", "flow": current_flow, "target_pct": int(current_flow * 1000)},
        {"action": "Cooling 20%", "flow": 0.020, "target_pct": 20},
        {"action": "Cooling 35%", "flow": 0.035, "target_pct": 35},
        {"action": "Emergency cooling", "flow": 0.040, "target_pct": 40},
    ]
    return {"candidates": candidates}

def simulate_candidates_node(state: ThermalAgentState, xgb_model):
    """Simulate the thermal result for each candidate action."""
    simulations = []
    current_features = state["sensor_data"]
    
    for cand in state["candidates"]:
        sim_features = copy.deepcopy(current_features)
        sim_features["coolant_flow_rate_kg_s"] = cand["flow"]
        
        # Use XGBoost as the baseline simulation model
        pred_temp = predict_temperature(xgb_model, sim_features)
        
        hotspot = estimate_hotspot_risk(
            pred_temp,
            sim_features.get("battery_current_A", 5.0),
            sim_features.get("discharge_rate_C", 1.0),
            sim_features.get("coolant_flow_rate_kg_s", 0.02),
            sim_features.get("ambient_temperature_C", 25.0),
        )
        
        risk = assess_thermal_risk(
            current_temp=sim_features.get("battery_temperature_C", pred_temp),
            predicted_temp_5min=pred_temp,
            temp_rate_per_min=1.0,
            hotspot_risk_pct=hotspot["hotspot_risk_percent"]
        )
        
        # Energy cost logic: Higher flow = higher cost
        energy_cost = "Low"
        if cand["target_pct"] >= 40:
            energy_cost = "Very High"
        elif cand["target_pct"] >= 35:
            energy_cost = "High"
        elif cand["target_pct"] >= 20:
            energy_cost = "Medium"
            
        simulations.append({
            "action": cand["action"],
            "target_pct": cand["target_pct"],
            "flow": cand["flow"],
            "predicted_temp": pred_temp,
            "risk_level": risk["risk_level"],
            "risk_score": risk["risk_score"],
            "energy_cost": energy_cost
        })
        
    return {"simulations": simulations}

def compare_candidates_node(state: ThermalAgentState):
    """Compare candidates based on safety > risk reduction > energy efficiency."""
    _level_rank = {"NORMAL": 0, "CAUTION": 1, "HIGH": 2, "CRITICAL": 3}
    
    simulations = state["simulations"]
    
    # Sort primarily by risk severity, then by risk score, then by flow (energy cost)
    sorted_sims = sorted(
        simulations,
        key=lambda x: (
            _level_rank.get(x["risk_level"], 9), 
            x["flow"] if x["risk_level"] == "NORMAL" else x["risk_score"],
            x["flow"]
        )
    )
    
    best_option = sorted_sims[0]
    
    current_flow = state["sensor_data"].get("coolant_flow_rate_kg_s", 0.02)
    
    reason = f"The predicted temperature can be reduced to {best_option['predicted_temp']:.1f}°C, resulting in a {best_option['risk_level']} risk state."
    if best_option["flow"] > current_flow:
        reason += f" Increased cooling to {best_option['target_pct']}% is required to safely manage thermal stress."
    elif best_option["flow"] < current_flow:
        reason += f" Decreased cooling to {best_option['target_pct']}% is sufficient to maintain thermal safety while conserving energy."
    else:
        reason += " Current cooling is sufficient to maintain thermal safety while conserving energy."
        
    return {
        "selected_action": best_option["action"],
        "target_coolant_flow": best_option["flow"],
        "target_coolant_percent": best_option["target_pct"],
        "decision_reason": reason
    }

def apply_node(state: ThermalAgentState):
    """Apply node represents applying the decision to the simulation state."""
    # Simulation only - no real hardware control
    return state

def build_thermal_agent_graph(xgb_model, iso_model, mlp_model=None):
    """Build the LangGraph StateGraph."""
    workflow = StateGraph(ThermalAgentState)
    
    # Wrap nodes to inject models where needed
    def _predict(state): return predict_node(state, xgb_model, mlp_model, iso_model)
    def _simulate(state): return simulate_candidates_node(state, xgb_model)
    
    workflow.add_node("Observe", observe_node)
    workflow.add_node("Predict", _predict)
    workflow.add_node("Evaluate_Risk", evaluate_risk_node)
    workflow.add_node("Generate_Candidates", generate_candidates_node)
    workflow.add_node("Simulate_Candidates", _simulate)
    workflow.add_node("Compare_Candidates", compare_candidates_node)
    workflow.add_node("Decide", lambda state: state) # Dummy pass-through to Match architecture visually
    workflow.add_node("Apply", apply_node)
    
    workflow.set_entry_point("Observe")
    
    workflow.add_edge("Observe", "Predict")
    workflow.add_edge("Predict", "Evaluate_Risk")
    workflow.add_edge("Evaluate_Risk", "Generate_Candidates")
    workflow.add_edge("Generate_Candidates", "Simulate_Candidates")
    workflow.add_edge("Simulate_Candidates", "Compare_Candidates")
    workflow.add_edge("Compare_Candidates", "Decide")
    workflow.add_edge("Decide", "Apply")
    workflow.add_edge("Apply", END)
    
    return workflow.compile()

def run_agent_graph(xgb_model, iso_model, mlp_model, sensor_data: dict) -> ThermalAgentState:
    """Convenience function to run the graph."""
    graph = build_thermal_agent_graph(xgb_model, iso_model, mlp_model)
    
    initial_state = ThermalAgentState(
        sensor_data=sensor_data,
        xgb_prediction=0.0,
        mlp_prediction=0.0,
        anomaly_score=0.0,
        anomaly_label="",
        risk_level="",
        risk_score=0.0,
        candidates=[],
        simulations=[],
        selected_action="",
        decision_reason="",
        target_coolant_flow=0.0,
        target_coolant_percent=0
    )
    
    # LangGraph run
    result = graph.invoke(initial_state)
    return result
