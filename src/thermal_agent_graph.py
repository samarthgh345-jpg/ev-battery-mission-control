import copy
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

from src.prediction_engine import predict_failure_probability
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

def observe_node(state: ThermalAgentState):
    """Passes the sensor data forward. Validates features."""
    return state

def predict_node(state: ThermalAgentState, xgb_model, mlp_model, ae_model_artifacts):
    """Predict failure probability using XGBoost and MLP, and detect anomalies."""
    sensor_data = state["sensor_data"]
    
    # XGBoost prediction
    xgb_pred = predict_failure_probability(xgb_model, sensor_data)
    
    # MLP prediction
    mlp_pred = xgb_pred
    if mlp_model:
        mlp_pred = predict_failure_probability(mlp_model, sensor_data)
        
    # Anomaly detection (Autoencoder)
    ae_model, ae_scaler, ae_imputer, ae_threshold = ae_model_artifacts
    anomaly_res = detect_anomaly(ae_model, ae_scaler, ae_imputer, ae_threshold, sensor_data)
    
    return {
        "xgb_prediction": xgb_pred,
        "mlp_prediction": mlp_pred,
        "anomaly_score": anomaly_res["score"],
        "anomaly_label": anomaly_res["label"]
    }

def evaluate_risk_node(state: ThermalAgentState):
    """Evaluate overall risk based on failure probability and anomaly."""
    # We use the MLP prediction as the primary failure risk
    pred_prob = state["mlp_prediction"]
    
    if pred_prob > 0.8:
        risk_level = "CRITICAL"
        risk_score = pred_prob * 100
    elif pred_prob > 0.4 or state["anomaly_label"] == "ANOMALY":
        risk_level = "HIGH"
        risk_score = max(pred_prob * 100, 60.0)
    elif pred_prob > 0.15:
        risk_level = "CAUTION"
        risk_score = pred_prob * 100
    else:
        risk_level = "NORMAL"
        risk_score = pred_prob * 100
        
    return {
        "risk_level": risk_level,
        "risk_score": risk_score
    }

def generate_candidates_node(state: ThermalAgentState):
    """Generate candidate simulated actions."""
    candidates = [
        {"action": "Maintain current operations", "mods": {}},
        {"action": "Increase virtual cooling", "mods": {"cooling_system_health": 100.0}},
        {"action": "Reduce simulated charging power", "mods": {"average_charge_power_kw": 10.0, "fast_charge_ratio": 0.0}},
        {"action": "Allow simulated cooldown", "mods": {"state_of_charge": max(0, state["sensor_data"].get("state_of_charge", 0) - 5), "average_charge_power_kw": 0.0}},
    ]
    return {"candidates": candidates}

def simulate_candidates_node(state: ThermalAgentState, mlp_model):
    """Simulate the failure probability result for each candidate action."""
    simulations = []
    current_features = state["sensor_data"]
    
    for cand in state["candidates"]:
        sim_features = copy.deepcopy(current_features)
        
        # Apply modifications
        for k, v in cand["mods"].items():
            sim_features[k] = v
            
        # Use MLP as the baseline simulation model
        pred_prob = predict_failure_probability(mlp_model, sim_features)
        
        if pred_prob > 0.8:
            risk_level = "CRITICAL"
        elif pred_prob > 0.4:
            risk_level = "HIGH"
        elif pred_prob > 0.15:
            risk_level = "CAUTION"
        else:
            risk_level = "NORMAL"
            
        # Simple energy cost heuristic
        energy_cost = "Low"
        if "cooling_system_health" in cand["mods"]:
            energy_cost = "High"
        elif cand["action"] == "Reduce simulated charging power":
            energy_cost = "Medium (Time penalty)"

        simulations.append({
            "action": cand["action"],
            "predicted_prob": pred_prob,
            "predicted_temp": pred_prob, # Alias to fix KeyError in UI temporarily
            "risk_level": risk_level,
            "risk_score": pred_prob * 100,
            "energy_cost": energy_cost,
            "mods": cand["mods"]
        })
        
    return {"simulations": simulations}

def compare_candidates_node(state: ThermalAgentState):
    """Compare candidates based on safety > operational disruption."""
    _level_rank = {"NORMAL": 0, "CAUTION": 1, "HIGH": 2, "CRITICAL": 3}
    
    simulations = state["simulations"]
    
    # Sort primarily by risk severity, then by predicted probability
    sorted_sims = sorted(
        simulations,
        key=lambda x: (
            _level_rank.get(x["risk_level"], 9), 
            x["predicted_prob"],
            len(x["mods"]) # penalize more disruptive actions if risk is the same
        )
    )
    
    best_option = sorted_sims[0]
    
    reason = f"The simulated action '{best_option['action']}' reduces predicted failure probability to {best_option['predicted_prob']*100:.1f}%, resulting in a {best_option['risk_level']} risk state."
    if state["anomaly_label"] == "ANOMALY":
        reason += " Monitoring of the Autoencoder anomaly is also recommended."
        
    return {
        "selected_action": best_option["action"],
        "decision_reason": reason
    }

def apply_node(state: ThermalAgentState):
    """Apply node represents applying the decision to the simulation state."""
    # Simulation only - no real hardware control
    return state

def build_thermal_agent_graph(xgb_model, ae_model_artifacts, mlp_model):
    """Build the LangGraph StateGraph."""
    workflow = StateGraph(ThermalAgentState)
    
    # Wrap nodes to inject models where needed
    def _predict(state): return predict_node(state, xgb_model, mlp_model, ae_model_artifacts)
    def _simulate(state): return simulate_candidates_node(state, mlp_model)
    
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

def run_agent_graph(xgb_model, ae_model_artifacts, mlp_model, sensor_data: dict) -> ThermalAgentState:
    """Convenience function to run the graph."""
    graph = build_thermal_agent_graph(xgb_model, ae_model_artifacts, mlp_model)
    
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
        decision_reason=""
    )
    
    # LangGraph run
    result = graph.invoke(initial_state)
    return result
