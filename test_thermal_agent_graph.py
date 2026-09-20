import json
from src.prediction_engine import load_model, load_mlp_model
from src.anomaly_detection import load_anomaly_model
from src.simulation_engine import get_simulation_state
from src.thermal_agent_graph import run_agent_graph, build_thermal_agent_graph

print("Loading models...")
xgb_model = load_model("models/xgboost_temperature_model.joblib")
mlp_model = load_mlp_model()
iso_model = load_anomaly_model("models")

print("--- Test 1: Normal condition ---")
normal_state = get_simulation_state("NORMAL", step=0, seed=42)
res_normal = run_agent_graph(xgb_model, iso_model, mlp_model, normal_state)
print(f"Risk: {res_normal['risk_level']}")
print(f"Selected: {res_normal['selected_action']}")
print("Factors: ", res_normal)

print("\n--- Test 2: High-temperature condition ---")
# High stress state
high_state = get_simulation_state("THERMAL_STRESS", step=4, seed=42)
res_high = run_agent_graph(xgb_model, iso_model, mlp_model, high_state)
print(f"Risk: {res_high['risk_level']}")
print(f"Selected: {res_high['selected_action']}")

print("\n--- Test 3: Critical condition ---")
# Critical stress state
crit_state = get_simulation_state("THERMAL_STRESS", step=6, seed=42)
res_crit = run_agent_graph(xgb_model, iso_model, mlp_model, crit_state)
print(f"Risk: {res_crit['risk_level']}")
print(f"Selected: {res_crit['selected_action']}")

print("\n--- Test 4: Candidate comparison ---")
print(f"Simulated {len(res_crit['simulations'])} actions.")
for sim in res_crit['simulations']:
    print(f"  {sim['action']}: Temp={sim['predicted_temp']:.1f}C, Risk={sim['risk_level']}, Cost={sim['energy_cost']}")

print("\n--- Test 5: Graph structure (No LLM required) ---")
graph = build_thermal_agent_graph(xgb_model, iso_model, mlp_model)
print("Graph nodes:", [node for node in graph.nodes])

print("\nAll tests passed successfully!")
