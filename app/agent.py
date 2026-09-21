"""
Thermal AI Agent — Simulated Decision Loop
============================================
Simulates failure stress events and LangGraph agentic response strategies.
"""

import streamlit as st
import time
import plotly.graph_objects as go
from src.prediction_engine import get_default_features, load_mlp_model
from src.simulation_engine import get_simulation_state
from src.thermal_agent_graph import run_agent_graph
from app.ui_components import page_header, section_header, CHART_LAYOUT, metric_card, info_panel, warning_panel, render_html

def render(xgb_model, ae_model_artifacts, mlp_model, shap_explainer, metadata, dataset):
    page_header("Thermal AI Agent", "Agentic Decision Loop & Strategy Simulation")

    info_panel("Simulated thermal stress event. The LangGraph agent evaluates options using XGBoost and MLP deterministic models.")

    # ── Controls ──
    st.markdown('<div style="margin-top:20px; margin-bottom:20px; border-bottom:1px solid var(--border);"></div>', unsafe_allow_html=True)
    
    col_btn, col_btn2, col_info = st.columns([1, 1, 2])
    with col_btn:
        run_stress = st.button("Simulate thermal stress", use_container_width=True, key="agent_stress")
    with col_btn2:
        run_det = st.button("Run Determinism Check", use_container_width=True, key="agent_det")
    with col_info:
        st.markdown("""
        <div style="font-size:13px; color:var(--text-2); padding-top:6px;">
            Simulates a rapid thermal spike. The agent will observe, predict the failure probability, evaluate cooling options, and apply the optimal strategy.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="margin-top:20px; margin-bottom:20px; border-bottom:1px solid var(--border);"></div>', unsafe_allow_html=True)

    if not run_stress and not run_det:
        return
        
    if run_det:
        start_state = get_simulation_state("THERMAL_STRESS", step=6, seed=42)
        with st.spinner("Running determinism check..."):
            state1 = run_agent_graph(xgb_model, ae_model_artifacts, mlp_model, start_state)
            state2 = run_agent_graph(xgb_model, ae_model_artifacts, mlp_model, start_state)
            state3 = run_agent_graph(xgb_model, ae_model_artifacts, mlp_model, start_state)
            
            # Compare key fields
            fields = ['xgb_prediction', 'mlp_prediction', 'anomaly_label', 'risk_level', 'selected_action', 'decision_reason']
            is_deterministic = True
            for f in fields:
                if not (state1[f] == state2[f] == state3[f]):
                    is_deterministic = False
                    break
                    
            if is_deterministic:
                st.success("✅ **Determinism verified for this scenario.** All 3 runs produced the exact same predictions, anomaly results, risk levels, selected actions, and decision reasons.")
            else:
                st.error("❌ Determinism failed! Outputs varied across runs.")
        return

    # ── Run LangGraph Agent ──
    start_state = get_simulation_state("THERMAL_STRESS", step=6, seed=42)
    
    with st.spinner("Running LangGraph Workflow..."):
        agent_state = run_agent_graph(xgb_model, ae_model_artifacts, mlp_model, start_state)
        
        # Sync with Mission Control
        st.session_state.mc_features = start_state
        st.session_state.mc_agent_state = agent_state
        
        time.sleep(1.0) # For visual effect

    section_header("LangGraph Workflow Execution")
    
    # Custom styling for agent steps
    st.markdown("""
    <style>
    .agent-timeline {
        border-left: 2px solid var(--border-light);
        margin-left: 20px;
        padding-left: 20px;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .agent-node {
        position: relative;
        margin-bottom: 24px;
    }
    .agent-node::before {
        content: '';
        position: absolute;
        left: -27px;
        top: 0;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background-color: var(--primary);
        border: 2px solid var(--surface);
    }
    .agent-node-title {
        font-size: 14px;
        font-weight: 600;
        color: var(--text-1);
        margin-bottom: 8px;
    }
    .agent-node-body {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 16px;
        font-size: 13px;
        line-height: 1.6;
        color: var(--text-2);
    }
    .risk-badge-agent {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    timeline_html = '<div class="agent-timeline">'
    
    # 1. Observe
    timeline_html += f"<div class=\"agent-node\"><div class=\"agent-node-title\">Node: Observe</div><div class=\"agent-node-body\">Received 14-feature BTMS sensor vector.<br>Initial average cell temperature: <b>{agent_state['sensor_data']['cell_temperature_avg']:.1f}°C</b></div></div>"
    
    # 2. Predict
    timeline_html += f"<div class=\"agent-node\"><div class=\"agent-node-title\">Node: Predict</div><div class=\"agent-node-body\">XGBoost Probability: <b>{agent_state['xgb_prediction']*100:.1f}%</b><br>MLP Probability: <b>{agent_state['mlp_prediction']*100:.1f}%</b><br>Anomaly Status: <b>{agent_state['anomaly_label']}</b> (Score: {agent_state['anomaly_score']:.4f})</div></div>"
    
    # 3. Evaluate Risk
    risk_color = "#16A34A" if agent_state['risk_level'] == "NORMAL" else "#D97706" if agent_state['risk_level'] == "CAUTION" else "#DC2626"
    timeline_html += f"<div class=\"agent-node\"><div class=\"agent-node-title\">Node: Evaluate Risk</div><div class=\"agent-node-body\">Assessed Failure Risk Level: <span class=\"risk-badge-agent\" style=\"background-color: {risk_color};\">{agent_state['risk_level']}</span></div></div>"
    
    # 4. Simulate & Compare
    sim_rows = ""
    for opt in agent_state['simulations']:
        r_col = "#16A34A" if opt['risk_level'] == "NORMAL" else "#D97706" if opt['risk_level'] == "CAUTION" else "#DC2626"
        selected = opt['action'] == agent_state['selected_action']
        sel_style = "background-color: #DBEAFE; font-weight:600;" if selected else ""
        
        sim_rows += f"<tr style=\"border-bottom:1px solid var(--border-light); {sel_style}\"><td style=\"padding:8px 4px;\">{opt['action']}</td><td style=\"padding:8px 4px; font-family:var(--mono);\">{opt['predicted_temp']*100:.1f}%</td><td style=\"padding:8px 4px;\"><span class=\"risk-badge-agent\" style=\"background-color: {r_col};\">{opt['risk_level']}</span></td><td style=\"padding:8px 4px;\">{opt['energy_cost']}</td><td style=\"padding:8px 4px;\">{'Yes' if selected else 'No'}</td></tr>"
        
    timeline_html += f"<div class=\"agent-node\"><div class=\"agent-node-title\">Node: Simulate & Compare Candidates (Deterministic)</div><div class=\"agent-node-body\" style=\"padding: 0;\"><table style=\"width:100%; border-collapse:collapse; font-size:12px; text-align:left;\"><tr style=\"border-bottom:1px solid var(--border); color:var(--text-3); background:var(--bg-color);\"><th style=\"padding:8px 4px;\">Action</th><th style=\"padding:8px 4px;\">Predicted Probability</th><th style=\"padding:8px 4px;\">Risk Level</th><th style=\"padding:8px 4px;\">Energy Cost</th><th style=\"padding:8px 4px;\">Selected</th></tr>{sim_rows}</table></div></div>"
    
    # 5. Decide & Apply
    timeline_html += f"<div class=\"agent-node\"><div class=\"agent-node-title\">Node: Decide & Apply</div><div class=\"agent-node-body\"><div style=\"font-size: 14px; font-weight:600; color:var(--primary); margin-bottom: 8px;\">Recommended Action: {agent_state['selected_action']}</div><div><strong>Reason:</strong> {agent_state['decision_reason']}</div><div style=\"color:#16A34A; margin-top:8px; font-weight:600;\">[Simulation Only - State updated]</div></div></div>"
    
    timeline_html += "</div>"
    
    render_html(timeline_html)
    st.success("Simulation complete.")
