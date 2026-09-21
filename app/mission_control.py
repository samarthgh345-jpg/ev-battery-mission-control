"""
Overview — Main Monitoring Panel
==================================
Battery failure status, forecasts, risk assessment, and control recommendation.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.prediction_engine import predict_failure_probability, get_default_features
from src.forecasting import forecast_failure_probability
from src.anomaly_detection import detect_anomaly
from src.thermal_agent_graph import run_agent_graph
from app.ui_components import page_header, section_header, metric_card, status_badge, info_panel, CHART_LAYOUT, render_html

def _risk_color(level: str) -> str:
    return {"NORMAL": "#16A34A", "CAUTION": "#D97706", "HIGH": "#D97706", "CRITICAL": "#DC2626"}.get(level, "#6B7280")

def render(xgb_model, ae_model_artifacts, mlp_model, shap_explainer, metadata, dataset):
    page_header("Mission Control", "System Overview & Top-Level Safety Analysis")

    # ── Current State ──────────────────────────────
    if "mc_features" not in st.session_state:
        st.session_state.mc_features = get_default_features()

    features = st.session_state.mc_features

    # ── Compute ────────────────────────────────────
    predicted_prob = predict_failure_probability(mlp_model, features)
    forecasts = forecast_failure_probability(mlp_model, features)

    ae_model, ae_scaler, ae_imputer, ae_threshold = ae_model_artifacts
    anomaly_result = detect_anomaly(ae_model, ae_scaler, ae_imputer, ae_threshold, features)

    if "mc_agent_state" in st.session_state:
        ag = st.session_state.mc_agent_state
    else:
        ag = run_agent_graph(xgb_model, ae_model_artifacts, mlp_model, features)
        
    risk_level = ag["risk_level"]
    risk_score = ag["risk_score"]
    
    agent_decision = {
        'decision': ag['selected_action'],
        'expected_result': ag['decision_reason'],
        'options_evaluated': ag['simulations']
    }

    rc = _risk_color(risk_level)

    # ── Top Metrics Row ────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("Failure Probability", f"{predicted_prob * 100:.1f}", " %")
    with m2:
        pred_5min = forecasts[-1]["predicted_prob"] if forecasts else predicted_prob
        metric_card("Predicted Prob (5m)", f"{pred_5min * 100:.1f}", " %")
    with m3:
        metric_card("Overall Risk Score", f"{risk_score:.0f}", " / 100")
    with m4:
        metric_card("Anomaly Score", f"{anomaly_result['score']:.4f}", "")

    st.markdown('<div style="margin-top:20px; margin-bottom:20px; border-bottom:1px solid var(--border);"></div>', unsafe_allow_html=True)

    # ── System Status Badges ────────────────────────
    st.markdown("<div style='display:flex; flex-direction:row; flex-wrap:wrap;'>", unsafe_allow_html=True)
    status_badge("XGBoost Predictive Engine", "NORMAL")
    status_badge("MLP Predictive Engine", "NORMAL")
    status_badge("Autoencoder Anomaly", anomaly_result["label"])
    status_badge("Risk Evaluator", risk_level)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div style="margin-top:20px; margin-bottom:20px; border-bottom:1px solid var(--border);"></div>', unsafe_allow_html=True)

    # ── Main Two-Column Layout ─────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        section_header("Failure Risk Horizon")
        horizon_labels = ["Now"] + [f"+{fc['horizon_min']} min" for fc in forecasts]
        horizon_probs  = [predicted_prob * 100] + [fc["predicted_prob"] * 100 for fc in forecasts]

        bar_colors = []
        for p in horizon_probs:
            if p < 15: bar_colors.append("#16A34A")
            elif p < 40: bar_colors.append("#D97706")
            else: bar_colors.append("#DC2626")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=horizon_labels,
            y=horizon_probs,
            marker_color=bar_colors,
            text=[f"{p:.1f}%" for p in horizon_probs],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=12, color="#111827"),
            cliponaxis=False,
        ))
        fig.add_hline(y=15, line_dash="dot", line_color="#D97706", line_width=1,
                      annotation_text="Caution 15%", annotation_font_color="#D97706",
                      annotation_font_size=10)
        fig.add_hline(y=40, line_dash="dot", line_color="#DC2626", line_width=1,
                      annotation_text="High 40%", annotation_font_color="#DC2626",
                      annotation_font_size=10)
        
        layout_opts = CHART_LAYOUT.copy()
        layout_opts.update(
            height=280,
            xaxis=dict(showgrid=False),
            yaxis=dict(title="Failure Probability (%)", range=[0, 100]),
            showlegend=False,
        )
        fig.update_layout(**layout_opts)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        section_header("Current Recommendation")
        
        # Recommendation Callout Box
        rec_html = f"""
        <div style="background-color: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 16px;">
            <div style="font-size: 11px; font-weight: 600; color: var(--text-2); text-transform: uppercase; margin-bottom: 8px;">Action Recommended by LangGraph</div>
            <div style="font-size: 18px; font-weight: 700; color: {rc}; margin-bottom: 12px;">{agent_decision['decision']}</div>
            <div style="font-size: 13px; color: var(--text-2); margin-bottom: 16px;">{agent_decision['expected_result']}</div>
        </div>
        """
        render_html(rec_html)
        
        st.markdown("<br/>", unsafe_allow_html=True)
        info_panel("The above action is calculated via LangGraph utilizing real-time MLP predictions and simulation trees.")

    st.markdown('<div style="margin-top:20px; margin-bottom:20px; border-bottom:1px solid var(--border);"></div>', unsafe_allow_html=True)
    
    # ── Simulation Buttons ─────────────────────────
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if st.button("Why this prediction?", use_container_width=True, key="mc_why"):
            st.session_state.page_to_open = "XAI"
            st.rerun()
    with b2:
        if st.button("Run What-If", use_container_width=True, key="mc_whatif"):
            st.session_state.page_to_open = "What-If Simulator"
            st.rerun()
    with b3:
        if st.button("Start live simulation", use_container_width=True, key="mc_live"):
            st.session_state.page_to_open = "Thermal Analysis"
            st.rerun()
    with b4:
        st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
        if st.button("Simulate thermal stress", use_container_width=True, key="mc_stress"):
            st.session_state.page_to_open = "AI Agent Decision"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
