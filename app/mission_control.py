"""
Overview — Main Monitoring Panel
==================================
Battery thermal status, forecasts, risk assessment, and control recommendation.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.prediction_engine import predict_temperature, get_default_features
from src.forecasting import forecast_temperatures
from src.anomaly_detection import detect_anomaly
from src.risk_engine import assess_thermal_risk
from src.hotspot_risk import estimate_hotspot_risk, generate_cell_temperature_grid
from src.thermal_agent import create_agent_decision
from app.ui_components import page_header, section_header, metric_card, status_badge, info_panel, CHART_LAYOUT, render_html

def _risk_color(level: str) -> str:
    return {"NORMAL": "#16A34A", "CAUTION": "#D97706", "HIGH": "#D97706", "CRITICAL": "#DC2626"}.get(level, "#6B7280")

def render(xgb_model, iso_model, shap_explainer, metadata, dataset):
    page_header("Mission Control", "System Overview & Top-Level Safety Analysis")

    # ── Current State ──────────────────────────────
    if "mc_features" not in st.session_state:
        st.session_state.mc_features = get_default_features()
        st.session_state.mc_features["battery_current_A"] = 8.5
        st.session_state.mc_features["ambient_temperature_C"] = 32.0
        st.session_state.mc_features["battery_temperature_C"] = 38.0
        st.session_state.mc_features["coolant_flow_rate_kg_s"] = 0.015
        st.session_state.mc_features["discharge_rate_C"] = 2.5
        st.session_state.mc_features["coolant_inlet_temperature_C"] = 26.0

    features = st.session_state.mc_features

    # ── Compute ────────────────────────────────────
    predicted_temp = predict_temperature(xgb_model, features)
    forecasts = forecast_temperatures(xgb_model, features)

    anomaly_features = {**features, "max_battery_temperature_C": predicted_temp}
    anomaly_result = (
        detect_anomaly(iso_model, anomaly_features)
        if iso_model else {"label": "NORMAL", "score": 0, "message": ""}
    )

    hotspot = estimate_hotspot_risk(
        predicted_temp, features["battery_current_A"],
        features["discharge_rate_C"], features["coolant_flow_rate_kg_s"],
        features["ambient_temperature_C"],
    )

    pred_5min = forecasts[-1]["predicted_C"] if forecasts else predicted_temp
    risk = assess_thermal_risk(
        predicted_temp, pred_5min, temp_rate_per_min=0.5,
        anomaly_score=anomaly_result["score"],
        hotspot_risk_pct=hotspot["hotspot_risk_percent"],
    )

    rc = _risk_color(risk["risk_level"])

    if "mc_agent_state" in st.session_state:
        ag = st.session_state.mc_agent_state
        agent_decision = {
            'decision': ag['selected_action'],
            'expected_result': ag['decision_reason'],
            'target_coolant_percent': ag['target_coolant_percent'],
            'options_evaluated': ag['simulations']
        }
    else:
        agent_decision = create_agent_decision(
            xgb_model=xgb_model,
            current_features=features,
            forecasts=forecasts,
            temp_trend=0.5,
            current_risk_level=risk["risk_level"],
            hotspot_risk_pct=hotspot["hotspot_risk_percent"],
            anomaly_label=anomaly_result["label"],
        )

    # ── Top Metrics Row ────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("Battery Temperature", f"{predicted_temp:.1f}", " °C")
    with m2:
        metric_card("Predicted Temp (5m)", f"{pred_5min:.1f}", " °C")
    with m3:
        metric_card("Thermal Risk Score", f"{risk['risk_score']:.0f}", " / 100")
    with m4:
        metric_card("Hotspot Risk", f"{hotspot['hotspot_risk_percent']:.0f}", " %")

    st.markdown('<div style="margin-top:20px; margin-bottom:20px; border-bottom:1px solid var(--border);"></div>', unsafe_allow_html=True)

    # ── System Status Badges ────────────────────────
    st.markdown("<div style='display:flex; flex-direction:row; flex-wrap:wrap;'>", unsafe_allow_html=True)
    status_badge("XGBoost Predictive Engine", "NORMAL")
    status_badge("MLP Predictive Engine", "NORMAL")
    status_badge("Isolation Forest Anomaly", anomaly_result["label"])
    status_badge("Risk Evaluator", risk["risk_level"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div style="margin-top:20px; margin-bottom:20px; border-bottom:1px solid var(--border);"></div>', unsafe_allow_html=True)

    # ── Main Two-Column Layout ─────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        section_header("Thermal Forecast Horizon")
        horizon_labels = ["Now"] + [f"+{fc['horizon_min']} min" for fc in forecasts]
        horizon_temps  = [predicted_temp] + [fc["predicted_C"] for fc in forecasts]

        bar_colors = []
        for t in horizon_temps:
            if t < 40: bar_colors.append("#16A34A")
            elif t < 45: bar_colors.append("#D97706")
            else: bar_colors.append("#DC2626")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=horizon_labels,
            y=horizon_temps,
            marker_color=bar_colors,
            text=[f"{t:.1f}°C" for t in horizon_temps],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=12, color="#111827"),
            cliponaxis=False,
        ))
        fig.add_hline(y=40, line_dash="dot", line_color="#D97706", line_width=1,
                      annotation_text="Caution 40°C", annotation_font_color="#D97706",
                      annotation_font_size=10)
        fig.add_hline(y=50, line_dash="dot", line_color="#DC2626", line_width=1,
                      annotation_text="High 50°C", annotation_font_color="#DC2626",
                      annotation_font_size=10)
        
        layout_opts = CHART_LAYOUT.copy()
        layout_opts.update(
            height=280,
            xaxis=dict(showgrid=False),
            yaxis=dict(title="Temperature (°C)", range=[25, max(horizon_temps) + 6]),
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
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--border-light);">
                <div>
                    <div style="font-size: 11px; color: var(--text-3);">TARGET COOLANT</div>
                    <div style="font-family: var(--mono); font-weight: 600; font-size: 14px;">{agent_decision['target_coolant_percent']}%</div>
                </div>
                <div>
                    <div style="font-size: 11px; color: var(--text-3);">EXPECTED TEMP</div>
                    <div style="font-family: var(--mono); font-weight: 600; font-size: 14px;">{next((o['predicted_temp'] for o in agent_decision['options_evaluated'] if o['target_pct'] == agent_decision['target_coolant_percent']), agent_decision['options_evaluated'][0]['predicted_temp']):.1f} °C</div>
                </div>
            </div>
        </div>
        """
        render_html(rec_html)
        
        st.markdown("<br/>", unsafe_allow_html=True)
        info_panel("The above action is calculated deterministically via numerical simulation logic, entirely isolated from generative LLMs.")

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
