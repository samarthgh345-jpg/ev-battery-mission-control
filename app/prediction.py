"""
Thermal Analysis — Live Monitoring & Forecasting Page
=====================================================
Failure probability forecasting, live sensor simulation, and real-time charts.
"""

import streamlit as st
import time
import numpy as np
import plotly.graph_objects as go
from src.prediction_engine import predict_failure_probability, get_default_features
from src.forecasting import forecast_failure_probability
from src.anomaly_detection import detect_anomaly
from src.simulation_engine import get_simulation_state
from app.ui_components import page_header, CHART_LAYOUT

_CHART_LAYOUT = CHART_LAYOUT.copy()
_CHART_LAYOUT.pop("margin", None)
_CHART_LAYOUT.pop("xaxis", None)
_CHART_LAYOUT.pop("yaxis", None)
_AXIS = dict(color="#9CA3AF", gridcolor="#F3F4F6", linecolor="#E5E7EB", showgrid=True)

def render(xgb_model, ae_model_artifacts, mlp_model, metadata, dataset):
    page_header("Thermal Analysis", "Real-Time Monitoring & Forecasting Engine")

    # ── Simulation Controls ────────────────────────
    with st.container(border=True):
        col_mode, col_status, col_action = st.columns([2, 1, 1])
    
        with col_mode:
            sim_mode = st.selectbox(
                "Simulation Mode",
                ["NORMAL", "HIGH_LOAD", "THERMAL_STRESS"],
                key="pred_sim_mode",
            )
        with col_status:
            is_running = st.session_state.get("pred_running", False)
            is_completed = st.session_state.get("pred_completed", False)
            
            if is_running:
                status_text = "🟢 RUNNING"
                status_color = "#16A34A"
            elif is_completed:
                status_text = "✅ COMPLETE"
                status_color = "#2563EB"
            else:
                status_text = "⚪ STOPPED"
                status_color = "#6B7280"
                
            st.markdown(f'<div style="padding-top: 28px; font-weight: 600; font-size:13px; color: {status_color};">{status_text}</div>', unsafe_allow_html=True)
    
        with col_action:
            st.markdown('<div style="padding-top: 20px;"></div>', unsafe_allow_html=True)
            if not is_running:
                if st.button("Start Live Simulation", type="primary", key="start_sim", use_container_width=True):
                    st.session_state.pred_running = True
                    st.session_state.pred_completed = False
                    st.session_state.pred_step = 0
                    st.session_state.pred_history = []
                    st.rerun()
            else:
                if st.button("Stop", key="stop_sim", use_container_width=True):
                    st.session_state.pred_running = False
                    st.session_state.pred_completed = False
                    st.rerun()

    ae_model, ae_scaler, ae_imputer, ae_threshold = ae_model_artifacts

    # ── Live Simulation Loop ───────────────────────
    if is_running or is_completed:
        step = st.session_state.get("pred_step", 0)
        history = st.session_state.get("pred_history", [])

        # Generate current sensor state
        features = get_simulation_state(sim_mode, step)
        predicted_prob = predict_failure_probability(mlp_model, features)

        trend = {
            "cell_temperature_avg": 0.5 if sim_mode == "THERMAL_STRESS" else 0.1,
            "cell_temperature_max": 0.8 if sim_mode == "THERMAL_STRESS" else 0.1,
            "internal_resistance": 0.05 if sim_mode == "THERMAL_STRESS" else 0.01,
            "cooling_system_health": -1.0 if sim_mode == "THERMAL_STRESS" else -0.1,
        }

        forecasts = forecast_failure_probability(mlp_model, features, trend=trend)
        anomaly = detect_anomaly(ae_model, ae_scaler, ae_imputer, ae_threshold, features)
        
        # Calculate Risk
        if predicted_prob > 0.8: risk_level = "CRITICAL"
        elif predicted_prob > 0.4: risk_level = "HIGH"
        elif predicted_prob > 0.15: risk_level = "CAUTION"
        else: risk_level = "NORMAL"

        history.append({"step": step, "prob": predicted_prob * 100, "time": step * 2})
        if len(history) > 60:
            history = history[-60:]
        st.session_state.pred_history = history
        st.session_state.pred_step = step + 1
        
        # Sync with Mission Control
        st.session_state.mc_features = features
        st.session_state.pop("mc_agent_state", None)

        # ── Sensor Gauges ──────────────────────────
        with st.container(border=True):
            cols = st.columns(6)
            gauges = [
                ("Avg Temp", f"{features['cell_temperature_avg']:.1f} °C"),
                ("Max Temp", f"{features['cell_temperature_max']:.1f} °C"),
                ("Charge Power", f"{features['average_charge_power_kw']:.1f} kW"),
                ("SOC", f"{features['state_of_charge']:.1f} %"),
                ("Resistance", f"{features['internal_resistance']:.2f} Ω"),
                ("Cooling Health", f"{features['cooling_system_health']:.1f} %"),
            ]
            for col, (label, value) in zip(cols, gauges):
                with col:
                    st.metric(label, value)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # ── Probability Chart ──────────────────────
        col_chart, col_pred = st.columns([2, 1])

        with col_chart:
            with st.container(border=True):
                st.markdown('<div class="eng-card-header">Real-Time Failure Probability Monitor</div>', unsafe_allow_html=True)
                fig = go.Figure()
                times = [h["time"] for h in history]
                probs = [h["prob"] for h in history]
                fig.add_trace(go.Scatter(
                    x=times, y=probs, mode="lines+markers",
                    line=dict(color="#2563EB", width=2),
                    marker=dict(size=4, color="#2563EB"),
                    name="Failure Probability",
                    fill="tozeroy",
                    fillcolor="rgba(37, 99, 235, 0.1)",
                ))
    
                # Add threshold lines
                fig.add_hline(y=15, line_dash="dash", line_color="#D97706",
                             annotation_text="CAUTION", annotation_font_color="#D97706")
                fig.add_hline(y=40, line_dash="dash", line_color="#F97316",
                             annotation_text="HIGH", annotation_font_color="#F97316")
                fig.add_hline(y=80, line_dash="dash", line_color="#DC2626",
                             annotation_text="CRITICAL", annotation_font_color="#DC2626")

            fig.update_layout(
                **_CHART_LAYOUT,
                height=350,
                margin=dict(l=40, r=20, t=10, b=40),
                xaxis=dict(**_AXIS, title="Time (s)"),
                yaxis=dict(**_AXIS, title="Failure Probability (%)", range=[0, 100]),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_pred:
            with st.container(border=True):
                st.markdown('<div class="eng-card-header">Forecast</div>', unsafe_allow_html=True)
                for fc in forecasts:
                    fc_prob = fc["predicted_prob"] * 100
                    delta_color = "#16A34A" if fc_prob < 15 else "#D97706" if fc_prob < 40 else "#DC2626"
                    st.markdown(f'''
                    <div style="display: flex; justify-content: space-between; padding: 10px 0;
                                border-bottom: 1px solid #F3F4F6;">
                        <span style="color: #6B7280; font-size: 13px;">+{fc["horizon_min"]} min</span>
                        <span style="color: {delta_color}; font-family: 'JetBrains Mono'; font-weight: 600;">{fc_prob:.1f}%</span>
                    </div>
                    ''', unsafe_allow_html=True)
    
                # Status
                st.markdown('<div class="eng-card-header" style="margin-top:16px;">Status</div>', unsafe_allow_html=True)
                st.markdown(f'''
                <div class="eng-card">
                    <div><span class="risk-badge risk-{risk_level.lower()}">{risk_level}</span></div>
                    <div style="margin-top: 8px; font-size: 12px; color: #6B7280;">
                        {"🔴 " + anomaly["label"] if anomaly["label"] == "ANOMALY" else "🟢 " + anomaly["label"]}
                    </div>
                </div>
                ''', unsafe_allow_html=True)

        # Auto-refresh or complete
        if is_running:
            if step < 50:
                time.sleep(1.5)
                st.rerun()
            else:
                st.session_state.pred_running = False
                st.session_state.pred_completed = True
                st.rerun()
        elif is_completed:
            st.success("✅ Simulation complete — 50 steps reached.")

    else:
        # ── Static View ────────────────────────────
        features = get_default_features()
        predicted_prob = predict_failure_probability(mlp_model, features)
        forecasts = forecast_failure_probability(mlp_model, features)

        col_curr, col_f1, col_f3, col_f5 = st.columns(4)
        with col_curr:
            st.metric("Current Probability", f"{predicted_prob * 100:.1f}%")
        with col_f1:
            st.metric(f"+{forecasts[0]['horizon_min']} min", f"{forecasts[0]['predicted_prob']*100:.1f}%")
        with col_f3:
            st.metric(f"+{forecasts[1]['horizon_min']} min", f"{forecasts[1]['predicted_prob']*100:.1f}%")
        with col_f5:
            st.metric(f"+{forecasts[2]['horizon_min']} min", f"{forecasts[2]['predicted_prob']*100:.1f}%")

        st.info("Select a simulation mode and click **Start Live Simulation** to begin real-time monitoring.")

        # Show history if available
        history = st.session_state.get("pred_history", [])
        if history:
            st.markdown('<div class="eng-card-header" style="margin-top:20px;">Previous Simulation</div>', unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[h["time"] for h in history],
                y=[h["prob"] for h in history],
                mode="lines+markers",
                line=dict(color="#2563EB", width=2),
                marker=dict(size=4),
                fill="tozeroy",
                fillcolor="rgba(37, 99, 235, 0.1)",
            ))
            fig.update_layout(
                **_CHART_LAYOUT,
                height=300,
                margin=dict(l=40, r=20, t=10, b=40),
                xaxis=dict(**_AXIS, title="Time (s)"),
                yaxis=dict(**_AXIS, title="Failure Probability (%)", range=[0, 100]),
            )
            st.plotly_chart(fig, use_container_width=True)
