"""
Thermal Analysis — Live Monitoring & Forecasting Page
=====================================================
Temperature forecasting, live sensor simulation, and real-time charts.
"""

import streamlit as st
import time
import numpy as np
import plotly.graph_objects as go
from src.prediction_engine import predict_temperature, get_default_features
from src.forecasting import forecast_temperatures
from src.anomaly_detection import detect_anomaly
from src.risk_engine import assess_thermal_risk
from src.simulation_engine import get_simulation_state
from app.ui_components import page_header, CHART_LAYOUT

_CHART_LAYOUT = CHART_LAYOUT.copy()
# Remove keys that each chart overrides explicitly to avoid 'multiple values' errors
_CHART_LAYOUT.pop("margin", None)
_CHART_LAYOUT.pop("xaxis", None)
_CHART_LAYOUT.pop("yaxis", None)
_AXIS = dict(color="#9CA3AF", gridcolor="#F3F4F6", linecolor="#E5E7EB", showgrid=True)


def render(xgb_model, iso_model, metadata, dataset):
    page_header("Thermal Analysis", "Real-Time Monitoring & Forecasting Engine")

    # ── Simulation Controls ────────────────────────
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
            if st.button("Start Live Simulation", key="start_sim", use_container_width=True):
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

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Live Simulation Loop ───────────────────────
    if is_running or is_completed:
        step = st.session_state.get("pred_step", 0)
        history = st.session_state.get("pred_history", [])

        # Generate current sensor state
        features = get_simulation_state(sim_mode, step)
        predicted_temp = predict_temperature(xgb_model, features)

        # Trend for forecasting
        if len(history) >= 2:
            trend_rate = (history[-1]["temp"] - history[-2]["temp"])
        else:
            trend_rate = 0.3 if sim_mode != "NORMAL" else 0.0

        trend = {
            "battery_current_A": 0.1 if sim_mode == "THERMAL_STRESS" else 0.05,
            "ambient_temperature_C": 0.1 if sim_mode == "THERMAL_STRESS" else 0.02,
            "battery_temperature_C": max(0, trend_rate),
            "coolant_flow_rate_kg_s": -0.0003 if sim_mode == "THERMAL_STRESS" else 0.0,
            "discharge_rate_C": 0.05,
        }

        forecasts = forecast_temperatures(xgb_model, features, trend=trend)
        anomaly_features = {**features, "max_battery_temperature_C": predicted_temp}
        anomaly = detect_anomaly(iso_model, anomaly_features) if iso_model else {"label": "NORMAL", "score": 0}
        risk = assess_thermal_risk(predicted_temp, forecasts[-1]["predicted_C"], temp_rate_per_min=trend_rate)

        history.append({"step": step, "temp": predicted_temp, "time": step * 2})
        if len(history) > 60:
            history = history[-60:]
        st.session_state.pred_history = history
        st.session_state.pred_step = step + 1
        
        # Sync with Mission Control
        st.session_state.mc_features = features
        st.session_state.pop("mc_agent_state", None)

        # ── Sensor Gauges ──────────────────────────
        cols = st.columns(6)
        gauges = [
            ("Current", f"{features['battery_current_A']:.1f} A"),
            ("Voltage", f"{features['battery_voltage_V']:.2f} V"),
            ("Battery Temp", f"{predicted_temp:.1f}°C"),
            ("Ambient", f"{features['ambient_temperature_C']:.1f}°C"),
            ("Coolant Flow", f"{features['coolant_flow_rate_kg_s']:.4f} kg/s"),
            ("Inlet Temp", f"{features['coolant_inlet_temperature_C']:.1f}°C"),
        ]
        for col, (label, value) in zip(cols, gauges):
            with col:
                st.metric(label, value)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # ── Temperature Chart ──────────────────────
        col_chart, col_pred = st.columns([2, 1])

        with col_chart:
            st.markdown('<div class="eng-card-header">Real-Time Temperature Monitor</div>', unsafe_allow_html=True)
            fig = go.Figure()
            times = [h["time"] for h in history]
            temps = [h["temp"] for h in history]
            fig.add_trace(go.Scatter(
                x=times, y=temps, mode="lines+markers",
                line=dict(color="#2563EB", width=2),
                marker=dict(size=4, color="#2563EB"),
                name="Temperature",
                fill="tozeroy",
                fillcolor="rgba(37, 99, 235, 0.1)",
            ))

            # Add threshold lines
            fig.add_hline(y=40, line_dash="dash", line_color="#D97706",
                         annotation_text="CAUTION", annotation_font_color="#D97706")
            fig.add_hline(y=45, line_dash="dash", line_color="#F97316",
                         annotation_text="HIGH", annotation_font_color="#F97316")
            fig.add_hline(y=50, line_dash="dash", line_color="#DC2626",
                         annotation_text="CRITICAL", annotation_font_color="#DC2626")

            fig.update_layout(
                **_CHART_LAYOUT,
                height=350,
                margin=dict(l=40, r=20, t=10, b=40),
                xaxis=dict(**_AXIS, title="Time (s)"),
                yaxis=dict(**_AXIS, title="Temperature (°C)", range=[25, 60]),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_pred:
            st.markdown('<div class="eng-card-header">Forecast</div>', unsafe_allow_html=True)
            st.markdown('<div class="eng-card">', unsafe_allow_html=True)
            for fc in forecasts:
                delta_color = "#16A34A" if fc["predicted_C"] < 42 else "#D97706" if fc["predicted_C"] < 48 else "#DC2626"
                st.markdown(f'''
                <div style="display: flex; justify-content: space-between; padding: 10px 0;
                            border-bottom: 1px solid #F3F4F6;">
                    <span style="color: #6B7280; font-size: 13px;">+{fc["horizon_min"]} min</span>
                    <span style="color: {delta_color}; font-family: 'JetBrains Mono'; font-weight: 600;">{fc["predicted_C"]:.1f}°C</span>
                </div>
                ''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Status
            st.markdown('<div class="eng-card-header" style="margin-top:16px;">Status</div>', unsafe_allow_html=True)
            st.markdown(f'''
            <div class="eng-card">
                <div><span class="risk-badge risk-{risk["risk_level"].lower()}">{risk["risk_level"]}</span></div>
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
        features["battery_current_A"] = 6.0
        features["ambient_temperature_C"] = 28.0
        features["battery_temperature_C"] = 35.0
        predicted_temp = predict_temperature(xgb_model, features)
        forecasts = forecast_temperatures(xgb_model, features)

        col_curr, col_f1, col_f3, col_f5 = st.columns(4)
        with col_curr:
            st.metric("Current Temperature", f"{predicted_temp:.1f}°C")
        with col_f1:
            st.metric(f"+{forecasts[0]['horizon_min']} min", f"{forecasts[0]['predicted_C']:.1f}°C")
        with col_f3:
            st.metric(f"+{forecasts[1]['horizon_min']} min", f"{forecasts[1]['predicted_C']:.1f}°C")
        with col_f5:
            st.metric(f"+{forecasts[2]['horizon_min']} min", f"{forecasts[2]['predicted_C']:.1f}°C")

        st.info("Select a simulation mode and click **Start Live Simulation** to begin real-time monitoring.")

        # Show history if available
        history = st.session_state.get("pred_history", [])
        if history:
            st.markdown('<div class="eng-card-header" style="margin-top:20px;">Previous Simulation</div>', unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[h["time"] for h in history],
                y=[h["temp"] for h in history],
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
                yaxis=dict(**_AXIS, title="Temperature (°C)"),
            )
            st.plotly_chart(fig, use_container_width=True)
