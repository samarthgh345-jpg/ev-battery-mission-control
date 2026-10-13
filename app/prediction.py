"""
Thermal Analysis — Live Monitoring & Forecasting Engine
========================================================
Real-time thermal tracking, failure probability forecasting,
subsystem parameter telemetry, and multi-model anomaly detection.
Clean, modern, spacious, engineering-focused design.
"""

import time
import streamlit as st
import numpy as np
import plotly.graph_objects as go

from src.prediction_engine import predict_failure_probability, get_default_features
from src.forecasting import forecast_failure_probability
from src.anomaly_detection import detect_anomaly
from src.simulation_engine import get_simulation_state
from src.risk_engine import assess_thermal_risk
from src.thermal_simulator import BatteryPackThermalSimulator
from app.ui_components import (
    CHART_LAYOUT,
    page_header,
    metric_card,
    status_badge,
    info_panel,
    warning_panel,
    error_panel,
)
from app.styles import clean_html


def render(xgb_model, ae_model_artifacts, mlp_model, metadata, dataset):
    page_header(
        title="Thermal Analysis",
        subtitle="Real-time thermal tracking, dynamic forecasting, and stress simulation",
        badge_label="SIMULATION MONITOR",
    )

    # ── Simulation Controls Header Card ────────────────────────
    st.markdown(
        clean_html(
            """
            <div class="white-card" style="padding: 14px 20px; margin-bottom: 12px;">
                <div class="white-card-title">Telemetry stream controls</div>
                <div class="white-card-subtitle">Select simulation mode, advance steps, or stream dynamic thermal progression</div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.5, 1, 1, 1])

    with ctrl_col1:
        sim_mode = st.selectbox(
            "Telemetry Simulation Mode",
            ["NORMAL", "HIGH_LOAD", "THERMAL_STRESS"],
            key="pred_sim_mode_select",
            help="NORMAL = nominal cruising load; HIGH_LOAD = sustained fast-charging load; THERMAL_STRESS = degraded cooling and runaway risk.",
        )

    is_running = st.session_state.get("pred_running", False)
    is_completed = st.session_state.get("pred_completed", False)
    current_step = st.session_state.get("pred_step", 0)

    with ctrl_col2:
        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
        if not is_running:
            if st.button("▶ Start Stream", type="primary", use_container_width=True, key="btn_start_thermal_sim"):
                st.session_state.pred_running = True
                st.session_state.pred_completed = False
                if "pred_history" not in st.session_state:
                    st.session_state.pred_history = []
                st.rerun()
        else:
            if st.button("⏹ Pause Stream", use_container_width=True, key="btn_pause_thermal_sim"):
                st.session_state.pred_running = False
                st.rerun()

    with ctrl_col3:
        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
        if st.button("⏭ Single Step", use_container_width=True, key="btn_step_thermal_sim"):
            st.session_state.pred_running = False
            st.session_state.pred_step = current_step + 1
            st.session_state.pred_manual_step = True
            st.rerun()

    with ctrl_col4:
        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Reset Monitor", use_container_width=True, key="btn_reset_thermal_sim"):
            st.session_state.pred_running = False
            st.session_state.pred_completed = False
            st.session_state.pred_step = 0
            st.session_state.pred_history = []
            st.session_state.pop("pred_manual_step", None)
            st.rerun()

    # Model artifact unpacking
    ae_model, ae_scaler, ae_imputer, ae_threshold = ae_model_artifacts

    # Step progression state
    step = st.session_state.get("pred_step", 0)
    history = st.session_state.get("pred_history", [])

    # Advance state if running or manual step
    manual_stepped = st.session_state.pop("pred_manual_step", False)
    if is_running or manual_stepped or (step > 0):
        active_step = step
    else:
        active_step = 0

    # Generate current physical telemetry features from simulation engine
    features = get_simulation_state(sim_mode, active_step)

    # Real ML Model Predictions on current features
    prob_mlp = predict_failure_probability(mlp_model, features)
    prob_xgb = predict_failure_probability(xgb_model, features) if xgb_model is not None else prob_mlp

    trend_config = {
        "cell_temperature_avg": 0.6 if sim_mode == "THERMAL_STRESS" else 0.2 if sim_mode == "HIGH_LOAD" else 0.05,
        "cell_temperature_max": 0.9 if sim_mode == "THERMAL_STRESS" else 0.3 if sim_mode == "HIGH_LOAD" else 0.08,
        "internal_resistance": 0.06 if sim_mode == "THERMAL_STRESS" else 0.02 if sim_mode == "HIGH_LOAD" else 0.005,
        "cooling_system_health": -1.2 if sim_mode == "THERMAL_STRESS" else -0.3 if sim_mode == "HIGH_LOAD" else -0.05,
    }

    forecasts = forecast_failure_probability(mlp_model, features, trend=trend_config)
    anomaly = detect_anomaly(ae_model, ae_scaler, ae_imputer, ae_threshold, features)

    # Multi-factor Thermal Risk Assessment
    risk_assessment = assess_thermal_risk(
        current_temp=features["cell_temperature_max"],
        predicted_temp_5min=features["cell_temperature_max"] + (trend_config["cell_temperature_max"] * 5),
        temp_rate_per_min=trend_config["cell_temperature_max"],
        anomaly_score=anomaly["score"],
        hotspot_risk_pct=prob_mlp * 100,
    )
    risk_level = risk_assessment["risk_level"]
    risk_score = risk_assessment["risk_score"]

    # Record history snapshot
    if is_running or manual_stepped:
        history.append({
            "step": active_step,
            "time_s": active_step * 2,
            "prob_pct": prob_mlp * 100,
            "temp_avg": features["cell_temperature_avg"],
            "temp_max": features["cell_temperature_max"],
            "power": features["average_charge_power_kw"],
            "soc": features["state_of_charge"],
            "res": features["internal_resistance"],
            "cool_health": features["cooling_system_health"],
        })
        if len(history) > 60:
            history = history[-60:]
        st.session_state.pred_history = history
        if is_running:
            st.session_state.pred_step = active_step + 1

    # Keep Mission Control state synchronized
    st.session_state.mc_features = features

    # Stream progress indicator
    p_col1, p_col2 = st.columns([3, 1])
    with p_col1:
        prog = min(1.0, active_step / 50.0)
        st.progress(prog)
    with p_col2:
        status_txt = "🟢 STREAMING" if is_running else "✅ COMPLETED (50 STEPS)" if is_completed else f"⏸ PAUSED (Step {active_step}/50)"
        st.markdown(f"<div style='font-size: 11px; font-family: monospace; color: #16866A; font-weight: 700; text-align: right;'>{status_txt}</div>", unsafe_allow_html=True)

    # ── Overview Metrics ───────────────────────────────────────
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    g1, g2, g3, g4, g5, g6 = st.columns(6)

    with g1:
        metric_card("AVG CELL TEMP", f"{features['cell_temperature_avg']:.1f}", "°C", color="#326CE5", subcaption="Pack mean")
    with g2:
        max_c = "#D94841" if features["cell_temperature_max"] >= 48 else "#C7771B" if features["cell_temperature_max"] >= 42 else "#16866A"
        metric_card("HOTSPOT TEMP", f"{features['cell_temperature_max']:.1f}", "°C", color=max_c, subcaption="Hottest cell")
    with g3:
        p_c = "#D94841" if prob_mlp >= 0.4 else "#C7771B" if prob_mlp >= 0.15 else "#16866A"
        metric_card("FAILURE PROB", f"{prob_mlp*100:.1f}", "%", color=p_c, subcaption="MLP estimate")
    with g4:
        metric_card("CHARGE POWER", f"{features['average_charge_power_kw']:.1f}", "kW", color="#172033", subcaption="Fast charge")
    with g5:
        metric_card("RESISTANCE", f"{features['internal_resistance']:.2f}", "mΩ", color="#172033", subcaption="Internal impedance")
    with g6:
        c_c = "#16866A" if features["cooling_system_health"] >= 60 else "#D94841"
        metric_card("COOLING HEALTH", f"{features['cooling_system_health']:.0f}", "%", color=c_c, subcaption="BTMS capacity")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # ── Synchronized Plots: Temperature Tracking & Failure Probability ──
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown(
            clean_html(
                """
                <div class="white-card" style="margin-bottom: 0;">
                    <div class="white-card-title">Temperature trend</div>
                    <div class="white-card-subtitle">Simulated pack temperature and hottest-cell temperature over time</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        fig_temp = go.Figure()
        if history:
            times = [h["time_s"] for h in history]
            t_avgs = [h["temp_avg"] for h in history]
            t_maxs = [h["temp_max"] for h in history]

            fig_temp.add_trace(
                go.Scatter(
                    x=times,
                    y=t_maxs,
                    mode="lines+markers",
                    name="Max Cell Temp (°C)",
                    line=dict(color="#D94841", width=2.4),
                    marker=dict(size=4),
                )
            )
            fig_temp.add_trace(
                go.Scatter(
                    x=times,
                    y=t_avgs,
                    mode="lines",
                    name="Avg Cell Temp (°C)",
                    line=dict(color="#326CE5", width=2.0),
                )
            )
        else:
            fig_temp.add_trace(
                go.Scatter(
                    x=[0],
                    y=[features["cell_temperature_max"]],
                    mode="markers",
                    name="Initial Hotspot",
                    marker=dict(color="#D94841", size=8),
                )
            )

        fig_temp.add_hline(y=42.0, line_dash="dash", line_color="#C7771B", annotation_text="Caution (42°C)", annotation_font_size=9)
        fig_temp.add_hline(y=48.0, line_dash="dash", line_color="#D94841", annotation_text="Critical (48°C)", annotation_font_size=9)

        t_layout = CHART_LAYOUT.copy()
        t_layout.update(
            height=260,
            margin=dict(l=38, r=12, t=15, b=25),
            xaxis=dict(title="Time (Seconds)", showgrid=True, gridcolor="#F3F4F6", tickfont=dict(color="#667085", size=9)),
            yaxis=dict(title="Temperature (°C)", showgrid=True, gridcolor="#F3F4F6", tickfont=dict(color="#667085", size=9), range=[15, 60]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#172033", size=9)),
        )
        fig_temp.update_layout(**t_layout)
        st.plotly_chart(fig_temp, use_container_width=True)

    with chart_col2:
        st.markdown(
            clean_html(
                """
                <div class="white-card" style="margin-bottom: 0;">
                    <div class="white-card-title">Failure risk horizon</div>
                    <div class="white-card-subtitle">PyTorch MLP inference on simulated feature progression</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        fig_prob = go.Figure()
        if history:
            times = [h["time_s"] for h in history]
            probs = [h["prob_pct"] for h in history]

            fig_prob.add_trace(
                go.Scatter(
                    x=times,
                    y=probs,
                    mode="lines+markers",
                    name="Failure Prob (%)",
                    line=dict(color="#D94841" if probs[-1] >= 40 else "#326CE5", width=2.4),
                    fill="tozeroy",
                    fillcolor="rgba(217, 72, 65, 0.1)" if probs[-1] >= 40 else "rgba(50, 108, 229, 0.08)",
                    marker=dict(size=4),
                )
            )
        else:
            fig_prob.add_trace(
                go.Scatter(
                    x=[0],
                    y=[prob_mlp * 100],
                    mode="markers",
                    name="Current Prob",
                    marker=dict(color="#326CE5", size=8),
                )
            )

        fig_prob.add_hline(y=15.0, line_dash="dash", line_color="#C7771B", annotation_text="Caution (15%)", annotation_font_size=9)
        fig_prob.add_hline(y=40.0, line_dash="dash", line_color="#D94841", annotation_text="High (40%)", annotation_font_size=9)

        p_layout = CHART_LAYOUT.copy()
        p_layout.update(
            height=260,
            margin=dict(l=38, r=12, t=15, b=25),
            xaxis=dict(title="Time (Seconds)", showgrid=True, gridcolor="#F3F4F6", tickfont=dict(color="#667085", size=9)),
            yaxis=dict(title="Failure Prob (%)", showgrid=True, gridcolor="#F3F4F6", tickfont=dict(color="#667085", size=9), range=[0, 100]),
            showlegend=False,
        )
        fig_prob.update_layout(**p_layout)
        st.plotly_chart(fig_prob, use_container_width=True)

    # ── Multi-Horizon Forecast & Anomaly Intelligence ──────────
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    f_col1, f_col2, f_col3 = st.columns([1.2, 1, 1.2])

    with f_col1:
        st.markdown(
            clean_html(
                f"""
                <div class="white-card">
                    <div class="white-card-title">Multi-horizon failure forecast</div>
                    <div class="white-card-subtitle">Simulated feature projection into PyTorch MLP</div>
                    <div style="margin-top: 12px; display: flex; flex-direction: column; gap: 8px; font-size: 12px;">
                        <div style="display: flex; justify-content: space-between; padding-bottom: 6px; border-bottom: 1px solid #F3F4F6;">
                            <span style="color: #667085;">Current State:</span>
                            <b style="font-family: 'JetBrains Mono', monospace; color: #172033;">{prob_mlp*100:.1f}%</b>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding-bottom: 6px; border-bottom: 1px solid #F3F4F6;">
                            <span style="color: #667085;">+1 min Horizon:</span>
                            <b style="font-family: 'JetBrains Mono', monospace; color: #326CE5;">{forecasts[0]['predicted_prob']*100:.1f}%</b>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding-bottom: 6px; border-bottom: 1px solid #F3F4F6;">
                            <span style="color: #667085;">+3 min Horizon:</span>
                            <b style="font-family: 'JetBrains Mono', monospace; color: #C7771B;">{forecasts[1]['predicted_prob']*100:.1f}%</b>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #667085;">+5 min Horizon:</span>
                            <b style="font-family: 'JetBrains Mono', monospace; color: #D94841;">{forecasts[2]['predicted_prob']*100:.1f}%</b>
                        </div>
                    </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    with f_col2:
        st.markdown(
            clean_html(
                f"""
                <div class="white-card">
                    <div class="white-card-title">Autoencoder &amp; risk engine</div>
                    <div class="white-card-subtitle">Anomaly detector and multi-factor evaluation</div>
                    <div style="margin-top: 12px;">
                        <div style="font-size: 10px; font-weight: 700; color: #667085; text-transform: uppercase;">Anomaly Classification</div>
                        <div style="font-size: 16px; font-weight: 800; color: {'#16866A' if anomaly['label'] == 'NORMAL' else '#D94841'}; font-family: 'JetBrains Mono', monospace; margin: 2px 0;">
                            ● {anomaly['label']}
                        </div>
                        <div style="font-size: 11px; color: #667085;">
                            MSE: <b>{anomaly['score']:.4f}</b> (Threshold: {anomaly['threshold']:.4f})
                        </div>
                    </div>
                    <div style="margin-top: 10px; border-top: 1px solid #F3F4F6; padding-top: 8px;">
                        <div style="font-size: 10px; font-weight: 700; color: #667085; text-transform: uppercase;">Thermal Risk Level</div>
                        <div style="font-size: 16px; font-weight: 800; color: {'#16866A' if risk_level == 'NORMAL' else '#C7771B' if risk_level == 'CAUTION' else '#D94841'}; font-family: 'JetBrains Mono', monospace; margin: 2px 0;">
                            {risk_level} ({risk_score:.0f}/100)
                        </div>
                    </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    with f_col3:
        info_panel(
            "**Data & Simulation Notice:**\n\n"
            "• **Sensor Telemetry:** Cell temperatures, power, resistance, and cooling health are generated via the calibrated lumped-capacitance simulation engine.\n\n"
            "• **Probabilities & Anomaly Detection:** Calculated directly from the trained PyTorch MLP failure predictor and Autoencoder anomaly detector.\n\n"
            "• No simulated values are claimed to be physical real-world sensor hardware."
        )

    # ── Auto-refresh live loop ──────────────────────────────────
    if is_running:
        if active_step < 50:
            time.sleep(1.2)
            st.rerun()
        else:
            st.session_state.pred_running = False
            st.session_state.pred_completed = True
            st.toast("✅ Live simulation reached 50 steps.", icon="🏁")
            st.rerun()
