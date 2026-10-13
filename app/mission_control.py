"""
Mission Control — Primary Battery Monitoring Console
======================================================
AI-Powered EV Battery Thermal Safety, Monitoring, and Prediction.
Clean, modern, spacious, engineering-focused dashboard interface.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.prediction_engine import predict_failure_probability, get_default_features
from src.forecasting import forecast_failure_probability
from src.anomaly_detection import detect_anomaly
from src.risk_engine import assess_thermal_risk
from src.simulation_engine import get_simulation_state
from src.thermal_simulator import BatteryPackThermalSimulator
from src.decision_engine import evaluate_thermal_recommendation
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

def _on_flow_slider_change():
    """Streamlit callback triggered when the coolant flow slider changes."""
    if "sim" in st.session_state and "mc_flow_rate_slider" in st.session_state:
        target_flow = float(st.session_state["mc_flow_rate_slider"])
        sim_obj: BatteryPackThermalSimulator = st.session_state.sim
        if abs(target_flow - sim_obj.coolant_flow) > 1e-4:
            res = sim_obj.set_coolant_flow(target_flow)
            if "mc_features" in st.session_state:
                pack_s = sim_obj.get_pack_summary()
                hotspot_s = sim_obj.get_hotspot_info()
                st.session_state.mc_features["cell_temperature_avg"] = round(float(pack_s["avg_temp"]), 1)
                st.session_state.mc_features["cell_temperature_max"] = round(float(hotspot_s["temperature"]), 1)
                st.session_state.mc_features["cooling_system_health"] = max(10.0, min(100.0, round((target_flow / 0.02) * 74.0, 1)))
            st.session_state.mc_recalc_diff = {
                "old_flow": res["old_flow"],
                "new_flow": res["new_flow"],
                "old_hotspot": res["old_hotspot_temp"],
                "new_hotspot": res["new_hotspot_temp"],
                "delta_temp": res["delta_temp"],
            }


def render(xgb_model, ae_model_artifacts, mlp_model, shap_explainer, metadata, dataset):
    sim: BatteryPackThermalSimulator = st.session_state.sim
    sim_mode = st.session_state.get("mc_sim_mode", "NORMAL")

    # ── A. Header Card ──────────────────────────────────────────
    page_header(
        title="EV Battery Mission Control",
        subtitle="Battery health · Thermal monitoring · AI-assisted safety",
        badge_label=f"PROTOTYPE • MODE: {sim_mode}",
    )

    # Synchronize simulated physical state into active telemetry feature vector
    if "mc_features" not in st.session_state:
        st.session_state.mc_features = get_default_features()

    features = st.session_state.mc_features

    # Sync battery physical state from simulator to feature vector
    pack_summary = sim.get_pack_summary()
    hotspot_info = sim.get_hotspot_info()
    features["cell_temperature_avg"] = round(float(pack_summary["avg_temp"]), 1)
    features["cell_temperature_max"] = round(float(hotspot_info["temperature"]), 1)
    features["cooling_system_health"] = max(10.0, min(100.0, round((sim.coolant_flow / 0.02) * 74.0, 1)))

    # ── ML Model Inference ─────────────────────────────────────
    # 1. PyTorch MLP Predictor
    prob_mlp = predict_failure_probability(mlp_model, features)

    # 2. XGBoost Predictor
    prob_xgb = predict_failure_probability(xgb_model, features) if xgb_model is not None else prob_mlp

    # 3. Horizon Forecast (MLP projected features over +1, +3, +5 min)
    forecasts = forecast_failure_probability(mlp_model, features)
    prob_5min = forecasts[-1]["predicted_prob"] if forecasts else prob_mlp

    # 4. Autoencoder Anomaly Detector
    ae_model, ae_scaler, ae_imputer, ae_threshold = ae_model_artifacts
    anomaly_res = detect_anomaly(ae_model, ae_scaler, ae_imputer, ae_threshold, features)

    # 5. Multi-factor Thermal Risk Engine
    risk_assessment = assess_thermal_risk(
        current_temp=hotspot_info["temperature"],
        predicted_temp_5min=None,
        temp_rate_per_min=0.0,
        anomaly_score=anomaly_res["score"],
        hotspot_risk_pct=prob_mlp * 100,
    )
    risk_level = risk_assessment["risk_level"]
    risk_score = risk_assessment["risk_score"]

    # ── B. Four Aligned Battery Overview Metrics ────────────────
    m1, m2, m3, m4 = st.columns(4)

    prob_mlp_pct = prob_mlp * 100
    prob_color = "#16866A" if prob_mlp_pct < 15 else "#C7771B" if prob_mlp_pct < 40 else "#D94841"

    avg_temp = features["cell_temperature_avg"]
    temp_color = "#16866A" if avg_temp < 36 else "#326CE5" if avg_temp < 42 else "#C7771B" if avg_temp < 48 else "#D94841"

    risk_color = "#16866A" if risk_score < 30 else "#C7771B" if risk_score < 60 else "#D94841"

    cooling_health = features["cooling_system_health"]
    cool_color = "#326CE5" if cooling_health >= 70 else "#C7771B" if cooling_health >= 40 else "#D94841"

    with m1:
        metric_card(
            label="FAILURE PROBABILITY",
            value=f"{prob_mlp_pct:.1f}",
            suffix="%",
            subcaption="Model estimate",
            help_text="Output of the trained PyTorch MLP failure classifier for current telemetry.",
            color=prob_color,
        )

    with m2:
        metric_card(
            label="PACK TEMPERATURE",
            value=f"{avg_temp:.1f}",
            suffix="°C",
            subcaption="Simulated telemetry",
            help_text="Mean pack temperature across all 24 cells from the physics-based thermal simulator.",
            color=temp_color,
        )

    with m3:
        metric_card(
            label="OVERALL RISK SCORE",
            value=f"{risk_score:.0f}",
            suffix="/100",
            subcaption="Deterministic score",
            help_text="Multi-factor deterministic Risk Engine score (0–100) combining hotspot boundaries and rate.",
            color=risk_color,
        )

    with m4:
        metric_card(
            label="COOLING HEALTH",
            value=f"{cooling_health:.0f}",
            suffix="%",
            subcaption="Simulated status",
            help_text="Convective cooling system operational capacity based on coolant flow rate.",
            color=cool_color,
        )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # ── C & D. Main Content Row: Battery Visualization & Decision ──
    col_left, col_right = st.columns([1.18, 0.82])

    with col_left:
        # ── C. Battery Pack Visualization Card ──────────────────
        cell_matrix = sim.get_cell_matrix()
        cells_html = []
        for row in cell_matrix:
            for cell in row:
                cid = cell["id"]
                t = cell["temp"]

                # Pastel palette matching reference design
                if t >= 48.0:
                    bg = "#FEE2E2"
                    border = "#FECACA"
                    text_color = "#991B1B"
                elif t >= 42.0:
                    bg = "#FEF3C7"
                    border = "#FDE68A"
                    text_color = "#92400E"
                elif t >= 36.0:
                    bg = "#E0F2FE"
                    border = "#BAE6FD"
                    text_color = "#0369A1"
                else:
                    bg = "#D1FAE5"
                    border = "#A7F3D0"
                    text_color = "#065F46"

                is_hotspot = (cid == hotspot_info["cell_id"])
                hotspot_ring = "box-shadow: 0 0 0 2px #D94841;" if is_hotspot and t >= 42.0 else ""

                cells_html.append(
                    f"""<div style="background: {bg}; border: 1px solid {border}; border-radius: 8px; padding: 10px 4px; text-align: center; {hotspot_ring}">"""
                    f"""<div style="font-size: 11px; font-weight: 600; color: #4B5563;">C{cid}</div>"""
                    f"""<div style="font-size: 13px; font-weight: 800; color: {text_color}; font-family: 'JetBrains Mono', monospace; margin-top: 2px;">{t:.0f}°</div>"""
                    f"""</div>"""
                )

        grid_html = "".join(cells_html)

        st.markdown(
            clean_html(
                f"""
                <div class="white-card">
                    <div class="white-card-header">
                        <div>
                            <div class="white-card-title">Battery pack health</div>
                            <div class="white-card-subtitle">24-cell planar module (4×6 grid)</div>
                        </div>
                        <div style="font-size: 12px; font-weight: 700; color: #326CE5; font-family: 'JetBrains Mono', monospace;">
                            Avg: {pack_summary['avg_temp']:.1f}°C · Hotspot: {hotspot_info['temperature']:.1f}°C
                        </div>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(6, 1fr); gap: 8px; margin: 12px 0 14px 0;">
                        {grid_html}
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid #F3F4F6; padding-top: 10px; font-size: 11px; color: #667085;">
                        <div style="display: flex; gap: 12px;">
                            <span><b style="color: #065F46;">■</b> &lt;36°C Normal</span>
                            <span><b style="color: #0369A1;">■</b> 36–42°C Moderate</span>
                            <span><b style="color: #92400E;">■</b> 42–48°C Caution</span>
                            <span><b style="color: #991B1B;">■</b> ≥48°C Critical</span>
                        </div>
                        <div style="font-style: italic; font-size: 10px;">
                            Illustrative layout only — actual readings are driven by the physics simulator.
                        </div>
                    </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        # Hotspot Alert beneath grid if applicable
        if hotspot_info["temperature"] >= 42.0:
            is_crit = hotspot_info["temperature"] >= 48.0
            alert_prefix = "🚨 CRITICAL THERMAL RUNAWAY RISK" if is_crit else "⚠️ THERMAL HOTSPOT CAUTION"
            alert_text = (
                f"{alert_prefix}: Cell C{hotspot_info['cell_id']} at {hotspot_info['temperature']:.1f}°C "
                f"(+{hotspot_info['temperature'] - pack_summary['avg_temp']:.1f}°C above pack average). "
                f"Location: {hotspot_info['location']}. Active cooling flow adjustment recommended."
            )
            if is_crit:
                error_panel(alert_text)
            else:
                warning_panel(alert_text)

    with col_right:
        # ── D. Thermal AI Decision Card ─────────────────────────
        decision = evaluate_thermal_recommendation(
            hotspot_temp=hotspot_info["temperature"],
            avg_temp=pack_summary["avg_temp"],
            cooling_health=features["cooling_system_health"],
            failure_prob=prob_mlp,
            anomaly_score=anomaly_res["score"],
            anomaly_threshold=anomaly_res["threshold"],
            coolant_flow=sim.coolant_flow,
            current=sim.current,
        )

        # Status badge color and safe field fallbacks
        status_colors = {
            "normal": "#16866A",
            "caution": "#C7771B",
            "high": "#D97706",
            "critical": "#D94841",
        }
        raw_status = decision.get("status") or decision.get("severity") or "NORMAL"
        dec_status = str(raw_status).lower()
        dec_status_display = str(raw_status).upper()
        badge_c = status_colors.get(dec_status, "#326CE5")

        suggested_action = decision.get("suggested_action") or decision.get("recommended_action") or "MAINTAIN NOMINAL OPERATION"
        reason_text = decision.get("reason") or "All monitored thermal and electrical parameters within safe operating thresholds."
        obs_text = decision.get("observation") or f"Peak temperature {hotspot_info['temperature']:.1f}°C in Cell C{hotspot_info['cell_id']} ({hotspot_info['location']}), pack mean {pack_summary['avg_temp']:.1f}°C."
        boost_pct = float(decision.get("recommended_flow_boost_pct", 0.0))

        st.markdown(
            clean_html(
                f"""
                <div class="white-card">
                    <div class="white-card-header">
                        <div>
                            <div class="white-card-title">Thermal AI decision</div>
                            <div class="white-card-subtitle">Observation · Prediction · Recommended cooling action</div>
                        </div>
                        <span style="background: #F3F4F6; color: {badge_c}; font-size: 11px; font-weight: 800; padding: 4px 10px; border-radius: 6px; font-family: 'JetBrains Mono', monospace;">
                            ● {dec_status_display}
                        </span>
                    </div>

                    <div class="decision-callout">
                        {suggested_action}
                    </div>

                    <div style="font-size: 12px; line-height: 1.6; color: #172033; display: flex; flex-direction: column; gap: 8px;">
                        <div>
                            <strong style="color: #667085; text-transform: uppercase; font-size: 10px; letter-spacing: 0.04em;">Observation</strong><br>
                            {obs_text}
                        </div>
                        <div>
                            <strong style="color: #667085; text-transform: uppercase; font-size: 10px; letter-spacing: 0.04em;">Reason</strong><br>
                            {reason_text}
                        </div>
                        <div>
                            <strong style="color: #667085; text-transform: uppercase; font-size: 10px; letter-spacing: 0.04em;">Prediction</strong><br>
                            PyTorch MLP failure probability: <b>{prob_mlp*100:.1f}%</b> · 5-minute projected horizon: <b>{prob_5min*100:.1f}%</b>
                        </div>
                        <div>
                            <strong style="color: #667085; text-transform: uppercase; font-size: 10px; letter-spacing: 0.04em;">Recommended Action</strong><br>
                            Coolant flow boost of +{boost_pct:.0f}% suggested by rule-based safety governor.
                        </div>
                    </div>

                    <div style="border-top: 1px solid #F3F4F6; margin-top: 12px; padding-top: 8px; font-size: 10px; color: #667085;">
                        Deterministic rule-based safety decision engine. Differentiated from ML predictions; no autonomous physical actuator.
                    </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        # Active Operating Telemetry Tile
        st.markdown(
            clean_html(
                f"""
                <div class="white-card" style="padding: 14px 18px;">
                    <div style="font-size: 11px; font-weight: 700; color: #667085; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 8px;">
                        Active Operating Telemetry
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 12px;">
                        <div style="background: #F9FAFB; padding: 8px 10px; border-radius: 6px; border: 1px solid #E5E7EB;">
                            <span style="color: #667085; font-size: 10px;">Charge Power:</span><br>
                            <b style="font-family: 'JetBrains Mono', monospace; color: #172033;">{features['average_charge_power_kw']:.1f} kW</b>
                        </div>
                        <div style="background: #F9FAFB; padding: 8px 10px; border-radius: 6px; border: 1px solid #E5E7EB;">
                            <span style="color: #667085; font-size: 10px;">Internal Res:</span><br>
                            <b style="font-family: 'JetBrains Mono', monospace; color: #172033;">{features['internal_resistance']:.2f} mΩ</b>
                        </div>
                        <div style="background: #F9FAFB; padding: 8px 10px; border-radius: 6px; border: 1px solid #E5E7EB;">
                            <span style="color: #667085; font-size: 10px;">State of Charge:</span><br>
                            <b style="font-family: 'JetBrains Mono', monospace; color: #172033;">{features['state_of_charge']:.1f}%</b>
                        </div>
                        <div style="background: #F9FAFB; padding: 8px 10px; border-radius: 6px; border: 1px solid #E5E7EB;">
                            <span style="color: #667085; font-size: 10px;">Autoencoder MSE:</span><br>
                            <b style="font-family: 'JetBrains Mono', monospace; color: {'#D94841' if anomaly_res['label'] == 'ANOMALY' else '#16866A'};">{anomaly_res['score']:.4f}</b>
                        </div>
                    </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # ── E. Two Balanced Chart Cards ─────────────────────────────
    ch_col1, ch_col2 = st.columns(2)

    with ch_col1:
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

        traj = sim.simulate_trajectory(duration_minutes=5, dt_minutes=1.0)
        minutes = [t["minute"] for t in traj]
        hotspot_curve = [t["hotspot_temp"] for t in traj]
        avg_curve = [t["avg_temp"] for t in traj]

        fig_t = go.Figure()
        fig_t.add_trace(
            go.Scatter(
                x=minutes,
                y=hotspot_curve,
                mode="lines+markers",
                name="Hotspot Cell (°C)",
                line=dict(color="#D94841", width=2.4),
                marker=dict(size=6, color="#D94841"),
            )
        )
        fig_t.add_trace(
            go.Scatter(
                x=minutes,
                y=avg_curve,
                mode="lines",
                name="Pack Average (°C)",
                line=dict(color="#326CE5", width=2.0),
            )
        )
        fig_t.add_hline(y=42.0, line_dash="dot", line_color="#C7771B", annotation_text="Caution (42°C)", annotation_font_size=9)
        fig_t.add_hline(y=48.0, line_dash="dot", line_color="#D94841", annotation_text="Runaway (48°C)", annotation_font_size=9)

        t_layout = CHART_LAYOUT.copy()
        t_layout.update(
            height=230,
            margin=dict(l=38, r=12, t=15, b=25),
            xaxis=dict(title="Time (Minutes)", showgrid=True, gridcolor="#F3F4F6", tickfont=dict(color="#667085", size=9)),
            yaxis=dict(title="Temperature (°C)", showgrid=True, gridcolor="#F3F4F6", tickfont=dict(color="#667085", size=9), range=[20, 60]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#172033", size=9)),
        )
        fig_t.update_layout(**t_layout)
        st.plotly_chart(fig_t, use_container_width=True)

    with ch_col2:
        st.markdown(
            clean_html(
                """
                <div class="white-card" style="margin-bottom: 0;">
                    <div class="white-card-title">Failure risk horizon</div>
                    <div class="white-card-subtitle">Current and projected model probabilities (+1m, +3m, +5m)</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        horizon_labels = ["Now"] + [f"+{fc['horizon_min']}m" for fc in forecasts]
        horizon_probs = [prob_mlp * 100] + [fc["predicted_prob"] * 100 for fc in forecasts]
        bar_colors = ["#16866A" if p < 15 else "#C7771B" if p < 40 else "#D94841" for p in horizon_probs]

        fig_p = go.Figure()
        fig_p.add_trace(
            go.Bar(
                x=horizon_labels,
                y=horizon_probs,
                marker_color=bar_colors,
                text=[f"{p:.1f}%" for p in horizon_probs],
                textposition="outside",
                textfont=dict(family="JetBrains Mono", size=10, color="#172033"),
                cliponaxis=False,
            )
        )
        fig_p.add_hline(y=15.0, line_dash="dot", line_color="#C7771B", annotation_text="Caution (15%)", annotation_font_size=9)
        fig_p.add_hline(y=40.0, line_dash="dot", line_color="#D94841", annotation_text="High (40%)", annotation_font_size=9)

        p_layout = CHART_LAYOUT.copy()
        p_layout.update(
            height=230,
            margin=dict(l=38, r=12, t=15, b=25),
            xaxis=dict(showgrid=False),
            yaxis=dict(title="Probability (%)", showgrid=True, gridcolor="#F3F4F6", range=[0, 100]),
            showlegend=False,
        )
        fig_p.update_layout(**p_layout)
        st.plotly_chart(fig_p, use_container_width=True)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # ── F & G. Cooling Controls & Scenario Selection ────────────
    ctrl_col1, ctrl_col2 = st.columns(2)

    with ctrl_col1:
        st.markdown(
            clean_html(
                f"""
                <div class="white-card">
                    <div class="white-card-title">Cooling controls</div>
                    <div class="white-card-subtitle">Adjust simulated coolant flow and observe post-action recalculation</div>
                    <div style="font-size: 12px; color: #172033; margin: 8px 0 4px 0;">
                        Current Flow Rate: <b style="font-family: 'JetBrains Mono', monospace; color: #326CE5;">{sim.coolant_flow:.3f} kg/s</b>
                    </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        flow_sub1, flow_sub2 = st.columns([1.3, 1])
        with flow_sub1:
            if "mc_flow_rate_slider_pending" in st.session_state:
                st.session_state["mc_flow_rate_slider"] = float(st.session_state.pop("mc_flow_rate_slider_pending"))
            elif "mc_flow_rate_slider" not in st.session_state:
                st.session_state["mc_flow_rate_slider"] = float(sim.coolant_flow)

            st.slider(
                "Coolant Flow Rate (kg/s)",
                min_value=0.010,
                max_value=0.040,
                step=0.002,
                format="%.3f",
                key="mc_flow_rate_slider",
                on_change=_on_flow_slider_change,
                help="Adjusts coolant flow rate. Applies an immediate quasi-steady convective thermal adjustment without advancing the simulation clock.",
            )

        with flow_sub2:
            st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
            if st.button("Apply Suggested Cooling", type="primary", use_container_width=True, key="btn_apply_rec_cooling"):
                old_h = hotspot_info["temperature"]
                boost_pct = float(decision.get("recommended_flow_boost_pct", 10.0))
                res = sim.apply_recommendation(boost_percentage=boost_pct)
                new_h = sim.get_hotspot_info()["temperature"]

                # Queue slider update for next rerun to avoid modifying instantiated widget
                st.session_state["mc_flow_rate_slider_pending"] = round(float(res["new_flow"]), 3)

                # Synchronize features
                features["cell_temperature_avg"] = round(float(sim.get_pack_summary()["avg_temp"]), 1)
                features["cell_temperature_max"] = round(float(new_h), 1)
                features["cooling_system_health"] = max(10.0, min(100.0, round((res["new_flow"] / 0.02) * 74.0, 1)))

                st.session_state.mc_recalc_diff = {
                    "old_flow": res["old_flow"],
                    "new_flow": res["new_flow"],
                    "old_hotspot": old_h,
                    "new_hotspot": new_h,
                    "delta_temp": new_h - old_h,
                }
                st.toast(f"Cooling flow updated to {res['new_flow']:.3f} kg/s (Hotspot: {new_h - old_h:+.1f}°C)", icon="❄️")
                st.rerun()

        if "mc_recalc_diff" in st.session_state:
            diff = st.session_state.mc_recalc_diff
            st.markdown(
                clean_html(
                    f"""
                    <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 8px; padding: 8px 12px; margin-top: 6px; font-size: 11px; color: #166534;">
                        <b>Quasi-Steady Convective Shift:</b> Flow {diff['old_flow']:.3f} → <b>{diff['new_flow']:.3f} kg/s</b> | Hotspot shift: {diff['old_hotspot']:.1f}°C → <b>{diff['new_hotspot']:.1f}°C</b> ({diff['delta_temp']:+.1f}°C). Simulation time remains fixed at {sim.time_elapsed_seconds:.0f}s.
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )

    with ctrl_col2:
        st.markdown(
            clean_html(
                """
                <div class="white-card">
                    <div class="white-card-title">Scenario and reset controls</div>
                    <div class="white-card-subtitle">Switch operating conditions or restore nominal calibrated baseline</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        sc_sub1, sc_sub2 = st.columns([1.3, 1])
        with sc_sub1:
            if "mc_scenario_select_pending" in st.session_state:
                st.session_state["mc_scenario_select"] = st.session_state.pop("mc_scenario_select_pending")
            elif "mc_scenario_select" not in st.session_state:
                st.session_state["mc_scenario_select"] = sim_mode

            selected_mode = st.selectbox(
                "Operating Scenario",
                ["NORMAL", "HIGH_LOAD", "THERMAL_STRESS"],
                key="mc_scenario_select",
                help="NORMAL = nominal cruising; HIGH_LOAD = fast charging load; THERMAL_STRESS = cooling degradation.",
            )

            if selected_mode != sim_mode:
                st.session_state.mc_sim_mode = selected_mode
                if selected_mode == "NORMAL":
                    new_sim = BatteryPackThermalSimulator(ambient_temp=30.0, coolant_temp=22.0, current=10.0, coolant_flow=0.02, seed=42)
                elif selected_mode == "HIGH_LOAD":
                    new_sim = BatteryPackThermalSimulator(ambient_temp=32.0, coolant_temp=24.0, current=35.0, coolant_flow=0.02, seed=42)
                elif selected_mode == "THERMAL_STRESS":
                    new_sim = BatteryPackThermalSimulator(ambient_temp=38.0, coolant_temp=28.0, current=45.0, coolant_flow=0.012, seed=42)
                st.session_state.sim = new_sim
                st.session_state["mc_flow_rate_slider_pending"] = float(new_sim.coolant_flow)
                st.session_state.mc_features = get_simulation_state(selected_mode, 0)
                st.session_state.pop("mc_recalc_diff", None)
                st.rerun()

        with sc_sub2:
            st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
            if st.button("Reset to Baseline", use_container_width=True, key="btn_reset_baseline"):
                st.session_state.mc_sim_mode = "NORMAL"
                st.session_state["mc_scenario_select_pending"] = "NORMAL"
                st.session_state["mc_flow_rate_slider_pending"] = 0.020
                st.session_state.sim = BatteryPackThermalSimulator(
                    ambient_temp=30.0,
                    coolant_temp=22.0,
                    current=10.0,
                    coolant_flow=0.02,
                    seed=42,
                )
                st.session_state.mc_features = get_default_features()
                st.session_state.pop("mc_recalc_diff", None)
                st.toast("Restored nominal calibrated normal baseline.", icon="⏮")
                st.rerun()


        # Quick navigation buttons
        nav_sub1, nav_sub2 = st.columns(2)
        with nav_sub1:
            if st.button("Open Thermal Analysis", use_container_width=True, key="btn_nav_thermal_analysis"):
                st.session_state.current_page = "Thermal Analysis"
                st.rerun()
        with nav_sub2:
            if st.button("Open What-If Simulator", use_container_width=True, key="btn_nav_what_if"):
                st.session_state.current_page = "What-If Simulator"
                st.rerun()
