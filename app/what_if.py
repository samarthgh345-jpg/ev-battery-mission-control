"""
What-If Simulator Page
========================
Compare baseline vs hypothetical operating conditions.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from src.prediction_engine import predict_temperature, get_default_features
from src.risk_engine import assess_thermal_risk
from src.hotspot_risk import estimate_hotspot_risk, generate_cell_temperature_grid
from src.shap_explainer import explain_local
from app.ui_components import page_header, section_header, CHART_LAYOUT, metric_card, status_badge

def render(xgb_model, iso_model, shap_explainer, metadata, dataset):
    page_header("What-If Simulator", "Explore Hypothetical Operating Scenarios")

    # ── Controls ───────────────────────────────────
    section_header("Adjust Operating Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        wi_current = st.slider("Battery Current (A)", 1.0, 15.0, 5.0, 0.5, key="wi_current")
        wi_ambient = st.slider("Ambient Temperature (°C)", 20.0, 45.0, 28.0, 0.5, key="wi_ambient")

    with col2:
        wi_coolant_flow = st.slider("Coolant Flow Rate (kg/s)", 0.005, 0.040, 0.020, 0.001, key="wi_flow")
        wi_coolant_inlet = st.slider("Coolant Inlet Temp (°C)", 18.0, 40.0, 23.0, 0.5, key="wi_inlet")

    with col3:
        wi_soc = st.slider("State of Charge (%)", 10.0, 100.0, 60.0, 5.0, key="wi_soc")
        wi_discharge = st.slider("Discharge Rate (C)", 0.5, 5.0, 1.5, 0.1, key="wi_discharge")

    run_whatif = st.button("Run What-If Simulation", use_container_width=True, key="run_whatif")

    st.markdown('<div style="margin-top:20px; margin-bottom:20px; border-bottom:1px solid var(--border);"></div>', unsafe_allow_html=True)

    # ── Baseline ───────────────────────────────────
    baseline = get_default_features()
    baseline_temp = predict_temperature(xgb_model, baseline)
    baseline_risk = assess_thermal_risk(baseline_temp)
    baseline_hotspot = estimate_hotspot_risk(
        baseline_temp, baseline["battery_current_A"],
        baseline["discharge_rate_C"], baseline["coolant_flow_rate_kg_s"],
        baseline["ambient_temperature_C"],
    )

    if run_whatif or st.session_state.get("wi_ran", False):
        st.session_state.wi_ran = True

        # What-if scenario
        whatif = get_default_features()
        whatif["battery_current_A"] = wi_current
        whatif["ambient_temperature_C"] = wi_ambient
        whatif["battery_temperature_C"] = wi_ambient + 6
        whatif["coolant_flow_rate_kg_s"] = wi_coolant_flow
        whatif["coolant_inlet_temperature_C"] = wi_coolant_inlet
        whatif["state_of_charge_percent"] = wi_soc
        whatif["discharge_rate_C"] = wi_discharge

        whatif_temp = predict_temperature(xgb_model, whatif)
        delta_temp = whatif_temp - baseline_temp

        whatif_risk = assess_thermal_risk(whatif_temp)
        whatif_hotspot = estimate_hotspot_risk(
            whatif_temp, wi_current, wi_discharge, wi_coolant_flow, wi_ambient
        )

        # ── Parameter Comparison ───────────────────────
        section_header("Parameter Comparison")
        
        comp_data = [
            {"Parameter": "Battery Current", "Baseline": f'{baseline["battery_current_A"]:.1f} A', "Scenario": f"{wi_current:.1f} A", "Change": f"{wi_current - baseline['battery_current_A']:+.1f} A"},
            {"Parameter": "Ambient Temp", "Baseline": f'{baseline["ambient_temperature_C"]:.1f} °C', "Scenario": f"{wi_ambient:.1f} °C", "Change": f"{wi_ambient - baseline['ambient_temperature_C']:+.1f} °C"},
            {"Parameter": "Coolant Flow", "Baseline": f'{baseline["coolant_flow_rate_kg_s"]:.4f} kg/s', "Scenario": f"{wi_coolant_flow:.4f} kg/s", "Change": f"{wi_coolant_flow - baseline['coolant_flow_rate_kg_s']:+.4f} kg/s"},
            {"Parameter": "Coolant Inlet", "Baseline": f'{baseline["coolant_inlet_temperature_C"]:.1f} °C', "Scenario": f"{wi_coolant_inlet:.1f} °C", "Change": f"{wi_coolant_inlet - baseline['coolant_inlet_temperature_C']:+.1f} °C"},
        ]
        
        df_comp = pd.DataFrame(comp_data)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        st.markdown('<div style="margin-top:20px; margin-bottom:20px; border-bottom:1px solid var(--border);"></div>', unsafe_allow_html=True)

        # ── Comparison ─────────────────────────────
        col_b, col_arrow, col_w = st.columns([2, 1, 2])

        with col_b:
            section_header("Baseline Prediction")
            metric_card("Predicted Temp", f"{baseline_temp:.1f}", " °C")
            st.markdown("<br/>", unsafe_allow_html=True)
            status_badge("Thermal Risk", baseline_risk["risk_level"])
            status_badge("Hotspot Variance", "NORMAL" if baseline_hotspot["hotspot_risk_percent"] < 30 else "CAUTION")

        with col_arrow:
            delta_color = "#DC2626" if delta_temp > 0 else "#16A34A"
            delta_icon = "↑" if delta_temp > 0 else "↓"
            st.markdown(f'''
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; margin-top: 50px;">
                <div style="font-size: 24px;">{delta_icon}</div>
                <div style="font-family: var(--mono); font-weight: 700; font-size: 24px; color: {delta_color};">{delta_temp:+.1f} °C</div>
                <div style="font-size: 11px; color: var(--text-2); text-transform: uppercase; letter-spacing: 0.05em;">IMPACT</div>
            </div>
            ''', unsafe_allow_html=True)

        with col_w:
            section_header("What-If Prediction")
            metric_card("Predicted Temp", f"{whatif_temp:.1f}", " °C")
            st.markdown("<br/>", unsafe_allow_html=True)
            status_badge("Thermal Risk", whatif_risk["risk_level"])
            status_badge("Hotspot Variance", "NORMAL" if whatif_hotspot["hotspot_risk_percent"] < 30 else "CAUTION")

        # ── Battery Visualization Comparison ───────
        st.markdown('<div style="margin-top:20px; margin-bottom:20px; border-bottom:1px solid var(--border);"></div>', unsafe_allow_html=True)
        col_grid_b, col_grid_w = st.columns(2)

        with col_grid_b:
            section_header("Baseline Grid")
            grid_b = generate_cell_temperature_grid(baseline_temp, baseline_hotspot["hotspot_risk_percent"], seed=42)
            _render_heatmap(grid_b)

        with col_grid_w:
            section_header("What-If Grid")
            grid_w = generate_cell_temperature_grid(whatif_temp, whatif_hotspot["hotspot_risk_percent"], seed=42)
            _render_heatmap(grid_w)

    else:
        st.info("Adjust the parameters above and click **Run What-If Simulation** to compare scenarios.")

def _render_heatmap(grid):
    # Heatmap color scale for light theme
    fig = go.Figure(data=go.Heatmap(
        z=grid[::-1],
        text=[[f"{v:.1f}°" for v in row] for row in grid[::-1]],
        texttemplate="%{text}",
        textfont=dict(size=14, family="JetBrains Mono", color="#111827"),
        colorscale=[[0, "#E0F2FE"], [0.3, "#BAE6FD"], [0.5, "#BBF7D0"],
                    [0.7, "#FEF08A"], [0.85, "#FDBA74"], [1, "#FCA5A5"]],
        zmin=28, zmax=55,
        showscale=False,
        xgap=3, ygap=3,
    ))
    layout_opts = CHART_LAYOUT.copy()
    layout_opts.update(
        height=220,
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False),
        margin=dict(l=10, r=10, t=10, b=10)
    )
    fig.update_layout(**layout_opts)
    st.plotly_chart(fig, use_container_width=True)
