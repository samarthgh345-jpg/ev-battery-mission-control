"""
What-If Simulator Page
========================
Compare baseline vs hypothetical operating conditions for failure probability.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from src.prediction_engine import predict_failure_probability, get_default_features
from src.hotspot_risk import estimate_hotspot_risk, generate_cell_temperature_grid
from app.ui_components import page_header, section_header, CHART_LAYOUT, metric_card, status_badge

def render(xgb_model, ae_model_artifacts, mlp_model, shap_explainer, metadata, dataset):
    page_header("What-If Simulator", "Explore Hypothetical Operating Scenarios")

    # ── Controls ───────────────────────────────────
    section_header("Adjust Operating Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        wi_avg_temp = st.slider("Avg Cell Temperature (°C)", 15.0, 60.0, 30.0, 0.5, key="wi_avg")
        wi_max_temp = st.slider("Max Cell Temperature (°C)", 15.0, 85.0, max(32.0, wi_avg_temp + 2), 0.5, key="wi_max")

    with col2:
        wi_power = st.slider("Charge Power (kW)", 0.0, 200.0, 50.0, 5.0, key="wi_power")
        wi_resistance = st.slider("Internal Resistance (Ω)", 0.5, 10.0, 1.5, 0.1, key="wi_res")

    with col3:
        wi_soc = st.slider("State of Charge (%)", 0.0, 100.0, 80.0, 5.0, key="wi_soc")
        wi_cooling = st.slider("Cooling Health (%)", 0.0, 100.0, 100.0, 5.0, key="wi_cool")

    run_whatif = st.button("Run What-If Simulation", use_container_width=True, key="run_whatif")

    st.markdown('<div style="margin-top:20px; margin-bottom:20px; border-bottom:1px solid var(--border);"></div>', unsafe_allow_html=True)

    # ── Baseline ───────────────────────────────────
    baseline = get_default_features()
    baseline_prob = predict_failure_probability(mlp_model, baseline)
    
    if baseline_prob > 0.8: base_risk = "CRITICAL"
    elif baseline_prob > 0.4: base_risk = "HIGH"
    elif baseline_prob > 0.15: base_risk = "CAUTION"
    else: base_risk = "NORMAL"
    
    baseline_hotspot = estimate_hotspot_risk(
        baseline["cell_temperature_avg"], baseline["cell_temperature_max"]
    )

    if run_whatif or st.session_state.get("wi_ran", False):
        st.session_state.wi_ran = True

        # What-if scenario
        whatif = get_default_features()
        whatif["cell_temperature_avg"] = wi_avg_temp
        whatif["cell_temperature_max"] = max(wi_avg_temp, wi_max_temp)
        whatif["average_charge_power_kw"] = wi_power
        whatif["internal_resistance"] = wi_resistance
        whatif["state_of_charge"] = wi_soc
        whatif["cooling_system_health"] = wi_cooling

        whatif_prob = predict_failure_probability(mlp_model, whatif)
        delta_prob = whatif_prob - baseline_prob
        
        if whatif_prob > 0.8: whatif_risk = "CRITICAL"
        elif whatif_prob > 0.4: whatif_risk = "HIGH"
        elif whatif_prob > 0.15: whatif_risk = "CAUTION"
        else: whatif_risk = "NORMAL"

        whatif_hotspot = estimate_hotspot_risk(
            whatif["cell_temperature_avg"], whatif["cell_temperature_max"]
        )

        # ── Parameter Comparison ───────────────────────
        section_header("Parameter Comparison")
        
        comp_data = [
            {"Parameter": "Avg Temp", "Baseline": f'{baseline["cell_temperature_avg"]:.1f} °C', "Scenario": f"{wi_avg_temp:.1f} °C", "Change": f"{wi_avg_temp - baseline['cell_temperature_avg']:+.1f} °C"},
            {"Parameter": "Max Temp", "Baseline": f'{baseline["cell_temperature_max"]:.1f} °C', "Scenario": f"{wi_max_temp:.1f} °C", "Change": f"{wi_max_temp - baseline['cell_temperature_max']:+.1f} °C"},
            {"Parameter": "Charge Power", "Baseline": f'{baseline["average_charge_power_kw"]:.1f} kW', "Scenario": f"{wi_power:.1f} kW", "Change": f"{wi_power - baseline['average_charge_power_kw']:+.1f} kW"},
            {"Parameter": "Resistance", "Baseline": f'{baseline["internal_resistance"]:.2f} Ω', "Scenario": f"{wi_resistance:.2f} Ω", "Change": f"{wi_resistance - baseline['internal_resistance']:+.2f} Ω"},
        ]
        
        df_comp = pd.DataFrame(comp_data)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        st.markdown('<div style="margin-top:20px; margin-bottom:20px; border-bottom:1px solid var(--border);"></div>', unsafe_allow_html=True)

        # ── Comparison ─────────────────────────────
        col_b, col_arrow, col_w = st.columns([2, 1, 2])

        with col_b:
            section_header("Baseline Prediction")
            metric_card("Failure Prob", f"{baseline_prob*100:.1f}", " %")
            st.markdown("<br/>", unsafe_allow_html=True)
            status_badge("Failure Risk", base_risk)
            status_badge("Hotspot Variance", "NORMAL" if baseline_hotspot["hotspot_risk_percent"] < 30 else "CAUTION")

        with col_arrow:
            delta_color = "#DC2626" if delta_prob > 0 else "#16A34A"
            delta_icon = "↑" if delta_prob > 0 else "↓"
            st.markdown(f'''
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; margin-top: 50px;">
                <div style="font-size: 24px;">{delta_icon}</div>
                <div style="font-family: var(--mono); font-weight: 700; font-size: 24px; color: {delta_color};">{delta_prob*100:+.1f} %</div>
                <div style="font-size: 11px; color: var(--text-2); text-transform: uppercase; letter-spacing: 0.05em;">IMPACT</div>
            </div>
            ''', unsafe_allow_html=True)

        with col_w:
            section_header("What-If Prediction")
            metric_card("Failure Prob", f"{whatif_prob*100:.1f}", " %")
            st.markdown("<br/>", unsafe_allow_html=True)
            status_badge("Failure Risk", whatif_risk)
            status_badge("Hotspot Variance", "NORMAL" if whatif_hotspot["hotspot_risk_percent"] < 30 else "CAUTION")

        # ── Battery Visualization Comparison ───────
        st.markdown('<div style="margin-top:20px; margin-bottom:20px; border-bottom:1px solid var(--border);"></div>', unsafe_allow_html=True)
        col_grid_b, col_grid_w = st.columns(2)

        with col_grid_b:
            section_header("Baseline Grid")
            grid_b = generate_cell_temperature_grid(baseline["cell_temperature_avg"], baseline["cell_temperature_max"], seed=42)
            _render_heatmap(grid_b)

        with col_grid_w:
            section_header("What-If Grid")
            grid_w = generate_cell_temperature_grid(whatif["cell_temperature_avg"], whatif["cell_temperature_max"], seed=42)
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
        zmin=15, zmax=85,
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
