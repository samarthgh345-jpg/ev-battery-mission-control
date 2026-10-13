"""
What-If Simulator — Live Parameter Explanation & Scenario Comparison
====================================================================
Real-time, parameter-specific thermal and battery health explanations
as sliders adjust, paired with on-demand multi-model scenario comparisons.
Clean, modern, spacious, engineering-focused design.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from src.prediction_engine import get_default_features
from src.hotspot_risk import generate_cell_temperature_grid
from src.what_if_explainer import (
    get_live_parameter_explanations,
    compute_simulation_comparison,
)
from app.ui_components import page_header, CHART_LAYOUT
from app.styles import clean_html


def render(xgb_model, ae_model_artifacts, mlp_model, shap_explainer, metadata, dataset):
    page_header(
        title="What-If Simulator",
        subtitle="Simulate and compare hypothetical thermal and operating scenarios against nominal baseline",
        badge_label="SCENARIO LAB",
    )

    baseline = get_default_features()

    # ── Initialize session state slider keys if not present ────
    if "wi_slider_avg" not in st.session_state:
        st.session_state.wi_slider_avg = float(baseline["cell_temperature_avg"])
    if "wi_slider_max" not in st.session_state:
        st.session_state.wi_slider_max = float(baseline["cell_temperature_max"])
    if "wi_slider_power" not in st.session_state:
        st.session_state.wi_slider_power = float(baseline["average_charge_power_kw"])
    if "wi_slider_res" not in st.session_state:
        st.session_state.wi_slider_res = float(baseline["internal_resistance"])
    if "wi_slider_cool" not in st.session_state:
        st.session_state.wi_slider_cool = float(baseline["cooling_system_health"])
    if "wi_slider_soc" not in st.session_state:
        st.session_state.wi_slider_soc = float(baseline["state_of_charge"])
    if "wi_has_run" not in st.session_state:
        st.session_state.wi_has_run = False

    # ── Operating Condition Sliders Card ───────────────────────
    st.markdown(
        clean_html(
            """
            <div class="white-card" style="margin-bottom: 12px;">
                <div class="white-card-title">Operating conditions &amp; thermal controls</div>
                <div class="white-card-subtitle">Adjust simulated electrical and thermal operating variables</div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    s_col1, s_col2 = st.columns(2)
    with s_col1:
        wi_avg_temp = st.slider(
            "Average Cell Temperature (°C)",
            min_value=15.0,
            max_value=60.0,
            value=float(st.session_state.get("wi_slider_avg", baseline["cell_temperature_avg"])),
            step=0.5,
            key="wi_slider_avg",
            help="Baseline nominal value: 21.0 °C",
        )
        wi_max_temp = st.slider(
            "Maximum Hotspot Temperature (°C)",
            min_value=15.0,
            max_value=85.0,
            value=float(st.session_state.get("wi_slider_max", baseline["cell_temperature_max"])),
            step=0.5,
            key="wi_slider_max",
            help="Must be ≥ Average Cell Temperature. Baseline nominal value: 28.5 °C",
        )
        wi_power = st.slider(
            "Average Charge Power (kW)",
            min_value=0.0,
            max_value=200.0,
            value=float(st.session_state.get("wi_slider_power", baseline["average_charge_power_kw"])),
            step=5.0,
            key="wi_slider_power",
            help="Baseline nominal value: 40.0 kW",
        )

    with s_col2:
        wi_resistance = st.slider(
            "Internal Resistance (mΩ)",
            min_value=0.2,
            max_value=6.0,
            value=float(st.session_state.get("wi_slider_res", baseline["internal_resistance"])),
            step=0.05,
            key="wi_slider_res",
            help="Baseline nominal value: 0.53 mΩ",
        )
        wi_cooling = st.slider(
            "Cooling System Health (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state.get("wi_slider_cool", baseline["cooling_system_health"])),
            step=1.0,
            key="wi_slider_cool",
            help="Baseline nominal value: 74.0 %",
        )
        wi_soc = st.slider(
            "State of Charge (%)",
            min_value=5.0,
            max_value=100.0,
            value=float(st.session_state.get("wi_slider_soc", baseline["state_of_charge"])),
            step=1.0,
            key="wi_slider_soc",
            help="Baseline nominal value: 55.0 %",
        )

    # Hotspot must physically be at least average temperature
    eff_max_temp = max(wi_avg_temp, wi_max_temp)

    current_vals = {
        "cell_temperature_avg": wi_avg_temp,
        "cell_temperature_max": eff_max_temp,
        "average_charge_power_kw": wi_power,
        "internal_resistance": wi_resistance,
        "cooling_system_health": wi_cooling,
        "state_of_charge": wi_soc,
    }

    # ── Live Parameter Explanation Panel ───────────────────────
    # Evaluated immediately on every rerun / slider adjustment
    live_explanations = get_live_parameter_explanations(baseline, current_vals)

    if not live_explanations:
        st.markdown(
            clean_html(
                """
                <div class="white-card" style="margin-top: 4px; margin-bottom: 12px;">
                    <div class="white-card-header">
                        <div>
                            <div class="white-card-title">Live Parameter Explanation</div>
                            <div class="white-card-subtitle">Real-time thermal mechanism breakdown as sliders adjust</div>
                        </div>
                        <span style="background: #F0FDF4; color: #16866A; font-weight: 700; font-size: 11px; padding: 3px 8px; border-radius: 4px;">
                            NOMINAL BASELINE
                        </span>
                    </div>
                    <div style="font-size: 13px; color: #475569; margin-top: 8px; line-height: 1.5;">
                        All operating parameters match the calibrated baseline. Adjust any slider above to immediately see a concise explanation of how that parameter affects battery physics and thermal risk.
                    </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
    else:
        n_changed = len(live_explanations)
        badge_txt = f"{n_changed} PARAMETER{'S' if n_changed > 1 else ''} CHANGED"

        items_html = []
        for item in live_explanations:
            is_inc = item["direction"] == "INCREASE"
            is_cooling = item["param_key"] == "cooling_system_health"
            is_riskier = (is_inc and not is_cooling) or (not is_inc and is_cooling)
            badge_color = "#D94841" if is_riskier else "#16866A"
            badge_bg = "#FEF2F2" if is_riskier else "#F0FDF4"

            items_html.append(
                f"""
                <div style="background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 8px; padding: 12px 14px; margin-top: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; border-bottom: 1px solid #EEF2F6; padding-bottom: 6px;">
                        <div style="font-size: 13px; font-weight: 700; color: #172033;">
                            <span style="color: #64748B; font-weight: 600;">Parameter changed:</span> {item['param_name']}
                        </div>
                        <span style="font-size: 12px; font-weight: 700; font-family: 'JetBrains Mono', monospace; color: {badge_color}; background: {badge_bg}; padding: 3px 8px; border-radius: 4px;">
                            {item['change_summary']}
                        </span>
                    </div>
                    <div style="font-size: 12px; color: #334155; line-height: 1.55; display: flex; flex-direction: column; gap: 5px;">
                        <div><b style="color: #172033;">What this means:</b> {item['what_this_means']}</div>
                        <div><b style="color: #172033;">Why it matters:</b> {item['why_it_matters']}</div>
                        <div><b style="color: #172033;">Expected effect:</b> {item['expected_effect']}</div>
                        <div><b style="color: #172033;">Suggested adjustment:</b> {item['suggested_adjustment']}</div>
                    </div>
                </div>
                """
            )

        all_items_str = "\n".join(items_html)

        st.markdown(
            clean_html(
                f"""
                <div class="white-card" style="margin-top: 4px; margin-bottom: 12px;">
                    <div class="white-card-header">
                        <div>
                            <div class="white-card-title">Live Parameter Explanation</div>
                            <div class="white-card-subtitle">Immediate explanation for {n_changed} changed parameter(s)</div>
                        </div>
                        <span style="background: #EFF6FF; color: #1D4ED8; font-weight: 700; font-size: 11px; padding: 3px 8px; border-radius: 4px;">
                            {badge_txt}
                        </span>
                    </div>
                    {all_items_str}
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    # ── Action Buttons ─────────────────────────────────────────
    b_col1, b_col2 = st.columns([1, 1])
    with b_col1:
        if st.button("Run Simulation & Compare", type="primary", use_container_width=True, key="btn_run_wi"):
            st.session_state.wi_has_run = True

    with b_col2:
        if st.button("Reset to Baseline", use_container_width=True, key="btn_reset_wi"):
            st.session_state.wi_slider_avg = float(baseline["cell_temperature_avg"])
            st.session_state.wi_slider_max = float(baseline["cell_temperature_max"])
            st.session_state.wi_slider_power = float(baseline["average_charge_power_kw"])
            st.session_state.wi_slider_res = float(baseline["internal_resistance"])
            st.session_state.wi_slider_cool = float(baseline["cooling_system_health"])
            st.session_state.wi_slider_soc = float(baseline["state_of_charge"])
            st.session_state.wi_has_run = False
            st.rerun()

    # ── Simulation Comparison Results ──────────────────────────
    # Displayed after the user clicks "Run Simulation & Compare"
    if st.session_state.get("wi_has_run", False):
        scenario = baseline.copy()
        scenario.update(current_vals)

        comp = compute_simulation_comparison(
            baseline=baseline,
            scenario=scenario,
            mlp_model=mlp_model,
            xgb_model=xgb_model,
            ae_artifacts=ae_model_artifacts,
        )

        mlp_delta = comp["mlp_delta_pp"]
        delta_color = "#D94841" if mlp_delta > 0.5 else "#16866A" if mlp_delta < -0.5 else "#667085"
        delta_sign = "+" if mlp_delta > 0 else ""

        # Optional XGBoost comparison line if model is loaded
        xgb_sub = ""
        if comp["base_xgb_pct"] is not None and comp["scen_xgb_pct"] is not None:
            xgb_sign = "+" if comp["xgb_delta_pp"] > 0 else ""
            xgb_sub = (
                f'<div style="font-size: 11px; color: #64748B; margin-top: 8px; border-top: 1px solid #F1F5F9; padding-top: 6px;">'
                f'<b>XGBoost Predictor:</b> Baseline {comp["base_xgb_pct"]:.2f}% → Scenario {comp["scen_xgb_pct"]:.2f}% '
                f'({xgb_sign}{comp["xgb_delta_pp"]:.2f} pp)'
                f'</div>'
            )

        st.markdown(
            clean_html(
                f"""
                <div class="white-card" style="margin-top: 12px; margin-bottom: 12px;">
                    <div class="white-card-header">
                        <div>
                            <div class="white-card-title">Simulation Comparison Results</div>
                            <div class="white-card-subtitle">Calculated model predictions comparing baseline against hypothetical scenario</div>
                        </div>
                        <span style="background: #EFF6FF; color: #1D4ED8; font-weight: 700; font-size: 11px; padding: 3px 8px; border-radius: 4px;">
                            MODEL ESTIMATE
                        </span>
                    </div>

                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 12px;">
                        <div style="background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 6px; padding: 10px 12px;">
                            <div style="font-size: 10px; font-weight: 700; color: #64748B; text-transform: uppercase;">BASELINE FAILURE PROB (MLP)</div>
                            <div style="font-size: 22px; font-weight: 800; color: #16866A; font-family: 'JetBrains Mono', monospace; margin-top: 2px;">
                                {comp['base_mlp_pct']:.2f}%
                            </div>
                        </div>
                        <div style="background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 6px; padding: 10px 12px;">
                            <div style="font-size: 10px; font-weight: 700; color: #64748B; text-transform: uppercase;">SCENARIO FAILURE PROB (MLP)</div>
                            <div style="font-size: 22px; font-weight: 800; color: {delta_color}; font-family: 'JetBrains Mono', monospace; margin-top: 2px;">
                                {comp['scen_mlp_pct']:.2f}%
                            </div>
                        </div>
                        <div style="background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 6px; padding: 10px 12px;">
                            <div style="font-size: 10px; font-weight: 700; color: #64748B; text-transform: uppercase;">DIFFERENCE (DELTA)</div>
                            <div style="font-size: 22px; font-weight: 800; color: {delta_color}; font-family: 'JetBrains Mono', monospace; margin-top: 2px;">
                                {delta_sign}{comp['mlp_delta_pp']:.2f} pp
                            </div>
                        </div>
                    </div>

                    {xgb_sub}

                    <div style="margin-top: 12px; background: #F8FAFC; border-left: 3px solid #3B82F6; padding: 8px 12px; border-radius: 0 6px 6px 0;">
                        <div style="font-size: 12px; color: #1E293B; line-height: 1.45;">
                            <b>Calculated outcome:</b> {comp['result_explanation']}
                        </div>
                    </div>

                    <div style="margin-top: 8px; background: #F8FAFC; border-left: 3px solid #10B981; padding: 8px 12px; border-radius: 0 6px 6px 0;">
                        <div style="font-size: 12px; color: #1E293B; line-height: 1.45;">
                            <b>Recommended next adjustment:</b> {comp['recommended_action']}
                        </div>
                    </div>

                    <div style="margin-top: 10px; font-size: 11px; color: #64748B; font-style: italic;">
                        {comp['disclaimer']}
                    </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    # ── Thermal Grid Comparison (Side by Side) ─────────────────
    st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
    g_col1, g_col2 = st.columns(2)

    with g_col1:
        st.markdown(
            clean_html(
                f"""
                <div class="white-card" style="margin-bottom: 0;">
                    <div class="white-card-title">Baseline thermal grid (nominal)</div>
                    <div class="white-card-subtitle">Illustrative 3×4 module map · Mean: {baseline['cell_temperature_avg']:.1f}°C · Peak: {baseline['cell_temperature_max']:.1f}°C</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
        grid_base = generate_cell_temperature_grid(
            baseline["cell_temperature_avg"],
            baseline["cell_temperature_max"],
            rows=3,
            cols=4,
            seed=42,
        )
        _render_heatmap(grid_base, key="heatmap_wi_baseline")

    with g_col2:
        st.markdown(
            clean_html(
                f"""
                <div class="white-card" style="margin-bottom: 0;">
                    <div class="white-card-title">What-If thermal grid (hypothetical)</div>
                    <div class="white-card-subtitle">Illustrative 3×4 module map · Mean: {current_vals['cell_temperature_avg']:.1f}°C · Peak: {current_vals['cell_temperature_max']:.1f}°C</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
        grid_scen = generate_cell_temperature_grid(
            current_vals["cell_temperature_avg"],
            current_vals["cell_temperature_max"],
            rows=3,
            cols=4,
            seed=42,
        )
        _render_heatmap(grid_scen, key="heatmap_wi_scenario")

    # ── Illustrative Disclaimer ───────────────────────────────────
    st.markdown(
        clean_html(
            """
            <div style="font-size: 11px; color: #64748B; font-style: italic; margin-top: 6px; padding: 4px 6px;">
                <b>Visualization Note:</b> These 3×4 heatmaps are illustrative synthetic visualizations generated from slider inputs (center hotspot at peak temperature, balanced to pack average). They are illustrative representations and not the full 24-cell lumped-capacitance ODE simulation, which runs on Mission Control.
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def _render_heatmap(grid, key):
    fig = go.Figure(
        data=go.Heatmap(
            z=grid[::-1],
            text=[[f"{v:.1f}°C" for v in row] for row in grid[::-1]],
            texttemplate="%{text}",
            textfont=dict(size=12, family="JetBrains Mono", color="#FFFFFF"),
            colorscale=[
                [0.0, "#065F46"],
                [0.3, "#0D9488"],
                [0.55, "#326CE5"],
                [0.75, "#C7771B"],
                [1.0, "#D94841"],
            ],
            zmin=15,
            zmax=80,
            showscale=False,
            xgap=3,
            ygap=3,
        )
    )
    layout = CHART_LAYOUT.copy()
    layout.update(
        height=200,
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True, key=key)
