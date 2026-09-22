"""
Digital Twin — Battery Visualization Page
===========================================
Simulated digital twin with 3×4 cell temperature grid,
temperature scale, and failure probability status.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.prediction_engine import predict_failure_probability, get_default_features
from src.hotspot_risk import estimate_hotspot_risk, generate_cell_temperature_grid
from app.ui_components import page_header, section_header, metric_card, status_badge, info_panel, CHART_LAYOUT

def render(xgb_model, ae_model_artifacts, mlp_model, metadata, dataset):
    page_header("Digital Twin", "Simulated Cell-Level Thermal Visualization")
    info_panel("SIMULATED DIGITAL TWIN — This is a visualization/model representation, not a physical hardware digital twin. Cell temperatures are simulated.")

    features = get_default_features()
    
    # We will use session state to track the sliders, but since they are at the top, we just define them.
    # Wait, they are supposed to be under Simulation Parameters.
    # But they define variables for the rest of the page.
    # We can just define the layout.
    
    # Actually, we need to declare the variables before computing, but the UI is rendered from top to bottom.
    # So we'll render the KPIs first with dummy or default data if not submitted?
    # Or just use the current values of the sliders. Since Streamlit is reactive, we can just put the sliders in the sidebar or just at the top of the main area, or we can use st.session_state!
    
    # To put KPIs at the top that depend on sliders below them, we either compute them early (by putting sliders early but visually late? not possible unless using empty placeholders).
    # Since Streamlit renders linearly, the easiest way to put sliders below KPIs is to use placeholders, OR just keep sliders above KPIs.
    # But the user asked for:
    # Top: KPI cards
    # Below: Simulation Parameters | Digital Twin Visualization
    
    placeholder_kpis = st.empty()
    
    st.markdown('<div style="margin-top:20px;"></div>', unsafe_allow_html=True)
    
    # ── Main Two-Column Layout ─────────────────────
    col_left, col_right = st.columns([1, 2])

    with col_left:
        with st.container(border=True):
            section_header("Simulation Parameters")
            avg_temp = st.slider("Avg Cell Temperature (°C)", 15.0, 60.0, 30.0, 0.5, key="dt_avg")
            max_temp = st.slider("Max Cell Temperature (°C)", 15.0, 85.0, max(32.0, avg_temp + 2), 0.5, key="dt_max")
            soc = st.slider("State of Charge (%)", 0.0, 100.0, 80.0, 1.0, key="dt_soc")
            
            features["cell_temperature_avg"] = avg_temp
            features["cell_temperature_max"] = max(avg_temp, max_temp)
            features["state_of_charge"] = soc

            predicted_prob = predict_failure_probability(mlp_model, features)
            hotspot = estimate_hotspot_risk(avg_temp, features["cell_temperature_max"])
            
            st.markdown("<br/>", unsafe_allow_html=True)
            status_badge("Hotspot Status", "NORMAL" if hotspot['hotspot_risk_percent'] < 30 else "CAUTION")

    with col_right:
        with st.container(border=True):
            section_header("Digital Twin Visualization (3×4)")
            cell_grid = generate_cell_temperature_grid(avg_temp, features["cell_temperature_max"], seed=42)

            # Create Plotly heatmap
            fig = go.Figure(data=go.Heatmap(
                z=cell_grid[::-1],
                text=[[f"{v:.1f}°C" for v in row] for row in cell_grid[::-1]],
                texttemplate="%{text}",
                textfont=dict(size=14, family="JetBrains Mono", color="#111827"),
                colorscale=[
                    [0.0, "#E0F2FE"],
                    [0.25, "#BAE6FD"],
                    [0.45, "#BBF7D0"],
                    [0.6, "#FEF08A"],
                    [0.8, "#FDBA74"],
                    [1.0, "#FCA5A5"],
                ],
                zmin=15, zmax=85,
                colorbar=dict(
                    title=dict(text="°C", font=dict(color="#6B7280", size=12)),
                    tickfont=dict(color="#6B7280", size=11),
                    thickness=10,
                    len=0.8,
                ),
                hovertemplate="Cell [%{y}, %{x}]: %{z:.1f}°C<extra></extra>",
                xgap=4, ygap=4,
            ))
            
            layout_opts = CHART_LAYOUT.copy()
            layout_opts.update(
                height=320,
                xaxis=dict(
                    showticklabels=True,
                    tickvals=[0, 1, 2, 3],
                    ticktext=["C1", "C2", "C3", "C4"],
                    tickfont=dict(color="#6B7280", family="Inter"),
                    showgrid=False
                ),
                yaxis=dict(
                    showticklabels=True,
                    tickvals=[0, 1, 2],
                    ticktext=["R3", "R2", "R1"],
                    tickfont=dict(color="#6B7280", family="Inter"),
                    showgrid=False
                ),
                margin=dict(l=20, r=20, t=10, b=20)
            )
            fig.update_layout(**layout_opts)
            st.plotly_chart(fig, use_container_width=True)

    # Now populate the KPIs placeholder at the top
    with placeholder_kpis.container():
        with st.container(border=True):
            m1, m2, m3 = st.columns(3)
            with m1:
                metric_card("Failure Probability", f"{predicted_prob * 100:.1f}", " %")
            with m2:
                metric_card("Thermal Delta (Max-Avg)", f"{features['cell_temperature_max'] - avg_temp:.1f}", " °C")
            with m3:
                metric_card("Hotspot Variance", f"{hotspot['hotspot_risk_percent']:.1f}", " %")
