"""
Digital Twin — Battery Visualization Page
===========================================
Simulated digital twin with 3×4 cell temperature grid,
temperature scale, and thermal status.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.prediction_engine import predict_temperature, get_default_features
from src.hotspot_risk import estimate_hotspot_risk, generate_cell_temperature_grid
from app.ui_components import page_header, section_header, metric_card, status_badge, info_panel, CHART_LAYOUT

def render(xgb_model, iso_model, metadata, dataset):
    page_header("Digital Twin", "Simulated Cell-Level Thermal Visualization")
    info_panel("SIMULATED DIGITAL TWIN — This is a visualization/model representation, not a physical hardware digital twin. Cell temperatures are simulated.")

    # ── Controls ───────────────────────────────────
    st.markdown('<div style="margin-top:20px; margin-bottom:20px; border-bottom:1px solid var(--border);"></div>', unsafe_allow_html=True)
    
    section_header("Operating Conditions")
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    with col_ctrl1:
        current = st.slider("Battery Current (A)", 1.0, 15.0, 8.0, 0.5, key="dt_current")
    with col_ctrl2:
        ambient = st.slider("Ambient Temperature (°C)", 20.0, 45.0, 30.0, 0.5, key="dt_ambient")
    with col_ctrl3:
        coolant_flow = st.slider("Coolant Flow (kg/s)", 0.005, 0.040, 0.018, 0.001, key="dt_flow")

    features = get_default_features()
    features["battery_current_A"] = current
    features["ambient_temperature_C"] = ambient
    features["battery_temperature_C"] = ambient + 6
    features["coolant_flow_rate_kg_s"] = coolant_flow
    features["discharge_rate_C"] = current / 4

    predicted_temp = predict_temperature(xgb_model, features)

    hotspot = estimate_hotspot_risk(
        predicted_temp, current, features["discharge_rate_C"], coolant_flow, ambient
    )

    st.markdown('<div style="margin-top:20px; margin-bottom:20px; border-bottom:1px solid var(--border);"></div>', unsafe_allow_html=True)

    # ── Battery Visualization ──────────────────────
    col_grid, col_info = st.columns([2, 1])

    with col_grid:
        section_header("Battery Pack — 12 Cells (3×4)")
        cell_grid = generate_cell_temperature_grid(predicted_temp, hotspot["hotspot_risk_percent"], seed=42)

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
            zmin=28, zmax=55,
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

    with col_info:
        section_header("Thermal Output")
        
        metric_card("Pack Max Temp", f"{predicted_temp:.1f}", " °C")
        st.markdown("<br/>", unsafe_allow_html=True)
        metric_card("Hotspot Risk", f"{hotspot['hotspot_risk_percent']:.0f}", " %")
        st.markdown("<br/>", unsafe_allow_html=True)
        
        # Determine status colors
        if predicted_temp < 40:
            status_badge("Thermal State", "NORMAL")
        elif predicted_temp < 45:
            status_badge("Thermal State", "CAUTION")
        else:
            status_badge("Thermal State", "CRITICAL")
            
        if hotspot["hotspot_risk_percent"] < 30:
            status_badge("Cell Variance", "NORMAL")
        elif hotspot["hotspot_risk_percent"] < 60:
            status_badge("Cell Variance", "CAUTION")
        else:
            status_badge("Cell Variance", "CRITICAL")
