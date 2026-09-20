"""
Dataset Explorer Page
======================
Browse, filter, and download the synthetic BTMS dataset.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN, get_feature_display_names
from app.ui_components import page_header, section_header, CHART_LAYOUT, metric_card, info_panel

def render(dataset):
    page_header("Dataset Explorer", "Synthetic BTMS Dataset — Browse, filter, and download")

    if dataset is None:
        st.error("Dataset not found. Please run: `python scripts/generate_dataset.py`")
        return

    # ── Dataset Info ───────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        metric_card("Dataset", "Synthetic BTMS")
    with col2:
        metric_card("Rows", f"{len(dataset):,}")
    with col3:
        metric_card("Dataset Columns", f"{dataset.shape[1]}")
    with col4:
        metric_card("Model Input Features", "14")
    with col5:
        metric_card("Target", "max_temp")

    st.markdown("<br/>", unsafe_allow_html=True)
    info_panel("Synthetic / Simulation-Inspired Dataset — created for demonstration and educational purposes. This is NOT experimentally validated battery sensor data.")

    st.markdown('<div style="margin-top:20px; margin-bottom:20px; border-bottom:1px solid var(--border);"></div>', unsafe_allow_html=True)

    # ── Tabs ───────────────────────────────────────
    tab_preview, tab_stats, tab_dist, tab_risk = st.tabs([
        "Preview", "Statistics", "Distributions", "Risk Analysis"
    ])

    with tab_preview:
        section_header("Data Preview")

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            risk_filter = st.multiselect(
                "Filter by Risk Label",
                ["NORMAL", "CAUTION", "HIGH", "CRITICAL"],
                default=["NORMAL", "CAUTION", "HIGH", "CRITICAL"],
                key="de_risk_filter",
            )
        with col_f2:
            n_rows = st.slider("Number of rows to display", 10, 500, 100, 10, key="de_nrows")

        filtered = dataset[dataset["thermal_risk_label"].isin(risk_filter)]
        st.dataframe(filtered.head(n_rows), use_container_width=True, height=400)

        csv = dataset.to_csv(index=False)
        st.download_button(
            "Download Full Dataset (CSV)",
            csv,
            "btms_dataset.csv",
            "text/csv",
        )

    with tab_stats:
        section_header("Descriptive Statistics")
        st.dataframe(dataset.describe().round(3), use_container_width=True)

        section_header("Missing Values")
        missing = dataset.isnull().sum()
        if missing.sum() == 0:
            st.success("No missing values in the dataset.")
        else:
            st.dataframe(missing[missing > 0])

    with tab_dist:
        section_header("Feature Distributions")
        display_names = get_feature_display_names()
        all_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
        selected_feature = st.selectbox(
            "Select feature",
            all_cols,
            format_func=lambda x: display_names.get(x, x),
            key="de_feature",
        )

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=dataset[selected_feature],
            nbinsx=50,
            marker_color="#2563EB",
            opacity=0.9,
        ))
        layout_opts = CHART_LAYOUT.copy()
        layout_opts.update(
            title=dict(text=f"Distribution: {display_names.get(selected_feature, selected_feature)}", font=dict(color="#111827", size=14)),
            height=350,
            margin=dict(l=50, r=20, t=50, b=50),
            xaxis=dict(title=selected_feature, showgrid=False),
            yaxis=dict(title="Count"),
        )
        fig.update_layout(**layout_opts)
        st.plotly_chart(fig, use_container_width=True)

        section_header("Feature vs Target")
        fig2 = go.Figure()
        n_sample = min(1000, len(dataset))
        sample = dataset.sample(n_sample, random_state=42)
        fig2.add_trace(go.Scatter(
            x=sample[selected_feature],
            y=sample[TARGET_COLUMN],
            mode="markers",
            marker=dict(size=5, color="#2563EB", opacity=0.6),
        ))
        layout_opts2 = CHART_LAYOUT.copy()
        layout_opts2.update(
            title=dict(text=f"{display_names.get(selected_feature, selected_feature)} vs Max Battery Temperature", font=dict(color="#111827", size=14)),
            height=350,
            margin=dict(l=50, r=20, t=50, b=50),
            xaxis=dict(title=selected_feature, showgrid=False),
            yaxis=dict(title="Max Battery Temp (°C)", showgrid=False),
        )
        fig2.update_layout(**layout_opts2)
        st.plotly_chart(fig2, use_container_width=True)

    with tab_risk:
        section_header("Risk Distribution")

        risk_counts = dataset["thermal_risk_label"].value_counts()
        risk_colors = {"NORMAL": "#16A34A", "CAUTION": "#D97706", "HIGH": "#D97706", "CRITICAL": "#DC2626"}

        fig = go.Figure()
        for label in ["NORMAL", "CAUTION", "HIGH", "CRITICAL"]:
            count = risk_counts.get(label, 0)
            fig.add_trace(go.Bar(
                x=[label], y=[count],
                marker_color=risk_colors.get(label, "#6B7280"),
                text=[f"{count:,}"],
                textposition="outside",
                textfont=dict(family="JetBrains Mono", size=12, color="#111827"),
                name=label,
            ))
        layout_opts3 = CHART_LAYOUT.copy()
        layout_opts3.update(
            title=dict(text="Thermal Risk Label Distribution", font=dict(color="#111827", size=14)),
            height=350,
            margin=dict(l=50, r=20, t=50, b=50),
            xaxis=dict(showgrid=False),
            yaxis=dict(title="Count"),
            showlegend=False,
        )
        fig.update_layout(**layout_opts3)
        st.plotly_chart(fig, use_container_width=True)

        section_header("Temperature by Risk Level")
        fig2 = go.Figure()
        for label in ["NORMAL", "CAUTION", "HIGH", "CRITICAL"]:
            subset = dataset[dataset["thermal_risk_label"] == label]
            fig2.add_trace(go.Box(
                y=subset[TARGET_COLUMN],
                name=label,
                marker_color=risk_colors.get(label, "#6B7280"),
            ))
        layout_opts4 = CHART_LAYOUT.copy()
        layout_opts4.update(
            title=dict(text="Max Battery Temperature by Risk Level", font=dict(color="#111827", size=14)),
            height=350,
            margin=dict(l=50, r=20, t=50, b=50),
            yaxis=dict(title="Max Battery Temp (°C)", showgrid=False),
            xaxis=dict(showgrid=False),
            showlegend=False,
        )
        fig2.update_layout(**layout_opts4)
        st.plotly_chart(fig2, use_container_width=True)
