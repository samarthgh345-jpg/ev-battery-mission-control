"""
Dataset Explorer — EV Battery Failure Data Intelligence
=========================================================
Inspect, filter, analyze distributions, and export the authoritative
200,000-sample EV Battery Failure Prediction dataset.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN, get_feature_display_names
from app.ui_components import (
    CHART_LAYOUT,
    page_header,
    section_header,
    metric_card,
    status_badge,
    info_panel,
    warning_panel,
)
from app.styles import clean_html


def render(dataset):
    page_header(
        title="Dataset Explorer",
        subtitle="200K EV Battery Failure Telemetry Dataset — Schema, Distributions & Exploratory Analysis",
        badge_label="DATA INTELLIGENCE",
    )

    if dataset is None:
        st.error("⚠️ Dataset not found at `data/ev_battery_failure_dataset.csv`. Please ensure the dataset file exists.")
        return

    n_rows = len(dataset)
    n_cols = dataset.shape[1]
    fail_count = int(dataset[TARGET_COLUMN].sum())
    healthy_count = n_rows - fail_count
    fail_rate = (fail_count / n_rows) * 100

    # ── Dataset Summary KPI Row ────────────────────────────────
    s1, s2, s3, s4, s5 = st.columns(5)

    with s1:
        metric_card("Total Samples", f"{n_rows:,}", "", "Total physical telemetry rows", color="#38BDF8")
    with s2:
        metric_card("Total Features", f"{n_cols}", " cols", "All metadata and sensor telemetry fields", color="#38BDF8")
    with s3:
        metric_card("Model Inputs", f"{len(FEATURE_COLUMNS)}", " features", "Feature vector used for MLP and XGBoost", color="#34D399")
    with s4:
        metric_card("Target Variable", TARGET_COLUMN, "", "Binary classification target", color="#FBBF24")
    with s5:
        metric_card("Failure Incidence", f"{fail_rate:.1f}", "%", f"{fail_count:,} failures vs {healthy_count:,} healthy", color="#F87171" if fail_rate > 20 else "#FB923C")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # ── Tabs Navigation ────────────────────────────────────────
    tab_preview, tab_stats, tab_dist, tab_target = st.tabs([
        "📋 Data Preview & Filter",
        "📊 Descriptive Statistics",
        "📈 Feature Distributions",
        "🎯 Failure Class Analysis",
    ])

    display_names = get_feature_display_names()

    # ── Tab 1: Preview & Filter ────────────────────────────────
    with tab_preview:
        with st.container():
            f_col1, f_col2, f_col3 = st.columns([1.2, 1, 1.2])

            with f_col1:
                label_filter = st.multiselect(
                    "Filter by Target Label (battery_failure)",
                    options=[0, 1],
                    default=[0, 1],
                    format_func=lambda x: "0 — Healthy Battery" if x == 0 else "1 — Failure / Fault",
                    key="de_label_filter",
                )
            with f_col2:
                row_limit = st.slider("Rows to Preview", min_value=10, max_value=500, value=50, step=10, key="de_row_limit")
            with f_col3:
                cols_to_show = st.selectbox(
                    "Columns View",
                    ["14 Model Features + Target", "All 70 Dataset Columns"],
                    key="de_cols_view",
                )

            selected_cols = (FEATURE_COLUMNS + [TARGET_COLUMN]) if cols_to_show == "14 Model Features + Target" else dataset.columns.tolist()

            filtered_df = dataset[dataset[TARGET_COLUMN].isin(label_filter)][selected_cols].head(row_limit)
            st.dataframe(filtered_df, use_container_width=True, height=380)

            d_col1, d_col2 = st.columns([2, 1])
            with d_col1:
                st.caption(f"Showing {len(filtered_df):,} rows from filtered subset of {len(dataset[dataset[TARGET_COLUMN].isin(label_filter)]):,} total matching rows.")
            with d_col2:
                csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Previewed Subset (CSV)",
                    data=csv_bytes,
                    file_name="ev_battery_filtered_subset.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    # ── Tab 2: Descriptive Statistics ──────────────────────────
    with tab_stats:
        with st.container():
            section_header("Descriptive Statistics (14 Model Features)")
            stats_df = dataset[FEATURE_COLUMNS].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
            stats_df.index = [display_names.get(idx, idx) for idx in stats_df.index]
            st.dataframe(stats_df.round(3), use_container_width=True, height=350)

            section_header("Missing Values Audit")
            missing_series = dataset[FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().sum()
            if missing_series.sum() == 0:
                st.success("✅ Clean dataset: 0 missing values detected across all 14 model input features and target column.")
            else:
                st.dataframe(missing_series[missing_series > 0])

    # ── Tab 3: Feature Distributions ───────────────────────────
    with tab_dist:
        section_header("Feature Distribution & Target Stratification")

        d_col1, d_col2 = st.columns([1.5, 2.5])
        with d_col1:
            selected_feature = st.selectbox(
                "Select Feature to Analyze",
                FEATURE_COLUMNS,
                format_func=lambda f: f"{display_names.get(f, f)} ({f})",
                key="de_feature_select",
            )
            feat_mean = float(dataset[selected_feature].mean())
            feat_std = float(dataset[selected_feature].std())
            feat_median = float(dataset[selected_feature].median())
            feat_min = float(dataset[selected_feature].min())
            feat_max = float(dataset[selected_feature].max())

            st.markdown(
                clean_html(
                    f"""
                    <div style="background: #0E1A34; border: 1px solid #1C2E52; border-radius: 8px; padding: 12px; margin-top: 10px; font-size: 11px;">
                        <div style="font-weight: 700; color: #38BDF8; font-size: 13px; margin-bottom: 6px;">{display_names.get(selected_feature, selected_feature)}</div>
                        <div style="display: flex; justify-content: space-between; padding: 3px 0; border-bottom: 1px solid #162646;">
                            <span style="color: #94A3B8;">Mean:</span>
                            <span style="font-family: monospace; color: #FFFFFF;">{feat_mean:.3f}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 3px 0; border-bottom: 1px solid #162646;">
                            <span style="color: #94A3B8;">Std Dev:</span>
                            <span style="font-family: monospace; color: #FFFFFF;">{feat_std:.3f}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 3px 0; border-bottom: 1px solid #162646;">
                            <span style="color: #94A3B8;">Median:</span>
                            <span style="font-family: monospace; color: #FFFFFF;">{feat_median:.3f}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 3px 0;">
                            <span style="color: #94A3B8;">Min / Max:</span>
                            <span style="font-family: monospace; color: #FFFFFF;">{feat_min:.2f} / {feat_max:.2f}</span>
                        </div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )

        with d_col2:
            fig_hist = go.Figure()
            # Stratified histogram by class
            h_healthy = dataset[dataset[TARGET_COLUMN] == 0][selected_feature]
            h_failed = dataset[dataset[TARGET_COLUMN] == 1][selected_feature]

            fig_hist.add_trace(
                go.Histogram(
                    x=h_healthy,
                    name="Healthy (Class 0)",
                    marker_color="#10B981",
                    opacity=0.65,
                    nbinsx=40,
                )
            )
            fig_hist.add_trace(
                go.Histogram(
                    x=h_failed,
                    name="Failure (Class 1)",
                    marker_color="#EF4444",
                    opacity=0.65,
                    nbinsx=40,
                )
            )

            hist_layout = CHART_LAYOUT.copy()
            hist_layout.update(
                barmode="overlay",
                height=320,
                margin=dict(l=40, r=20, t=20, b=35),
                title=dict(text=f"Distribution: {display_names.get(selected_feature, selected_feature)}", font=dict(color="#CBD5E1", size=12)),
                xaxis=dict(title=display_names.get(selected_feature, selected_feature), showgrid=True, gridcolor="#162646"),
                yaxis=dict(title="Sample Count", showgrid=True, gridcolor="#162646"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#CBD5E1", size=10)),
            )
            fig_hist.update_layout(**hist_layout)
            st.plotly_chart(fig_hist, use_container_width=True)

    # ── Tab 4: Target Class Distribution ───────────────────────
    with tab_target:
        section_header("Target Imbalance & Class Breakdown")

        t_c1, t_c2 = st.columns([1.5, 2.5])

        with t_c1:
            st.markdown(
                clean_html(
                    f"""
                    <div style="background: #0E1A34; border: 1px solid #1C2E52; border-radius: 8px; padding: 14px;">
                        <div style="font-size: 13px; font-weight: 700; color: #FFFFFF; margin-bottom: 8px;">Target Class Distribution</div>
                        <div style="margin-bottom: 8px; padding: 8px; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 6px;">
                            <div style="color: #34D399; font-weight: 700; font-size: 12px;">Class 0 — Healthy Operation</div>
                            <div style="font-family: monospace; font-size: 18px; font-weight: 800; color: #FFFFFF; margin-top: 2px;">
                                {healthy_count:,} <span style="font-size: 11px; color: #94A3B8;">({(healthy_count/n_rows)*100:.2f}%)</span>
                            </div>
                        </div>
                        <div style="padding: 8px; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 6px;">
                            <div style="color: #F87171; font-weight: 700; font-size: 12px;">Class 1 — Battery Failure</div>
                            <div style="font-family: monospace; font-size: 18px; font-weight: 800; color: #FFFFFF; margin-top: 2px;">
                                {fail_count:,} <span style="font-size: 11px; color: #94A3B8;">({(fail_count/n_rows)*100:.2f}%)</span>
                            </div>
                        </div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )
            info_panel(
                "**Class Imbalance Impact:** The dataset exhibits a realistic 9:1 imbalance ratio typical of automotive warranty reliability distributions. "
                "Models were trained with stratified splitting to ensure consistent failure incidence across train and test sets."
            )

        with t_c2:
            fig_bar = go.Figure()
            fig_bar.add_trace(
                go.Bar(
                    x=["Healthy (Class 0)", "Failure (Class 1)"],
                    y=[healthy_count, fail_count],
                    marker_color=["#10B981", "#EF4444"],
                    text=[f"{healthy_count:,}<br>({(healthy_count/n_rows)*100:.1f}%)", f"{fail_count:,}<br>({(fail_count/n_rows)*100:.1f}%)"],
                    textposition="outside",
                    textfont=dict(family="JetBrains Mono", size=11, color="#CBD5E1"),
                )
            )
            bar_layout = CHART_LAYOUT.copy()
            bar_layout.update(
                height=300,
                margin=dict(l=40, r=20, t=20, b=35),
                xaxis=dict(showgrid=False),
                yaxis=dict(title="Sample Count", showgrid=True, gridcolor="#162646", range=[0, 210000]),
                showlegend=False,
            )
            fig_bar.update_layout(**bar_layout)
            st.plotly_chart(fig_bar, use_container_width=True)
