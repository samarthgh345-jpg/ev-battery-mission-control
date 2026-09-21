"""
Model Performance Page
========================
Display trained failure classifier metrics, charts, and SHAP summary.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.preprocessing import FEATURE_COLUMNS, get_feature_display_names
from src.shap_explainer import create_explainer, explain_global
from app.ui_components import page_header, section_header, CHART_LAYOUT, metric_card, info_panel, warning_panel

def render(xgb_model, mlp_model, metadata, dataset):
    page_header("Model Performance", "Battery Failure Models — Training Results & Evaluation")

    if metadata is None:
        st.error("Model metadata not found. Please train the model first.")
        return

    metrics = metadata.get("metrics", {})

    # ── Model Info ─────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Model Type", metadata.get("model_type", "XGBoost"))
    with col2:
        metric_card("Dataset Rows", f"{metadata.get('n_total_samples', 'N/A'):,}")
    with col3:
        metric_card("Model Input Features", str(metadata.get("n_features", "N/A")))
    with col4:
        metric_card("Train / Test Split", f'{metadata.get("n_train_samples", "N/A"):,} / {metadata.get("n_test_samples", "N/A"):,}')

    st.markdown('<div style="margin-top:20px; margin-bottom:20px; border-bottom:1px solid var(--border);"></div>', unsafe_allow_html=True)

    # ── Metrics ────────────────────────────────────
    section_header("Evaluation Metrics (Test Set)")

    col_acc, col_f1, col_roc, col_pr = st.columns(4)
    with col_acc:
        metric_card("Accuracy", f"{metrics.get('Accuracy', 0) * 100:.1f}", " %")
    with col_f1:
        metric_card("F1 Score", f"{metrics.get('F1_Score', 0):.3f}")
    with col_roc:
        metric_card("ROC-AUC", f"{metrics.get('ROC_AUC', 0):.3f}")
    with col_pr:
        metric_card("PR-AUC", f"{metrics.get('PR_AUC', 0):.3f}")

    st.markdown('<div style="margin-top:20px; margin-bottom:20px; border-bottom:1px solid var(--border);"></div>', unsafe_allow_html=True)

    # ── Charts ─────────────────────────────────────
    tab_fi, tab_shap = st.tabs([
        "Feature Importance (XGBoost)", "SHAP Summary (MLP)"
    ])

    with tab_fi:
        if hasattr(xgb_model, "feature_importances_"):
            importances = xgb_model.feature_importances_
            display_names = get_feature_display_names()
            names = [display_names.get(f, f) for f in FEATURE_COLUMNS]

            sorted_idx = np.argsort(importances)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=[names[i] for i in sorted_idx],
                x=[importances[i] for i in sorted_idx],
                orientation="h",
                marker_color="#2563EB",
                text=[f"{importances[i]:.3f}" for i in sorted_idx],
                textposition="outside",
                textfont=dict(family="JetBrains Mono", size=11, color="#111827"),
                cliponaxis=False,
            ))
            layout_opts = CHART_LAYOUT.copy()
            layout_opts.update(
                title=dict(text="XGBoost Feature Importance (Gain)", font=dict(color="#111827", size=14)),
                height=600,
                margin=dict(l=200, r=80, t=50, b=40),
                xaxis=dict(title="Importance", zeroline=True, zerolinecolor="#E5E7EB", showgrid=False),
                yaxis=dict(showgrid=False),
                showlegend=False,
            )
            fig.update_layout(**layout_opts)
            st.plotly_chart(fig, use_container_width=True)

    with tab_shap:
        if dataset is not None:
            if st.button("Compute Global SHAP Summary"):
                with st.spinner("Computing global SHAP values for MLP over subset..."):
                    # For performance, only use a small subset
                    sample = dataset[FEATURE_COLUMNS].sample(min(150, len(dataset)), random_state=42)
                    bg = dataset[FEATURE_COLUMNS].sample(min(100, len(dataset)), random_state=123)
                    
                    explainer = create_explainer(mlp_model, bg)
                    global_exp = explain_global(explainer, sample)

                    sorted_idx = np.argsort(global_exp["importance"])
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        y=[global_exp["display_names"][i] for i in sorted_idx],
                        x=[global_exp["importance"][i] for i in sorted_idx],
                        orientation="h",
                        marker_color="#10B981",
                        text=[f"{global_exp['importance'][i]:.3f}" for i in sorted_idx],
                        textposition="outside",
                        textfont=dict(family="JetBrains Mono", size=11, color="#111827"),
                        cliponaxis=False,
                    ))
                    layout_opts = CHART_LAYOUT.copy()
                    layout_opts.update(
                        title=dict(text="PyTorch MLP SHAP Feature Importance (mean |SHAP|)", font=dict(color="#111827", size=14)),
                        height=600,
                        margin=dict(l=200, r=80, t=50, b=40),
                        xaxis=dict(title="Mean |SHAP Value|", zeroline=True, zerolinecolor="#E5E7EB", showgrid=False),
                        yaxis=dict(showgrid=False),
                        showlegend=False,
                    )
                    fig.update_layout(**layout_opts)
                    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    info_panel("Models trained on authoritative EV battery failure data. Metrics reflect test-set evaluation.")
