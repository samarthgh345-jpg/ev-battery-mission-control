"""
Model Performance Page
========================
Display trained model metrics, charts, and SHAP summary.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.preprocessing import FEATURE_COLUMNS, get_feature_display_names
from src.shap_explainer import create_explainer, explain_global
from app.ui_components import page_header, section_header, CHART_LAYOUT, metric_card, info_panel, warning_panel

def render(xgb_model, metadata, dataset):
    page_header("Model Performance", "XGBoost Temperature Model — Training Results & Evaluation")

    if metadata is None:
        st.error("Model metadata not found. Please train the model first.")
        return

    metrics = metadata.get("metrics", {})

    # ── Model Info ─────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        metric_card("Model Type", metadata.get("model_type", "XGBoost"))
    with col2:
        metric_card("Dataset Rows", f"{metadata.get('n_total_samples', 'N/A'):,}")
    with col3:
        metric_card("Dataset Columns", "19")
    with col4:
        metric_card("Model Input Features", str(metadata.get("n_features", "N/A")))
    with col5:
        metric_card("Train / Test Split", f'{metadata.get("n_train_samples", "N/A"):,} / {metadata.get("n_test_samples", "N/A"):,}')

    st.markdown('<div style="margin-top:20px; margin-bottom:20px; border-bottom:1px solid var(--border);"></div>', unsafe_allow_html=True)

    # ── Metrics ────────────────────────────────────
    section_header("Evaluation Metrics (Test Set)")

    col_mae, col_rmse, col_r2 = st.columns(3)
    with col_mae:
        metric_card("MAE (Mean Absolute Error)", f"{metrics.get('MAE', 'N/A')}", " °C")
    with col_rmse:
        metric_card("RMSE (Root Mean Square Error)", f"{metrics.get('RMSE', 'N/A')}", " °C")
    with col_r2:
        metric_card("R² (Coefficient of Determination)", f"{metrics.get('R2', 'N/A')}")

    st.markdown('<div style="margin-top:20px; margin-bottom:20px; border-bottom:1px solid var(--border);"></div>', unsafe_allow_html=True)

    # ── Charts ─────────────────────────────────────
    tab_avp, tab_residual, tab_fi, tab_shap = st.tabs([
        "Actual vs Predicted", "Residuals", "Feature Importance", "SHAP Summary"
    ])

    test_actual = metadata.get("test_actual", [])
    test_predicted = metadata.get("test_predicted", [])

    with tab_avp:
        if test_actual and test_predicted:
            n = min(500, len(test_actual))
            idx = np.random.RandomState(42).choice(len(test_actual), n, replace=False)
            actual = [test_actual[i] for i in idx]
            predicted = [test_predicted[i] for i in idx]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=actual, y=predicted, mode="markers",
                marker=dict(size=6, color="#2563EB", opacity=0.7),
                name="Predictions",
            ))
            min_val = min(min(actual), min(predicted))
            max_val = max(max(actual), max(predicted))
            fig.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode="lines",
                line=dict(color="#DC2626", width=2, dash="dash"),
                name="Perfect Prediction",
            ))
            layout_opts = CHART_LAYOUT.copy()
            layout_opts.update(
                title=dict(text="Actual vs Predicted Temperature", font=dict(color="#111827", size=14)),
                height=450,
                margin=dict(l=50, r=20, t=50, b=50),
                xaxis=dict(title="Actual (°C)"),
                yaxis=dict(title="Predicted (°C)"),
                legend=dict(font=dict(color="#6B7280"), yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h"),
            )
            fig.update_layout(**layout_opts)
            st.plotly_chart(fig, use_container_width=True)
            
            warning_panel("Note: Evaluation metrics shown are for the XGBoost predictive model.")
        else:
            st.warning("Test data not available in metadata.")

    with tab_residual:
        if test_actual and test_predicted:
            residuals = [a - p for a, p in zip(test_actual, test_predicted)]
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=residuals, nbinsx=50,
                marker_color="#2563EB",
                opacity=0.9,
            ))
            layout_opts = CHART_LAYOUT.copy()
            layout_opts.update(
                title=dict(text="Residual Distribution", font=dict(color="#111827", size=14)),
                height=400,
                margin=dict(l=50, r=20, t=50, b=50),
                xaxis=dict(title="Residual (Actual − Predicted) °C"),
                yaxis=dict(title="Count"),
            )
            fig.update_layout(**layout_opts)
            st.plotly_chart(fig, use_container_width=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                metric_card("Residual Mean", f"{np.mean(residuals):.4f}", " °C")
            with col2:
                metric_card("Residual Std", f"{np.std(residuals):.4f}", " °C")
            with col3:
                metric_card("Residual Median", f"{np.median(residuals):.4f}", " °C")

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
                height=500,
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
                with st.spinner("Computing global SHAP values over subset..."):
                    sample = dataset[FEATURE_COLUMNS].sample(min(300, len(dataset)), random_state=42)
                    explainer = create_explainer(xgb_model)
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
                        title=dict(text="SHAP Feature Importance (mean |SHAP|)", font=dict(color="#111827", size=14)),
                        height=500,
                        margin=dict(l=200, r=80, t=50, b=40),
                        xaxis=dict(title="Mean |SHAP Value|", zeroline=True, zerolinecolor="#E5E7EB", showgrid=False),
                        yaxis=dict(showgrid=False),
                        showlegend=False,
                    )
                    fig.update_layout(**layout_opts)
                    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    info_panel("Model trained on synthetic/simulation-inspired data. Metrics reflect model performance on synthetic data, not experimentally validated predictions.")
