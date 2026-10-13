"""
XAI — Explainable AI & Feature Attribution
===========================================
Local and global feature attribution for the PyTorch Failure Classifier
using SHAP (SHapley Additive exPlanations) DeepExplainer.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.prediction_engine import predict_failure_probability, get_default_features
from src.shap_explainer import explain_local
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


def render(xgb_model, mlp_model, shap_explainer, metadata, dataset):
    page_header(
        title="Model Explainability (XAI)",
        subtitle="SHAP Feature Attributions & Decision Transparency for Battery Safety",
        badge_label="EXPLAINABILITY LAB",
    )

    baseline = get_default_features()

    # ── Operating Condition Sliders ────────────────────────────
    with st.container():
        st.markdown(
            clean_html(
                """
                <div class="mission-card">
                    <div class="card-header-row">
                        <div class="card-title">🎛️ Configure Operating Condition for Explanation</div>
                        <span style="font-size: 10px; color: #94A3B8;">FEATURE VECTOR INPUTS</span>
                    </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        x_col1, x_col2, x_col3 = st.columns(3)

        with x_col1:
            x_avg = st.slider("Avg Cell Temp (°C)", 15.0, 60.0, float(st.session_state.get("xai_avg", 30.0)), 0.5, key="xai_slider_avg")
            x_max = st.slider("Max Cell Temp (°C)", 15.0, 85.0, float(max(x_avg, st.session_state.get("xai_max", 35.0))), 0.5, key="xai_slider_max")

        with x_col2:
            x_power = st.slider("Charge Power (kW)", 0.0, 200.0, float(st.session_state.get("xai_power", 50.0)), 5.0, key="xai_slider_power")
            x_res = st.slider("Internal Resistance (mΩ)", 0.2, 5.0, float(st.session_state.get("xai_res", 1.2)), 0.05, key="xai_slider_res")

        with x_col3:
            x_soc = st.slider("State of Charge (%)", 5.0, 100.0, float(st.session_state.get("xai_soc", 80.0)), 1.0, key="xai_slider_soc")
            x_cool = st.slider("Cooling Health (%)", 0.0, 100.0, float(st.session_state.get("xai_cool", 70.0)), 1.0, key="xai_slider_cool")

    # Form feature dictionary
    features = baseline.copy()
    features["cell_temperature_avg"] = x_avg
    features["cell_temperature_max"] = max(x_avg, x_max)
    features["average_charge_power_kw"] = x_power
    features["internal_resistance"] = x_res
    features["state_of_charge"] = x_soc
    features["cooling_system_health"] = x_cool

    # Compute current predictions
    prob_mlp = predict_failure_probability(mlp_model, features)
    prob_xgb = predict_failure_probability(xgb_model, features) if xgb_model is not None else prob_mlp

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # ── Prediction Summary & Trigger Button ────────────────────
    c_left, c_right = st.columns([1, 2])

    with c_left:
        st.markdown(
            clean_html(
                f"""
                <div class="mission-card">
                    <div class="card-header-row">
                        <div class="card-title">📈 Current Prediction</div>
                    </div>
                    <div style="margin-top: 4px;">
                        <div style="font-size: 10px; color: #94A3B8; text-transform: uppercase;">PYTORCH MLP FAILURE PROB</div>
                        <div style="font-size: 28px; font-weight: 800; color: {'#34D399' if prob_mlp < 0.15 else '#FBBF24' if prob_mlp < 0.4 else '#F87171'}; font-family: 'JetBrains Mono', monospace;">
                            {prob_mlp * 100:.1f}%
                        </div>
                    </div>
                    <div style="margin-top: 8px; font-size: 11px; color: #94A3B8; border-top: 1px solid #162646; padding-top: 6px;">
                        XGBoost Comparison: <b style="color: #FFFFFF;">{prob_xgb * 100:.1f}%</b>
                    </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        if st.button("🧠 GENERATE SHAP EXPLANATION", type="primary", use_container_width=True, key="btn_trigger_shap"):
            st.session_state.xai_computed = True

    with c_right:
        if st.session_state.get("xai_computed", True):
            if shap_explainer is not None:
                with st.spinner("Computing Shapley values across background reference..."):
                    try:
                        exp = explain_local(shap_explainer, mlp_model, features)
                    except Exception as e:
                        st.error(f"Error computing SHAP values: {e}")
                        exp = None

                if exp is not None:
                    st.markdown(
                        clean_html(
                            f"""
                            <div class="mission-card">
                                <div class="card-header-row">
                                    <div class="card-title">⚖️ SHAP Local Feature Attribution</div>
                                    <span style="font-size: 10px; color: #38BDF8; font-family: monospace;">
                                        Base Logit: {exp['base_value_logit']:.2f} → Pred: {exp['predicted_prob']*100:.1f}%
                                    </span>
                                </div>
                            </div>
                            """
                        ),
                        unsafe_allow_html=True,
                    )

                    contribs = exp["contributions"]
                    names = [c["display_name"] for c in contribs]
                    values = [c["shap_value"] for c in contribs]
                    colors = ["#EF4444" if v > 0 else "#10B981" for v in values]

                    fig = go.Figure()
                    fig.add_trace(
                        go.Bar(
                            y=names[::-1],
                            x=values[::-1],
                            orientation="h",
                            marker_color=colors[::-1],
                            text=[f"{v:+.3f}" for v in values[::-1]],
                            textposition="outside",
                            textfont=dict(family="JetBrains Mono", size=10, color="#CBD5E1"),
                            cliponaxis=False,
                        )
                    )

                    exp_layout = CHART_LAYOUT.copy()
                    exp_layout.update(
                        height=440,
                        margin=dict(l=170, r=40, t=10, b=30),
                        xaxis=dict(
                            title="SHAP Value on Model Logit (Red = Increases Risk, Green = Decreases Risk)",
                            zeroline=True,
                            zerolinecolor="#1C2E52",
                            showgrid=True,
                            gridcolor="#162646",
                        ),
                        yaxis=dict(showgrid=False),
                        showlegend=False,
                    )
                    fig.update_layout(**exp_layout)
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown(
                        clean_html(
                            f"""
                            <div style="background: #0E1A34; border: 1px solid #1C2E52; border-radius: 8px; padding: 12px; margin-top: -6px;">
                                <div style="font-size: 11px; font-weight: 700; color: #38BDF8; margin-bottom: 4px;">KEY FACTOR SYNTHESIS:</div>
                                <div style="font-size: 12px; color: #CBD5E1; line-height: 1.5;">{exp['explanation_text']}</div>
                            </div>
                            """
                        ),
                        unsafe_allow_html=True,
                    )
            else:
                info_panel("SHAP explainer initializing. Please click 'Generate SHAP Explanation' to compute attributions.")
        else:
            info_panel("Adjust the parameters on the left and click **Generate SHAP Explanation** to see feature contributions.")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    info_panel(
        "**Methodological Attribution Disclaimer:**\n\n"
        "• SHAP (SHapley Additive exPlanations) values represent the marginal contribution of each input feature toward shifting the model's output logit from the baseline expectation.\n"
        "• Positive SHAP values (Red) push the prediction toward failure. Negative SHAP values (Green) pull the prediction toward normal safety.\n"
        "• These values reflect statistical associations within the trained model, **not physical proof of causation** in real battery cells."
    )
