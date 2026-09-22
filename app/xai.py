"""
Explainability Page
=================================
Global and local SHAP/LIME explanations for the MLP classifier model.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.prediction_engine import predict_failure_probability, get_default_features
from src.shap_explainer import explain_global, explain_local
from src.lime_explainer import create_lime_explainer
from app.ui_components import page_header, section_header, CHART_LAYOUT, metric_card

def render(xgb_model, mlp_model, shap_explainer, metadata, dataset):
    page_header("Model Explainability", "SHAP/LIME Feature Contributions & Model Transparency")

    tab1, tab2, tab3 = st.tabs(["Local Explanation (SHAP)", "Local Explanation (LIME)", "Global Importance (SHAP)"])

    # ── Local Explanation (SHAP) ──────────────────────────
    with tab1:
        with st.container(border=True):
            section_header("Adjust Operating Conditions")
            col1, col2, col3 = st.columns(3)
    
            with col1:
                wi_avg_temp = st.slider("Avg Cell Temperature (°C)", 15.0, 60.0, 30.0, 0.5, key="xai_avg")
                wi_max_temp = st.slider("Max Cell Temperature (°C)", 15.0, 85.0, max(32.0, wi_avg_temp + 2), 0.5, key="xai_max")
    
            with col2:
                wi_power = st.slider("Charge Power (kW)", 0.0, 200.0, 50.0, 5.0, key="xai_power")
                wi_resistance = st.slider("Internal Resistance (Ω)", 0.5, 10.0, 1.5, 0.1, key="xai_res")
    
            with col3:
                wi_soc = st.slider("State of Charge (%)", 0.0, 100.0, 80.0, 5.0, key="xai_soc")
                wi_cooling = st.slider("Cooling Health (%)", 0.0, 100.0, 100.0, 5.0, key="xai_cool")
    
            features = get_default_features()
            features["cell_temperature_avg"] = wi_avg_temp
            features["cell_temperature_max"] = max(wi_avg_temp, wi_max_temp)
            features["average_charge_power_kw"] = wi_power
            features["internal_resistance"] = wi_resistance
            features["state_of_charge"] = wi_soc
            features["cooling_system_health"] = wi_cooling
    
            predicted_prob = predict_failure_probability(mlp_model, features)
        
        c1, c2 = st.columns([1, 2])
        with c1:
            metric_card("Failure Prob", f"{predicted_prob*100:.1f}", " %")
            
            st.markdown("<br/>", unsafe_allow_html=True)
            if st.button("Generate SHAP Explanation", type="primary", use_container_width=True, key="xai_explain"):
                st.session_state.xai_run = True
            
        with c2:
            if st.session_state.get("xai_run", False):
                with st.spinner("Computing SHAP values..."):
                    exp = explain_local(shap_explainer, mlp_model, features)

                st.markdown(f'<div style="font-size:13px; color:var(--text-2); margin-bottom:12px;">Base value: <b>{exp["base_value_logit"]:.4f}</b> (logit) → Predicted: <b>{exp["predicted_prob"]:.4f}</b> (prob)</div>', unsafe_allow_html=True)

                contribs = exp["contributions"]
                fig = go.Figure()

                names = [c["display_name"] for c in contribs]
                values = [c["shap_value"] for c in contribs]
                colors = ["#DC2626" if v > 0 else "#16A34A" for v in values]

                fig.add_trace(go.Bar(
                    y=names[::-1],
                    x=values[::-1],
                    orientation="h",
                    marker_color=colors[::-1],
                    text=[f"{v:+.3f}" for v in values[::-1]],
                    textposition="outside",
                    textfont=dict(family="JetBrains Mono", size=12, color="#111827"),
                    cliponaxis=False,
                ))
                layout_opts = CHART_LAYOUT.copy()
                layout_opts.update(
                    height=500,
                    margin=dict(l=200, r=80, t=20, b=40),
                    xaxis=dict(title="SHAP Value (impact on probability)", zeroline=True, zerolinecolor="#E5E7EB", showgrid=False),
                    yaxis=dict(showgrid=False),
                    showlegend=False,
                )
                fig.update_layout(**layout_opts)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown(f'<div style="font-size:12px; color:var(--text-2);">{exp["explanation_text"]}</div>', unsafe_allow_html=True)

    # ── Local Explanation (LIME) ──────────────────────────
    with tab2:
        with st.container(border=True):
            section_header("LIME Local Surrogate Explanation")
            st.markdown("<div style='font-size:13px; color:var(--text-2); margin-bottom:12px;'>LIME approximates the model locally using a linear surrogate.</div>", unsafe_allow_html=True)
            
            if st.button("Generate LIME Explanation", type="primary", key="lime_btn"):
                with st.spinner("Training LIME surrogate..."):
                    lime_exp = create_lime_explainer(mlp_model)
                    if lime_exp is None:
                        st.error("Failed to create LIME explainer. Check that the dataset file exists.")
                    else:
                        lime_result = lime_exp.explain_prediction(features, num_features=10)
                        
                        lime_html = "<div style='display:flex; flex-direction:column; gap:8px; margin-top:20px;'>"
                        for contrib in lime_result["contributions"]:
                            condition = contrib["condition"]
                            weight = contrib["weight"]
                            color = "#DC2626" if weight > 0 else "#16A34A"
                            bar_w = min(100, abs(weight) * 300)
                            
                            lime_html += f"<div style='display:flex; align-items:center; background-color:var(--surface); border:1px solid var(--border); padding:10px; border-radius:6px; margin-bottom:8px;'>"
                            lime_html += f"<div style='width:300px; font-family:var(--font); font-size:13px; font-weight:500;'>{condition}</div>"
                            lime_html += f"<div style='width:100px; font-family:var(--mono); font-size:13px; color:{color}; font-weight:700; text-align:right;'>{weight:+.3f}</div>"
                            lime_html += f"<div style='flex-grow:1; margin-left:15px; height:8px; background-color:var(--border-light); border-radius:4px; overflow:hidden;'>"
                            lime_html += f"<div style='width:{bar_w}%; height:100%; background-color:{color};'></div>"
                            lime_html += "</div></div>"
                        lime_html += "</div>"
                        st.markdown(lime_html, unsafe_allow_html=True)

    # ── Global Importance (SHAP) ──────────────────────────
    with tab3:
        with st.container(border=True):
            section_header("Global Feature Importance")
            if st.button("Compute Global Importance (Slow)", type="primary"):
                with st.spinner("Computing global SHAP values over subset..."):
                    sample = dataset.sample(min(150, len(dataset)), random_state=42)
                    global_exp = explain_global(shap_explainer, sample)
                    
                    st.markdown('<div style="margin-top:20px; margin-bottom:20px; border-bottom:1px solid var(--border);"></div>', unsafe_allow_html=True)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        y=global_exp["display_names"][::-1],
                        x=global_exp["importance"][::-1],
                        orientation="h",
                        marker_color="#2563EB",
                        text=[f"{v:.3f}" for v in global_exp["importance"][::-1]],
                        textposition="outside",
                        textfont=dict(family="JetBrains Mono", size=12, color="#111827"),
                    cliponaxis=False,
                ))
                layout_opts = CHART_LAYOUT.copy()
                layout_opts.update(
                    height=500,
                    margin=dict(l=200, r=40, t=20, b=40),
                    xaxis=dict(title="Mean |SHAP Value| (average impact on prediction)", zeroline=True, zerolinecolor="#E5E7EB", showgrid=False),
                    yaxis=dict(showgrid=False),
                    showlegend=False,
                )
                fig.update_layout(**layout_opts)
                st.plotly_chart(fig, use_container_width=True)
