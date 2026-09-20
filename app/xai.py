"""
Explainability Page
=================================
Global and local SHAP/LIME explanations for the XGBoost model.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.prediction_engine import predict_temperature, get_default_features
from src.shap_explainer import explain_global, explain_local
from src.lime_explainer import create_lime_explainer
from app.ui_components import page_header, section_header, CHART_LAYOUT, metric_card

def render(xgb_model, shap_explainer, metadata, dataset):
    page_header("Model Explainability", "SHAP/LIME Feature Contributions & Model Transparency")

    tab1, tab2, tab3 = st.tabs(["Local Explanation (SHAP)", "Local Explanation (LIME)", "Global Importance (SHAP)"])

    # ── Local Explanation (SHAP) ──────────────────────────
    with tab1:
        section_header("Adjust Operating Conditions")
        col1, col2 = st.columns(2)
        with col1:
            current = st.slider("Battery Current (A)", 1.0, 15.0, 8.5, 0.5, key="xai_current")
            ambient = st.slider("Ambient Temperature (°C)", 20.0, 45.0, 32.0, 0.5, key="xai_ambient")
            discharge = st.slider("Discharge Rate (C)", 0.5, 5.0, 2.0, 0.1, key="xai_discharge")
        with col2:
            coolant_flow = st.slider("Coolant Flow (kg/s)", 0.005, 0.040, 0.015, 0.001, key="xai_flow")
            coolant_inlet = st.slider("Coolant Inlet Temp (°C)", 15.0, 40.0, 25.0, 0.5, key="xai_inlet")
            soc = st.slider("State of Charge (%)", 10.0, 100.0, 60.0, 5.0, key="xai_soc")

        features = get_default_features()
        features["battery_current_A"] = current
        features["ambient_temperature_C"] = ambient
        features["battery_temperature_C"] = ambient + 6
        features["coolant_flow_rate_kg_s"] = coolant_flow
        features["coolant_inlet_temperature_C"] = coolant_inlet
        features["discharge_rate_C"] = discharge
        features["state_of_charge_percent"] = soc

        predicted_temp = predict_temperature(xgb_model, features)

        st.markdown('<div style="margin-top:20px; margin-bottom:20px; border-bottom:1px solid var(--border);"></div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns([1, 2])
        with c1:
            metric_card("Predicted Temp", f"{predicted_temp:.1f}", " °C")
            
            st.markdown("<br/>", unsafe_allow_html=True)
            if st.button("Generate SHAP Explanation", use_container_width=True, key="xai_explain"):
                st.session_state.xai_run = True
            
        with c2:
            if st.session_state.get("xai_run", False):
                with st.spinner("Computing SHAP values..."):
                    exp = explain_local(shap_explainer, features)

                st.markdown(f'<div style="font-size:13px; color:var(--text-2); margin-bottom:12px;">Base value: <b>{exp["base_value"]:.2f}°C</b> → Predicted: <b>{exp["predicted_value"]:.2f}°C</b></div>', unsafe_allow_html=True)

                contribs = exp["contributions"] # Show all features, do not truncate
                fig = go.Figure()

                names = [c["display_name"] for c in contribs]
                values = [c["shap_value"] for c in contribs]
                colors = ["#DC2626" if v > 0 else "#16A34A" for v in values]

                fig.add_trace(go.Bar(
                    y=names[::-1],
                    x=values[::-1],
                    orientation="h",
                    marker_color=colors[::-1],
                    text=[f"{v:+.2f}°C" for v in values[::-1]],
                    textposition="outside",
                    textfont=dict(family="JetBrains Mono", size=12, color="#111827"),
                    cliponaxis=False,
                ))
                layout_opts = CHART_LAYOUT.copy()
                layout_opts.update(
                    height=400,
                    margin=dict(l=200, r=80, t=20, b=40),
                    xaxis=dict(title="SHAP Value (°C impact)", zeroline=True, zerolinecolor="#E5E7EB", showgrid=False),
                    yaxis=dict(showgrid=False),
                    showlegend=False,
                )
                fig.update_layout(**layout_opts)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown(f'<div style="font-size:12px; color:var(--text-2);">{exp["explanation_text"]}</div>', unsafe_allow_html=True)

    # ── Local Explanation (LIME) ──────────────────────────
    with tab2:
        section_header("LIME Local Surrogate Explanation")
        st.markdown("<div style='font-size:13px; color:var(--text-2); margin-bottom:12px;'>LIME approximates the model locally using a linear surrogate.</div>", unsafe_allow_html=True)
        
        if st.button("Generate LIME Explanation", key="lime_btn"):
            with st.spinner("Training LIME surrogate..."):
                lime_exp = create_lime_explainer(xgb_model)
                if lime_exp is None:
                    st.error("Failed to create LIME explainer. Check that the dataset file exists.")
                else:
                    lime_result = lime_exp.explain_prediction(features, num_features=14)
                    
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
        section_header("Global Feature Importance")
        if st.button("Compute Global Importance (Slow)"):
            with st.spinner("Computing global SHAP values over subset..."):
                global_exp = explain_global(shap_explainer, dataset)
                
                st.markdown('<div style="margin-top:20px; margin-bottom:20px; border-bottom:1px solid var(--border);"></div>', unsafe_allow_html=True)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=global_exp["display_names"][::-1],
                    x=global_exp["importance"][::-1],
                    orientation="h",
                    marker_color="#2563EB",
                    text=[f"{v:.2f}" for v in global_exp["importance"][::-1]],
                    textposition="outside",
                    textfont=dict(family="JetBrains Mono", size=12, color="#111827"),
                    cliponaxis=False,
                ))
                layout_opts = CHART_LAYOUT.copy()
                layout_opts.update(
                    height=450,
                    margin=dict(l=200, r=40, t=20, b=40),
                    xaxis=dict(title="Mean |SHAP Value| (average impact on prediction)", zeroline=True, zerolinecolor="#E5E7EB", showgrid=False),
                    yaxis=dict(showgrid=False),
                    showlegend=False,
                )
                fig.update_layout(**layout_opts)
                st.plotly_chart(fig, use_container_width=True)
