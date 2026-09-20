"""
VAE Generator Page
====================
Demonstrate VAE reconstruction limits. Input vector → Latent space → Reconstructed vector.
"""

import streamlit as st
import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
import plotly.graph_objects as go
from src.preprocessing import FEATURE_COLUMNS, get_feature_display_names
from app.ui_components import page_header, section_header, CHART_LAYOUT, info_panel, metric_card, render_html

PROJECT_ROOT = Path(__file__).resolve().parent.parent

@st.cache_resource
def load_vae():
    try:
        from src.models.vae import VAE
        vae_path = PROJECT_ROOT / "models" / "vae" / "vae.pt"
        scaler_path = PROJECT_ROOT / "models" / "vae" / "scaler.pkl"
        
        if not vae_path.exists() or not scaler_path.exists():
            return None, None
            
        scaler = joblib.load(scaler_path)
        
        model = VAE(input_dim=14, latent_dim=4)
        model.load_state_dict(torch.load(vae_path, weights_only=True))
        model.eval()
        
        return model, scaler
    except Exception as e:
        print(f"Error loading VAE: {e}")
        return None, None

def render():
    page_header("Variational Autoencoder (VAE)", "Latent Space Compression & Signal Reconstruction")
    
    vae_model, scaler = load_vae()
    
    if vae_model is None:
        st.error("VAE models not found. Please run: `python src/train_vae.py`")
        return
        
    info_panel("This tool demonstrates compression of the 14-dimensional telemetry vector into a 4-dimensional latent space, and evaluates the reconstruction accuracy.")
    
    # ── Controls ───────────────────────────────────
    st.markdown('<div style="margin-top:20px; margin-bottom:20px; border-bottom:1px solid var(--border);"></div>', unsafe_allow_html=True)
    
    section_header("Input Signal Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        current = st.slider("Battery Current (A)", 1.0, 15.0, 8.5, 0.5)
        ambient = st.slider("Ambient Temperature (°C)", 20.0, 45.0, 32.0, 0.5)
        discharge = st.slider("Discharge Rate (C)", 0.5, 5.0, 2.0, 0.1)
    with col2:
        coolant_flow = st.slider("Coolant Flow (kg/s)", 0.005, 0.040, 0.015, 0.001)
        coolant_inlet = st.slider("Coolant Inlet Temp (°C)", 15.0, 40.0, 25.0, 0.5)
        soc = st.slider("State of Charge (%)", 10.0, 100.0, 60.0, 5.0)

    # Recreate the default features exactly as the model expects
    from src.prediction_engine import get_default_features
    features = get_default_features()
    features["battery_current_A"] = current
    features["ambient_temperature_C"] = ambient
    features["battery_temperature_C"] = ambient + 6
    features["coolant_flow_rate_kg_s"] = coolant_flow
    features["coolant_inlet_temperature_C"] = coolant_inlet
    features["discharge_rate_C"] = discharge
    features["state_of_charge_percent"] = soc

    input_vector = np.array([[features[col] for col in FEATURE_COLUMNS]])

    st.markdown('<div style="margin-top:20px; margin-bottom:20px; border-bottom:1px solid var(--border);"></div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    
    with c1:
        if st.button("Encode & Reconstruct", use_container_width=True):
            st.session_state.vae_run = True
            
            # Perform inference
            with torch.no_grad():
                scaled_input = scaler.transform(input_vector)
                tensor_input = torch.FloatTensor(scaled_input)
                
                # Encode
                mu, logvar = vae_model.encode(tensor_input)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std
                
                # Decode
                recon_scaled = vae_model.decode(z)
                recon_output = scaler.inverse_transform(recon_scaled.numpy())
                
                st.session_state.vae_results = {
                    "input": input_vector[0],
                    "scaled_input": scaled_input[0],
                    "z": z.numpy()[0],
                    "recon": recon_output[0],
                    "recon_scaled": recon_scaled.numpy()[0]
                }
                
    if st.session_state.get("vae_run", False):
        res = st.session_state.vae_results
        
        # ── Results ────────────────────────────────────
        section_header("Latent Space Representation (4D)")
        
        z = res["z"]
        html_z = "<div style='display:flex; justify-content:space-around; margin: 20px 0;'>"
        for i, val in enumerate(z):
            color = "#2563EB" if val >= 0 else "#DC2626"
            html_z += f"<div style='text-align:center; padding:15px; border:1px solid var(--border); border-radius:8px; width:22%; background:var(--surface);'><div style='font-size:11px; color:var(--text-2);'>DIMENSION {i+1}</div><div style='font-family:var(--mono); font-size:20px; font-weight:700; color:{color};'>{val:+.3f}</div></div>"
        html_z += "</div>"
        render_html(html_z)

        section_header("Reconstruction Error")
        
        display_names = get_feature_display_names()
        names = [display_names.get(f, f) for f in FEATURE_COLUMNS]
        
        fig = go.Figure()
        
        # Actual values
        fig.add_trace(go.Scatter(
            x=names,
            y=res["input"],
            mode='markers+lines',
            name='Original Input',
            marker=dict(color="#111827", size=8),
            line=dict(width=1)
        ))
        
        # Reconstructed
        fig.add_trace(go.Scatter(
            x=names,
            y=res["recon"],
            mode='markers+lines',
            name='Reconstructed',
            marker=dict(color="#10B981", size=8),
            line=dict(width=1, dash="dash")
        ))
        
        layout_opts = CHART_LAYOUT.copy()
        layout_opts.update(
            height=400,
            margin=dict(l=40, r=40, t=30, b=120),
            xaxis=dict(showgrid=False, tickangle=45),
            yaxis=dict(title="Standardized Value", showgrid=True, type="log"),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        fig.update_layout(**layout_opts)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Error metrics
        norm_mse = np.mean((res["scaled_input"] - res["recon_scaled"])**2)
        norm_mae = np.mean(np.abs(res["scaled_input"] - res["recon_scaled"]))
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            metric_card("Normalized MSE", f"{norm_mse:.4f}")
        with col_m2:
            metric_card("Normalized MAE", f"{norm_mae:.4f}")
            
        raw_mse = np.mean((res["input"] - res["recon"])**2)
        raw_mae = np.mean(np.abs(res["input"] - res["recon"]))
        
        st.markdown("<div style='font-size:12px; color:var(--text-2); margin-top:10px;'>Raw-scale MSE (which can be very large due to features like pressure): " + f"{raw_mse:.2f}" + ", Raw-scale MAE: " + f"{raw_mae:.2f}</div>", unsafe_allow_html=True)
