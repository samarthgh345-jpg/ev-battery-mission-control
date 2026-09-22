"""
GenAI Generator Page
====================
Demonstrate generating synthetic battery telemetry using a Conditional GAN (cGAN).
"""

import streamlit as st
import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
import plotly.graph_objects as go
from src.preprocessing import FEATURE_COLUMNS, get_feature_display_names
from app.ui_components import page_header, section_header, CHART_LAYOUT, info_panel, metric_card

PROJECT_ROOT = Path(__file__).resolve().parent.parent

@st.cache_resource(show_spinner=False)
def load_cgan_model():
    try:
        import sys
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from src.models.cgan import Generator
        gen_path = PROJECT_ROOT / "models" / "cgan" / "generator.pt"
        scaler_path = PROJECT_ROOT / "models" / "cgan" / "scaler.pkl"
        
        if not gen_path.exists() or not scaler_path.exists():
            st.error(f"gen_path exists: {gen_path.exists()} ({gen_path}), scaler_path exists: {scaler_path.exists()} ({scaler_path})")
            return None, None
            
        scaler = joblib.load(scaler_path)
        
        # Generator init: noise_dim=16, num_classes=2, feature_dim=14
        generator = Generator(16, 2, 14)
        generator.load_state_dict(torch.load(gen_path, weights_only=True))
        generator.eval()
        
        return generator, scaler
    except Exception as e:
        st.error(f"Error loading cGAN details: {e}")
        print(f"Error loading cGAN: {e}")
        return None, None

def render():
    page_header("Generative AI Data Synthesizer", "Conditional GAN (cGAN) for Synthetic Telemetry Generation")
    
    generator, scaler = load_cgan_model()
    
    if generator is None:
        st.error("cGAN models not found. Please run: `python src/train_cgan.py`")
        return
        
    info_panel("This tool uses a trained Conditional Generative Adversarial Network (cGAN) to synthesize realistic battery sensor data conditioned on specific thermal risk profiles.")
    
    # ── Controls ───────────────────────────────────
    with st.container(border=True):
        section_header("Generation Parameters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            target_risk = st.selectbox(
                "Condition (Target Risk Level)", 
                ["NORMAL", "FAILURE"],
                help="The cGAN will generate telemetry matching this risk profile."
            )
        with col2:
            n_samples = st.slider("Number of samples to generate", 10, 500, 100, 10)
        with col3:
            random_seed = st.number_input("Random Seed", value=42, min_value=1, max_value=9999)
            
        generate_btn = st.button("Generate Synthetic Data", type="primary", use_container_width=True)
        
    if generate_btn:
        with st.spinner(f"Generating {n_samples} {target_risk} samples..."):
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
            
            label_map = {"NORMAL": 0, "FAILURE": 1}
            class_idx = label_map[target_risk]
            
            noise = torch.randn(n_samples, 16)
            labels = torch.full((n_samples,), class_idx, dtype=torch.long)
            
            with torch.no_grad():
                fake_data_scaled = generator(noise, labels).numpy()
                
            fake_data = scaler.inverse_transform(fake_data_scaled)
            
            # Reconstruct DataFrame
            df_gen = pd.DataFrame(fake_data, columns=FEATURE_COLUMNS)
            
            # Validation and Constraints
            violations = {}
            for col in FEATURE_COLUMNS:
                violations[col] = 0
            
            # Physically positive constraints
            positive_cols = [
                "cycle_count", "state_of_charge", "depth_of_discharge", 
                "cell_voltage_avg", "cell_voltage_std", "cell_temperature_avg", 
                "cell_temperature_max", "internal_resistance", "charge_efficiency", 
                "fast_charge_ratio", "average_charge_power_kw", "charging_interruptions", 
                "cooling_system_health"
            ]
            
            for col in positive_cols:
                invalid_mask = df_gen[col] < 0
                if invalid_mask.any():
                    violations[col] += invalid_mask.sum()
                    # Clip to 0 since negative values for these are physically impossible
                    df_gen.loc[invalid_mask, col] = 0.0
                    
            # SOC constraints
            soc_low = df_gen["state_of_charge"] < 0
            if soc_low.any():
                violations["state_of_charge"] += soc_low.sum()
                df_gen.loc[soc_low, "state_of_charge"] = 0.0
                
            soc_high = df_gen["state_of_charge"] > 100
            if soc_high.any():
                violations["state_of_charge"] += soc_high.sum()
                df_gen.loc[soc_high, "state_of_charge"] = 100.0
                
            total_violations = sum(violations.values())
            
            st.session_state.cgan_df = df_gen
            st.session_state.cgan_risk = target_risk
            st.session_state.cgan_violations = violations
            st.session_state.cgan_total_violations = total_violations
            
    # ── Results ────────────────────────────────────
    if "cgan_df" in st.session_state:
        df_gen = st.session_state.cgan_df
        target_risk = st.session_state.cgan_risk
        violations = st.session_state.cgan_violations
        total_violations = st.session_state.cgan_total_violations
        
        with st.container(border=True):
            section_header("Generated Telemetry")
            
            if total_violations > 0:
                viol_details = ", ".join([f"{k}: {v}" for k, v in violations.items() if v > 0])
                st.warning(f"Validation: {total_violations} values violated physical constraints (e.g., < 0) and were corrected. Details: {viol_details}")
            else:
                st.success("Validation: All generated samples satisfied physical constraints without correction.")
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                metric_card("Samples Generated", str(len(df_gen)))
            with c2:
                metric_card("Condition", target_risk)
            with c3:
                metric_card("Mean Temp", f"{df_gen['cell_temperature_avg'].mean():.1f}", " °C")
            with c4:
                metric_card("Mean Voltage", f"{df_gen['cell_voltage_avg'].mean():.2f}", " V")
            
        st.markdown("<br/>", unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Data Table", "Feature Distributions"])
        
        with tab1:
            st.dataframe(df_gen.head(100), use_container_width=True, height=300)
            
            csv = df_gen.to_csv(index=False)
            st.download_button(
                "Download Synthetic Data (CSV)",
                csv,
                f"synthetic_{target_risk.lower()}_samples.csv",
                "text/csv"
            )
            
        with tab2:
            display_names = get_feature_display_names()
            selected_feature = st.selectbox(
                "Select feature to view distribution",
                FEATURE_COLUMNS,
                format_func=lambda x: display_names.get(x, x),
            )
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df_gen[selected_feature],
                nbinsx=30,
                marker_color="#8B5CF6",
                opacity=0.9,
            ))
            layout_opts = CHART_LAYOUT.copy()
            layout_opts.update(
                title=dict(text=f"Synthetic Distribution: {display_names.get(selected_feature, selected_feature)}", font=dict(color="#111827", size=14)),
                height=350,
                margin=dict(l=50, r=20, t=50, b=50),
                xaxis=dict(title=selected_feature, showgrid=False),
                yaxis=dict(title="Count"),
            )
            fig.update_layout(**layout_opts)
            st.plotly_chart(fig, use_container_width=True)
