"""
EV Battery Mission Control — Main Application & Navigation Router
==================================================================
AI-Powered EV Battery Thermal Safety, Monitoring, Prediction & Explainability.
Futuristic industrial dark telemetry mission control console.
"""

import sys
from pathlib import Path
import json
import pandas as pd
import joblib
import streamlit as st

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prediction_engine import load_mlp_model, load_model as load_xgb_engine, get_default_features
from src.anomaly_detection import load_autoencoder
from src.thermal_simulator import BatteryPackThermalSimulator
from src.agent import ThermalAIAgent
from app.styles import CLEAN_ENGINEERING_CSS, clean_html

# ── Streamlit Page Configuration ────────────────────────────
st.set_page_config(
    page_title="EV Battery Mission Control",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply unified clean engineering design system
st.markdown(CLEAN_ENGINEERING_CSS, unsafe_allow_html=True)

# ── Session State Initialization ────────────────────────────
if "sim" not in st.session_state:
    st.session_state.sim = BatteryPackThermalSimulator(
        ambient_temp=30.0,
        coolant_temp=22.0,
        current=10.0,
        coolant_flow=0.02,
        seed=42,
    )

if "mc_flow_rate_slider" not in st.session_state:
    st.session_state.mc_flow_rate_slider = 0.020

if "mc_sim_mode" not in st.session_state:
    st.session_state.mc_sim_mode = "NORMAL"

if "agent" not in st.session_state:
    st.session_state.agent = ThermalAIAgent()

if "mc_features" not in st.session_state:
    st.session_state.mc_features = get_default_features()

if "current_page" not in st.session_state:
    st.session_state.current_page = "Mission Control"

# Page redirection helper
if "page_to_open" in st.session_state:
    st.session_state.current_page = st.session_state.page_to_open
    del st.session_state.page_to_open


# ── Model & Data Caching ─────────────────────────────────────
@st.cache_resource
def load_xgboost_model():
    model_path = PROJECT_ROOT / "models" / "xgboost_failure_model.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    return None


@st.cache_resource
def load_pytorch_mlp_model():
    mlp_dir = PROJECT_ROOT / "models" / "failure_predictor"
    if (mlp_dir / "mlp.pt").exists():
        return load_mlp_model(str(mlp_dir))
    return None


@st.cache_resource
def load_anomaly_autoencoder():
    ae_dir = PROJECT_ROOT / "models" / "autoencoder"
    if (ae_dir / "ae.pt").exists():
        return load_autoencoder(str(ae_dir))
    return None


@st.cache_resource
def load_model_metadata():
    meta_path = PROJECT_ROOT / "models" / "model_metadata_xgb.json"
    if not meta_path.exists():
        meta_path = PROJECT_ROOT / "models" / "model_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_resource
def load_dataset():
    data_path = PROJECT_ROOT / "data" / "ev_battery_failure_dataset.csv"
    if data_path.exists():
        # Load dataset efficiently
        return pd.read_csv(data_path)
    return None


@st.cache_resource
def load_shap_explainer(_model):
    try:
        from src.shap_explainer import create_explainer
        data_path = PROJECT_ROOT / "data" / "ev_battery_failure_dataset.csv"
        if data_path.exists() and _model is not None:
            df = pd.read_csv(data_path, nrows=500)
            bg = df.sample(min(80, len(df)), random_state=42)
            return create_explainer(_model, bg)
    except Exception as e:
        print(f"SHAP explainer init note: {e}")
    return None


# ── Load Model Artifacts ─────────────────────────────────────
xgb_model = load_xgboost_model()
mlp_model = load_pytorch_mlp_model()
ae_model_artifacts = load_anomaly_autoencoder()
metadata = load_model_metadata()
dataset = load_dataset()

# Graceful loading check
models_loaded = (xgb_model is not None) and (mlp_model is not None) and (ae_model_artifacts is not None)
shap_explainer = load_shap_explainer(mlp_model) if mlp_model is not None else None

# ── Compact Dark Sidebar Navigation ──────────────────────────
with st.sidebar:
    st.markdown(
        clean_html(
            """
            <div class="sidebar-brand-box">
                <div class="brand-icon-badge">⚡</div>
                <div>
                    <div class="sidebar-brand-title">EV Battery Mission Control</div>
                    <div class="sidebar-brand-subtitle">AI-Powered Battery Safety</div>
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='font-size: 10px; font-weight: 700; color: #64748B; letter-spacing: 0.08em; margin: 8px 0 6px 0; text-transform: uppercase;'>NAVIGATION</div>",
        unsafe_allow_html=True,
    )

    PAGES = [
        "Mission Control",
        "Thermal Analysis",
        "What-If Simulator",
    ]

    PAGE_ICONS = {
        "Mission Control": "🎛️",
        "Thermal Analysis": "🌡️",
        "What-If Simulator": "🧪",
    }

    # Ensure valid page selection among the 3 retained pages
    if st.session_state.current_page not in PAGES:
        st.session_state.current_page = "Mission Control"

    page_index = PAGES.index(st.session_state.current_page)
    selected_page = st.radio(
        "Navigation",
        PAGES,
        index=page_index,
        format_func=lambda p: f"{PAGE_ICONS.get(p, '•')}  {p}",
        label_visibility="collapsed",
        key="sidebar_nav_radio",
    )

    if selected_page != st.session_state.current_page:
        st.session_state.current_page = selected_page
        st.rerun()

    page = st.session_state.current_page

    # Pipeline & System Telemetry Health Badges in Sidebar
    st.markdown("<div style='border-top: 1px solid #232838; margin: 18px 0 12px 0;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size: 10px; font-weight: 700; color: #64748B; letter-spacing: 0.08em; margin-bottom: 8px; text-transform: uppercase;'>SYSTEM STATUS</div>",
        unsafe_allow_html=True,
    )

    xgb_online = xgb_model is not None
    mlp_online = mlp_model is not None
    ae_online = ae_model_artifacts is not None
    data_online = dataset is not None

    xgb_badge_color = "#34D399" if xgb_online else "#F87171"
    mlp_badge_color = "#34D399" if mlp_online else "#F87171"
    ae_badge_color = "#34D399" if ae_online else "#F87171"
    data_badge_color = "#38BDF8" if data_online else "#F87171"

    xgb_text = "ONLINE" if xgb_online else "OFFLINE"
    mlp_text = "ONLINE" if mlp_online else "OFFLINE"
    ae_text = "ONLINE" if ae_online else "OFFLINE"
    data_text = f"ONLINE ({len(dataset):,} ROWS)" if data_online else "OFFLINE"

    st.markdown(
        clean_html(
            f"""
            <div style="font-size: 11px; font-family: 'JetBrains Mono', monospace; display: flex; flex-direction: column; gap: 6px;">
                <div style="display: flex; justify-content: space-between; align-items: center; background: #1B2130; padding: 6px 10px; border-radius: 6px; border: 1px solid #242D42;">
                    <span style="color: #94A3B8;">XGBoost</span>
                    <span style="color: {xgb_badge_color}; font-weight: 700;">● {xgb_text}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; background: #1B2130; padding: 6px 10px; border-radius: 6px; border: 1px solid #242D42;">
                    <span style="color: #94A3B8;">PyTorch MLP</span>
                    <span style="color: {mlp_badge_color}; font-weight: 700;">● {mlp_text}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; background: #1B2130; padding: 6px 10px; border-radius: 6px; border: 1px solid #242D42;">
                    <span style="color: #94A3B8;">Autoencoder</span>
                    <span style="color: {ae_badge_color}; font-weight: 700;">● {ae_text}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; background: #1B2130; padding: 6px 10px; border-radius: 6px; border: 1px solid #242D42;">
                    <span style="color: #94A3B8;">Dataset</span>
                    <span style="color: {data_badge_color}; font-weight: 600;">{data_text}</span>
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    

# Fallback error notice if models are missing
if not models_loaded:
    st.error("⚠️ One or more required ML model artifacts were not found in `models/`. Please verify trained model files.")
    st.stop()

# ── Dynamic Page Routing ─────────────────────────────────────
try:
    if page == "Mission Control":
        from app.mission_control import render
        render(xgb_model, ae_model_artifacts, mlp_model, shap_explainer, metadata, dataset)
    elif page == "Thermal Analysis":
        from app.prediction import render
        render(xgb_model, ae_model_artifacts, mlp_model, metadata, dataset)
    elif page == "What-If Simulator":
        from app.what_if import render
        render(xgb_model, ae_model_artifacts, mlp_model, shap_explainer, metadata, dataset)
    elif page == "Dataset Explorer":
        from app.dataset_explorer import render
        render(dataset)
    elif page == "Model Performance":
        from app.model_performance import render
        render(xgb_model, mlp_model, metadata, dataset)
    elif page == "XAI":
        from app.xai import render
        render(xgb_model, mlp_model, shap_explainer, metadata, dataset)
except Exception as e:
    st.error(f"❌ Error rendering page '{page}': {e}")
    import traceback
    st.code(traceback.format_exc(), language="python")
