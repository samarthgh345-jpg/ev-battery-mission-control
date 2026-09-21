"""
EV Battery Mission Control — Main Application
================================================
Entry point for the Streamlit dashboard.
Professional engineering monitoring theme.
"""

import sys
from pathlib import Path
import streamlit as st
import json
import joblib
import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.prediction_engine import load_mlp_model, load_model
from src.anomaly_detection import load_autoencoder

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="EV Battery Mission Control",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg:           #F7F8FA;
    --surface:      #FFFFFF;
    --border:       #E5E7EB;
    --border-light: #F3F4F6;
    --text-1:       #111827;
    --text-2:       #6B7280;
    --text-3:       #9CA3AF;
}

/* Global reset */
.stApp {
    background: var(--bg) !important;
    font-family: var(--font) !important;
    color: var(--text-1) !important;
}

/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--text-1) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    padding: 8px 16px !important;
}
.stButton > button:hover {
    opacity: 0.85 !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── Caching functions ──────────────────────────────────────
@st.cache_resource
def load_xgboost_model():
    model_path = PROJECT_ROOT / "models" / "xgboost_failure_model.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    return None

@st.cache_resource
def load_pytorch_mlp_model():
    return load_mlp_model(str(PROJECT_ROOT / "models" / "failure_predictor"))

@st.cache_resource
def load_anomaly_autoencoder():
    ae_dir = PROJECT_ROOT / "models" / "autoencoder"
    if (ae_dir / "ae.pt").exists():
        return load_autoencoder(str(ae_dir))
    return None

@st.cache_resource
def load_model_metadata():
    meta_path = PROJECT_ROOT / "models" / "model_metadata_xgb.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            return json.load(f)
    return None

@st.cache_resource
def load_dataset():
    data_path = PROJECT_ROOT / "data" / "ev_battery_failure_dataset.csv"
    if data_path.exists():
        return pd.read_csv(data_path)
    return None

@st.cache_resource
def load_shap_explainer(_model):
    from src.shap_explainer import create_explainer
    data_path = PROJECT_ROOT / "data" / "ev_battery_failure_dataset.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
        bg = df.sample(min(100, len(df)), random_state=42)
        return create_explainer(_model, bg)
    return None

# ── Sidebar Navigation ───────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 4px 0 16px 0;">
        <div style="font-family: 'Inter', sans-serif; font-weight: 700; font-size: 15px; color: #111827;">
            EV Battery Mission Control
        </div>
        <div style="font-size: 11px; color: #6B7280; margin-top: 2px;">
            Battery Thermal Management System<br/>Research & Simulation Platform
        </div>
    </div>
    <div style="border-top: 1px solid #E5E7EB; margin-bottom: 12px;"></div>
    """, unsafe_allow_html=True)

    PAGES = [
        "Mission Control",
        "Digital Twin",
        "Thermal Analysis",
        "What-If Simulator",
        "Dataset Explorer",
        "Model Performance",
        "GenAI Generator",
        "VAE Generator",
        "XAI",
        "Responsible AI",
        "RAG Assistant",
        "AI Agent Decision"
    ]

    if "current_page" not in st.session_state:
        st.session_state.current_page = "Mission Control"

    if "page_to_open" in st.session_state:
        st.session_state.current_page = st.session_state.page_to_open
        del st.session_state.page_to_open

    page = st.selectbox(
        "Navigation",
        PAGES,
        key="current_page",
        label_visibility="collapsed"
    )

    st.markdown('<div style="border-top: 1px solid #E5E7EB; margin-top: 12px; margin-bottom: 12px;"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 11px; color: #6B7280;">
        Prototype — synthetic/simulation-inspired data. Not experimentally validated.
    </div>
    """, unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────
xgb_model = load_xgboost_model()
mlp_model = load_pytorch_mlp_model()
ae_model_artifacts = load_anomaly_autoencoder()
metadata = load_model_metadata()
dataset = load_dataset()

if xgb_model is None or mlp_model is None or ae_model_artifacts is None:
    st.error("Models not found. Run: python scripts/train_model.py")
    st.stop()

shap_explainer = load_shap_explainer(mlp_model)

# ── Page Routing ──────────────────────────────────────────
try:
    if page == "Mission Control":
        from app.mission_control import render
        render(xgb_model, ae_model_artifacts, mlp_model, shap_explainer, metadata, dataset)
    elif page == "Digital Twin":
        from app.digital_twin import render
        render(xgb_model, ae_model_artifacts, mlp_model, metadata, dataset)
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
    elif page == "GenAI Generator":
        from app.genai_generator import render
        render()
    elif page == "VAE Generator":
        from app.vae_ui import render
        render()
    elif page == "XAI":
        from app.xai import render
        render(xgb_model, mlp_model, shap_explainer, metadata, dataset)
    elif page == "Responsible AI":
        from app.responsible_ai import render
        render()
    elif page == "RAG Assistant":
        from app.rag_assistant import render
        render()
    elif page == "AI Agent Decision":
        from app.agent import render
        render(xgb_model, ae_model_artifacts, mlp_model, shap_explainer, metadata, dataset)
except Exception as e:
    st.error(f"Error loading page '{page}': {e}")
