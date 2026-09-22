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
    --text-2:       #374151;
    --text-3:       #4B5563;
}

/* Global reset */
* {
    font-family: 'Inter', sans-serif;
}
body, .stApp {
    background-color: var(--bg) !important;
    color: var(--text-1) !important;
}

/* Typography */
h1 {
    color: #1E3A5F !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}
h2, h3 {
    color: var(--text-1) !important;
    font-weight: 600 !important;
}

/* Cards (Metrics, Expanders, Container Borders) */
div[data-testid="metric-container"],
div[data-testid="stExpander"],
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
}
div[data-testid="metric-container"] {
    padding: 16px !important;
}

/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
/* header { visibility: hidden; } REMOVED so sidebar toggle remains accessible */

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

/* Sidebar Navigation Menu Styling (st.radio) */
section[data-testid="stSidebar"] .stRadio > label {
    display: none; /* Hide the "Navigation" title */
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
    gap: 2px;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
    padding: 8px 12px;
    border-radius: 6px;
    background: transparent;
    border-left: 3px solid transparent;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
    background: var(--border-light);
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) {
    background: var(--border-light);
    border-left: 3px solid var(--text-1);
}
/* Hide the actual radio circle if possible, but leave it if not */
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label div[data-testid="stMarkdownContainer"] p {
    color: var(--text-1) !important;
    font-weight: 500 !important;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) div[data-testid="stMarkdownContainer"] p {
    font-weight: 700 !important;
}

/* ── Buttons ── */
/* Base button styles */
.stButton > button, 
.stDownloadButton > button,
button[kind="primaryFormSubmit"],
button[kind="secondaryFormSubmit"] {
    border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 8px 16px !important;
    transition: all 0.2s ease-in-out !important;
}

/* Secondary Button (Default) */
.stButton > button[kind="secondary"],
.stDownloadButton > button[kind="secondary"],
button[data-testid="baseButton-secondary"] {
    background: #F8FAFC !important;
    color: #1F2937 !important;
    border: 1px solid #CBD5E1 !important;
}
.stButton > button[kind="secondary"]:hover,
.stDownloadButton > button[kind="secondary"]:hover,
button[data-testid="baseButton-secondary"]:hover {
    background: #E2E8F0 !important;
    border-color: #CBD5E1 !important;
    color: #1F2937 !important;
}
.stButton > button[kind="secondary"] p,
.stDownloadButton > button[kind="secondary"] p,
button[data-testid="baseButton-secondary"] p {
    color: #1F2937 !important;
}

/* Primary Button */
.stButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"],
button[kind="primaryFormSubmit"],
button[data-testid="baseButton-primary"] {
    background: #1E3A5F !important;
    color: #FFFFFF !important;
    border: 1px solid #1E3A5F !important;
}
.stButton > button[kind="primary"]:hover,
.stDownloadButton > button[kind="primary"]:hover,
button[kind="primaryFormSubmit"]:hover,
button[data-testid="baseButton-primary"]:hover {
    background: #16324F !important;
    border-color: #16324F !important;
    color: #FFFFFF !important;
}
.stButton > button[kind="primary"] p,
.stDownloadButton > button[kind="primary"] p,
button[kind="primaryFormSubmit"] p,
button[data-testid="baseButton-primary"] p {
    color: #FFFFFF !important;
}

/* ── Widget Labels & Text ── */
/* Target sliders, number inputs, selectboxes, and their labels */
div[data-testid="stWidgetLabel"] p,
div[data-testid="stWidgetLabel"] label,
.stSlider label,
.stSelectbox label,
.stNumberInput label {
    color: var(--text-1) !important;
    font-weight: 600 !important;
}

/* Helper text / captions underneath widgets */
div[data-testid="stCaptionContainer"] p,
.st-emotion-cache-16idsys p {
    color: var(--text-2) !important;
    font-weight: 500 !important;
}

/* Metric Labels */
div[data-testid="stMetricLabel"] p, 
div[data-testid="stMetricLabel"] label {
    color: var(--text-2) !important;
    font-weight: 600 !important;
}

/* Tabs */
button[data-testid="stBaseButton-tab"] p {
    color: var(--text-2) !important;
    font-weight: 500 !important;
}
button[data-testid="stBaseButton-tab"][aria-selected="true"] p {
    color: var(--text-1) !important;
    font-weight: 700 !important;
}

/* Expander Headers */
summary[data-testid="stExpanderDetails"] p,
summary[data-testid="stExpanderDetails"] h1,
summary[data-testid="stExpanderDetails"] h2,
summary[data-testid="stExpanderDetails"] h3 {
    color: var(--text-1) !important;
    font-weight: 600 !important;
}

/* Markdown Text that gets forced to gray by Streamlit */
div[data-testid="stMarkdownContainer"] p {
    color: var(--text-1) !important;
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

    page = st.radio(
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
