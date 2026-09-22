"""
Responsible AI Page
====================
Documents the safety boundaries, model limitations, and decoupled architecture.
"""

import streamlit as st
from app.ui_components import page_header, section_header, info_panel, warning_panel, metric_card

def render():
    page_header("Responsible AI & Safety Boundaries", "Project Limitations and Decoupled Architecture")
    
    st.markdown("""
    <div style="font-size: 14px; color: var(--text-2); line-height: 1.6; margin-bottom: 24px;">
        This EV Battery Mission Control system is a prototype demonstrating the integration of traditional Machine Learning, 
        Generative AI, and Agentic workflows. It is governed by strict Responsible AI principles to ensure that 
        generative models do not hallucinate critical numerical safety decisions.
    </div>
    """, unsafe_allow_html=True)

    # ── Architectural Boundaries ──────────────────
    section_header("1. Architectural Boundaries")
    
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("""
            <div style="font-size: 15px; font-weight: 600; color: #16A34A; margin-bottom: 12px;">✅ Deterministic Safety Core</div>
            <ul style="font-size: 13px; color: var(--text-2); line-height: 1.6; padding-left: 20px;">
                <li><strong>XGBoost / MLP</strong>: Trained on tabular sensor data to predict numerical max temperature.</li>
                <li><strong>LangGraph Agent</strong>: Uses physics-based formulas (Newton's Law of Cooling) to evaluate control actions.</li>
                <li><strong>Decision Logic</strong>: 100% deterministic code. No LLM controls the cooling system.</li>
            </ul>
            """, unsafe_allow_html=True)
        
    with col2:
        with st.container(border=True):
            st.markdown("""
            <div style="font-size: 15px; font-weight: 600; color: #2563EB; margin-bottom: 12px;">✅ Generative Assistance Layer</div>
            <ul style="font-size: 13px; color: var(--text-2); line-height: 1.6; padding-left: 20px;">
                <li><strong>RAG LLM</strong>: Grounded purely on verified engineering documents. Restricted to Q&A.</li>
                <li><strong>cGAN / VAE</strong>: Used strictly for generating synthetic data for research and edge-case simulation, clearly labeled as synthetic.</li>
                <li><strong>Isolation</strong>: Generative models cannot trigger battery shutdown or alter flow rates.</li>
            </ul>
            """, unsafe_allow_html=True)

    st.markdown('<div style="margin-top:20px; margin-bottom:20px; border-bottom:1px solid var(--border);"></div>', unsafe_allow_html=True)

    # ── Model Limitations ─────────────────────────
    section_header("2. Known Limitations & Research Constraints")
    
    warning_panel("This is a research prototype. Do not use this software to monitor or control real-world lithium-ion battery packs.")
    
    st.markdown("""
    <div style="font-size: 14px; color: var(--text-1); line-height: 1.6; padding-left: 12px; border-left: 3px solid var(--border-light);">
        <strong>Data Authenticity:</strong> The dataset is synthesized based on thermodynamic principles and does not perfectly reflect the non-linear degradation of aging cells.<br/><br/>
        <strong>Model Disagreement:</strong> The dashboard explicitly displays both XGBoost and MLP predictions. We do not hide model divergence; users must manually investigate when the ensemble disagrees.<br/><br/>
        <strong>LLM Hallucinations:</strong> While RAG reduces hallucination, the LLM may still interpolate facts incorrectly. The prompt explicitly instructs the LLM to refuse answering if the context is insufficient.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="margin-top:20px; margin-bottom:20px; border-bottom:1px solid var(--border);"></div>', unsafe_allow_html=True)

    # ── Explainability ────────────────────────────
    section_header("3. Explainability & Trust")
    
    info_panel("Every prediction made by the system can be decomposed and audited.")
    
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("SHAP", "Global & Local")
            st.markdown("<div style='font-size: 11px; color: var(--text-3); text-align: center;'>Exact feature attribution</div>", unsafe_allow_html=True)
        with c2:
            metric_card("LIME", "Linear Surrogate")
            st.markdown("<div style='font-size: 11px; color: var(--text-3); text-align: center;'>Local boundary approximation</div>", unsafe_allow_html=True)
        with c3:
            metric_card("RAG Distances", "L2 Norms")
            st.markdown("<div style='font-size: 11px; color: var(--text-3); text-align: center;'>Visible confidence in retrieval</div>", unsafe_allow_html=True)
