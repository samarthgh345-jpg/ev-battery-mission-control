"""
Reusable UI Components
======================
Standardized UI elements for the EV Battery Mission Control dashboard.
"""

import streamlit as st

# Common Plotly Chart Layout for engineering aesthetic
CHART_LAYOUT = dict(
    font=dict(family="Inter, sans-serif", size=12, color="#4B5563"),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(showgrid=True, gridcolor="#E5E7EB", zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="#E5E7EB", zeroline=False)
)

def page_header(title: str, subtitle: str = None):
    """Render a standardized page header."""
    st.markdown(f"<h1>{title}</h1>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<p style='color: var(--text-2); font-size: 14px; margin-top: -10px; margin-bottom: 20px;'>{subtitle}</p>", unsafe_allow_html=True)
    st.markdown("<hr style='margin-top: 0px; margin-bottom: 20px; border-color: var(--border);'>", unsafe_allow_html=True)

def section_header(title: str):
    """Render a clean section header."""
    st.markdown(f"<h3 style='margin-top: 15px; margin-bottom: 10px; color: var(--text-1);'>{title}</h3>", unsafe_allow_html=True)

def metric_card(label: str, value: str, suffix: str = ""):
    """Render a distinct metric card. (Prefer standard st.metric unless custom HTML is explicitly needed)."""
    # Streamlit's native st.metric is usually best, but this can serve as a custom wrapper if needed.
    st.metric(label=label, value=f"{value}{suffix}")

def status_badge(label: str, status: str):
    """
    Render a status badge.
    status options: 'NORMAL', 'CAUTION', 'WARNING', 'CRITICAL', 'INFO'
    """
    colors = {
        'NORMAL': ('#16A34A', '#F0FDF4', '#BBF7D0'),
        'CAUTION': ('#D97706', '#FFFBEB', '#FDE68A'),
        'WARNING': ('#D97706', '#FFFBEB', '#FDE68A'),
        'CRITICAL': ('#DC2626', '#FEF2F2', '#FECACA'),
        'INFO': ('#2563EB', '#EFF6FF', '#BFDBFE'),
        'NEUTRAL': ('#4B5563', '#F3F4F6', '#E5E7EB')
    }
    
    text_color, bg_color, border_color = colors.get(status.upper(), colors['NEUTRAL'])
    
    html = f"""
    <div style="display: inline-flex; flex-direction: column; background-color: {bg_color}; 
                border: 1px solid {border_color}; border-radius: 6px; padding: 8px 12px; margin-right: 10px; margin-bottom: 10px;">
        <span style="font-size: 11px; font-weight: 600; color: {text_color}; text-transform: uppercase; letter-spacing: 0.05em;">{label}</span>
        <span style="font-size: 14px; font-weight: 700; color: var(--text-1); font-family: var(--mono);">{status.upper()}</span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def info_panel(text: str):
    """Render an informational panel."""
    st.info(text)

def warning_panel(text: str):
    """Render a warning panel for model disagreements or cautions."""
    st.warning(text)

def error_panel(text: str):
    """Render a critical error or danger panel."""
    st.error(text)

def render_html(html_str: str):
    """Safely render HTML by injecting it directly into the DOM, bypassing the Markdown parser."""
    st.html(html_str)
