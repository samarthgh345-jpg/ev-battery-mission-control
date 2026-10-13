"""
EV Battery Mission Control — Clean Engineering Design System
=============================================================
A modern, spacious, engineering-focused dashboard aesthetic:
- Light grey page background: #ECEDEF
- Deep dark vertical sidebar: #171B26
- Pure white rounded cards: #FFFFFF with subtle borders (#E5E7EB) and soft shadows
- Clean typography and consistent spacing
- Color used strictly for functional status communication
"""

import re


def clean_html(html_str: str) -> str:
    """
    Remove leading and trailing whitespace on each line to prevent
    Markdown parsers from misinterpreting indented HTML as preformatted code blocks.
    """
    lines = [line.strip() for line in html_str.strip().splitlines() if line.strip()]
    return "".join(lines)


CLEAN_ENGINEERING_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap');

:root {
    --bg-main: #ECEDEF;
    --bg-sidebar: #171B26;
    --bg-card: #FFFFFF;
    --border-card: #E5E7EB;
    --text-primary: #172033;
    --text-secondary: #667085;
    --color-blue: #326CE5;
    --color-green: #16866A;
    --color-amber: #C7771B;
    --color-red: #D94841;
}

/* Global Reset & Typography */
* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    box-sizing: border-box;
}

html, body, .stApp {
    background-color: var(--bg-main) !important;
    color: var(--text-primary) !important;
}

/* Eliminate excessive Streamlit default padding */
.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 2.5rem !important;
    padding-left: 2.0rem !important;
    padding-right: 2.0rem !important;
    max-width: 1300px !important;
    margin: 0 auto !important;
}

div[data-testid="stVerticalBlock"] {
    gap: 0.85rem !important;
}

div[data-testid="stHorizontalBlock"] {
    gap: 1.0rem !important;
}

/* Streamlit Header & Footer */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] {
    background: transparent !important;
}

/* ── Dark Vertical Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #171B26 !important;
    border-right: 1px solid #232838 !important;
    width: 270px !important;
}

section[data-testid="stSidebar"] * {
    color: #F3F4F6 !important;
}

section[data-testid="stSidebar"] .stRadio label span {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #94A3B8 !important;
    padding: 2px 0 !important;
}

section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
    gap: 6px !important;
}

section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
    border-radius: 8px !important;
    padding: 6px 10px !important;
    transition: all 0.15s ease !important;
}

section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
    background: #1F2536 !important;
}

section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) {
    background: #232A3E !important;
    border-left: 3px solid #326CE5 !important;
}

section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) span {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

/* Sidebar Brand Block */
.sidebar-brand-box {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 0 16px 0;
    border-bottom: 1px solid #232838;
    margin-bottom: 16px;
}

.brand-icon-badge {
    width: 36px;
    height: 36px;
    border-radius: 8px;
    background: #232A3E;
    border: 1px solid #2E3752;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    color: #F59E0B;
}

.sidebar-brand-title {
    font-size: 14px;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -0.01em;
    line-height: 1.2;
}

.sidebar-brand-subtitle {
    font-size: 11px;
    color: #94A3B8;
    margin-top: 2px;
}

/* ── White Engineering Cards ── */
.white-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04), 0 1px 2px rgba(0, 0, 0, 0.02);
    margin-bottom: 14px;
}

.white-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    padding-bottom: 10px;
    border-bottom: 1px solid #F3F4F6;
}

.white-card-title {
    font-size: 15px;
    font-weight: 700;
    color: #172033;
    letter-spacing: -0.01em;
}

.white-card-subtitle {
    font-size: 12px;
    color: #667085;
}

/* Metric Display Card */
.metric-tile {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 16px 18px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.metric-tile-label {
    font-size: 11px;
    font-weight: 700;
    color: #667085;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.metric-tile-value {
    font-size: 26px;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.1;
    margin: 8px 0 4px 0;
}

.metric-tile-sub {
    font-size: 11px;
    color: #667085;
}

/* Highlight / Decision Box */
.decision-callout {
    background: #EFF6FF;
    border: 1px solid #DBEAFE;
    border-radius: 10px;
    padding: 14px 16px;
    color: #1E40AF;
    font-size: 13px;
    font-weight: 600;
    line-height: 1.5;
    margin-bottom: 12px;
}

/* ── Buttons ── */
div.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 8px 16px !important;
    transition: all 0.15s ease !important;
}

div.stButton > button[kind="secondary"] {
    background: #FFFFFF !important;
    color: #172033 !important;
    border: 1px solid #D1D5DB !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
}

div.stButton > button[kind="secondary"]:hover {
    background: #F9FAFB !important;
    border-color: #9CA3AF !important;
}

div.stButton > button[kind="primary"] {
    background: #326CE5 !important;
    color: #FFFFFF !important;
    border: 1px solid #2554B8 !important;
    box-shadow: 0 1px 2px rgba(50, 108, 229, 0.25) !important;
}

div.stButton > button[kind="primary"]:hover {
    background: #2554B8 !important;
}

/* ── Form Controls & Labels ── */
div[data-testid="stWidgetLabel"] label,
div[data-testid="stWidgetLabel"] p {
    color: #172033 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
}

/* Clean Selectbox and Slider styling */
div[data-baseweb="select"] {
    border-radius: 8px !important;
}

/* Plotly chart container */
div[data-testid="stPlotlyChart"] {
    background: #FFFFFF !important;
    border-radius: 10px !important;
}
</style>
"""
