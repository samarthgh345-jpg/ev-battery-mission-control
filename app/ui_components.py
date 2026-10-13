"""
EV Battery Mission Control — Reusable UI Components
===================================================
Standardized UI elements, metric cards, status badges, and Plotly templates
for the clean, modern, spacious, engineering-focused dashboard design.
"""

import streamlit as st
from app.styles import clean_html

# Common Plotly Chart Layout matching the clean white card aesthetic
CHART_LAYOUT = dict(
    font=dict(family="Inter, -apple-system, sans-serif", size=11, color="#172033"),
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    margin=dict(l=42, r=16, t=28, b=32),
    xaxis=dict(
        showgrid=True,
        gridcolor="#F3F4F6",
        zeroline=False,
        linecolor="#E5E7EB",
        tickfont=dict(color="#667085", size=10, family="JetBrains Mono"),
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="#F3F4F6",
        zeroline=False,
        linecolor="#E5E7EB",
        tickfont=dict(color="#667085", size=10, family="JetBrains Mono"),
    ),
)


def page_header(title: str, subtitle: str = None, badge_label: str = "SIMULATION / PROTOTYPE"):
    """
    Render a clean white header card with title, subtitle, and prototype badge.
    Matches the top card in the reference design.
    """
    badge_html = (
        f"""<div style="background: #F3F4F6; border: 1px solid #E5E7EB; """
        f"""color: #4B5563; font-size: 11px; font-weight: 600; padding: 5px 12px; border-radius: 6px; """
        f"""display: inline-flex; align-items: center; gap: 6px; letter-spacing: 0.03em; font-family: 'JetBrains Mono', monospace;">"""
        f"""⚡ {badge_label}</div>"""
    )
    sub_html = f'<p style="color: #667085; font-size: 13px; margin: 4px 0 0 0;">{subtitle}</p>' if subtitle else ""
    html = clean_html(f"""
<div class="white-card" style="display: flex; justify-content: space-between; align-items: center; padding: 18px 22px; margin-bottom: 14px;">
<div>
<h1 style="font-size: 22px; font-weight: 800; color: #172033; margin: 0; letter-spacing: -0.02em;">{title}</h1>
{sub_html}
</div>
<div>{badge_html}</div>
</div>
""")
    st.markdown(html, unsafe_allow_html=True)


def section_header(title: str):
    """
    Render a clean section header.
    """
    html = clean_html(f"""
<div style="font-size: 14px; font-weight: 700; color: #172033; margin: 16px 0 8px 0; letter-spacing: -0.01em;">
    {title}
</div>
""")
    st.markdown(html, unsafe_allow_html=True)


def metric_card(label: str, value: str, suffix: str = "", help_text: str = None, color: str = "#326CE5", subcaption: str = None):
    """
    Render a clean white metric card with uppercase label, bold colored value,
    and a descriptive subcaption (e.g. 'Model estimate', 'Simulated telemetry').
    Wrapped in clean_html to prevent Markdown from interpreting indented lines as code blocks.
    """
    help_attr = f'title="{help_text}"' if help_text else ""
    info_icon = f'<span style="font-size: 10px; color: #9CA3AF; cursor: help;" title="{help_text}">ℹ</span>' if help_text else ""
    sub_html = f'<div style="font-size: 11px; color: #667085; margin-top: 3px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{subcaption}</div>' if subcaption else ""

    html = clean_html(f"""
<div class="metric-tile" {help_attr} style="padding: 12px 14px; min-width: 0;">
<div style="display: flex; justify-content: space-between; align-items: center;">
<div class="metric-tile-label" style="font-size: 10px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{label}</div>
{info_icon}
</div>
<div style="display: flex; align-items: baseline; gap: 3px; margin: 6px 0 2px 0;">
<span style="font-size: 22px; font-weight: 800; color: {color}; font-family: 'JetBrains Mono', monospace; line-height: 1.1;">{value}</span>
<span style="font-size: 13px; font-weight: 700; color: {color}; font-family: 'JetBrains Mono', monospace;">{suffix}</span>
</div>
{sub_html}
</div>
""")
    st.markdown(html, unsafe_allow_html=True)


def status_badge(label: str, status: str, subtext: str = None):
    """
    Render a clean functional status badge.
    status options: 'NORMAL', 'CAUTION', 'WARNING', 'HIGH', 'CRITICAL', 'ANOMALY', 'HEALTHY'
    """
    colors = {
        "NORMAL": ("#16866A", "#F0FDF4", "#BBF7D0"),
        "HEALTHY": ("#16866A", "#F0FDF4", "#BBF7D0"),
        "CAUTION": ("#C7771B", "#FFFBEB", "#FDE68A"),
        "WARNING": ("#C7771B", "#FFFBEB", "#FDE68A"),
        "HIGH": ("#D97706", "#FFF7ED", "#FED7AA"),
        "CRITICAL": ("#D94841", "#FEF2F2", "#FECACA"),
        "ANOMALY": ("#D94841", "#FEF2F2", "#FECACA"),
        "INFO": ("#326CE5", "#EFF6FF", "#BFDBFE"),
    }
    text_color, bg_color, border_color = colors.get(status.upper(), ("#667085", "#F9FAFB", "#E5E7EB"))
    sub_html = f'<div style="font-size: 10px; color: #667085; margin-top: 2px;">{subtext}</div>' if subtext else ""

    html = clean_html(f"""
<div style="background-color: {bg_color}; border: 1px solid {border_color}; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;">
<div style="font-size: 10px; font-weight: 700; color: #667085; text-transform: uppercase; letter-spacing: 0.04em;">{label}</div>
<div style="font-size: 14px; font-weight: 800; color: {text_color}; font-family: 'JetBrains Mono', monospace; margin-top: 2px;">{status.upper()}</div>
{sub_html}
</div>
""")
    st.markdown(html, unsafe_allow_html=True)


def info_panel(text: str):
    """Render a clean informational panel."""
    html = clean_html(f"""
<div style="background: #EFF6FF; border: 1px solid #DBEAFE; border-radius: 8px; padding: 12px 16px; font-size: 12px; color: #1E40AF; margin: 8px 0; line-height: 1.5;">
{text}
</div>
""")
    st.markdown(html, unsafe_allow_html=True)


def warning_panel(text: str):
    """Render a clean warning callout."""
    html = clean_html(f"""
<div style="background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 8px; padding: 12px 16px; font-size: 12px; color: #92400E; margin: 8px 0; line-height: 1.5;">
{text}
</div>
""")
    st.markdown(html, unsafe_allow_html=True)


def error_panel(text: str):
    """Render a critical risk callout."""
    html = clean_html(f"""
<div style="background: #FEF2F2; border: 1px solid #FECACA; border-radius: 8px; padding: 12px 16px; font-size: 12px; color: #991B1B; margin: 8px 0; line-height: 1.5;">
{text}
</div>
""")
    st.markdown(html, unsafe_allow_html=True)
