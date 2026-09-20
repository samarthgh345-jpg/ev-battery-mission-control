# UI Audit Report (Phase 11)

## Current Pages & Navigation
The application currently routes through `app/main.py` which presents a sidebar radio button menu.
Pages currently implemented:
1. Overview / Mission Control (`app/mission_control.py`)
2. Digital Twin (`app/digital_twin.py`)
3. Thermal Analysis (Maybe alias for Mission Control/Prediction?)
4. Explainability (`app/xai.py`)
5. What-If (`app/what_if.py`)
6. Agent (`app/agent.py`)
7. RAG Assistant (`app/rag_assistant.py`)
8. Model Performance (`app/model_performance.py`)
9. Dataset (`app/dataset_explorer.py`)

## Duplicated UI Code
- `CUSTOM_CSS` is injected centrally via `app/main.py`, but many pages hardcode HTML structures with classes like `<div class="eng-card">`, `<div class="section-divider">`, and `<div class="page-header">`. 
- Functions to render layout headers and metric cards are manually rewritten in each file (using `st.markdown(..., unsafe_allow_html=True)`).
- We lack a centralized `ui_components.py` to handle standard UI blocks, forcing developers to repeat long HTML strings.

## Inconsistent Spacing and Typography
- Although the global CSS establishes fonts (Inter & JetBrains Mono), raw markdown and hardcoded inline styles in various pages sometimes override or clash with these standard sizes.
- Sizing for subheaders varies per page; some use `st.markdown("### ")` while others use custom `<div class="eng-card-header">`.

## Inconsistent Cards and Colors
- Standard Streamlit metrics are styled nicely via CSS, but custom HTML cards (`eng-card`) sometimes drift in their padding and border-radius applications depending on the file.
- Risk level badges (`risk-normal`, `risk-critical`) are defined as CSS classes but manually typed into tables, making it prone to typo errors.
- Some pages rely heavily on `st.columns` leading to cluttered layouts on smaller screens, while others are entirely vertical walls of text.

## Components to Preserve
- The global clean CSS tokens in `app/main.py` (colors, fonts, zero-border aesthetics).
- The use of Streamlit `st.columns` for side-by-side metric comparisons.
- Plotly charts with custom `_CHART_LAYOUT` which keeps visualizations minimal and clean.

## Components to Replace or Refactor
- Raw HTML blocks inside `.py` files should be abstracted into Python functions (e.g., `card(title, content)`).
- The sidebar radio navigation can be grouped cleanly into distinct subheaders (MONITORING, ANALYSIS, AI & GENERATIVE) using `st.sidebar.markdown` or a custom selection UI.
- All pages need to adopt the uniform design system and eliminate any leftover glowing/neon elements.
