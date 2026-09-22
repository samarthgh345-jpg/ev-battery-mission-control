"""
RAG Assistant Page
==================
Engineering Knowledge Assistant powered by RAG.
"""

import streamlit as st
from src.rag_pipeline import RAGPipeline, get_llm_provider
from app.ui_components import page_header, section_header, info_panel, warning_panel, metric_card, render_html
import re

def clean_markdown(text: str, filename: str) -> tuple[str, str]:
    """Clean markdown text and extract a title and excerpt."""
    # Find the first heading for the title, or use filename
    title_match = re.search(r'^#+\s+(.+)$', text, flags=re.MULTILINE)
    if title_match:
        title = title_match.group(1).strip()
    else:
        title = filename.replace('.md', '').replace('_', ' ').title()
    
    # Remove headings completely from excerpt
    text = re.sub(r'^#+\s+.*$', '', text, flags=re.MULTILINE)
    
    # Remove markdown bold/italic markers but keep the text
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    
    # Clean up whitespace and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Truncate to a reasonable excerpt length (e.g. 150 chars)
    excerpt = text if len(text) <= 200 else text[:197] + "..."
    return title, excerpt

def render():
    page_header("Engineering Assistant (RAG)", "Retrieval-Augmented Generation for BTMS Knowledge")

    warning_panel("This assistant uses RAG-grounded responses from project knowledge documents. It does NOT make numerical safety decisions or override the deterministic agent.")

    @st.cache_resource(show_spinner=False)
    def get_rag_pipeline():
        rag = RAGPipeline()
        if not rag.index:
            try:
                rag.build_index()
            except Exception as e:
                st.error(f"Failed to build RAG index: {e}")
        return rag

    with st.spinner("Initializing Knowledge Base..."):
        rag = get_rag_pipeline()
        provider = get_llm_provider()

    st.markdown('<div style="margin-top:20px; margin-bottom:20px; border-bottom:1px solid var(--border);"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])

    with col2:
        with st.container(border=True):
            section_header("System Status")
    
            provider_name = provider.__class__.__name__ if provider else "No-LLM Fallback"
            metric_card("LLM Provider", provider_name)
            st.markdown("<br/>", unsafe_allow_html=True)
            metric_card("FAISS Chunks", f"{len(rag.metadata) if rag.index else 0}")
            st.markdown("<br/>", unsafe_allow_html=True)
            metric_card("Embedding Model", "Loaded")
    
            st.markdown("<br/>", unsafe_allow_html=True)
            section_header("Sample Queries")
            samples = [
                "What is BTMS?",
                "What is thermal runaway?",
                "Why is liquid cooling used?",
                "How does coolant flow affect heat transfer?",
            ]
    
            selected_sample = st.radio("Select a question:", samples, index=None, label_visibility="collapsed")

    with col1:
        with st.container(border=True):
            section_header("Ask an Engineering Question")
    
            user_query = st.text_input(
                "Question:",
                value=selected_sample if selected_sample else "",
                placeholder="e.g., Why does battery temperature matter?",
            )
    
            ask_btn = st.button("Ask Assistant", type="primary", use_container_width=True)
            
        if ask_btn:
            if not user_query.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Retrieving knowledge and generating answer..."):
                    res = rag.generate_answer(user_query)
    
                st.markdown('<div style="margin-top:20px; margin-bottom:20px; border-bottom:1px solid var(--border);"></div>', unsafe_allow_html=True)
    
                section_header("Answer")
                
                if res['provider'] == "Fallback (No LLM)":
                    fallback_html = """
                    <div style="background-color:rgba(217, 119, 6, 0.1); border:1px solid #D97706; border-radius:6px; padding:16px; margin-bottom:24px;">
                        <div style="color:#D97706; font-weight:700; font-size:14px; margin-bottom:8px;">⚠️ LLM Configuration Required</div>
                        <div style="color:var(--text-1); font-size:13px; line-height:1.5;">
                            Relevant information was successfully retrieved from the BTMS knowledge base.<br><br>
                            A language model is not currently configured, so a conversational answer cannot be generated.<br><br>
                            Configure OpenAI or Ollama to enable conversational answers.
                        </div>
                    </div>
                    """
                    render_html(fallback_html)
                    
                    st.markdown('<div style="font-size:16px; font-weight:600; color:var(--text-1); margin-bottom:16px; border-bottom:1px solid var(--border); padding-bottom:8px;">Retrieved Knowledge</div>', unsafe_allow_html=True)
                    
                    for source in res['sources']:
                        title, excerpt = clean_markdown(source['text'], source['source'])
                        card_html = f"""
                        <div style="background-color:var(--surface); border:1px solid var(--border); border-radius:6px; padding:16px; margin-bottom:12px;">
                            <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:8px;">
                                <div style="font-weight:600; font-size:15px; color:var(--text-1);">{title}</div>
                                <div style="font-size:10px; padding:2px 6px; background-color:rgba(37, 99, 235, 0.1); color:#2563EB; border-radius:4px; font-weight:600;">Knowledge Base</div>
                            </div>
                            <div style="font-size:13px; color:var(--text-2); line-height:1.5;">
                                {excerpt}
                            </div>
                        </div>
                        """
                        render_html(card_html)
                else:
                    st.markdown(f"""
                    <div style="background-color:var(--surface); border:1px solid var(--border); border-radius:6px; padding:20px; font-size:14px; color:var(--text-1); line-height:1.6;">
                        {res['answer']}
                    </div>
                    """, unsafe_allow_html=True)

                    if res['sources']:
                        st.markdown("<br/>", unsafe_allow_html=True)
                        section_header("Retrieved Sources")
                        for i, source in enumerate(res['sources']):
                            with st.expander(f"[{i+1}] {source['source']} — L2 distance: {source['score']:.4f}"):
                                st.markdown(f"<div style='font-size:12px; color:var(--text-2); font-family:var(--mono); white-space:pre-wrap;'>{source['text']}</div>", unsafe_allow_html=True)
