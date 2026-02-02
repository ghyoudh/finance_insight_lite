# ui_with_visualization.py
"""
Streamlit UI with Financial Data Visualization Support
"""

import streamlit as st
import plotly.graph_objects as go
import json
from pathlib import Path
import os

# Import your modules
from finance_insight_lite.modules.processor import load_documents_fastest
from finance_insight_lite.modules.verctor_store import build_vector_db
from finance_insight_lite.modules.rag import create_rag

# Page config
st.set_page_config(
    page_title="Finance Insight Lite",
    page_icon="💼",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .chart-container {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
if 'pending_question' not in st.session_state:
    st.session_state.pending_question = None


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.image("./images/logo.png", width=100)
    st.title("Finance Insight Lite")
    
    st.divider()
    
    # Document Upload
    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or Excel",
        type=['pdf', 'xlsx', 'xls'],
        accept_multiple_files=True,
        help="Upload financial reports (PDF) or data files (Excel)"
    )
    
    if uploaded_files:
        # Save files
        os.makedirs("data/uploaded", exist_ok=True)
        file_paths = []
        
        for file in uploaded_files:
            file_path = f"data/uploaded/{file.name}"
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            file_paths.append(file_path)
        
        st.success(f"✅ {len(file_paths)} file(s) uploaded")
        
        if st.button("🚀 Process Documents", use_container_width=True):
            with st.spinner("Processing..."):
                import time
                start = time.time()
                
                all_documents = []
                
                for path in file_paths:
                    result = load_documents_fastest(path, use_cache=True)
                    all_documents.extend(result['documents'])
                
                # Build vector DB
                st.session_state.vector_db = build_vector_db(
                    all_documents,
                    db_path="./database"
                )
                
                # Create enhanced agent with visualization
                st.session_state.agent = create_rag(
                    st.session_state.vector_db,
                    use_self_rag=True
                )
                
                elapsed = time.time() - start
                st.success(f"✅ Processed {len(all_documents)} documents in {elapsed:.1f}s!")
    
    st.divider()
    
    # Settings
    with st.expander("⚙️ Settings"):
        st.subheader("Visualization Settings")
        
        default_chart = st.selectbox(
            "Default Chart Type",
            ["bar", "line", "pie", "scatter", "area"],
            help="Default chart type for visualizations"
        )
        st.session_state.default_chart_type = default_chart
        
        show_data_table = st.checkbox(
            "Show Data Table",
            value=True,
            help="Display data table below charts"
        )
        st.session_state.show_data_table = show_data_table
    
    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ============================================================================
# MAIN AREA
# ============================================================================

st.markdown('<p class="main-header">💼 Finance Insight Lite</p>', unsafe_allow_html=True)

# Check if agent is ready
if st.session_state.agent is None:
    st.info("👈 Please upload and process documents to get started!")
    st.stop()

# Display chat history
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat['question'])
    
    with st.chat_message("assistant"):
        # Display answer
        st.markdown(f'<div class="answer-box">{chat["answer"]}</div>', unsafe_allow_html=True)
        
        # Display chart if available
        if chat.get('chart'):
            chart_data = chat['chart']
            
            if chart_data.get('success'):
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                
                # Display chart
                chart_json = chart_data.get('chart')
                if chart_json:
                    fig = go.Figure(json.loads(chart_json))
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display data table if enabled
                if st.session_state.get('show_data_table', True):
                    data_preview = chart_data.get('data_preview')
                    if data_preview:
                        with st.expander("📊 View Data Table"):
                            import pandas as pd
                            df = pd.DataFrame(data_preview)
                            st.dataframe(df, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Display metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            pages = chat.get('source_pages', [])
            if pages:
                st.caption(f"📄 Pages: {', '.join(map(str, pages))}")
        with col2:
            st.caption(f"🎯 Confidence: {chat.get('confidence', 'N/A')}")
        with col3:
            if chat.get('chart'):
                st.caption("📈 Visualization included")

# Chat input
user_question = st.chat_input("Ask a question or request a visualization...")

# Sample questions
if len(st.session_state.chat_history) == 0:
    st.subheader("💡 Try These Examples")
    
    sample_questions = [
        "What is the net income for Q3 2025?",
        "Show me a line chart of quarterly revenue",
        "Draw a pie chart of expense breakdown",
        "Plot the revenue trend over time",
        "Compare Q1 and Q2 performance with a bar chart",
        "Visualize the profit margins"
    ]
    
    def set_pending_question(q):
        st.session_state.pending_question = q
    
    cols = st.columns(2)
    for i, q in enumerate(sample_questions):
        with cols[i % 2]:
            st.button(q, key=f"sample_{i}", on_click=set_pending_question, args=(q,))

# Process pending question
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None
    
    with st.spinner("🤔 Thinking..."):
        result = st.session_state.agent.process_query(question)
    
    st.session_state.chat_history.append({
        'question': question,
        **result
    })
    st.rerun()

# Process regular input
if user_question:
    with st.spinner("🤔 Thinking..."):
        result = st.session_state.agent.process_query(user_question)
    
    st.session_state.chat_history.append({
        'question': user_question,
        **result
    })
    st.rerun()

# Footer
st.divider()
st.markdown(
    """
    <div style="text-align: center; color: #888;">
        <p>Powered by Advanced RAG + Financial Visualization | 💹 Charts: Plotly</p>
    </div>
    """,
    unsafe_allow_html=True
)