import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
import time
import sys

# Load environment variables
project_root = Path(__file__).parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)
current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, "finance_insight_lite")
sys.path.append(current_dir)
sys.path.append(package_path)

# Import your modules
from finance_insight_lite.modules.processor import pdf_to_documents
from finance_insight_lite.modules.verctor_store import build_vector_db
from finance_insight_lite.modules.rag_agent import create_advanced_rag_agent

# Page config
st.set_page_config(
    page_title="Finance Insight Lite",
    page_icon="./images/logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTextInput > div > div > input {
        font-size: 1.1rem;
    }
    .answer-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .source-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 5px;
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


st.markdown(
    """
    <style>
    .rounded-img {
        border-radius: 20px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_complete_formatting=True, # لا نحتاجه فعلياً هنا لكنه خيار متاح
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.markdown('<div class="rounded-img">', unsafe_allow_html=True)
    st.image("../images/logo.png", width=200)
    st.markdown('</div>', unsafe_allow_html=True)
    st.title("⚙️ Settings")

    # RAG Configuration
    st.subheader("RAG Configuration")
    use_self_rag = st.toggle("Enable Self-RAG", value=True, help="Higher accuracy but slower")

    relevance_threshold = st.slider(
        "Relevance Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.1,
        help="Higher = stricter filtering"
    )

    num_docs = st.slider(
        "Number of Documents",
        min_value=3,
        max_value=10,
        value=5,
        help="More docs = better coverage"
    )

    st.divider()

    # Upload PDF
    st.subheader("📄 Document Upload")
    uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])

    if uploaded_file:
        # Save uploaded file
        pdf_path = f"data/uploaded/{uploaded_file.name}"
        os.makedirs("data/uploaded", exist_ok=True)

        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("🔄 Process Document"):
            with st.spinner("Processing PDF..."):
                # Load and process
                documents = pdf_to_documents(pdf_path)
                st.session_state.vector_db = build_vector_db(
                    documents,
                    db_path="./database"
                )

                # Create agent
                st.session_state.agent = create_advanced_rag_agent(
                    st.session_state.vector_db,
                    use_self_rag=use_self_rag
                )
                
                st.success(f"✅ Processed {len(documents)} pages!")
    
    st.divider()
    
    # Clear history
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Main area
st.markdown('<p class="main-header">💼 Finance Insight Lite</p>', unsafe_allow_html=True)

# Check if agent is initialized
if st.session_state.agent is None:
    st.info("👈 Please upload and process a PDF document to get started!")
    
    # Quick start with default document
    st.subheader("Quick Start")
    if st.button("📊 Load Saudi Aramco Q3 2025 Report"):
        with st.spinner("Loading default document..."):
            try:
                pdf_path = "data/rew/saudi-aramco-q3-2025-interim-report-english.pdf"
                documents = pdf_to_documents(pdf_path)
                st.session_state.vector_db = build_vector_db(
                    documents,
                    db_path="./database"
                )
                st.session_state.agent = create_advanced_rag_agent(
                    st.session_state.vector_db,
                    use_self_rag=use_self_rag
                )
                st.success(f"✅ Loaded {len(documents)} pages!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
else:
    # Chat interface
    st.subheader("💬 Ask Questions")
    
    # Display chat history
    for i, chat in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(chat['question'])
        
        with st.chat_message("assistant"):
            st.markdown(f'<div class="answer-box">{chat["answer"]}</div>', unsafe_allow_html=True)
            
            # Display metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"📄 Pages: {', '.join(map(str, chat['source_pages']))}")
            with col2:
                st.caption(f"🎯 Confidence: {chat['confidence']}")
            with col3:
                st.caption(f"📊 Docs: {chat['relevant_docs_count']}")
            
            # Verification result
            if chat.get('verification'):
                with st.expander("🔍 View Verification"):
                    st.write(chat['verification']['verification'])
    
    # Input area
    user_question = st.chat_input("Type your question here...")
    
    if user_question:
        # Add user message to chat
        with st.chat_message("user"):
            st.write(user_question)
        
        # Process query
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                result = st.session_state.agent.process_query(user_question)
            
            # Display answer
            st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)
            
            # Display metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"📄 Pages: {', '.join(map(str, result['source_pages']))}")
            with col2:
                st.caption(f"🎯 Confidence: {result['confidence']}")
            with col3:
                st.caption(f"📊 Docs: {result['relevant_docs_count']}")
            
            # Verification result
            if result.get('verification'):
                with st.expander("🔍 View Verification"):
                    st.write(result['verification']['verification'])
        
        # Save to history
        st.session_state.chat_history.append({
            'question': user_question,
            **result
        })

# Footer
st.divider()
st.caption("Powered by Advanced RAG (CRAG + Agentic RAG + Self-RAG)")

# Sample questions
with st.expander("💡 Sample Questions"):
    sample_questions = [
        "What is the net income for Q3 2025?",
        "What is the free cash flow?",
        "What is the gearing ratio?",
        "How much was the dividend declared?",
        "What are the key financial highlights?"
    ]
    
    cols = st.columns(2)
    for i, q in enumerate(sample_questions):
        with cols[i % 2]:
            if st.button(q, key=f"sample_{i}"):
                st.session_state.current_question = q
                st.rerun()