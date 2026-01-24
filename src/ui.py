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
    page_icon="../images/logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 100px;
        font-weight: bold;
        color: #2E5D28;
        padding-bottom: 2rem;
        padding-top: 1.7rem;
    }
    .stTextInput > div > div > input {
        font-size: 1.1rem;
    }
    .answer-box {
        background-color: #789575;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #0D3908;
        color: #E1E1E1;
        font-size: 1.1rem;
    }
    .source-box {
        background-color: #465844;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
    }

    /* Target the button containers */
    div.stButton > button {
        background-color: #1e1e1e; /* Dark background */
        color: #e0e0e0;            /* Light text */
        border-radius: 50px;       /* Makes it a pill */
        border: 1px solid #333;    /* Subtle border */
        padding: 10px 25px;
        transition: all 0.3s ease;
        width: 100%;
    }

    /* Hover effect */
    div.stButton > button:hover {
        background-color: #333;
        border-color: #555;
        color: white;
    }
    
    /* Active/Focus state */
    div.stButton > button:active, div.stButton > button:focus {
        background-color: #444;
        color: white;
        border-color: #777;
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

# Sidebar
with st.sidebar:
    col1, col2 = st.columns([2.5, 4]) # Adjust ratios for width
    with col1:
        st.image("../images/logo.png", use_container_width=True)
    with col2:
        st.markdown('<p class="main-header" style="font-size:25px; font-weight:bold;">Finance Insight Lite</p>', unsafe_allow_html=True)

    # Upload PDF
    st.subheader("📄 Document Upload")
    uploaded_file = st.file_uploader("Upload PDF", type=['pdf'], help="Upload a financial report in PDF format.")

    if uploaded_file:
        # Save uploaded file
        pdf_path = f"data/uploaded/{uploaded_file.name}"
        os.makedirs("data/uploaded", exist_ok=True)

        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Process Document"):
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
                    use_self_rag=True
                )

                st.success(f"✅ Processed {len(documents)} pages!")

    st.divider()
    # Settings
    with st.expander("⚙️ Settings"):
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

    # Clear history
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.markdown(
        """
        <div style="text-align: center; margin-top: 2rem; color: #888;">
            <p>Finance Insight Lite © 2026. All rights reserved.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Display chat history
for i, chat in enumerate(st.session_state.chat_history):
    with st.chat_message("user", avatar="../images/user_icon.png"):
        st.write(chat['question'])

    with st.chat_message("assistant", avatar="../images/chatbots_icon.png"):
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

if len(st.session_state.chat_history) == 0:
    # Main area
    st.title('Hi there! 👋 Welcome to **Finance Insight Lite**. Upload a financial report to get started.')

    # Chat interface
    st.subheader("💬 Ask Questions")

    # Sample questions
    sample_questions = [
        "What is the net income for Q3 2025?",
        "What is the free cash flow?",
        "What is the gearing ratio?",
        "How much was the dividend declared?",
        "What are the key financial highlights?"
    ]

    # Function to set pending question
    def set_pending_question(q):
        st.session_state.pending_question = q

    cols = st.columns(2)
    for i, q in enumerate(sample_questions):
        with cols[i % 2]:
            st.button(q, key=f"sample_{i}", on_click=set_pending_question, args=(q,))

# Process pending question from sample buttons
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None
    
    if st.session_state.agent is None:
        st.warning("⚠️ Please upload and process a PDF document first!")
    else:
        # Process the question
        with st.spinner("🤔 Thinking..."):
            result = st.session_state.agent.process_query(question)
        
        # Save to history
        st.session_state.chat_history.append({
            'question': question,
            **result
        })
        st.rerun()

# Process regular chat input
if user_question:
    if st.session_state.agent is None:
        st.warning("⚠️ Please upload and process a PDF document first!")
        st.stop()

    # Process query
    with st.spinner("🤔 Thinking..."):
        result = st.session_state.agent.process_query(user_question)
    
    # Save to history
    st.session_state.chat_history.append({
        'question': user_question,
        **result
    })
    st.rerun()
