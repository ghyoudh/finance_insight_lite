import shutil
import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
import time
import sys
from finance_insight_lite.modules.processor import load_documents_fastest, clear_cache
import plotly.graph_objects as go
import json
import pandas as pd

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
from finance_insight_lite.modules.rag_agent import FinancialRAGAgent

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
        st.image("./images/logo.png", width='stretch')
    with col2:
        st.markdown('<p class="main-header" style="font-size:25px; font-weight:bold;">Finance Insight Lite</p>', unsafe_allow_html=True)

    # Upload PDF or Excel
    st.subheader("📄 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF or Excel",
        type=['pdf', 'xlsx', 'xls'],
        help="Upload a financial report in PDF or Excel format.",
        accept_multiple_files=True
    )

    if uploaded_files:
        # Clear and recreate upload directory
        if os.path.exists("./data/uploaded/"):
            shutil.rmtree("./data/uploaded")
        os.makedirs("./data/uploaded")

        # Save all uploaded files
        uploaded_file_paths = []
        for uploaded_file in uploaded_files:
            file_path = f"./data/uploaded/{uploaded_file.name}"

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            uploaded_file_paths.append(file_path)

        st.success(f"✅ Uploaded {len(uploaded_file_paths)} file(s) successfully!")

        # Process Document Button
        process_btn = st.button("🚀 Process All Documents", width='stretch')

        if process_btn:
            with st.spinner("Processing files..."):
                start_time = time.time()

                all_documents = []
                file_types = []

                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    for idx, file_path in enumerate(uploaded_file_paths):
                        # Update progress
                        progress = (idx + 1) / len(uploaded_file_paths)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing {idx + 1}/{len(uploaded_file_paths)}: {os.path.basename(file_path)}")

                        # Load documents
                        result = load_documents_fastest(
                            file_path,
                            use_cache=True,  # Enable caching
                            max_workers=2  # Reduced workers for lighter processing
                        )


                        all_documents.extend(result['documents'])
                        file_types.append(result['file_type'])

                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()

                    # Build vector database
                    with st.spinner("Building vector database..."):
                        st.session_state.vector_db = build_vector_db(
                            all_documents,
                            db_path="./database"
                        )

                    # Create agent with toggle for Self-RAG
                    with st.spinner("Initializing agent..."):
                        st.session_state.agent = FinancialRAGAgent(
                            st.session_state.vector_db,
                        )

                    processing_time = time.time() - start_time

                    st.success(f"Processed {len(uploaded_file_paths)} files ({len(all_documents)} documents) in {processing_time:.2f}s!")

                except Exception as e:
                    st.error(f"Error: {str(e)}")

    st.divider()
    # Settings
    with st.expander("⚙️ Settings"):
        # RAG Configuration
        st.subheader("RAG Configuration")
        use_self_rag = st.toggle("Enable Self-RAG", value=True, help="Higher accuracy but slower")

        # RAG parameters
        relevance_threshold = st.slider(
            "Relevance Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.1,
            help="Higher = stricter filtering"
        )

        # Number of documents to retrieve
        num_docs = st.slider(
            "Number of Documents",
            min_value=3,
            max_value=10,
            value=3,  # Reduced default value for faster processing
            help="More docs = better coverage"
        )

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

        # clear cache button and size display
        cache_path = Path("data/cache")
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("Clear Cache", width='stretch'):
                clear_cache()
                st.success("✅ Cleared!")
                st.rerun()
        with col2:
            if cache_path.exists():
                cache_size = sum(f.stat().st_size for f in cache_path.glob("*.pkl")) / (1024 * 1024)
                st.caption(f"{cache_size:.1f} MB cached")

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
    with st.chat_message("user", avatar="./images/user_icon.png"):
        st.write(chat.get('question'))

    with st.chat_message("assistant", avatar="./images/chatbots_icon.png"):
        answer = chat.get('answer')
        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

        # Display chart if available
        if chat.get('chart'):
            chart_data = chat['chart']
            
            if chart_data.get('success'):
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                
                # Display chart
                chart_json = chart_data.get('chart')
                if chart_json:
                    try:
                        fig = go.Figure(json.loads(chart_json))
                        st.plotly_chart(fig, width='stretch', key=f"chart_{i}")
                        print(f"✅ Chart: {chart_data.get('title', 'Visualization')}")
                    except Exception as e:
                        st.error(f"⚠️ Error rendering chart: {e}")
                else:
                    st.warning("⚠️ No chart data available")
                
                # Display data table if enabled
                if st.session_state.get('show_data_table', True):
                    data_preview = chart_data.get('data_preview')
                    if data_preview:
                        with st.expander("📊 View Data Table"):
                            try:
                                df = pd.DataFrame(data_preview)
                                st.dataframe(df, width='stretch')
                            except Exception as e:
                                st.error(f"⚠️ Error displaying table: {e}")
                
                st.markdown('</div>', unsafe_allow_html=True)
            elif chart_data.get('error'):
                st.info(f"📊 Chart unavailable: {chart_data.get('error')}")

        # Display metadata
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            source_pages = chat.get('source_pages', [])
            if source_pages:
                st.caption(f"📄 Pages: {', '.join(source_pages)}")
            else:
                st.caption("📄 Pages: N/A")  # Fallback if no pages are available
        with col2:
            st.caption(f"🎯 Confidence: {chat['confidence'] if chat.get('confidence') else 'N/A'}")
        with col3:
            st.caption(f"📊 Docs: {chat['relevant_docs_count'] if chat.get('relevant_docs_count') else 0}")
        with col4:
            if chat.get('chart'):
                st.caption("📈 Visualization included")

        # Verification result
        if chat.get('verification'):
            with st.expander("🔍 View Verification"):
                verification_result = chat['verification'].get('notes', "No verification notes available.")
                st.write(verification_result)

# Input area
user_question = st.chat_input("Type your question here...")

if len(st.session_state.chat_history) == 0:
    # Main area
    st.title('Hi there! 👋 Welcome to **Finance Insight Lite**. Upload a financial report to get started.')

    # Chat interface
    st.subheader("💬 Ask Questions")

    # Sample questions
    sample_questions = [
        "What is the net income?",
        "What is the free cash flow?",
        "What is the gearing ratio?",
        "How much was the dividend declared?",
        "Draw a pie chart of expense breakdown",
        "Visualize the profit margins"
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
            result = st.session_state.agent.process_query(question, chat_history=st.session_state.chat_history)
            st.write("DEBUG result:", result)

        # Save to history
        chat_entry = {
            'question': question,
            'answer': result.get('answer'),
            'source_pages': result.get('source_pages'),
            'confidence': result.get('confidence'),
            'verification': result.get('verification'),
            'relevant_docs_count': result.get('relevant_docs_count'),
            'chart': result.get('chart')  # Use chart from result
        }
        st.session_state.chat_history.append(chat_entry)
        st.rerun()

# Process regular chat input
if user_question:
    if st.session_state.agent is None:
        st.warning("⚠️ Please upload and process a PDF document first!")
        st.stop()

    # Process query
    with st.spinner("🤔 Thinking..."):
        result = st.session_state.agent.process_query(user_question, chat_history=st.session_state.chat_history)
        st.write("DEBUG result:", result)

    # Save to history
    chat_entry = {
        'question': user_question,
        'answer': result.get('answer'),
        'source_pages': result.get('source_pages'),
        'confidence': result.get('confidence'),
        'verification': result.get('verification'),
        'relevant_docs_count': result.get('relevant_docs_count'),
        'chart': result.get('chart')  # Use chart from result
    }
    st.session_state.chat_history.append(chat_entry)
    st.rerun()
