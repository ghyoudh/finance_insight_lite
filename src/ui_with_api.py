import streamlit as st
import requests
from pathlib import Path
import time
import sys
import os
from dotenv import load_dotenv

# Load environment variables
project_root = Path(__file__).parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, "finance_insight_lite")
sys.path.append(current_dir)
sys.path.append(package_path)

# API Configuration
API_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="Finance Insight Lite",
    page_icon=str(Path(__file__).parent.parent / "images" / "logo3.png"),
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        color: #2E5D28;
        text-align: center;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #789575;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #0D3908;
        color: #E1E1E1;
    }
    div.stButton > button {
        background-color: #2E5D28;
        color: #E1E1E1;
        border-radius: 50px;
        border: 1px solid #0D3908;
        padding: 10px 25px;
        transition: all 0.3s ease;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #0D3908;
        border-color: #2E5D28;
        color: white;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2E5D28;
    }
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-online {
        background-color: #28a745;
    }
    .status-offline {
        background-color: #dc3545;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pending_question' not in st.session_state:
    st.session_state.pending_question = None
if 'api_status' not in st.session_state:
    st.session_state.api_status = None
if 'use_self_rag' not in st.session_state:
    st.session_state.use_self_rag = True
if 'retry_count' not in st.session_state:
    st.session_state.retry_count = 0


# ============================================================================
# API FUNCTIONS
# ============================================================================

def check_api_health():
    """Check if API is running and healthy"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except Exception as e:
        return False, None


def upload_document(file):
    """Upload PDF to API"""
    try:
        files = {"file": (file.name, file, "application/pdf")}
        response = requests.post(
            f"{API_URL}/upload",
            files=files,
            timeout=300,
            params={"use_self_rag": st.session_state.use_self_rag}
        )
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json().get("detail", "Unknown error")
    except Exception as e:
        return False, str(e)


def query_api(question: str, use_self_rag: bool = True):
    """Send query to API"""
    try:
        payload = {
            "question": question,
            "use_self_rag": use_self_rag
        }
        response = requests.post(f"{API_URL}/query", json=payload, timeout=120)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json().get("detail", "Unknown error")
    except Exception as e:
        return False, str(e)


def get_document_info():
    """Get info about loaded document"""
    try:
        response = requests.get(f"{API_URL}/document/info", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except:
        return False, None


def get_sample_questions():
    """Get sample questions from API"""
    try:
        response = requests.get(f"{API_URL}/sample-questions", timeout=5)
        if response.status_code == 200:
            return response.json()["questions"]
        return []
    except:
        return []


# ============================================================================
# UI COMPONENTS
# ============================================================================

def show_api_status():
    """Show API connection status"""
    is_online, health_data = check_api_health()
    st.session_state.api_status = is_online
    
    if is_online:
        st.session_state.retry_count = 0  # Reset retry count on success
        st.sidebar.markdown(
            f'<div style="display: flex; align-items: center;">'
            f'<span class="status-indicator status-online"></span>'
            f'<span style="color: #28a745;">API Connected</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        
        if health_data:
            doc_loaded = health_data.get('document_loaded', False)
            if doc_loaded:
                st.sidebar.success("✅ Document Loaded")
            else:
                st.sidebar.warning("📄 No Document Loaded")
    else:
        st.sidebar.markdown(
            f'<div style="display: flex; align-items: center;">'
            f'<span class="status-indicator status-offline"></span>'
            f'<span style="color: #dc3545;">API Offline</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.sidebar.error("⚠️ Cannot connect to API")


# ============================================================================
# MAIN APP
# ============================================================================

# Sidebar
with st.sidebar:
    # Try to load logo if exists
    logo_path = Path(__file__).parent.parent / "images" / "logo.png"
    if logo_path.exists():
        st.image(str(logo_path), width=100)
    
    st.title("Finance Insight Lite")
    
    # API Status
    show_api_status()
    
    st.divider()
    
    # Document Upload
    st.subheader("📄 Upload Document")
    uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
    
    if uploaded_file:
        if st.button("🔄 Process Document", use_container_width=True):
            with st.spinner("Processing PDF..."):
                success, result = upload_document(uploaded_file)
                
                if success:
                    pages = result.get('pages', 'N/A') if isinstance(result, dict) else 'N/A'
                    st.success(f"✅ Processed {pages} pages!")
                    st.json(result)
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"❌ Error: {result}")
    
    st.divider()
    
    # Document Info
    st.subheader("📊 Document Info")
    success, doc_info = get_document_info()
    if success and doc_info:
        st.markdown(f"""
        <div class="metric-card">
            <strong>File:</strong> {doc_info.get('filename', 'N/A')}<br>
            <strong>Pages:</strong> {doc_info.get('pages', 'N/A')}<br>
            <strong>Chunks:</strong> {doc_info.get('chunks', 'N/A')}<br>
            <strong>Processed:</strong> {doc_info.get('processed_at', 'N/A')[:10]}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No document loaded yet")
    
    st.divider()
    
    # Settings
    with st.expander("⚙️ Settings"):
        use_self_rag = st.toggle(
            "Enable Self-RAG",
            value=True,
            help="Higher accuracy but slower"
        )
        st.session_state.use_self_rag = use_self_rag
    
    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Main area
st.markdown('<p class="main-header">💼 Finance Insight Lite</p>', unsafe_allow_html=True)

# Check API status - WITH RETRY LOGIC instead of st.stop()
if not st.session_state.api_status:
    st.markdown("""
    <div class="warning-box">
        <h3>⚠️ API Connection Issue</h3>
        <p>The FastAPI backend is not responding. This usually means:</p>
        <ul>
            <li>The API is still starting up (can take 30-60 seconds on first launch)</li>
            <li>The API process was not started</li>
            <li>There's a port conflict on port 8000</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info("💡 **If you just started the app**, please wait 30-60 seconds and click the retry button.")
    with col2:
        if st.button("🔄 Retry Connection", use_container_width=True):
            st.session_state.retry_count += 1
            st.rerun()
    
    # Show startup instructions
    with st.expander("📖 How to start the API"):
        st.markdown("""
        **Option 1: Using the startup script (Recommended)**
        ```bash
        python src/run_app.py
        ```
        
        **Option 2: Manual startup**
        ```bash
        # Terminal 1 - Start API
        python main.py
        
        # Terminal 2 - Start UI
        streamlit run src/ui_with_api.py
        ```
        
        **Option 3: Using uvicorn directly**
        ```bash
        uvicorn main:app --host 127.0.0.1 --port 8000 --reload
        ```
        """)
    
    # Show what features are disabled
    st.warning("⚠️ All features are disabled until API connection is established.")
    
    # Don't use st.stop() - just show a limited UI
    st.info("The page will automatically update when you click 'Retry Connection' above.")

else:
    # API IS ONLINE - Show full functionality
    
    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat['question'])
        
        with st.chat_message("assistant"):
            st.markdown(f'<div class="answer-box">{chat["answer"]}</div>', unsafe_allow_html=True)
            
            # Metadata
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                pages = chat.get('source_pages', [])
                st.caption(f"📄 Pages: {', '.join(map(str, pages)) if pages else 'N/A'}")
            with col2:
                st.caption(f"🎯 Confidence: {chat.get('confidence', 'N/A')}")
            with col3:
                st.caption(f"📊 Docs: {chat.get('relevant_docs_count', 'N/A')}")
            with col4:
                st.caption(f"⏱️ {chat.get('processing_time_ms', 'N/A')}ms")
            
            # Verification
            if chat.get('verification'):
                with st.expander("🔍 View Verification"):
                    st.write(chat['verification'].get('verification', 'N/A'))

    # Chat input
    user_question = st.chat_input("Type your question here...")

    # Sample questions (only show if no chat history AND document is loaded)
    success_info, doc_info = get_document_info()
    if len(st.session_state.chat_history) == 0 and success_info and doc_info:
        sample_questions = get_sample_questions()
        
        if sample_questions:
            st.subheader("💡 Sample Questions")
            
            def set_pending_question(q):
                st.session_state.pending_question = q
            
            cols = st.columns(2)
            for i, q in enumerate(sample_questions[:6]):  # Show first 6
                with cols[i % 2]:
                    st.button(q, key=f"sample_{i}", on_click=set_pending_question, args=(q,))

    # Process pending question
    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None
        
        with st.spinner("🤔 Thinking..."):
            success, result = query_api(
                question,
                use_self_rag=st.session_state.use_self_rag
            )
        
        if success:
            if isinstance(result, dict):
                st.session_state.chat_history.append({
                    'question': question,
                    **result
                })
                st.rerun()
            else:
                st.error(f"❌ Invalid response format")
        else:
            st.error(f"❌ Error: {result}")

    # Process regular input
    if user_question:
        with st.spinner("🤔 Thinking..."):
            success, result = query_api(
                user_question,
                use_self_rag=st.session_state.use_self_rag
            )
        
        if success:
            if isinstance(result, dict):
                st.session_state.chat_history.append({
                    'question': user_question,
                    **result
                })
                st.rerun()
            else:
                st.error(f"❌ Invalid response format")
        else:
            st.error(f"❌ Error: {result}")

# Footer
st.divider()
st.caption("💼 Finance Insight Lite - Powered by Finance Insight Lite API")