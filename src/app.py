import pathlib
import re
from finance_insight_lite.modules.processor import load_documents_fastest, pdf_to_documents
from finance_insight_lite.modules.verctor_store import build_vector_db
from finance_insight_lite.modules.rag_agent import OptimizedFinancialRAGAgent
import pandas as pd
import subprocess
import time
import sys
from pathlib import Path
from dotenv import load_dotenv
import os
from pathlib import Path
import uvicorn
import requests

# Load environment variables from .env file
project_root = Path(__file__).parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

# Debug: Check if the key is loaded
groq_key = os.getenv('GROQ_API_KEY')
if groq_key:
    print(f"✓ API Key loaded: {groq_key[:10]}...")
else:
    print("✗ API Key NOT loaded!")
    print(f"Looking for .env at: {env_path}")
    print(f".env exists: {env_path.exists()}")


def wait_for_api(url="http://localhost:8000/health", timeout=60, check_interval=2):
    """
    Wait for the API to be ready
    
    Args:
        url: Health check URL
        timeout: Maximum time to wait in seconds
        check_interval: Time between checks in seconds
        
    Returns:
        bool: True if API is ready, False if timeout
    """
    print(f"⏳ Waiting for API to be ready (timeout: {timeout}s)...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print("✅ API is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        elapsed = int(time.time() - start_time)
        print(f"   Waiting... ({elapsed}s)", end='\r')
        time.sleep(check_interval)
    
    print(f"\n❌ API did not become ready within {timeout} seconds")
    return False


def load_all_files_from_folder(folder_path):
    """
    Load all PDF and Excel files from a folder
    
    Args:
        folder_path: Path to the folder containing files
        
    Returns:
        dict: Contains 'documents' list and 'relevant_docs_count'
    """
    all_documents = []
    folder = Path(folder_path)
    
    # Search for all PDF and Excel files only
    files = list(folder.glob("*.pdf")) + list(folder.glob("*.xlsx")) + \
            list(folder.glob("*.xls"))
    
    print(f"Found {len(files)} files to process")
    print("=" * 60)
    
    for file in files:
        print(f"📄 Reading: {file.name}...")
        try:
            if file.suffix == '.pdf':
                # Use the load_documents_fastest function for PDF
                result = load_documents_fastest(str(file))
                all_documents.extend(result['documents'])
                print(f"   ✓ Loaded {result['relevant_docs_count']} pages")
                
            elif file.suffix in ['.xlsx', '.xls']:
                # Use the load_documents_fastest function for Excel
                result = load_documents_fastest(str(file))
                all_documents.extend(result['documents'])
                print(f"   ✓ Loaded {result['relevant_docs_count']} sheets")
                
        except Exception as e:
            print(f"   ❌ Error reading {file.name}: {e}")
    
    print("=" * 60)
    print(f"✅ Total documents loaded: {len(all_documents)}")
    
    return {
        'documents': all_documents,
        'relevant_docs_count': len(all_documents)
    }


if __name__ == "__main__":
    # Paths - always resolve relative to project root
    project_root = Path(__file__).resolve().parent.parent
    main_py = project_root / "main.py"
    streamlit_ui = project_root / "src" / "ui_with_api.py"

    print("=" * 70)
    print("🚀 Starting Finance Insight Lite Application")
    print("=" * 70)

    # Verify files exist
    if not main_py.exists():
        print(f"❌ Error: main.py not found at {main_py}")
        sys.exit(1)
    if not streamlit_ui.exists():
        print(f"❌ Error: ui_with_api.py not found at {streamlit_ui}")
        sys.exit(1)

    # Start FastAPI backend
    print("\n📡 Step 1: Starting FastAPI backend...")
    print(f"   Location: {main_py}")
    print(f"   Working directory: {project_root}")
    
    api_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000", "--reload"],
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for API to be ready
    if not wait_for_api(timeout=60, check_interval=2):
        print("❌ FastAPI failed to start. Terminating...")
        api_process.terminate()
        sys.exit(1)

    # Start Streamlit UI
    print("\n🖥️  Step 2: Starting Streamlit UI...")
    print(f"   Location: {streamlit_ui}")
    
    ui_process = subprocess.Popen(
        ["streamlit", "run", str(streamlit_ui)],
        cwd=str(project_root)
    )
    
    # Give Streamlit a moment to start
    time.sleep(3)

    print("\n" + "=" * 70)
    print("✅ Application is running!")
    print("=" * 70)
    print("📊 Access points:")
    print("   - FastAPI Docs:  http://localhost:8000/docs")
    print("   - Streamlit UI:  http://localhost:8501/")
    print("\n💡 Press Ctrl+C to stop both servers")
    print("=" * 70)

    try:
        # Wait for both processes
        api_process.wait()
        ui_process.wait()
    except KeyboardInterrupt:
        print("\n" + "=" * 70)
        print("🛑 Shutting down gracefully...")
        print("=" * 70)
        
        print("   Stopping Streamlit...")
        ui_process.terminate()
        ui_process.wait(timeout=5)
        
        print("   Stopping FastAPI...")
        api_process.terminate()
        api_process.wait(timeout=5)
        
        print("✅ All processes terminated successfully.")
        print("=" * 70)