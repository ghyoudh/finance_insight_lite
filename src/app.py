import pathlib
import re
from finance_insight_lite.modules.processor import load_documents_fastest, pdf_to_documents
from finance_insight_lite.modules.verctor_store import build_vector_db
from finance_insight_lite.modules.rag_agent import create_advanced_rag_agent
import pandas as pd
import subprocess
import time
import sys
from pathlib import Path
from dotenv import load_dotenv
import os
from pathlib import Path

# Load environment variables from .env file
# Get the project root directory (parent of src/)
project_root = Path(__file__).parent.parent
env_path = project_root / '.env'

# Load the .env file
load_dotenv(dotenv_path=env_path)

# Debug: Check if the key is loaded
groq_key = os.getenv('GROQ_API_KEY')
if groq_key:
    print(f"✓ API Key loaded: {groq_key[:10]}...")
else:
    print("✗ API Key NOT loaded!")
    print(f"Looking for .env at: {env_path}")
    print(f".env exists: {env_path.exists()}")


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
   # Paths
    # Always resolve main.py relative to project root
    project_root = Path(__file__).resolve().parent.parent
    main_py = project_root / "main.py"
    streamlit_ui = project_root / "src" / "ui_with_api.py"

    # Start FastAPI (main.py)
    print("🚀 Starting FastAPI backend (main.py)...")
    api_process = subprocess.Popen([sys.executable, str(main_py)])
    time.sleep(3)  # Wait a bit to ensure backend is up

    # Start Streamlit UI
    print("🖥️  Starting Streamlit UI (ui_with_api.py)...")
    ui_process = subprocess.Popen(["streamlit", "run", str(streamlit_ui)])

    print("✅ Both FastAPI and Streamlit UI are running.")
    print("- FastAPI:   http://localhost:8000/docs")
    print("- Streamlit: http://localhost:8501/")

    try:
        # Wait for both processes
        api_process.wait()
        ui_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        api_process.terminate()
        ui_process.terminate()
        print("✅ All processes terminated.")
    
    # (Optional) Keep the old test code for reference
    # To use the old test code, comment out the above and uncomment below
    # ...existing code...